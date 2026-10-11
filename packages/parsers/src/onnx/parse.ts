import type { Model, ModelLayer } from "../types";
import { ChunkBuffer, drain, type ModelStream } from "../common/chunks";
import { decodeAttrs } from "./attrs";
import { computeCost } from "./flops";
import { opMeta } from "./opmeta";
import { modelProtoType } from "./proto";
import {
  DT_BYTES,
  DT_NAME,
  num,
  numArr,
  type RawModel,
  type RawTensor,
} from "./raw";
import {
  foldConst,
  formatShape,
  inferShapes,
  prod,
  seedConstMap,
  seedShapeMap,
  tensorValues,
  type Shape,
} from "./shapes";
import { tensorParams, weightStat } from "./stats";

export type ParseOptions = {
  name?: string;
  /** Concrete shape for the first graph input; fills its dynamic dims (e.g. [1, 3, 640, 640]). */
  inputShape?: number[];
};

/**
 * Parse a serialized ONNX model (the bytes of a .onnx file) into the normalized
 * Model object the core graph renderer consumes. Pure function, no I/O.
 */
export function parseOnnx(input: ArrayBuffer | Uint8Array, opts: ParseOptions = {}): Model {
  return drain(onnxStream(input, opts, 0));
}

/**
 * Streaming form of parseOnnx: yields the header as soon as the protobuf is
 * decoded, then layers (+ their data-flow edges) every `batchSize` nodes in
 * topological order, then a final chunk with any fix-ups. Returns exactly what
 * parseOnnx returns. batchSize 0 yields nothing (the one-shot path).
 */
export function* onnxStream(input: ArrayBuffer | Uint8Array, opts: ParseOptions = {}, batchSize = 256): ModelStream {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const decoded = modelProtoType().decode(bytes) as unknown as RawModel;
  const graph = decoded.graph;
  if (!graph) throw new Error("ONNX model has no graph.");
  const chunks = new ChunkBuffer(batchSize);

  // Initializers (weights/constants), indexed by tensor name.
  const initByName = new Map<string, RawTensor>();
  for (const init of graph.initializer ?? []) {
    if (init.name) initByName.set(init.name, init);
  }

  const shapeMap = seedShapeMap(graph);
  // Declared input shapes, before any dynamic dims get resolved for counting.
  const declaredShape = new Map((graph.input ?? []).map((vi) => [vi.name ?? "", shapeMap.get(vi.name ?? "")]));
  const constMap = seedConstMap(graph);
  const dtypeByName = new Map<string, string>();
  for (const init of graph.initializer ?? []) {
    if (init.name) dtypeByName.set(init.name, DT_NAME[init.data_type ?? 1] ?? "float32");
  }
  const setViDtype = (vis = graph.input ?? []) => {
    for (const vi of vis) {
      const et = vi.type?.tensor_type?.elem_type;
      if (vi.name && et != null) dtypeByName.set(vi.name, DT_NAME[et] ?? "float32");
    }
  };
  setViDtype(graph.input);
  setViDtype(graph.value_info);
  setViDtype(graph.output);

  const outputNames = new Set((graph.output ?? []).map((o) => o.name).filter(Boolean) as string[]);

  const opset = (decoded.opset_import ?? []).find((o) => !o.domain || o.domain === "ai.onnx");
  const firstInput = (graph.input ?? []).find((vi) => vi.name && !initByName.has(vi.name));
  const notes = resolveDynamicInput(firstInput?.name, shapeMap, opts.inputShape, decoded.metadata_props);
  const firstOutput = (graph.output ?? [])[0];
  const header = {
    name: opts.name || graph.name || decoded.producer_name || "model",
    format: "ONNX",
    framework: decoded.producer_name || "unknown",
    opset: opset ? num(opset.version) : 0,
    sizeBytes: bytes.byteLength,
    producer: decoded.producer_name || undefined,
    irVersion: decoded.ir_version != null ? num(decoded.ir_version) : undefined,
  };
  const nodes = graph.node ?? [];
  if (chunks.enabled) {
    // Graph inputs carry declared shapes, so the input shape is known up front;
    // the output shape may need inference and arrives with the final chunk.
    chunks.setHeader({ ...header, inputShape: firstInput ? cleanShape(shapeMap.get(firstInput.name!)) : [] });
    yield chunks.take(0);
  }

  const layers: ModelLayer[] = [];
  const producerOf = new Map<string, number>(); // tensor name → layer id
  // Tensors computable without the input's values: initializers, Constant
  // outputs, Shape outputs, and anything derived only from those.
  const staticNames = new Set<string>(initByName.keys());
  let nextId = 0;

  // 1) Synthetic input layers for real graph inputs (initializers excluded).
  for (const vi of graph.input ?? []) {
    if (!vi.name || initByName.has(vi.name)) continue;
    const s = shapeMap.get(vi.name);
    const dt = dtypeByName.get(vi.name) ?? "float32";
    const declared = declaredShape.get(vi.name);
    const id = nextId++;
    producerOf.set(vi.name, id);
    layers.push({
      id,
      name: vi.name,
      type: "Tensor",
      op: "Input",
      shape: formatShape(s),
      dt,
      params: 0,
      flops: 0,
      macs: 0,
      mem: memBytes(s, dt),
      group: "input",
      ins: [],
      outs: [{ n: vi.name, s }],
      // Keep the file's own shape when dynamic dims were filled in for counting.
      attr: declared && formatShape(declared) !== formatShape(s) ? { declaredShape: formatShape(declared) } : {},
      w: null,
      math: "model input",
      insight: "Graph input tensor. Apply preprocessing (resize/normalize) upstream.",
      qSens: 0.1,
      compIssues: [],
    });
    chunks.layer(layers[layers.length - 1]);
  }

  // 2) One layer per node, in topological (file) order.
  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni];
    const op = node.op_type ?? "Unknown";
    const attrs = decodeAttrs(node);
    const inputs = node.input ?? [];
    const outputs = node.output ?? [];

    const inShapes = inputs.map((n) => shapeMap.get(n));
    const outName = outputs[0];

    // Infer output shapes value_info didn't already supply. Exporters often
    // omit intermediate value_info, so this is what keeps downstream shapes
    // (and therefore FLOPs) from collapsing to unknown.
    const inConsts = inputs.map((n) => constMap.get(n));
    if (op === "Constant") {
      const t = node.attribute?.find((a) => a.name === "value")?.t;
      if (t && outName) {
        shapeMap.set(outName, numArr(t.dims));
        const v = tensorValues(t);
        if (v) constMap.set(outName, v);
      }
    } else {
      const inferred = inferShapes(op, attrs, inShapes, inConsts, outputs.length);
      outputs.forEach((o, i) => {
        const inf = inferred[i];
        if (!o || !inf) return;
        const declared = shapeMap.get(o);
        // Declared shapes win, except where they hold a symbolic dim that
        // inference resolved (e.g. "height" once the input size is known).
        if (!declared) shapeMap.set(o, inf);
        else if (declared.length === inf.length && declared.some((d, j) => d < 0 && inf[j] >= 0)) {
          shapeMap.set(o, declared.map((d, j) => (d < 0 ? inf[j] : d)));
        }
      });
      const folded = outName ? foldConst(op, attrs, inShapes, inConsts) : undefined;
      if (folded && outName) constMap.set(outName, folded);
    }
    const outShape: Shape | undefined = outName ? shapeMap.get(outName) : undefined;

    // Attribute weight initializers to this node; pick the largest as "the weight".
    let params = 0;
    let weightTensor: RawTensor | null = null;
    let weightParams = 0;
    for (const inName of inputs) {
      const init = initByName.get(inName);
      if (!init) continue;
      const p = tensorParams(init);
      params += p;
      if (p > weightParams) {
        weightParams = p;
        weightTensor = init;
      }
    }

    const { flops, macs } = computeCost(op, attrs, inShapes, outShape);
    const meta = opMeta(op);
    const dt =
      (outName && dtypeByName.get(outName)) ||
      (inputs[0] && dtypeByName.get(inputs[0])) ||
      "float32";

    const id = nextId++;
    for (const o of outputs) if (o) producerOf.set(o, id);
    const realInputs = inputs.filter(Boolean);
    const constFolded =
      op === "Constant" || op === "Shape" || op === "Size" ||
      (realInputs.length > 0 && realInputs.every((n) => staticNames.has(n)));
    if (constFolded) for (const o of outputs) if (o) staticNames.add(o);

    const w = weightTensor ? weightStat(weightTensor) : null;

    layers.push({
      id,
      name: node.name || outName || `${op}_${id}`,
      type: op,
      op,
      shape: formatShape(outShape),
      dt,
      params,
      flops,
      macs,
      mem: memBytes(outShape, dt),
      group: outName && outputNames.has(outName) ? "output" : meta.group,
      ins: inputs
        .filter((n) => !initByName.has(n))
        .map((n) => ({ n, s: shapeMap.get(n) })),
      outs: outputs.map((n) => ({ n, s: shapeMap.get(n) })),
      attr: attrs,
      w,
      math: meta.math,
      insight: meta.insight,
      qSens: meta.qSens,
      compIssues: [],
      ...(constFolded ? { constFolded: true } : {}),
    });

    if (chunks.enabled) {
      // Stream this node's incoming data-flow edges now (producers precede
      // consumers in a topologically sorted graph). Step 3 stays the source
      // of truth; finish() emits anything this pass couldn't resolve.
      chunks.layer(layers[layers.length - 1]);
      if (outName) {
        for (const inName of inputs) {
          if (initByName.has(inName)) continue;
          const producerId = producerOf.get(inName);
          if (producerId != null && producerId !== id) chunks.edge(producerId, id);
        }
      }
      if (chunks.full) yield chunks.take((ni + 1) / (nodes.length + 1));
    }
  }

  // 3) Edges from data-flow: producer layer → consumer layer (weights excluded).
  const edgeSet = new Set<string>();
  const edges: [number, number][] = [];
  for (const node of nodes) {
    const consumerId = producerOf.get((node.output ?? [])[0] ?? "");
    if (consumerId == null) continue;
    for (const inName of node.input ?? []) {
      if (initByName.has(inName)) continue;
      const producerId = producerOf.get(inName);
      if (producerId == null || producerId === consumerId) continue;
      const key = `${producerId}->${consumerId}`;
      if (edgeSet.has(key)) continue;
      edgeSet.add(key);
      edges.push([producerId, consumerId]);
    }
  }

  const model: Model = {
    name: header.name,
    format: header.format,
    framework: header.framework,
    opset: header.opset,
    sizeBytes: header.sizeBytes,
    inputShape: firstInput ? cleanShape(shapeMap.get(firstInput.name!)) : [],
    outputShape: firstOutput ? cleanShape(shapeMap.get(firstOutput.name!)) : [],
    layers,
    edges,
    producer: header.producer,
    irVersion: header.irVersion,
    ...(notes.length ? { notes } : {}),
  };
  if (chunks.enabled) yield chunks.finish(model);
  return model;
}

/**
 * Fill dynamic non-batch dims of the first input so compute can be counted.
 * Sources, in order: the caller's inputShape, then an `imgsz` entry in the
 * model metadata (Ultralytics writes one). Mutates shapeMap; returns notes
 * describing what was assumed or what stays unknown.
 */
function resolveDynamicInput(
  name: string | undefined,
  shapeMap: Map<string, Shape>,
  override: number[] | undefined,
  meta: { key?: string; value?: string }[] | undefined,
): string[] {
  const s = name ? shapeMap.get(name) : undefined;
  if (!name || !s) return [];
  const dynIdx = s.map((d, i) => (d < 0 && i > 0 ? i : -1)).filter((i) => i >= 0);
  if (!dynIdx.length) return [];
  if (override && override.length === s.length) {
    shapeMap.set(name, s.map((d, i) => (d < 0 && i > 0 ? override[i] : d)));
    return [`Input "${name}" has dynamic dims; using the requested shape ${override.join("×")}.`];
  }
  const imgsz = meta?.find((p) => p.key === "imgsz")?.value?.match(/\d+/g)?.map(Number);
  if (imgsz?.length && s.length === 4 && dynIdx.every((i) => i >= 2)) {
    const [h, w] = imgsz.length === 1 ? [imgsz[0], imgsz[0]] : imgsz;
    shapeMap.set(name, [s[0], s[1], h, w]);
    return [`Input "${name}" has a dynamic height/width; FLOPs assume the ${h}×${w} recorded in the model metadata (imgsz).`];
  }
  return [
    `Input "${name}" has dynamic dims (${formatShape(s)}), so FLOPs and activation memory are counted as if each unknown dim were 1 and are far too low. Export with a fixed input size for real estimates.`,
  ];
}

function memBytes(shape: Shape | undefined, dt: string): number {
  if (!shape) return 0;
  const dtNum = Object.entries(DT_NAME).find(([, v]) => v === dt)?.[0];
  const bpe = dtNum ? DT_BYTES[Number(dtNum)] ?? 4 : 4;
  return prod(shape) * bpe;
}

function cleanShape(s: Shape | undefined): number[] {
  return (s ?? []).map((d) => (d < 0 ? 0 : d));
}

export { numArr };
