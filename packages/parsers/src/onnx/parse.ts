import type { Model, ModelLayer } from "../types";
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
import { formatShape, inferOutputShape, prod, seedShapeMap, type Shape } from "./shapes";
import { tensorParams, weightStat } from "./stats";

export type ParseOptions = { name?: string };

/**
 * Parse a serialized ONNX model (the bytes of a .onnx file) into the normalized
 * Model object the core graph renderer consumes. Pure function, no I/O.
 */
export function parseOnnx(input: ArrayBuffer | Uint8Array, opts: ParseOptions = {}): Model {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const decoded = modelProtoType().decode(bytes) as unknown as RawModel;
  const graph = decoded.graph;
  if (!graph) throw new Error("ONNX model has no graph.");

  // Initializers (weights/constants) — indexed by tensor name.
  const initByName = new Map<string, RawTensor>();
  for (const init of graph.initializer ?? []) {
    if (init.name) initByName.set(init.name, init);
  }

  const shapeMap = seedShapeMap(graph);
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

  const layers: ModelLayer[] = [];
  const producerOf = new Map<string, number>(); // tensor name → layer id
  let nextId = 0;

  // 1) Synthetic input layers for real graph inputs (initializers excluded).
  for (const vi of graph.input ?? []) {
    if (!vi.name || initByName.has(vi.name)) continue;
    const s = shapeMap.get(vi.name);
    const dt = dtypeByName.get(vi.name) ?? "float32";
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
      attr: {},
      w: null,
      math: "model input",
      insight: "Graph input tensor. Apply preprocessing (resize/normalize) upstream.",
      qSens: 0.1,
      compIssues: [],
    });
  }

  // 2) One layer per node, in topological (file) order.
  for (const node of graph.node ?? []) {
    const op = node.op_type ?? "Unknown";
    const attrs = decodeAttrs(node);
    const inputs = node.input ?? [];
    const outputs = node.output ?? [];

    const inShapes = inputs.map((n) => shapeMap.get(n));
    const outName = outputs[0];

    // Infer the primary output shape if value_info didn't already supply it.
    let outShape: Shape | undefined = outName ? shapeMap.get(outName) : undefined;
    if (!outShape) {
      const inferred = inferOutputShape(op, attrs, inShapes);
      if (inferred) {
        outShape = inferred;
        if (outName) shapeMap.set(outName, inferred);
      }
    }

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
    });
  }

  // 3) Edges from data-flow: producer layer → consumer layer (weights excluded).
  const edgeSet = new Set<string>();
  const edges: [number, number][] = [];
  for (const node of graph.node ?? []) {
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

  const opset = (decoded.opset_import ?? []).find((o) => !o.domain || o.domain === "ai.onnx");
  const firstInput = (graph.input ?? []).find((vi) => vi.name && !initByName.has(vi.name));
  const firstOutput = (graph.output ?? [])[0];

  return {
    name: opts.name || graph.name || decoded.producer_name || "model",
    format: "ONNX",
    framework: decoded.producer_name || "unknown",
    opset: opset ? num(opset.version) : 0,
    sizeBytes: bytes.byteLength,
    inputShape: firstInput ? cleanShape(shapeMap.get(firstInput.name!)) : [],
    outputShape: firstOutput ? cleanShape(shapeMap.get(firstOutput.name!)) : [],
    layers,
    edges,
    producer: decoded.producer_name || undefined,
    irVersion: decoded.ir_version != null ? num(decoded.ir_version) : undefined,
  };
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
