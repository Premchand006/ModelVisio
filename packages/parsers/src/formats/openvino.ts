import type { Model, ModelLayer } from "../types";
import { child, children, parseXml, type XmlElement } from "../common/xml";
import {
  DT_SIZE, EdgeSet, NO_COST, cleanShape, elemCost, linearCost, makeLayer, numel, sampleBytes, weightStats,
  type Cost, type Shape, type WStat,
} from "../common/graphKit";
import { parseContainer } from "./container";

// OpenVINO IR (.xml topology + optional .bin weights). The XML lists
// <layers> with typed <input>/<output> ports (dims give every tensor shape)
// and <edges> wiring layer ports together. Weights are `Const` layers that
// point into the .bin by offset/size. As with ONNX initializers, Const nodes
// (and constant-only subgraphs such as fp16 → Convert decompression chains)
// are folded into the op that consumes them: their element counts become that
// op's params, so the graph shows compute, not hundreds of constants.
// The parser only ever receives the .xml; pass the .bin as `weights` to also
// get min/max/mean/std statistics.

type Port = { id: string; dims: Shape; precision: string; names?: string };
type IrLayer = {
  id: string;
  name: string;
  type: string;
  version: string;
  data: Record<string, string>;
  ins: Port[];
  outs: Port[];
  blobs: { kind: string; offset: number; size: number; precision?: string }[];
  precision?: string; // IR v7 layer-level precision
};
type IrEdge = { from: string; fromPort: string; to: string; toPort: string };

const PRECISION: Record<string, string> = {
  FP32: "float32", FP16: "float16", BF16: "bfloat16", FP64: "float64",
  I64: "int64", I32: "int32", I16: "int16", I8: "int8", I4: "int4",
  U64: "uint64", U32: "uint32", U16: "uint16", U8: "uint8", U4: "uint4", U1: "uint1",
  BOOL: "bool", BOOLEAN: "bool",
  F32: "float32", F16: "float16", F64: "float64", "F8E4M3": "float8e4m3", "F8E5M2": "float8e5m2",
};
const dtOf = (p: string | undefined) => (p ? PRECISION[p.toUpperCase()] ?? p.toLowerCase() : "float32");

// Element types that count as learned parameters (shape/axis tensors don't).
const PARAM_DTYPES = new Set(["float32", "float16", "bfloat16", "float64", "int8", "uint8", "int4", "uint4", "uint1", "float8e4m3", "float8e5m2"]);

const dims = (s: string | undefined): Shape =>
  !s || !s.trim() ? [] : s.split(",").map((d) => { const n = Number(d.trim()); return Number.isFinite(n) && n >= 0 ? n : -1; });

function readPorts(el: XmlElement | undefined): Port[] {
  return children(el, "port").map((p) => ({
    id: p.attrs.id ?? "0",
    dims: children(p, "dim").map((d) => { const n = Number(d.text.trim()); return Number.isFinite(n) && n >= 0 ? n : -1; }),
    precision: dtOf(p.attrs.precision),
    names: p.attrs.names,
  }));
}

/** True when the text/bytes look like an OpenVINO IR document (root <net> with <layers>). */
export function isOpenvinoXml(input: Uint8Array | string): boolean {
  const head = typeof input === "string" ? input.slice(0, 16384) : new TextDecoder().decode(input.subarray(0, 16384));
  return /<net[\s>]/.test(head) && /<layers[\s>]/.test(head);
}

const CANON: Record<string, string> = {
  Convolution: "Conv", GroupConvolution: "Conv", ConvolutionBackpropData: "ConvTranspose",
  GroupConvolutionBackpropData: "ConvTranspose", Deconvolution: "ConvTranspose",
  MatMul: "MatMul", FullyConnected: "Gemm", InnerProduct: "Gemm",
  Add: "Add", Multiply: "Mul", Subtract: "Sub", Divide: "Div", Eltwise: "Add",
  ReLU: "Relu", Relu: "Relu", Sigmoid: "Sigmoid", Tanh: "Tanh", TanH: "Tanh", Clamp: "Clip", Elu: "Elu",
  Gelu: "Gelu", Erf: "Erf", Power: "Pow", PReLU: "LeakyRelu",
  MaxPool: "MaxPool", AvgPool: "AveragePool", Pooling: "MaxPool",
  BatchNormInference: "BatchNormalization", MVN: "LayerNormalization",
  SoftMax: "Softmax", Softmax: "Softmax", LogSoftmax: "LogSoftmax",
  Concat: "Concat", Reshape: "Reshape", Squeeze: "Reshape", Unsqueeze: "Reshape", Flatten: "Flatten",
  Transpose: "Transpose", Permute: "Transpose",
};

function cost(l: IrLayer, canon: string, out: Shape | undefined, ins: Shape[]): Cost {
  const x = ins[0];
  const w = ins[1];
  const area = (s: Shape) => s.reduce((a, b) => a * Math.max(1, b), 1);
  switch (l.type) {
    case "Convolution":
      if (w && w.length >= 3) return linearCost(out, Math.max(1, w[1]) * area(w.slice(2)));
      break;
    case "GroupConvolution":
      if (w && w.length >= 4) return linearCost(out, Math.max(1, w[2]) * area(w.slice(3)));
      break;
    case "ConvolutionBackpropData":
      if (x && w && w.length >= 3) return linearCost(x, Math.max(1, w[1]) * area(w.slice(2)));
      break;
    case "GroupConvolutionBackpropData":
      if (x && w && w.length >= 4) return linearCost(x, Math.max(1, w[2]) * area(w.slice(3)));
      break;
    case "MatMul": {
      if (!x || !x.length) break;
      const ta = /^(true|1)$/i.test(l.data.transpose_a ?? "");
      const K = x.length === 1 ? x[0] : ta ? x[x.length - 2] : x[x.length - 1];
      return linearCost(out, Math.max(1, K));
    }
    case "MaxPool":
    case "AvgPool": {
      const k = dims(l.data.kernel);
      return elemCost(out, k.length ? area(k) : 1);
    }
    case "ReduceMean":
    case "ReduceMax":
    case "ReduceSum":
      return elemCost(x, 1);
  }
  switch (canon) {
    case "Add": case "Mul": case "Sub": case "Div": case "Relu": case "Sigmoid": case "Tanh": case "Clip":
    case "Elu": case "Gelu": case "Erf": case "Pow": case "LeakyRelu":
      return elemCost(out, 1);
    case "BatchNormalization": case "LayerNormalization":
      return elemCost(out, 2);
    case "Softmax": case "LogSoftmax":
      return elemCost(out, 3);
  }
  if (/^(Swish|HSwish|HSigmoid|Mish|SoftPlus|Exp|Log|Sqrt|Abs|Negative|Maximum|Minimum|SquaredDifference|Floor|Ceiling)$/.test(l.type)) {
    return elemCost(out, 1);
  }
  return NO_COST;
}

function insightFor(type: string, canon: string | undefined): string | undefined {
  switch (type) {
    case "Parameter": return "Network input (Parameter). Preprocessing (layout, mean/scale) must match what the IR was converted with.";
    case "Result": return "Network output (Result).";
    case "Convert": return "Precision conversion. Between a Const and an op it is weight decompression; on activations it costs a pass over memory.";
    case "FakeQuantize": return "Quantization boundary (FakeQuantize): marks INT8 regions for the OpenVINO low-precision pipeline.";
    case "ShapeOf": return "Reads a tensor's runtime shape; part of a dynamic-shape subgraph.";
    case "Interpolate": return "Resize/upsample. Check mode and coordinate transform when targeting other compilers.";
    case "GroupConvolution": return "Grouped/depthwise convolution. Memory-bound on most edge NPUs; fuse with the following activation.";
    case "MatMul": return canon === "Gemm" ? "MatMul against constant weights: a dense/linear layer." : undefined;
  }
  return undefined;
}

/**
 * Parse an OpenVINO IR .xml (optionally with its .bin weights) into the
 * normalized Model. Non-IR XML falls back to detected-format metadata.
 */
export function parseOpenvino(
  input: ArrayBuffer | Uint8Array | string,
  name = "model.xml",
  weights?: ArrayBuffer | Uint8Array,
): Model {
  const bytes = typeof input === "string" ? null : input instanceof Uint8Array ? input : new Uint8Array(input);
  const text = typeof input === "string" ? input : new TextDecoder().decode(bytes!);
  const sizeBytes = bytes ? bytes.byteLength : text.length;
  const fallback = () => parseContainer(bytes ?? new TextEncoder().encode(text), name);

  let root: XmlElement;
  try {
    root = parseXml(text);
  } catch {
    return fallback();
  }
  const layersEl = child(root, "layers");
  if (root.tag !== "net" || !layersEl) return fallback();

  const bin = weights ? (weights instanceof Uint8Array ? weights : new Uint8Array(weights)) : undefined;

  // ── Read the IR ────────────────────────────────────────────────────────────
  const irLayers: IrLayer[] = children(layersEl, "layer").map((el) => ({
    id: el.attrs.id,
    name: el.attrs.name ?? `layer_${el.attrs.id}`,
    type: el.attrs.type ?? "Unknown",
    version: el.attrs.version ?? "",
    data: { ...(child(el, "data")?.attrs ?? {}) },
    ins: readPorts(child(el, "input")),
    outs: readPorts(child(el, "output")),
    blobs: (child(el, "blobs")?.children ?? []).map((b) => ({
      kind: b.tag, offset: Number(b.attrs.offset ?? 0), size: Number(b.attrs.size ?? 0), precision: b.attrs.precision,
    })),
    precision: el.attrs.precision,
  }));
  if (!irLayers.length) return fallback();
  const irEdges: IrEdge[] = children(child(root, "edges"), "edge").map((e) => ({
    from: e.attrs["from-layer"], fromPort: e.attrs["from-port"], to: e.attrs["to-layer"], toPort: e.attrs["to-port"],
  }));

  const byId = new Map(irLayers.map((l) => [l.id, l]));
  const inEdges = new Map<string, IrEdge[]>();
  const outEdges = new Map<string, IrEdge[]>();
  for (const e of irEdges) {
    if (!inEdges.has(e.to)) inEdges.set(e.to, []);
    inEdges.get(e.to)!.push(e);
    if (!outEdges.has(e.from)) outEdges.set(e.from, []);
    outEdges.get(e.from)!.push(e);
  }

  // ── Constant folding: a layer is constant when it is a Const, or when every
  // input comes from a constant layer (e.g. Const(f16) → Convert, or the
  // Const → Subtract → Multiply weight-decompression chains of INT8 IRs).
  const constMemo = new Map<string, boolean>();
  const isConst = (id: string, depth = 0): boolean => {
    const hit = constMemo.get(id);
    if (hit != null) return hit;
    const l = byId.get(id);
    let v = false;
    if (l && depth < 10_000) {
      if (l.type === "Const") v = true;
      else if (l.type === "Parameter" || l.type === "Result" || l.type === "ReadValue" || l.type === "Assign") v = false;
      else {
        const ins = inEdges.get(id) ?? [];
        constMemo.set(id, false); // cycle guard
        v = ins.length > 0 && ins.every((e) => isConst(e.from, depth + 1));
      }
    }
    constMemo.set(id, v);
    return v;
  };

  /** Root Const layers feeding (transitively) into a constant layer. */
  const constRoots = (id: string, acc: IrLayer[], seen: Set<string>) => {
    if (seen.has(id)) return;
    seen.add(id);
    const l = byId.get(id);
    if (!l) return;
    if (l.type === "Const") { acc.push(l); return; }
    for (const e of inEdges.get(id) ?? []) constRoots(e.from, acc, seen);
  };

  // ── Build visible layers ───────────────────────────────────────────────────
  const visible = irLayers.filter((l) => !isConst(l.id));
  const idMap = new Map<string, number>();
  visible.forEach((l, i) => idMap.set(l.id, i));
  const claimed = new Set<string>(); // Const roots already attributed

  const portName = (layerId: string, portId: string): string => {
    const l = byId.get(layerId);
    const p = l?.outs.find((o) => o.id === portId);
    const n = p?.names?.split(",")[0]?.trim();
    if (n) return n;
    const multi = (l?.outs.length ?? 0) > 1;
    return l ? (multi ? `${l.name}:${portId}` : l.name) : `#${layerId}`;
  };

  const layers: ModelLayer[] = [];
  const edgeSet = new EdgeSet();
  let maxOpset = 0;

  for (const l of visible) {
    const id = idMap.get(l.id)!;
    const m = /^opset(\d+)$/.exec(l.version);
    if (m) maxOpset = Math.max(maxOpset, Number(m[1]));

    const ine = (inEdges.get(l.id) ?? []).slice().sort((a, b) => Number(a.toPort) - Number(b.toPort));
    const inShapes = l.ins.map((p) => p.dims);
    const outShape = l.outs[0]?.dims ?? (l.type === "Result" ? l.ins[0]?.dims : undefined);

    // Fold constant inputs → params / weight stats.
    let params = 0;
    let wl: IrLayer | null = null;
    const weightShapes: Shape[] = [];
    for (const e of ine) {
      if (!isConst(e.from)) continue;
      const roots: IrLayer[] = [];
      constRoots(e.from, roots, new Set());
      for (const c of roots) {
        if (claimed.has(c.id)) continue;
        claimed.add(c.id);
        const dt = dtOf(c.data.element_type);
        if (!PARAM_DTYPES.has(dt)) continue;
        const s = dims(c.data.shape ?? "").length ? dims(c.data.shape) : c.outs[0]?.dims ?? [];
        const n = numel(s);
        params += n;
        weightShapes.push(s);
        if (!wl || n > numel(dims(wl.data.shape ?? ""))) wl = c;
      }
    }
    // IR v7: weights live in <blobs> on the layer itself.
    for (const b of l.blobs) {
      const dt = dtOf(b.precision ?? l.precision ?? l.outs[0]?.precision);
      params += Math.floor(b.size / (DT_SIZE[dt] ?? 4));
    }

    let w: WStat | null = null;
    if (wl) {
      const dt = dtOf(wl.data.element_type);
      const shape = dims(wl.data.shape ?? "");
      const off = Number(wl.data.offset);
      const size = Number(wl.data.size);
      const raw = bin && Number.isFinite(off) && Number.isFinite(size) && off + size <= bin.byteLength
        ? bin.subarray(off, off + size) : null;
      w = weightStats(shape, raw ? sampleBytes(raw, dt) : null);
    } else if (l.blobs.length && bin) {
      const b = l.blobs.reduce((a, c) => (c.size > a.size ? c : a));
      const dt = dtOf(b.precision ?? l.precision ?? l.outs[0]?.precision);
      const raw = b.offset + b.size <= bin.byteLength ? bin.subarray(b.offset, b.offset + b.size) : null;
      w = weightStats([Math.floor(b.size / (DT_SIZE[dt] ?? 4))], raw ? sampleBytes(raw, dt) : null);
    } else if (l.blobs.length) {
      w = { shape: "[multi]", size: params };
    }

    // Data inputs (non-constant) → ins + edges.
    const ins: { n: string; s?: number[] }[] = [];
    for (const e of ine) {
      if (isConst(e.from)) continue;
      const port = l.ins.find((p) => p.id === e.toPort);
      ins.push({ n: portName(e.from, e.fromPort), s: port?.dims });
      const from = idMap.get(e.from);
      if (from != null) edgeSet.add(from, id);
    }

    let canon = CANON[l.type] ?? l.type;
    const wIn = ine.find((e) => e.toPort === l.ins[1]?.id);
    if (l.type === "MatMul" && wIn && isConst(wIn.from)) canon = "Gemm";
    if (l.type === "ReduceMean" && outShape && outShape.length === 4 && outShape[2] === 1 && outShape[3] === 1) canon = "GlobalAveragePool";
    if (l.type === "ReduceMax" && outShape && outShape.length === 4 && outShape[2] === 1 && outShape[3] === 1) canon = "GlobalMaxPool";

    let group: ModelLayer["group"] | undefined;
    if (l.type === "Parameter") group = "input";
    else if (l.type === "Result") group = "output";
    else if ((outEdges.get(l.id) ?? []).some((e) => byId.get(e.to)?.type === "Result")) group = "output";
    else if (canon === "Gemm" || canon === "MatMul") group = "head";

    const layer = makeLayer({
      id,
      name: l.name,
      type: l.type,
      canon,
      shape: outShape,
      dt: l.outs[0]?.precision ?? l.ins[0]?.precision ?? dtOf(l.data.element_type),
      params,
      cost: l.type === "Result" ? NO_COST : cost(l, canon, outShape, inShapes),
      group,
      ins,
      outs: l.outs.map((p) => ({ n: portName(l.id, p.id), s: p.dims })),
      attr: { ...l.data, ...(l.version ? { version: l.version } : {}), ...(weightShapes.length ? { weights: weightShapes.map((s) => s.join("×")) } : {}) },
      w,
      insight: insightFor(l.type, canon),
      framework: "OpenVINO",
    });
    if (l.type === "Result") layer.mem = 0;
    layers.push(layer);
  }

  const firstIn = visible.find((l) => l.type === "Parameter");
  const firstOut = visible.find((l) => l.type === "Result");
  const rt = child(root, "rt_info");
  const runtime =
    children(rt, "info").find((i) => i.attrs.name === "OpenVINO Runtime")?.attrs.value ??
    child(rt, "Runtime_version")?.attrs.value ??
    child(child(root, "meta_data"), "MO_version")?.attrs.value;
  const irVersion = Number(root.attrs.version);

  return {
    name,
    format: "OpenVINO IR",
    framework: "OpenVINO",
    opset: maxOpset,
    sizeBytes,
    inputShape: cleanShape(firstIn?.outs[0]?.dims),
    outputShape: cleanShape(firstOut?.ins[0]?.dims),
    layers,
    edges: edgeSet.edges,
    producer: runtime ? `OpenVINO ${runtime.split("-")[0]}` : undefined,
    irVersion: Number.isFinite(irVersion) ? irVersion : undefined,
    doc: root.attrs.name,
  };
}
