import type { Model, ModelLayer } from "../types";
import { FlatBuf } from "../common/flatbuffers";
import { findJsonObject } from "../common/jsonScan";
import { EdgeSet, NO_COST, cleanShape, elemCost, linearCost, makeLayer, numel, weightStats, type Cost, type Shape } from "../common/graphKit";

// Rockchip RKNN (`.rknn`). Container layout (rknn-toolkit / rknn-toolkit2,
// cross-checked against Netron's source/rknn.js):
//   "RKNN\0\0\0\0" | u64 version | u64 data_size | [40-byte header if
//   (version & 0xff) > 1 and data_size > 0] | data[data_size] | u64 json_size | json
// `json` describes the network (version, network_platform, target_platform,
// nodes, connection, graph, norm_tensor / const_tensor / virtual_tensor).
// For rknn-toolkit2 models the JSON graph is a single compiled "NNBG" node and
// the real operator graph lives in `data`: a FlatBuffer with file identifier
// "RKNN" (undocumented schema; field order as reverse-engineered by Netron):
//   Model: format 1, graphs 2, generator 3, compiler 7, runtime 8, source 9
//   Graph: tensors 0, nodes 1, inputs 2, outputs 3
//   Node:  type 1, name 2, inputs 4, outputs 5
//   Tensor: data_type 0, kind 2, shape 4, name 5, size 12
// Newer "RKNX" (RKNN3) containers and encrypted "CYPTRKNN" files only expose
// header metadata here.

const FRAMEWORK = "Rockchip RKNN";
const FB_DTYPES = ["?", "float32", "uint8", "int8", "uint16", "int16", "int32", "int64", "string", "bool", "float16", "float64", "uint32", "uint64", "complex64", "complex128", "bfloat16"];
const CONST_KINDS = new Set([4, 5]);

const ascii = (buf: Uint8Array, off: number, s: string) => [...s].every((c, i) => buf[off + i] === c.charCodeAt(0));

/** True for RKNN containers (RKNN / RKNX / CYPTRKNN) and bare RKNN FlatBuffers. */
export function isRknn(buf: Uint8Array): boolean {
  if (buf.length < 8) return false;
  return ascii(buf, 0, "RKNN") || ascii(buf, 0, "RKNX") || ascii(buf, 0, "CYPTRKNN") || ascii(buf, 4, "RKNN");
}

type RknnJson = {
  version?: string; name?: string; network_platform?: string; ori_network_platform?: string;
  target_platform?: string[]; ovxlib_version?: string;
  nodes?: { name?: string; op?: string; nn?: Record<string, Record<string, unknown>> }[];
  connection?: { node_id: number; left: string; left_tensor_id: number; right_node?: { node_id: number; tensor_id: number }; right_tensor?: { type: string; tensor_id: number } }[];
  graph?: { left: string; left_tensor_id: number; right: string; right_tensor_id: number }[];
  norm_tensor?: { tensor_id: number; size?: number[]; dtype?: { vx_type?: string }; url?: string }[];
  const_tensor?: { tensor_id: number; size?: number[]; dtype?: { vx_type?: string } }[];
  virtual_tensor?: { node_id: number; output_port: number }[];
};

type Container = { kind: "rknn" | "rknx" | "encrypted" | "fb"; json: RknnJson | null; fb: Uint8Array | null; version: number };

function u64(dv: DataView, o: number): number {
  return Number(dv.getBigUint64(o, true));
}

function readContainer(buf: Uint8Array): Container {
  if (ascii(buf, 0, "CYPTRKNN")) return { kind: "encrypted", json: null, fb: null, version: 0 };
  if (ascii(buf, 0, "RKNX")) return { kind: "rknx", json: null, fb: buf.subarray(0x50), version: 3 };
  if (!ascii(buf, 0, "RKNN") && ascii(buf, 4, "RKNN")) return { kind: "fb", json: null, fb: buf, version: 0 };
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let json: RknnJson | null = null;
  let fb: Uint8Array | null = null;
  let version = 0;
  try {
    version = u64(dv, 8);
    const dataSize = u64(dv, 16);
    let off = 24;
    if ((version & 0xff) > 1 && dataSize > 0) off += 40;
    if (off + dataSize + 8 <= buf.length) {
      const data = buf.subarray(off, off + dataSize);
      if (data.length >= 8 && ascii(data, 4, "RKNN")) fb = data;
      const jsonSize = u64(dv, off + dataSize);
      const start = off + dataSize + 8;
      if (jsonSize > 0 && start + jsonSize <= buf.length) {
        json = JSON.parse(new TextDecoder().decode(buf.subarray(start, start + jsonSize))) as RknnJson;
      }
    }
  } catch {
    json = null;
  }
  if (!json) {
    // Robust fallback: brace-balanced scan for the model-description object.
    json = findJsonObject(buf, ['{"'], (o) => {
      const r = o as RknnJson;
      return !!r && Array.isArray(r.nodes) && (typeof r.network_platform === "string" || typeof r.version === "string");
    }) as RknnJson | null;
  }
  return { kind: "rknn", json, fb, version };
}

function canonOf(type: string): string {
  const t = type.toLowerCase();
  if (/deconv|convtranspose/.test(t)) return "ConvTranspose";
  if (/conv/.test(t)) return "Conv";
  if (/gemm|fullyconnected|fc$|innerproduct/.test(t)) return "Gemm";
  if (/matmul/.test(t)) return "MatMul";
  if (/globalaverage|globalavg/.test(t)) return "GlobalAveragePool";
  if (/maxpool/.test(t)) return "MaxPool";
  if (/avgpool|averagepool/.test(t)) return "AveragePool";
  if (/softmax/.test(t)) return "Softmax";
  if (/concat/.test(t)) return "Concat";
  if (/reshape/.test(t)) return "Reshape";
  if (/transpose|permute/.test(t)) return "Transpose";
  if (/^(add|eltwiseadd)$/.test(t)) return "Add";
  if (/^mul$/.test(t)) return "Mul";
  if (/relu/.test(t)) return "Relu";
  if (/sigmoid/.test(t)) return "Sigmoid";
  if (/input/.test(t)) return "Input";
  return type;
}

function costOf(canon: string, type: string, inS: (Shape | undefined)[], wS: Shape | undefined, out: Shape | undefined): Cost {
  if (canon === "Conv" && wS && wS.length >= 2 && wS[0] > 0) {
    const c = linearCost(out, numel(wS) / wS[0]);
    // Fused tails (ConvAdd / ConvRelu / ConvClip …) add one elementwise pass.
    return /conv.+/i.test(type) ? { ...c, flops: c.flops + numel(out) } : c;
  }
  if ((canon === "Gemm" || canon === "MatMul") && out) {
    const k = wS && wS.length >= 2 ? wS[wS.length - 1] : inS[0]?.[inS[0].length - 1] ?? 0;
    return linearCost(out, k);
  }
  if (/Pool/.test(canon)) return elemCost(inS[0]);
  if (canon === "Softmax") return elemCost(out, 3);
  if (["Add", "Mul", "Relu", "Sigmoid"].includes(canon) || /^(sub|div|clip|leakyrelu|tanh|swish|hardswish|silu|exp|max|min)$/i.test(type)) return elemCost(out);
  return NO_COST;
}

const groupOf = (canon: string): ModelLayer["group"] | undefined =>
  canon === "Conv" ? "backbone" : /Gemm|MatMul|Softmax/.test(canon) ? "head" : /Pool|Concat|Reshape|Transpose|Resize/.test(canon) ? "neck" : undefined;

/** Operator graph from the embedded RKNN FlatBuffer (rknn-toolkit2 models). */
function fbGraph(data: Uint8Array): { layers: ModelLayer[]; edges: [number, number][]; meta: Record<string, string>; inShape?: Shape; outShape?: Shape } | null {
  const fb = new FlatBuf(data);
  const root = fb.root();
  const graphs = fb.tables(root, 2);
  if (!graphs.length) return null;
  const meta = { format: fb.str(root, 1), compiler: fb.str(root, 7), runtime: fb.str(root, 8), source: fb.str(root, 9) };
  const g = graphs[0];
  type T = { name: string; dt: string; kind: number; shape: Shape; size: number };
  const tensors: T[] = fb.tables(g, 0).map((t, i) => ({
    name: fb.str(t, 5) || `tensor_${i}`,
    dt: FB_DTYPES[fb.int8(t, 0, 0)] ?? "?",
    kind: fb.int8(t, 2, 0),
    shape: fb.ints(t, 4),
    size: fb.int32(t, 12, 0),
  }));
  const graphOutputs = new Set(fb.ints(g, 3));
  const layers: ModelLayer[] = [];
  const edges = new EdgeSet();
  const producer = new Map<number, number>();
  let inShape: Shape | undefined, outShape: Shape | undefined;

  for (const n of fb.tables(g, 1)) {
    const type = fb.str(n, 1) || "Unknown";
    const ins = fb.ints(n, 4).filter((i) => i >= 0 && i < tensors.length);
    const outs = fb.ints(n, 5).filter((i) => i >= 0 && i < tensors.length);
    if (type === "OutputOperator") {
      // Mark the producer of each graph output instead of adding a pseudo-node.
      for (const i of ins) {
        const p = producer.get(i);
        if (p != null && tensors[i].kind !== 3) { layers[p].group = "output"; outShape = outShape ?? tensors[i].shape; }
      }
      continue;
    }
    const isInput = type === "InputOperator";
    const canon = isInput ? "Input" : canonOf(type);
    const consts = ins.filter((i) => CONST_KINDS.has(tensors[i].kind));
    const acts = ins.filter((i) => !CONST_KINDS.has(tensors[i].kind));
    const params = consts.reduce((s, i) => s + numel(tensors[i].shape), 0);
    const w = consts.map((i) => tensors[i]).sort((a, b) => numel(b.shape) - numel(a.shape))[0];
    const out = outs.length ? tensors[outs[0]].shape : undefined;
    const id = layers.length;
    for (const i of acts) edges.add(producer.get(i), id);
    for (const o of outs) producer.set(o, id);
    if (isInput && out && !inShape) inShape = out;
    const attr: Record<string, unknown> = {};
    if (consts.length) attr.constants = consts.map((i) => `${tensors[i].name} ${tensors[i].dt} ${tensors[i].shape.join("×")}`);
    const name = fb.str(n, 2) || `${type}_${id}`;
    layers.push(makeLayer({
      id, name, type: isInput ? "Input" : type, canon, shape: out,
      dt: outs.length ? tensors[outs[0]].dt : "float32",
      params, cost: isInput ? NO_COST : costOf(canon, type, acts.map((i) => tensors[i].shape), w?.shape, out),
      group: isInput ? "input" : outs.some((o) => graphOutputs.has(o)) ? "output" : groupOf(canon),
      ins: ins.map((i) => ({ n: tensors[i].name, s: tensors[i].shape })),
      outs: outs.map((i) => ({ n: tensors[i].name, s: tensors[i].shape })),
      attr, w: w ? { ...weightStats(w.shape, null) } : null,
      insight: w && /int8|uint8/.test(w.dt) ? `RKNN ${type} with ${w.dt}-quantized weights (NPU-compiled).` : undefined,
      framework: FRAMEWORK,
    }));
  }
  if (!outShape) {
    const lastOut = [...layers].reverse().find((l) => l.group === "output");
    outShape = lastOut?.outs[0]?.s;
  }
  return { layers, edges: edges.edges, meta, inShape, outShape };
}

/** Graph from the JSON description (rknn-toolkit v1 models, or NNBG fallback). */
function jsonGraph(j: RknnJson): { layers: ModelLayer[]; edges: [number, number][]; inShape?: Shape; outShape?: Shape } {
  // ovxlib (toolkit v1) sizes are innermost-first (w, h, c, n); flip them.
  const flip = !!j.ovxlib_version;
  const shapeOf = (s?: number[]) => (s ? (flip ? [...s].reverse() : [...s]) : undefined);
  const dtOf = (d?: { vx_type?: string }) => (d?.vx_type ? d.vx_type.replace(/^VSI_NN_TYPE_/, "").toLowerCase() : "") || "float32";
  const tensorInfo = new Map<string, { shape?: Shape; dt: string; isConst: boolean; label: string }>();
  for (const t of j.norm_tensor ?? []) tensorInfo.set(`norm_tensor:${t.tensor_id}`, { shape: shapeOf(t.size), dt: dtOf(t.dtype), isConst: false, label: t.url || `norm_tensor:${t.tensor_id}` });
  for (const t of j.const_tensor ?? []) tensorInfo.set(`const_tensor:${t.tensor_id}`, { shape: shapeOf(t.size), dt: dtOf(t.dtype), isConst: true, label: `const_tensor:${t.tensor_id}` });

  const nodes = j.nodes ?? [];
  const nodeIns: { key: string; port: number }[][] = nodes.map(() => []);
  const nodeOuts: { key: string; port: number }[][] = nodes.map(() => []);
  for (const c of j.connection ?? []) {
    if (c.node_id == null || c.node_id >= nodes.length) continue;
    const key = c.right_tensor ? `${c.right_tensor.type}:${c.right_tensor.tensor_id}` : c.right_node ? `${c.right_node.node_id}:${c.right_node.tensor_id}` : "";
    if (!key) continue;
    if (c.left === "input") nodeIns[c.node_id].push({ key, port: c.left_tensor_id });
    else if (c.left === "output") nodeOuts[c.node_id].push({ key, port: c.left_tensor_id });
  }
  // Virtual (intermediate) tensors are keyed by producing node + output port.
  nodeOuts.forEach((outs, i) => { if (!outs.length) outs.push({ key: `${i}:0`, port: 0 }); });

  const layers: ModelLayer[] = [];
  const edges = new EdgeSet();
  const producer = new Map<string, number>();
  const graphIns = (j.graph ?? []).filter((g) => g.left === "input").sort((a, b) => a.left_tensor_id - b.left_tensor_id);
  const graphOuts = new Set((j.graph ?? []).filter((g) => g.left === "output").map((g) => `${g.right}:${g.right_tensor_id}`));
  for (const gi of graphIns) {
    const key = `${gi.right}:${gi.right_tensor_id}`;
    const t = tensorInfo.get(key);
    const id = layers.length;
    producer.set(key, id);
    layers.push(makeLayer({
      id, name: t?.label ?? `input${gi.left_tensor_id}`, type: "Input", canon: "Input", shape: t?.shape, dt: t?.dt,
      group: "input", outs: [{ n: t?.label ?? key, s: t?.shape }], framework: FRAMEWORK,
    }));
  }
  const base = layers.length;
  nodes.forEach((_, i) => nodeOuts[i].forEach((o) => producer.set(o.key, base + i)));
  // Inputs referencing another node's output port (right_node) also map here.
  nodes.forEach((n, i) => {
    const type = (n.op ?? "Unknown").replace(/^(VSI_NN_OP_|RKNN_OP_)/, "");
    const ins = [...nodeIns[i]].sort((a, b) => a.port - b.port);
    const outs = [...nodeOuts[i]].sort((a, b) => a.port - b.port);
    const outT = tensorInfo.get(outs[0]?.key ?? "");
    const consts = ins.map((x) => tensorInfo.get(x.key)).filter((t) => t?.isConst);
    const params = consts.reduce((s, t) => s + numel(t!.shape), 0);
    const attr: Record<string, unknown> = {};
    for (const params of Object.values(n.nn ?? {})) Object.assign(attr, params);
    const id = base + i;
    for (const x of ins) if (!tensorInfo.get(x.key)?.isConst) edges.add(producer.get(x.key), id);
    const canon = canonOf(type);
    const inS = ins.map((x) => tensorInfo.get(x.key)?.shape);
    const isNbg = /NNBG|NBG/.test(type);
    layers.push(makeLayer({
      id, name: n.name || `${type}_${i}`, type, canon, shape: outT?.shape, dt: outT?.dt,
      params, cost: costOf(canon, type, inS, consts[0]?.shape, outT?.shape),
      group: outs.some((o) => graphOuts.has(o.key)) ? "output" : groupOf(canon),
      ins: ins.map((x) => ({ n: tensorInfo.get(x.key)?.label ?? x.key, s: tensorInfo.get(x.key)?.shape })),
      outs: outs.map((x) => ({ n: tensorInfo.get(x.key)?.label ?? x.key, s: tensorInfo.get(x.key)?.shape })),
      attr, w: params ? weightStats(consts[0]!.shape ?? [params], null) : null,
      insight: isNbg ? "Precompiled NPU network binary graph (NBG): operators are compiled into one opaque NPU program." : undefined,
      framework: FRAMEWORK,
    }));
  });
  const inShape = layers.find((l) => l.group === "input")?.outs[0]?.s;
  const outKey = [...graphOuts][0];
  const outShape = outKey ? tensorInfo.get(outKey)?.shape : undefined;
  return { layers, edges: edges.edges, inShape, outShape };
}

function infoModel(buf: Uint8Array, name: string, format: string, note: string, attr: Record<string, unknown>): Model {
  const layer = makeLayer({
    id: 0, name, type: format, canon: "Detected", group: "input", attr: { sizeBytes: buf.byteLength, ...attr },
    insight: note, framework: FRAMEWORK,
  });
  layer.mem = buf.byteLength;
  return { name, format, framework: FRAMEWORK, opset: 0, sizeBytes: buf.byteLength, inputShape: [], outputShape: [], layers: [layer], edges: [], doc: note };
}

/** Parse a Rockchip `.rknn` model into the normalized graph. */
export function parseRknn(input: ArrayBuffer | Uint8Array, name = "model.rknn"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (!isRknn(buf)) throw new Error("Not an RKNN file (missing RKNN/RKNX magic).");
  const c = readContainer(buf);
  if (c.kind === "encrypted") {
    return infoModel(buf, name, "RKNN (encrypted)", "Encrypted RKNN model (CYPTRKNN): the graph is not readable without the vendor key.", { magic: "CYPTRKNN" });
  }
  if (c.kind === "rknx") {
    const attr: Record<string, unknown> = { magic: "RKNX" };
    try {
      const fb = new FlatBuf(c.fb!);
      const h = fb.root();
      attr.description = fb.str(h, 1); attr.compiler = fb.str(h, 2); attr.toolkit = fb.str(h, 3);
      const hw = fb.table(h, 5);
      if (hw != null) attr.hardware = fb.str(hw, 0);
    } catch { /* header is best effort */ }
    return infoModel(buf, name, "RKNN3 (RKNX)", "RKNX (RKNN3) container: header metadata only; its compiled task graph has no operator-level structure to show.", attr);
  }

  const j = c.json;
  let g: ReturnType<typeof fbGraph> = null;
  if (c.fb) {
    try { g = fbGraph(c.fb); } catch { g = null; }
  }
  const jsonIsCompiledOnly = !j?.nodes?.length || j.nodes.every((n) => /NNBG|NBG/.test(n.op ?? ""));
  const useFb = !!g && g.layers.length > 0 && (jsonIsCompiledOnly || !j);
  const graph = useFb ? g! : j ? jsonGraph(j) : null;
  if (!graph) {
    return infoModel(buf, name, "RKNN", "RKNN container without a readable model description.", { version: c.version });
  }
  const version = j?.version?.split("-")[0] ?? "";
  const meta = useFb ? g!.meta : null;
  const producer = [
    version && `rknn-toolkit ${version}`,
    (j?.target_platform ?? []).length ? `target ${(j!.target_platform ?? []).join(",")}` : meta?.runtime && `target ${meta.runtime}`,
    (j?.ori_network_platform || meta?.source) && `from ${j?.ori_network_platform || meta?.source}`,
  ].filter(Boolean).join(", ");
  return {
    name, format: version ? `RKNN v${version}` : "RKNN", framework: FRAMEWORK, opset: 0, sizeBytes: buf.byteLength,
    inputShape: cleanShape(graph.inShape), outputShape: cleanShape(graph.outShape),
    layers: graph.layers, edges: graph.edges, producer,
    doc: useFb ? `Operator graph from the embedded RKNN FlatBuffer${meta?.compiler ? ` (compiler ${meta.compiler.split("(")[0].trim()})` : ""}.` : undefined,
  };
}
