import type { Model, ModelLayer } from "../types";
import { ProtoReader, WT_I64, WT_LEN } from "../common/protobufWire";
import {
  cleanShape, DT_SIZE, EdgeSet, elemCost, linearCost, makeLayer, NO_COST, numel, prod,
  sampleBytes, weightStats, type Cost, type Shape,
} from "../common/graphKit";

// PaddlePaddle static-graph inference program (`.pdmodel`): a protobuf
// ProgramDesc. Field numbers follow paddle/fluid/framework/framework.proto:
//   ProgramDesc { blocks=1, version=4 → Version{version=1} }
//   BlockDesc   { idx=1, parent_idx=2, vars=3, ops=4 }
//   VarDesc     { name=1, type=2 → VarType, persistable=3 }
//   VarType     { type=1, selected_rows=2, dense_tensor(lod_tensor)=3 → {tensor=1 → TensorDesc{data_type=1, dims=2}} }
//   OpDesc      { inputs=1, outputs=2 (Var{parameter=1, arguments=2}), type=3, attrs=4 }
//   OpDesc.Attr { name=1, type=2, i=3, f=4, s=5, ints=6, floats=7, strings=8, b=10, bools=11,
//                 block_idx=12, l=13, blocks_idx=14, longs=15, float64s=16, var_name=17, vars_name=18, float64=19 }
// Weights live in the sibling `.pdiparams`; persistable vars give their shapes.

type PVar = { name: string; vtype: number; dt: string; dims: Shape; persistable: boolean };
type PArg = { param: string; args: string[] };
type POp = { type: string; inputs: PArg[]; outputs: PArg[]; attrs: Record<string, unknown> };
type PBlock = { idx: number; parent: number; vars: PVar[]; ops: POp[] };

const VT_DTYPE: Record<number, string> = {
  0: "bool", 1: "int16", 2: "int32", 3: "int64", 4: "float16", 5: "float32", 6: "float64",
  20: "uint8", 21: "int8", 22: "bfloat16", 23: "complex64", 24: "complex128", 36: "uint16", 37: "uint32", 38: "uint64",
};

function readTensorDesc(r: ProtoReader): { dt: string; dims: Shape } {
  let dt = "float32";
  const dims: Shape = [];
  for (let t; (t = r.next()); ) {
    if (t.field === 1) dt = VT_DTYPE[r.varint()] ?? "float32";
    else if (t.field === 2) dims.push(...r.packedVarints(t.wireType));
    else r.skip(t.wireType);
  }
  return { dt, dims };
}

function readVarType(r: ProtoReader): { vtype: number; dt: string; dims: Shape } {
  let vtype = 7;
  let td: { dt: string; dims: Shape } = { dt: "float32", dims: [] };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) vtype = r.varint();
    else if (t.field === 2) td = readTensorDesc(r.sub()); // selected_rows
    else if (t.field === 3 || t.field === 4) {
      // dense_tensor / tensor_array: { tensor=1, lod_level=2 }
      const s = r.sub();
      for (let u; (u = s.next()); ) {
        if (u.field === 1) td = readTensorDesc(s.sub());
        else s.skip(u.wireType);
      }
    } else r.skip(t.wireType);
  }
  return { vtype, ...td };
}

function readVar(r: ProtoReader): PVar {
  const v: PVar = { name: "", vtype: 7, dt: "float32", dims: [], persistable: false };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) v.name = r.string();
    else if (t.field === 2) Object.assign(v, readVarType(r.sub()));
    else if (t.field === 3) v.persistable = r.bool();
    else r.skip(t.wireType);
  }
  return v;
}

function readArg(r: ProtoReader): PArg {
  const a: PArg = { param: "", args: [] };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) a.param = r.string();
    else if (t.field === 2) a.args.push(r.string());
    else r.skip(t.wireType);
  }
  return a;
}

// Attribute names that are bookkeeping (call stacks, roles) rather than op semantics.
const NOISY_ATTRS = /^(op_callstack|op_role|op_role_var|op_namescope|op_device|with_quant_attr|use_mkldnn|use_cudnn|use_quantizer|mkldnn_data_type|is_test|name|struct_name)$/;

function readAttr(r: ProtoReader): [string, unknown] {
  let name = "";
  let type = -1;
  const vals: Record<number, unknown> = {};
  const push = (k: number, v: unknown[]) => { vals[k] = [...((vals[k] as unknown[]) ?? []), ...v]; };
  for (let t; (t = r.next()); ) {
    switch (t.field) {
      case 1: name = r.string(); break;
      case 2: type = r.varint(); break;
      case 3: vals[3] = r.int32(); break;
      case 4: vals[4] = round6(r.float()); break;
      case 5: vals[5] = r.string(); break;
      case 6: push(6, r.packedVarints(t.wireType).map((x) => x | 0)); break;
      case 7: push(7, r.packedFloats(t.wireType).map(round6)); break;
      case 8: push(8, [r.string()]); break;
      case 10: vals[10] = r.bool(); break;
      case 11: push(11, r.packedVarints(t.wireType).map((x) => x !== 0)); break;
      case 12: vals[12] = r.int32(); break;
      case 13: vals[13] = r.int64(); break;
      case 14: push(14, r.packedVarints(t.wireType)); break;
      case 15: push(15, r.packedVarints(t.wireType)); break;
      case 16:
        if (t.wireType === WT_I64) push(16, [r.double()]);
        else { const s = r.sub(); const d: number[] = []; while (!s.eof) d.push(s.double()); push(16, d); }
        break;
      case 17: vals[17] = r.string(); break;
      case 18: push(18, [r.string()]); break;
      case 19: vals[19] = r.double(); break;
      default: r.skip(t.wireType);
    }
  }
  // AttrType enum → which field carries the value.
  const FIELD: Record<number, number> = { 0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 10, 7: 11, 8: 12, 9: 13, 10: 14, 11: 15, 12: 16, 13: 17, 14: 18, 15: 19 };
  const f = FIELD[type];
  let v = f != null ? vals[f] : undefined;
  if (v === undefined) {
    // Empty repeated attrs are legal; default by type.
    if ([3, 4, 5, 7, 10, 11, 12, 14].includes(type)) v = [];
    else v = Object.values(vals)[0];
  }
  return [name, v];
}

const round6 = (x: number) => Math.round(x * 1e6) / 1e6;

function readOp(r: ProtoReader): POp {
  const op: POp = { type: "", inputs: [], outputs: [], attrs: {} };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) op.inputs.push(readArg(r.sub()));
    else if (t.field === 2) op.outputs.push(readArg(r.sub()));
    else if (t.field === 3) op.type = r.string();
    else if (t.field === 4) {
      const [k, v] = readAttr(r.sub());
      if (k && !NOISY_ATTRS.test(k)) op.attrs[k] = v;
    } else r.skip(t.wireType);
  }
  return op;
}

function readBlock(r: ProtoReader): PBlock {
  const b: PBlock = { idx: 0, parent: -1, vars: [], ops: [] };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) b.idx = r.int32();
    else if (t.field === 2) b.parent = r.int32();
    else if (t.field === 3) b.vars.push(readVar(r.sub()));
    else if (t.field === 4) b.ops.push(readOp(r.sub()));
    else r.skip(t.wireType);
  }
  return b;
}

export type PaddleProgram = { blocks: PBlock[]; version: number };

/** Decode a ProgramDesc. Throws if the bytes are not a plausible program. */
export function decodePaddleProgram(buf: Uint8Array): PaddleProgram {
  const r = new ProtoReader(buf);
  const blocks: PBlock[] = [];
  let version = 0;
  for (let t; (t = r.next()); ) {
    if (t.field === 1 && t.wireType === WT_LEN) blocks.push(readBlock(r.sub()));
    else if (t.field === 4 && t.wireType === WT_LEN) {
      const s = r.sub();
      for (let u; (u = s.next()); ) {
        if (u.field === 1) version = s.int64();
        else s.skip(u.wireType);
      }
    } else r.skip(t.wireType);
  }
  if (!blocks.length) throw new Error("PaddlePaddle program has no blocks.");
  return { blocks, version };
}

/** Cheap sniff: does this look like a ProgramDesc (block with ops that have types)? */
export function isPaddleProgram(buf: Uint8Array): boolean {
  try {
    const p = decodePaddleProgram(buf);
    return p.blocks[0].ops.length > 0 && p.blocks[0].ops.every((o) => o.type.length > 0);
  } catch {
    return false;
  }
}

// ── Op semantics ─────────────────────────────────────────────────────────────

const CANON: Record<string, string> = {
  conv2d: "Conv", depthwise_conv2d: "Conv", conv3d: "Conv", fused_conv2d: "Conv",
  conv2d_transpose: "ConvTranspose", depthwise_conv2d_transpose: "ConvTranspose",
  matmul: "MatMul", matmul_v2: "MatMul", bmm: "MatMul", mul: "Gemm", fc: "Gemm",
  batch_norm: "BatchNormalization", sync_batch_norm: "BatchNormalization",
  layer_norm: "LayerNormalization", instance_norm: "InstanceNormalization", group_norm: "GroupNormalization",
  relu: "Relu", relu6: "Relu", leaky_relu: "LeakyRelu", sigmoid: "Sigmoid", tanh: "Tanh",
  hard_swish: "HardSwish", hard_sigmoid: "HardSigmoid", swish: "Swish", silu: "Swish", gelu: "Gelu",
  softmax: "Softmax", log_softmax: "LogSoftmax",
  elementwise_add: "Add", elementwise_sub: "Sub", elementwise_mul: "Mul", elementwise_div: "Div",
  add: "Add", subtract: "Sub", multiply: "Mul", divide: "Div", sum: "Add",
  flatten_contiguous_range: "Flatten", flatten2: "Flatten", flatten: "Flatten",
  reshape2: "Reshape", reshape: "Reshape", squeeze2: "Reshape", unsqueeze2: "Reshape",
  transpose2: "Transpose", transpose: "Transpose", concat: "Concat",
  nearest_interp_v2: "Resize", bilinear_interp_v2: "Resize", nearest_interp: "Resize", bilinear_interp: "Resize",
  scale: "Mul", dropout: "Identity", pad3d: "Pad", split: "Split", slice: "Slice",
};

function canonical(op: POp): string {
  if (op.type === "pool2d" || op.type === "pool3d") {
    const avg = op.attrs.pooling_type === "avg";
    if (op.attrs.global_pooling === true) return avg ? "GlobalAveragePool" : "GlobalMaxPool";
    return avg ? "AveragePool" : "MaxPool";
  }
  return CANON[op.type] ?? op.type;
}

const PRIMARY_OUT = ["Out", "Output", "Y"];

function cost(canon: string, op: POp, inShape: (p: string) => Shape | undefined, out: Shape | undefined): Cost {
  switch (canon) {
    case "Conv": {
      // Filter [Cout, Cin/groups, kh, kw]: each output accumulates Cin/g·kh·kw products.
      const f = inShape("Filter");
      return f ? linearCost(out, prod(f.slice(1))) : NO_COST;
    }
    case "ConvTranspose": {
      // Filter [Cin, Cout/groups, kh, kw]: each input scatters Cout/g·kh·kw products.
      const f = inShape("Filter");
      const x = inShape("Input");
      return f && x ? linearCost(x, prod(f.slice(1))) : NO_COST;
    }
    case "MatMul":
    case "Gemm": {
      const y = inShape("Y") ?? inShape("W");
      const x = inShape("X") ?? inShape("Input");
      let k = -1;
      if (op.type === "mul" && y) k = y[0];
      else if (y && y.length >= 2) {
        const ty = op.attrs.trans_y === true || op.attrs.transpose_Y === true;
        k = ty ? y[y.length - 1] : y[y.length - 2];
      } else if (x) k = x[x.length - 1];
      return linearCost(out, k);
    }
    case "MaxPool":
    case "AveragePool": {
      const ks = Array.isArray(op.attrs.ksize) ? (op.attrs.ksize as number[]) : [1, 1];
      return elemCost(out, prod(ks));
    }
    case "GlobalAveragePool":
    case "GlobalMaxPool":
      return elemCost(inShape("X"));
    case "BatchNormalization":
    case "LayerNormalization":
    case "InstanceNormalization":
    case "GroupNormalization":
      return elemCost(out, 2);
    case "Softmax":
    case "LogSoftmax":
      return elemCost(out, 3);
    case "Add": case "Sub": case "Mul": case "Div": case "Relu": case "LeakyRelu":
    case "Sigmoid": case "Tanh": case "HardSwish": case "HardSigmoid": case "Swish": case "Gelu":
      return elemCost(out);
    default:
      return NO_COST;
  }
}

/**
 * Read a combined `.pdiparams` file (paddle.static.save_inference_model /
 * save_combine): the persistable tensors back to back, ordered by variable
 * name. Each is serialized as
 *   u32 version · u64 lod_level · lod_level × (u64 nbytes, offsets)
 *   u32 version · i32 desc_size · TensorDesc proto · raw little-endian data
 * The stream carries no names, so every tensor's dims are checked against the
 * program's; on any mismatch this returns null rather than mislabel weights.
 */
export function readPdiparams(buf: Uint8Array, vars: { name: string; dims: Shape }[]): Map<string, { dt: string; bytes: Uint8Array }> | null {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const names = vars.map((v) => v.name).sort();
  const dimsOf = new Map(vars.map((v) => [v.name, v.dims]));
  const out = new Map<string, { dt: string; bytes: Uint8Array }>();
  let o = 0;
  const need = (n: number) => { if (o + n > buf.byteLength) throw new Error("truncated"); };
  try {
    for (const name of names) {
      need(12);
      o += 4; // LoDTensor version
      const lodLevel = Number(view.getBigUint64(o, true)); o += 8;
      for (let i = 0; i < lodLevel; i++) {
        need(8);
        const n = Number(view.getBigUint64(o, true)); o += 8;
        need(n); o += n;
      }
      need(8);
      o += 4; // Tensor version
      const descSize = view.getInt32(o, true); o += 4;
      need(descSize);
      const desc = readTensorDesc(new ProtoReader(buf, o, o + descSize));
      o += descSize;
      const want = dimsOf.get(name) ?? [];
      // Program dims may hold -1 for a dynamic batch; stored tensors never do.
      if (desc.dims.length !== want.length || desc.dims.some((d, i) => want[i] >= 0 && d !== want[i])) return null;
      const n = prod(desc.dims) * (DT_SIZE[desc.dt] ?? 4);
      need(n);
      out.set(name, { dt: desc.dt, bytes: buf.subarray(o, o + n) });
      o += n;
    }
  } catch {
    return null;
  }
  return o === buf.byteLength ? out : null;
}

/** Parse a PaddlePaddle `.pdmodel` (ProgramDesc protobuf) into the normalized graph. */
export function parsePaddle(input: ArrayBuffer | Uint8Array, name = "model.pdmodel", params?: Uint8Array): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const prog = decodePaddleProgram(buf);
  const block = prog.blocks[0];

  const vars = new Map<string, PVar>();
  for (const b of prog.blocks) for (const v of b.vars) if (!vars.has(v.name)) vars.set(v.name, v);
  // Weights are persistable dense tensors (feed/fetch holders are persistable too, but typed 9/10).
  const isWeight = (n: string) => {
    const v = vars.get(n);
    return !!v && v.persistable && v.vtype !== 9 && v.vtype !== 10 && v.vtype !== 28;
  };
  const shapeOf = (n: string) => vars.get(n)?.dims;
  const weightVars = [...vars.values()].filter((v) => isWeight(v.name));
  const values = params ? readPdiparams(params, weightVars) : null;

  const layers: ModelLayer[] = [];
  const edges = new EdgeSet();
  const producer = new Map<string, number>();
  const counted = new Set<string>();
  let nextId = 0;

  // Feed/fetch ops define the graph's inputs and outputs (ordered by `col`).
  const feeds = block.ops.filter((o) => o.type === "feed")
    .sort((a, b) => Number(a.attrs.col ?? 0) - Number(b.attrs.col ?? 0))
    .flatMap((o) => o.outputs.flatMap((a) => a.args));
  const fetches = block.ops.filter((o) => o.type === "fetch")
    .sort((a, b) => Number(a.attrs.col ?? 0) - Number(b.attrs.col ?? 0))
    .flatMap((o) => o.inputs.flatMap((a) => a.args));
  const fetchSet = new Set(fetches);

  for (const f of feeds) {
    const v = vars.get(f);
    const id = nextId++;
    producer.set(f, id);
    layers.push(makeLayer({
      id, name: f, type: "feed", canon: "Input", shape: v?.dims, dt: v?.dt, group: "input",
      outs: [{ n: f, s: v?.dims }], framework: "PaddlePaddle",
      insight: "Graph input (feed target). Apply preprocessing (resize/normalize) upstream.",
    }));
    layers[layers.length - 1].math = "model input";
  }

  // Constant-folded vars: outputs of ops fed only by weights/constants (e.g. the
  // reshape2 of a conv bias, fill_constant shape tensors) → their source weights.
  const derived = new Map<string, string[]>();
  const weightSources = (n: string): string[] | null => (isWeight(n) ? [n] : derived.get(n) ?? null);

  for (const op of block.ops) {
    if (op.type === "feed" || op.type === "fetch") continue;
    const inArgs = op.inputs.flatMap((a) => a.args);
    if (inArgs.every((n) => weightSources(n))) {
      const srcs = [...new Set(inArgs.flatMap((n) => weightSources(n)!))];
      for (const a of op.outputs) for (const n of a.args) if (!isWeight(n)) derived.set(n, srcs);
      continue;
    }
    const canon = canonical(op);
    const id = nextId++;

    let params = 0;
    let bigW: { n: string; s: Shape } | null = null;
    const ins: { n: string; s?: Shape }[] = [];
    for (const n of inArgs) {
      const srcs = weightSources(n);
      if (srcs) {
        for (const w of srcs) {
          const s = shapeOf(w) ?? [];
          if (!counted.has(w)) { counted.add(w); params += numel(s); }
          if (!bigW || numel(s) > numel(bigW.s)) bigW = { n: w, s };
        }
        continue;
      }
      ins.push({ n, s: shapeOf(n) });
      edges.add(producer.get(n), id);
    }

    const outArgs = PRIMARY_OUT.map((p) => op.outputs.find((a) => a.param === p && a.args.length))
      .find(Boolean) ?? op.outputs.find((a) => a.args.length && a.param !== "XShape");
    const outName = outArgs?.args[0];
    const outVar = outName ? vars.get(outName) : undefined;
    const outShape = outVar?.dims;
    const outs = op.outputs.filter((a) => a.param !== "XShape")
      .flatMap((a) => a.args.filter((n) => !isWeight(n)).map((n) => ({ n, s: shapeOf(n) })));
    // Assign producers after resolving inputs so in-place ops chain correctly.
    for (const o of outs) producer.set(o.n, id);

    const inShape = (p: string) => {
      const a = op.inputs.find((x) => x.param === p);
      return a?.args.length ? shapeOf(a.args[0]) : undefined;
    };
    const isOut = outs.some((o) => fetchSet.has(o.n));
    layers.push(makeLayer({
      id, name: outName ?? `${op.type}_${id}`, type: op.type, canon,
      shape: outShape, dt: outVar?.dt, params, cost: cost(canon, op, inShape, outShape),
      group: isOut ? "output" : undefined, ins, outs,
      attr: { ...op.attrs, ...(bigW ? { weight: bigW.n } : {}) },
      w: bigW ? weightStats(bigW.s, statValues(values?.get(bigW.n))) : null, framework: "PaddlePaddle",
    }));
  }

  const first = feeds[0];
  const last = fetches[0];
  const ver = prog.version;
  return {
    name, format: "PaddlePaddle", framework: "PaddlePaddle", opset: ver,
    sizeBytes: buf.byteLength,
    inputShape: first ? cleanShape(shapeOf(first)) : [],
    outputShape: last ? cleanShape(shapeOf(last)) : [],
    layers, edges: edges.edges,
    producer: ver ? `Paddle ${formatVersion(ver)}` : undefined,
  };
}

// Program version is encoded as major*1e6 + minor*1e3 + patch (e.g. 2006000 → 2.6.0).
function formatVersion(v: number): string {
  if (v < 1000) return String(v);
  return `${Math.floor(v / 1e6)}.${Math.floor((v % 1e6) / 1e3)}.${v % 1e3}`;
}

const statValues = (t: { dt: string; bytes: Uint8Array } | undefined) => (t ? sampleBytes(t.bytes, t.dt) : null);
