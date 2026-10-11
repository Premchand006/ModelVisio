import { attrInt, attrInts, attrStr } from "./attrs";
import { numArr, type RawGraph, type RawTensor, type RawValueInfo } from "./raw";

// Shapes use numbers; a symbolic/dynamic dim (e.g. batch "N") is stored as -1.
export type Shape = number[];
export type ShapeMap = Map<string, Shape>;

const DYN = -1;

function shapeFromValueInfo(vi: RawValueInfo): Shape | null {
  const dims = vi.type?.tensor_type?.shape?.dim;
  if (!dims) return null;
  return dims.map((d) => {
    if (d.dim_value != null) {
      const n = typeof d.dim_value === "number" ? d.dim_value : d.dim_value.toNumber();
      return n > 0 ? n : DYN;
    }
    return DYN; // symbolic dim_param
  });
}

/** Seed a shape map from declared inputs, outputs, value_info, and initializers. */
export function seedShapeMap(graph: RawGraph): ShapeMap {
  const m: ShapeMap = new Map();
  const add = (vis: RawValueInfo[] | undefined) => {
    for (const vi of vis ?? []) {
      if (!vi.name) continue;
      const s = shapeFromValueInfo(vi);
      if (s) m.set(vi.name, s);
    }
  };
  add(graph.input);
  add(graph.value_info);
  add(graph.output);
  for (const init of graph.initializer ?? []) {
    if (init.name) m.set(init.name, numArr(init.dims));
  }
  return m;
}

const prod = (s: Shape) => s.reduce((a, b) => (b === DYN ? a : a * b), 1);

/** Output spatial size for conv/pool along one axis. */
function spatialOut(
  inSize: number,
  kernel: number,
  stride: number,
  padBegin: number,
  padEnd: number,
  dilation: number,
  autoPad: string | undefined,
  ceil = false,
): number {
  if (inSize === DYN) return DYN;
  if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
    return Math.ceil(inSize / stride);
  }
  const effK = dilation * (kernel - 1) + 1;
  const round = ceil ? Math.ceil : Math.floor;
  return round((inSize + padBegin + padEnd - effK) / stride) + 1;
}

function broadcast(a: Shape, b: Shape): Shape {
  const out: Shape = [];
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da === DYN || db === DYN) out.unshift(DYN);
    else out.unshift(Math.max(da, db));
  }
  return out;
}

/** Small constant tensor values (shape vectors, axes, split sizes, scales) keyed by tensor name. */
export type ConstMap = Map<string, number[]>;

// Constants above this size are never shape metadata; skip decoding them.
const MAX_CONST = 64;

/** Decode a small numeric tensor (initializer or Constant attribute) to plain numbers. */
export function tensorValues(t: RawTensor): number[] | null {
  const count = numArr(t.dims).reduce((a, b) => a * b, 1);
  if (count > MAX_CONST) return null;
  if (t.int64_data?.length) return numArr(t.int64_data);
  if (t.int32_data?.length) return [...t.int32_data];
  if (t.float_data?.length) return [...t.float_data];
  if (t.double_data?.length) return [...t.double_data];
  const raw = t.raw_data;
  if (!raw || !raw.length) return count === 0 ? [] : null;
  const v = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const out: number[] = [];
  switch (t.data_type) {
    case 7: for (let o = 0; o + 8 <= raw.length; o += 8) out.push(Number(v.getBigInt64(o, true))); break;
    case 6: for (let o = 0; o + 4 <= raw.length; o += 4) out.push(v.getInt32(o, true)); break;
    case 1: for (let o = 0; o + 4 <= raw.length; o += 4) out.push(v.getFloat32(o, true)); break;
    case 11: for (let o = 0; o + 8 <= raw.length; o += 8) out.push(v.getFloat64(o, true)); break;
    default: return null;
  }
  return out;
}

/** Seed constants from initializers. Constant nodes are added as they are visited. */
export function seedConstMap(graph: RawGraph): ConstMap {
  const m: ConstMap = new Map();
  for (const init of graph.initializer ?? []) {
    if (!init.name) continue;
    const v = tensorValues(init);
    if (v) m.set(init.name, v);
  }
  return m;
}

const UNARY = new Set([
  "Relu", "LeakyRelu", "PRelu", "Sigmoid", "HardSigmoid", "HardSwish", "Tanh", "Clip", "Elu", "Selu", "Celu",
  "Softmax", "LogSoftmax", "Hardmax", "BatchNormalization", "InstanceNormalization", "LayerNormalization",
  "GroupNormalization", "LRN", "Dropout", "Identity", "Erf", "Gelu", "Mish", "Softplus", "Softsign", "ThresholdedRelu",
  "Neg", "Abs", "Sqrt", "Exp", "Log", "Reciprocal", "Floor", "Ceil", "Round", "Sign", "Not", "Sin", "Cos",
  "Cast", "CastLike", "QuantizeLinear", "DequantizeLinear", "IsNaN", "IsInf", "CumSum",
]);

const BINARY = new Set([
  "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Mean", "Sum", "Mod", "And", "Or", "Xor",
  "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "BitShift", "Where",
]);

const REDUCE = new Set([
  "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
  "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare", "ArgMax", "ArgMin",
]);

const normAxis = (a: number, rank: number) => (a < 0 ? a + rank : a);
const known = (s: Shape) => s.every((d) => d !== DYN);

/** Spatial output geometry shared by the ops that slide a window (conv, deconv, pool). */
function windowOut(x: Shape, kernel: number[], attrs: Record<string, unknown>, cout: number, transposed = false): Shape {
  const nsp = x.length - 2;
  const strides = attrInts(attrs, "strides") ?? [];
  const dil = attrInts(attrs, "dilations") ?? [];
  const pads = attrInts(attrs, "pads") ?? [];
  const autoPad = attrStr(attrs, "auto_pad");
  const outPad = attrInts(attrs, "output_padding") ?? [];
  const explicit = attrInts(attrs, "output_shape");
  const ceil = attrInt(attrs, "ceil_mode", 0) === 1;
  const sp: number[] = [];
  for (let i = 0; i < nsp; i++) {
    const inSize = x[2 + i];
    const k = kernel[i] ?? kernel[0] ?? 1;
    const s = strides[i] ?? strides[0] ?? 1;
    const d = dil[i] ?? dil[0] ?? 1;
    const pb = pads[i] ?? 0;
    const pe = pads[i + nsp] ?? 0;
    if (!transposed) sp.push(spatialOut(inSize, k, s, pb, pe, d, autoPad, ceil));
    else if (explicit?.[i] != null) sp.push(explicit[i]);
    else if (inSize === DYN) sp.push(DYN);
    else if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") sp.push(inSize * s);
    else sp.push(s * (inSize - 1) + (outPad[i] ?? 0) + d * (k - 1) + 1 - pb - pe);
  }
  return [x[0], cout, ...sp];
}

/**
 * Best-effort shape inference for every output of a node, given resolved input
 * shapes and any constant input values (reshape targets, split sizes, resize
 * scales, slice bounds). Entries are undefined where the op isn't modeled;
 * callers keep whatever value_info already provided. Covers the ops common in
 * vision/transformer exports; it is not a full ONNX shape-inference engine.
 */
export function inferShapes(
  op: string,
  attrs: Record<string, unknown>,
  inShapes: (Shape | undefined)[],
  inConsts: (number[] | undefined)[],
  nOut: number,
): (Shape | undefined)[] {
  const x = inShapes[0];
  const out: (Shape | undefined)[] = new Array(nOut).fill(undefined);

  if (op === "Split") {
    if (!x) return out;
    const a = normAxis(attrInt(attrs, "axis", 0), x.length);
    const dim = x[a];
    let sizes = inConsts[1] ?? attrInts(attrs, "split");
    if (!sizes || sizes.length !== nOut) {
      if (dim === DYN) sizes = new Array(nOut).fill(DYN);
      else {
        const chunk = Math.ceil(dim / nOut);
        sizes = Array.from({ length: nOut }, (_, i) => Math.max(0, Math.min(chunk, dim - i * chunk)));
      }
    }
    return sizes.map((n) => {
      const s = [...x];
      s[a] = n;
      return s;
    });
  }

  out[0] = inferOutputShape(op, attrs, inShapes, inConsts) ?? undefined;
  // TopK's indices output matches its values output.
  if (op === "TopK" && out[0] && nOut > 1) out[1] = [...out[0]];
  return out;
}

/** Shape of a node's first output, or null when the op isn't modeled. */
export function inferOutputShape(
  op: string,
  attrs: Record<string, unknown>,
  inShapes: (Shape | undefined)[],
  inConsts: (number[] | undefined)[] = [],
): Shape | null {
  const x = inShapes[0];
  const w = inShapes[1];

  if (UNARY.has(op)) return x ? [...x] : null;

  if (BINARY.has(op)) {
    // With an operand missing, assume the known one is the larger (the common
    // activation-times-constant case) rather than giving up on the chain.
    const valid = inShapes.filter((s): s is Shape => !!s);
    return valid.length ? valid.reduce((a, b) => broadcast(a, b)) : null;
  }

  if (REDUCE.has(op)) {
    if (!x) return null;
    const keep = attrInt(attrs, "keepdims", 1);
    const axesAttr = attrInts(attrs, "axes") ?? (op.startsWith("Arg") ? [attrInt(attrs, "axis", 0)] : undefined);
    const axes = (axesAttr ?? inConsts[1] ?? x.map((_, i) => i)).map((a) => normAxis(a, x.length));
    const s: Shape = [];
    x.forEach((d, i) => {
      if (!axes.includes(i)) s.push(d);
      else if (keep) s.push(1);
    });
    return s;
  }

  switch (op) {
    case "Conv":
    case "ConvInteger":
    case "QLinearConv": {
      const wi = op === "QLinearConv" ? inShapes[3] : w;
      if (!x || !wi || x.length < 3) return null;
      const kernel = attrInts(attrs, "kernel_shape") ?? wi.slice(2);
      return windowOut(x, kernel, attrs, wi[0]);
    }

    case "ConvTranspose": {
      if (!x || !w || x.length < 3) return null;
      const group = attrInt(attrs, "group", 1);
      const kernel = attrInts(attrs, "kernel_shape") ?? w.slice(2);
      return windowOut(x, kernel, attrs, w[1] * group, true);
    }

    case "MaxPool":
    case "AveragePool":
    case "LpPool": {
      if (!x || x.length < 3) return null;
      const kernel = attrInts(attrs, "kernel_shape") ?? [1, 1];
      // Pool strides default to 1 per the spec.
      return windowOut(x, kernel, attrs, x[1]);
    }

    case "GlobalAveragePool":
    case "GlobalMaxPool":
    case "GlobalLpPool": {
      if (!x || x.length < 2) return null;
      return [x[0], x[1], ...x.slice(2).map(() => 1)];
    }

    case "Gemm": {
      if (!x || !w) return null;
      const transA = attrInt(attrs, "transA", 0);
      const transB = attrInt(attrs, "transB", 0);
      const M = transA ? x[1] : x[0];
      const N = transB ? w[0] : w[1];
      return [M, N];
    }

    case "MatMul":
    case "MatMulInteger": {
      if (!x || !w) return null;
      if (w.length === 1) return x.slice(0, -1);
      if (x.length === 1) return [...w.slice(0, -2), w[w.length - 1]];
      const M = x[x.length - 2] ?? DYN;
      const N = w[w.length - 1] ?? DYN;
      const batch = broadcast(x.slice(0, -2), w.slice(0, -2));
      return [...batch, M, N];
    }

    case "Flatten": {
      if (!x) return null;
      const a = normAxis(attrInt(attrs, "axis", 1), x.length);
      const left = x.slice(0, a);
      const right = x.slice(a);
      return [known(left) ? prod(left) : DYN, known(right) ? prod(right) : DYN];
    }

    case "Concat": {
      const valid = inShapes.filter((s): s is Shape => !!s);
      if (valid.length === 0) return null;
      const base = [...valid[0]];
      const a = normAxis(attrInt(attrs, "axis", 0), base.length);
      // An input with an unknown shape makes the extent unknown, rather than
      // silently dropping its contribution (which undercounts channels).
      if (valid.length < inShapes.length) {
        base[a] = DYN;
        return base;
      }
      let sum = 0;
      for (const s of valid) {
        if (s[a] === DYN) {
          sum = DYN;
          break;
        }
        sum += s[a];
      }
      base[a] = sum;
      return base;
    }

    case "Reshape": {
      const target = inConsts[1];
      if (!target) return null;
      const allowZero = attrInt(attrs, "allowzero", 0) === 1;
      const s = target.map((d, i) => (d === 0 && !allowZero ? (x?.[i] ?? DYN) : d));
      const inferIdx = s.indexOf(-1);
      if (inferIdx >= 0) {
        const rest = s.filter((_, i) => i !== inferIdx);
        s[inferIdx] = x && known(x) && known(rest) ? prod(x) / Math.max(1, prod(rest)) : DYN;
      }
      return s.map((d) => (d < 0 ? DYN : d));
    }

    case "Transpose": {
      if (!x) return null;
      const perm = attrInts(attrs, "perm") ?? x.map((_, i) => x.length - 1 - i);
      return perm.map((p) => x[p] ?? DYN);
    }

    case "Squeeze": {
      if (!x) return null;
      const axes = (attrInts(attrs, "axes") ?? inConsts[1])?.map((a) => normAxis(a, x.length));
      return axes ? x.filter((_, i) => !axes.includes(i)) : x.filter((d) => d !== 1);
    }

    case "Unsqueeze": {
      if (!x) return null;
      const axesRaw = attrInts(attrs, "axes") ?? inConsts[1];
      if (!axesRaw) return null;
      const rank = x.length + axesRaw.length;
      const s = [...x];
      for (const a of axesRaw.map((v) => normAxis(v, rank)).sort((p, q) => p - q)) s.splice(a, 0, 1);
      return s;
    }

    case "Slice": {
      if (!x) return null;
      const starts = attrInts(attrs, "starts") ?? inConsts[1];
      const ends = attrInts(attrs, "ends") ?? inConsts[2];
      if (!starts || !ends) return null;
      const axes = (attrInts(attrs, "axes") ?? inConsts[3] ?? starts.map((_, i) => i)).map((a) => normAxis(a, x.length));
      const steps = inConsts[4] ?? starts.map(() => 1);
      const s = [...x];
      axes.forEach((a, i) => {
        const dim = x[a];
        if (dim === DYN) return;
        const step = steps[i] ?? 1;
        const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v < 0 ? v + dim : v));
        s[a] = step > 0
          ? Math.max(0, Math.ceil((clamp(ends[i], 0, dim) - clamp(starts[i], 0, dim)) / step))
          : Math.max(0, Math.ceil((clamp(starts[i], -1, dim - 1) - clamp(ends[i], -1, dim - 1)) / -step));
      });
      return s;
    }

    case "Resize":
    case "Upsample": {
      if (!x) return null;
      // Resize inputs: X, roi, scales, sizes. Upsample (deprecated): X, scales.
      const sizes = op === "Resize" ? inConsts[3] : undefined;
      if (sizes && sizes.length === x.length) return [...sizes];
      const scales = (op === "Resize" ? inConsts[2] : inConsts[1]) ?? (attrs.scales as number[] | undefined);
      if (!scales || scales.length !== x.length) return null;
      return x.map((d, i) => (d === DYN ? DYN : Math.floor(d * scales[i])));
    }

    case "Pad": {
      if (!x) return null;
      const pads = attrInts(attrs, "pads") ?? inConsts[1];
      if (!pads) return null;
      const axes = inConsts[3]?.map((a) => normAxis(a, x.length)) ?? x.map((_, i) => i);
      const n = axes.length;
      const s = [...x];
      axes.forEach((a, i) => {
        if (s[a] !== DYN) s[a] += (pads[i] ?? 0) + (pads[i + n] ?? 0);
      });
      return s;
    }

    case "Expand": {
      const target = inConsts[1];
      if (!target) return x ? [...x] : null;
      const t = target.map((d) => (d < 0 ? DYN : d));
      return x ? broadcast(x, t) : t;
    }

    case "Tile": {
      const reps = inConsts[1];
      if (!x || !reps) return null;
      return x.map((d, i) => (d === DYN ? DYN : d * (reps[i] ?? 1)));
    }

    case "Shape":
      return x ? [x.length] : null;

    case "Gather": {
      const idx = inShapes[1];
      if (!x || !idx) return null;
      const a = normAxis(attrInt(attrs, "axis", 0), x.length);
      return [...x.slice(0, a), ...idx, ...x.slice(a + 1)];
    }

    case "DepthToSpace":
    case "SpaceToDepth": {
      if (!x || x.length !== 4) return null;
      const b = attrInt(attrs, "blocksize", 1);
      const [N, C, H, W] = x;
      const mul = (d: number, f: number) => (d === DYN ? DYN : d * f);
      const div = (d: number, f: number) => (d === DYN ? DYN : Math.floor(d / f));
      return op === "DepthToSpace"
        ? [N, div(C, b * b), mul(H, b), mul(W, b)]
        : [N, mul(C, b * b), div(H, b), div(W, b)];
    }

    case "TopK": {
      const k = inConsts[1]?.[0];
      if (!x || k == null) return null;
      const s = [...x];
      s[normAxis(attrInt(attrs, "axis", -1), x.length)] = k;
      return s;
    }

    case "ConstantOfShape": {
      const target = inConsts[0];
      return target ? target.map((d) => (d < 0 ? DYN : d)) : null;
    }

    default:
      return null;
  }
}

/**
 * Fold the small integer computations exporters use to build shape vectors
 * (Shape → Gather → Unsqueeze → Concat → Reshape). Returns the first output's
 * value when every needed input is a known constant, else undefined.
 */
export function foldConst(
  op: string,
  attrs: Record<string, unknown>,
  inShapes: (Shape | undefined)[],
  inConsts: (number[] | undefined)[],
): number[] | undefined {
  const c0 = inConsts[0];
  const c1 = inConsts[1];
  switch (op) {
    case "Shape": {
      // Symbolic dims fold to -1 (DYN): Gather/Concat carry it through, and a
      // Reshape reads it as "infer this dim", which is what it means here.
      const x = inShapes[0];
      if (!x) return undefined;
      const r = x.length;
      const start = normAxis(attrInt(attrs, "start", 0), r);
      const end = attrs.end != null ? normAxis(attrInt(attrs, "end", r), r) : r;
      return x.slice(start, end);
    }
    case "Size": {
      const x = inShapes[0];
      return x && known(x) ? [prod(x)] : undefined;
    }
    case "Identity":
    case "Cast":
    case "Squeeze":
    case "Unsqueeze":
    case "Flatten":
    case "Reshape":
      return c0 ? [...c0] : undefined;
    case "Gather": {
      if (!c0 || !c1) return undefined;
      const out = c1.map((i) => c0[i < 0 ? i + c0.length : i]);
      return out.every((v) => v != null) ? out : undefined;
    }
    case "Concat":
      if (inConsts.length === 0 || inConsts.some((c) => !c)) return undefined;
      return inConsts.flatMap((c) => c!);
    case "Slice": {
      const ends = inConsts[2];
      if (!c0 || !c1 || !ends || (inConsts[4]?.[0] ?? 1) !== 1) return undefined;
      const n = c0.length;
      const fix = (v: number) => Math.max(0, Math.min(n, v < 0 ? v + n : v));
      return c0.slice(fix(c1[0]), fix(ends[0]));
    }
    case "Add":
    case "Sub":
    case "Mul":
    case "Div": {
      // Arithmetic on an unknown (-1) dim would produce a plausible-looking wrong value.
      if (!c0 || !c1 || c0.includes(DYN) || c1.includes(DYN)) return undefined;
      const len = Math.max(c0.length, c1.length);
      if (len > MAX_CONST) return undefined;
      const at = (c: number[], i: number) => (c.length === 1 ? c[0] : c[i]);
      const f =
        op === "Add" ? (a: number, b: number) => a + b
        : op === "Sub" ? (a: number, b: number) => a - b
        : op === "Mul" ? (a: number, b: number) => a * b
        : (a: number, b: number) => Math.trunc(a / b);
      return Array.from({ length: len }, (_, i) => f(at(c0, i), at(c1, i)));
    }
    default:
      return undefined;
  }
}

/** Render a shape for display: "1×3×224×224", dynamic dims as "?". */
export function formatShape(s: Shape | undefined): string {
  if (!s || s.length === 0) return "?";
  return s.map((d) => (d === DYN ? "?" : String(d))).join("×");
}

export { DYN, prod };
