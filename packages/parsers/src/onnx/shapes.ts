import { attrInt, attrInts, attrStr } from "./attrs";
import { numArr, type RawGraph, type RawValueInfo } from "./raw";

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
): number {
  if (inSize === DYN) return DYN;
  if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
    return Math.ceil(inSize / stride);
  }
  const effK = dilation * (kernel - 1) + 1;
  return Math.floor((inSize + padBegin + padEnd - effK) / stride) + 1;
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

/**
 * Best-effort shape inference for a single node's first output, given resolved
 * input shapes. Returns null when the op isn't modeled — callers keep whatever
 * value_info already provided. This covers the ops common in vision/transformer
 * exports; it is not a full ONNX shape-inference engine.
 */
export function inferOutputShape(
  op: string,
  attrs: Record<string, unknown>,
  inShapes: (Shape | undefined)[],
): Shape | null {
  const x = inShapes[0];
  const w = inShapes[1];

  switch (op) {
    case "Relu":
    case "LeakyRelu":
    case "Sigmoid":
    case "Tanh":
    case "Clip":
    case "Elu":
    case "Softmax":
    case "LogSoftmax":
    case "BatchNormalization":
    case "InstanceNormalization":
    case "LRN":
    case "Dropout":
    case "Identity":
    case "Erf":
    case "Gelu":
      return x ? [...x] : null;

    case "Add":
    case "Sub":
    case "Mul":
    case "Div":
    case "Pow":
    case "Max":
    case "Min":
      if (x && w) return broadcast(x, w);
      return x ? [...x] : null;

    case "Conv":
    case "ConvInteger": {
      if (!x || !w || x.length < 3) return null;
      const [N, , H, W] = x;
      const cout = w[0];
      const kernel = attrInts(attrs, "kernel_shape") ?? [w[2], w[3] ?? w[2]];
      const strides = attrInts(attrs, "strides") ?? [1, 1];
      const dil = attrInts(attrs, "dilations") ?? [1, 1];
      const pads = attrInts(attrs, "pads") ?? [0, 0, 0, 0];
      const autoPad = attrStr(attrs, "auto_pad");
      const outH = spatialOut(H, kernel[0], strides[0], pads[0], pads[2], dil[0], autoPad);
      const outW = spatialOut(W, kernel[1], strides[1] ?? strides[0], pads[1] ?? pads[0], pads[3] ?? pads[2], dil[1] ?? dil[0], autoPad);
      return [N, cout, outH, outW];
    }

    case "MaxPool":
    case "AveragePool": {
      if (!x || x.length < 4) return null;
      const [N, C, H, W] = x;
      const kernel = attrInts(attrs, "kernel_shape") ?? [1, 1];
      const strides = attrInts(attrs, "strides") ?? kernel;
      const pads = attrInts(attrs, "pads") ?? [0, 0, 0, 0];
      const autoPad = attrStr(attrs, "auto_pad");
      const outH = spatialOut(H, kernel[0], strides[0], pads[0], pads[2], 1, autoPad);
      const outW = spatialOut(W, kernel[1], strides[1] ?? strides[0], pads[1] ?? pads[0], pads[3] ?? pads[2], 1, autoPad);
      return [N, C, outH, outW];
    }

    case "GlobalAveragePool":
    case "GlobalMaxPool": {
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

    case "MatMul": {
      if (!x || !w) return null;
      const M = x[x.length - 2] ?? DYN;
      const N = w[w.length - 1] ?? DYN;
      const batch = x.slice(0, -2);
      return [...batch, M, N];
    }

    case "Flatten": {
      if (!x) return null;
      const axis = attrInt(attrs, "axis", 1);
      const a = axis < 0 ? x.length + axis : axis;
      const left = x.slice(0, a);
      const right = x.slice(a);
      const has = (arr: Shape) => arr.some((d) => d === DYN);
      return [has(left) ? DYN : prod(left), has(right) ? DYN : prod(right)];
    }

    case "Concat": {
      const axis = attrInt(attrs, "axis", 0);
      const valid = inShapes.filter((s): s is Shape => !!s);
      if (valid.length === 0) return null;
      const base = [...valid[0]];
      const a = axis < 0 ? base.length + axis : axis;
      let sum = 0;
      for (const s of valid) {
        if (s[a] === DYN) { sum = DYN; break; }
        sum += s[a];
      }
      base[a] = sum;
      return base;
    }

    case "GlobalLpPool":
      return x ? [x[0], x[1], 1, 1] : null;

    default:
      return null;
  }
}

/** Render a shape for display: "1×3×224×224", dynamic dims as "?". */
export function formatShape(s: Shape | undefined): string {
  if (!s || s.length === 0) return "?";
  return s.map((d) => (d === DYN ? "?" : String(d))).join("×");
}

export { DYN, prod };
