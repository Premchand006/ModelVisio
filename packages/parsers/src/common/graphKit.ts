import type { ModelLayer } from "../types";
import { opMeta } from "../onnx/opmeta";
import { DYN, formatShape, prod } from "../onnx/shapes";
import { fromHalf } from "./float16";

// Shared building blocks for the graph parsers that are not ONNX-backed
// (TensorFlow GraphDef, Keras config, PaddlePaddle ProgramDesc). Shapes are
// number[] with -1 for a dynamic/unknown dim, the same convention as onnx/.

export type Shape = number[];
export { DYN, formatShape, prod };

export const DT_SIZE: Record<string, number> = {
  float32: 4, float16: 2, bfloat16: 2, float64: 8, int8: 1, uint8: 1, int16: 2, uint16: 2,
  int32: 4, uint32: 4, int64: 8, uint64: 8, bool: 1, complex64: 8, complex128: 16,
};

export const isFloatDtype = (dt: string) => /^(float|bfloat|half|double)/.test(dt);

/** Bytes of one activation of `shape` (dynamic dims count as 1). */
export function memBytes(shape: Shape | undefined, dt: string): number {
  if (!shape || shape.length === 0) return 0;
  return prod(shape) * (DT_SIZE[dt] ?? 4);
}

/** Number of elements; dynamic dims are treated as 1 (per-sample cost). */
export const numel = (shape: Shape | undefined) => (shape && shape.length ? prod(shape) : shape ? 1 : 0);

// ── Cost formulas (layout-agnostic: callers pass the per-output reduction size) ──

export type Cost = { flops: number; macs: number };
export const NO_COST: Cost = { flops: 0, macs: 0 };

/** Linear ops (conv, dense, matmul): every output element accumulates `k` products. */
export function linearCost(out: Shape | undefined, k: number): Cost {
  if (!out || !(k > 0)) return NO_COST;
  const macs = prod(out) * k;
  return { macs, flops: 2 * macs };
}

/** Elementwise / normalization / pooling: `perElem` flops per output element. */
export function elemCost(out: Shape | undefined, perElem = 1): Cost {
  if (!out) return NO_COST;
  return { macs: 0, flops: prod(out) * perElem };
}

// ── Shape helpers ────────────────────────────────────────────────────────────

/** Spatial output size for SAME/VALID (TF/Keras) padding. */
export function padOut(inSize: number, k: number, stride: number, padding: string, dilation = 1): number {
  if (inSize < 0) return DYN;
  if (/same/i.test(padding)) return Math.ceil(inSize / stride);
  const effK = dilation * (k - 1) + 1;
  return Math.floor((inSize - effK) / stride) + 1;
}

/** Numpy-style broadcast of two shapes. */
export function broadcast(a: Shape, b: Shape): Shape {
  const out: Shape = [];
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    out.unshift(da === DYN || db === DYN ? (da === 1 ? db : db === 1 ? da : DYN) : Math.max(da, db));
  }
  return out;
}

// ── Weight statistics ────────────────────────────────────────────────────────

export type WStat = NonNullable<ModelLayer["w"]>;
const MAX_SAMPLES = 100_000;
const r4 = (n: number) => Math.round(n * 1e4) / 1e4;
const BF_F32 = new Float32Array(1);
const BF_U32 = new Uint32Array(BF_F32.buffer);

/** Read sampled numeric values from little-endian raw bytes of a given dtype. */
export function sampleBytes(bytes: Uint8Array, dt: string): number[] | null {
  const bpe = DT_SIZE[dt];
  if (!bpe || bytes.byteLength < bpe) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = Math.floor(bytes.byteLength / bpe);
  const step = Math.max(1, Math.ceil(count / MAX_SAMPLES));
  const out: number[] = [];
  for (let i = 0; i < count; i += step) {
    const o = i * bpe;
    switch (dt) {
      case "float32": out.push(view.getFloat32(o, true)); break;
      case "float64": out.push(view.getFloat64(o, true)); break;
      case "float16": out.push(fromHalf(view.getUint16(o, true))); break;
      case "bfloat16": {
        // bfloat16 is the top half of a float32 bit pattern.
        BF_U32[0] = view.getUint16(o, true) << 16;
        out.push(BF_F32[0]);
        break;
      }
      case "int8": out.push(view.getInt8(o)); break;
      case "uint8": out.push(view.getUint8(o)); break;
      case "int16": out.push(view.getInt16(o, true)); break;
      case "uint16": out.push(view.getUint16(o, true)); break;
      case "int32": out.push(view.getInt32(o, true)); break;
      case "uint32": out.push(view.getUint32(o, true)); break;
      case "int64": out.push(Number(view.getBigInt64(o, true))); break;
      default: return null;
    }
  }
  return out;
}

/** Shape + size + (when values are given) min/max/mean/std/sparsity. */
export function weightStats(shape: Shape, values?: ArrayLike<number> | null): WStat {
  const size = numel(shape);
  if (!values || values.length === 0) return { shape, size };
  const step = Math.max(1, Math.ceil(values.length / MAX_SAMPLES));
  let min = Infinity, max = -Infinity, sum = 0, zeros = 0, n = 0;
  for (let i = 0; i < values.length; i += step) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    if (Math.abs(v) < 1e-9) zeros++;
    n++;
  }
  const mean = sum / n;
  let variance = 0;
  for (let i = 0; i < values.length; i += step) variance += (values[i] - mean) ** 2;
  variance /= n;
  return { shape, size, min: r4(min), max: r4(max), mean: r4(mean), std: r4(Math.sqrt(variance)), sparse: r4(zeros / n) };
}

// ── Layer construction ───────────────────────────────────────────────────────

export type LayerSpec = {
  id: number;
  name: string;
  /** Framework-native op name shown in the UI (e.g. "Conv2D", "conv2d"). */
  type: string;
  /** ONNX-equivalent op used for math/insight/qSens/group priors. */
  canon: string;
  shape?: Shape;
  dt?: string;
  params?: number;
  cost?: Cost;
  group?: ModelLayer["group"];
  ins?: { n: string; s?: Shape }[];
  outs?: { n: string; s?: Shape }[];
  attr?: Record<string, unknown>;
  w?: WStat | null;
  insight?: string;
  framework: string;
};

export function makeLayer(s: LayerSpec): ModelLayer {
  const meta = opMeta(s.canon);
  const dt = s.dt ?? "float32";
  const known = s.canon in KNOWN_CANON;
  return {
    id: s.id,
    name: s.name,
    type: s.type,
    op: s.type,
    shape: formatShape(s.shape),
    dt,
    params: s.params ?? 0,
    flops: s.cost?.flops ?? 0,
    macs: s.cost?.macs ?? 0,
    mem: memBytes(s.shape, dt),
    group: s.group ?? meta.group,
    ins: s.ins ?? [],
    outs: s.outs ?? [],
    attr: s.attr ?? {},
    w: s.w ?? null,
    math: meta.math,
    insight: s.insight ?? (known ? meta.insight : `${s.framework} ${s.type} operator.`),
    qSens: meta.qSens,
    compIssues: [],
  };
}

// ONNX op names that opMeta has curated text for (others fall back to a generic insight).
const KNOWN_CANON: Record<string, true> = Object.fromEntries(
  [
    "Conv", "ConvTranspose", "Gemm", "MatMul", "BatchNormalization", "Relu", "LeakyRelu", "Sigmoid", "Tanh",
    "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "Flatten", "Reshape", "Transpose",
    "Concat", "Add", "Mul", "Softmax", "LogSoftmax",
  ].map((k) => [k, true]),
);

/** Deduplicating edge collector (no self-loops). */
export class EdgeSet {
  readonly edges: [number, number][] = [];
  private seen = new Set<string>();
  add(from: number | undefined, to: number): void {
    if (from == null || from === to) return;
    const k = `${from}>${to}`;
    if (this.seen.has(k)) return;
    this.seen.add(k);
    this.edges.push([from, to]);
  }
}

/** Replace dynamic dims with 0 for Model.inputShape/outputShape. */
export const cleanShape = (s: Shape | undefined): number[] => (s ?? []).map((d) => (d < 0 ? 0 : d));
