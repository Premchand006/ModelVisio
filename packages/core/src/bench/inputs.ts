// Graph-input discovery, dynamic-dimension resolution and synthetic input
// data for live benchmarking. Pure: no React, no DOM, no onnxruntime import.

import type { Model } from "@modelvisio/parsers";

/** A dimension as declared by the model: a fixed size, a symbolic name, or unknown (-1 / null). */
export type DeclaredDim = number | string | null;

/** One graph input as declared (before dynamic dims are resolved). */
export type InputSpec = { name: string; dtype: string; shape: DeclaredDim[] };

/** A graph input with every dimension resolved to a concrete size. */
export type ResolvedInput = {
  name: string;
  dtype: string;
  dims: number[];
  /** Total byte size of the synthesized tensor. */
  bytes: number;
  /** Axes whose size came from an override or a default, not the model. */
  dynamicAxes: number[];
};

/**
 * User-editable sizes for dynamic dims. Keyed by `"<input>:<axis>"` (always
 * available) or by the symbolic dim name (e.g. `"sequence_length"`) when the
 * runtime reports one. The per-axis key wins.
 */
export type DimOverrides = Record<string, number>;

export type InputFill = "random" | "zeros";

/** Hard cap on synthesized elements per input, so a bad override can't OOM the tab. */
export const MAX_INPUT_ELEMENTS = 256 * 1024 * 1024;

/** Key used in DimOverrides for one axis of one input. */
export const dimKey = (input: string, axis: number) => `${input}:${axis}`;

const isFixed = (d: DeclaredDim): d is number => typeof d === "number" && Number.isFinite(d) && d > 0;

/**
 * Graph inputs of a parsed Model. The ONNX parser emits one synthetic layer per
 * real graph input (op "Input", group "input") whose output carries the shape
 * with dynamic dims stored as -1. Falls back to `model.inputShape` when a
 * parser emits no input layers. Shapes here are a UI preview: the benchmark
 * worker re-reads the authoritative input metadata from the runtime session.
 */
export function inputSpecsFromModel(model: Model): InputSpec[] {
  const specs: InputSpec[] = [];
  for (const l of model.layers) {
    if (l.op !== "Input" || l.group !== "input") continue;
    const s = l.outs[0]?.s;
    specs.push({ name: l.outs[0]?.n || l.name, dtype: l.dt || "float32", shape: (s ?? []).map((d) => (d > 0 ? d : null)) });
  }
  if (specs.length === 0 && model.inputShape.length > 0) {
    specs.push({ name: "input", dtype: "float32", shape: model.inputShape.map((d) => (d > 0 ? d : null)) });
  }
  return specs;
}

/**
 * Default for a dynamic dim nobody overrode. Axis 0 is treated as batch → 1.
 * Image-like rank-4 inputs get 3 channels and 224×224 spatial; the second axis
 * of rank-2/3 inputs (token / sequence length) gets 128; anything else 1.
 */
export function defaultDynamicDim(shape: readonly DeclaredDim[], axis: number): number {
  if (axis === 0) return 1;
  const rank = shape.length;
  if (rank === 4) return axis === 1 ? 3 : 224;
  if ((rank === 2 || rank === 3) && axis === 1) return 128;
  return 1;
}

/** Resolve every dim of one input to a concrete size (override → declared → default). */
export function resolveDims(spec: InputSpec, overrides: DimOverrides = {}): { dims: number[]; dynamicAxes: number[] } {
  const dims: number[] = [];
  const dynamicAxes: number[] = [];
  spec.shape.forEach((d, axis) => {
    if (isFixed(d)) {
      dims.push(d);
      return;
    }
    dynamicAxes.push(axis);
    const byAxis = overrides[dimKey(spec.name, axis)];
    const bySym = typeof d === "string" && d ? overrides[d] : undefined;
    const pick = [byAxis, bySym].find((v) => typeof v === "number" && Number.isFinite(v) && v >= 1);
    dims.push(pick != null ? Math.floor(pick) : defaultDynamicDim(spec.shape, axis));
  });
  return { dims, dynamicAxes };
}

/** Bytes per element for the dtypes the benchmark can synthesize (strings counted as 1). */
export const DTYPE_BYTES: Record<string, number> = {
  float32: 4, float64: 8, float16: 2,
  int8: 1, uint8: 1, int16: 2, uint16: 2, int32: 4, uint32: 4, int64: 8, uint64: 8,
  bool: 1, string: 1,
};

export const isSupportedDtype = (dt: string) => dt in DTYPE_BYTES;

const elemCount = (dims: readonly number[]) => dims.reduce((a, b) => a * b, 1);

/** Resolve all inputs, validating dtypes and size. Throws a user-facing Error. */
export function resolveInputs(specs: readonly InputSpec[], overrides: DimOverrides = {}): ResolvedInput[] {
  return specs.map((spec) => {
    if (!isSupportedDtype(spec.dtype)) {
      throw new Error(`Input "${spec.name}" has dtype ${spec.dtype}, which the benchmark can't synthesize yet.`);
    }
    const { dims, dynamicAxes } = resolveDims(spec, overrides);
    const n = elemCount(dims);
    if (n > MAX_INPUT_ELEMENTS) {
      throw new Error(`Input "${spec.name}" [${dims.join("×")}] is too large to synthesize (${n.toLocaleString()} elements). Lower the dynamic dims.`);
    }
    return { name: spec.name, dtype: spec.dtype, dims, bytes: n * DTYPE_BYTES[spec.dtype], dynamicAxes };
  });
}

/** Deterministic PRNG (mulberry32) so runs are reproducible from a seed. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** float32 → IEEE-754 binary16 bits, round-to-nearest-even. */
export function toHalfBits(v: number): number {
  F32[0] = v;
  const x = U32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / NaN
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00; // overflow → inf
  if (exp <= 0) {
    // subnormal half (or underflow to signed zero)
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && h & 1)) h++;
    return sign | h;
  }
  let h = (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++; // a carry into the exponent is still correct
  return sign | h;
}
const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

export type SynthData =
  | Float32Array | Float64Array | Uint16Array | Int8Array | Uint8Array | Int16Array
  | Int32Array | Uint32Array | BigInt64Array | BigUint64Array | string[];

/**
 * Synthetic data for one input. Floats: uniform [0, 1) (random) or 0. Integers
 * and bools: {0, 1} (random) or 0, small enough to be valid indices / masks for
 * embedding, gather and attention-mask inputs. float16 is returned as raw
 * binary16 bits in a Uint16Array, which onnxruntime accepts for float16.
 */
export function synthesizeData(dtype: string, count: number, fill: InputFill, rand: () => number = Math.random): SynthData {
  const r = fill === "random";
  const bit = () => (r && rand() >= 0.5 ? 1 : 0);
  switch (dtype) {
    case "float32": { const a = new Float32Array(count); if (r) for (let i = 0; i < count; i++) a[i] = rand(); return a; }
    case "float64": { const a = new Float64Array(count); if (r) for (let i = 0; i < count; i++) a[i] = rand(); return a; }
    case "float16": { const a = new Uint16Array(count); if (r) for (let i = 0; i < count; i++) a[i] = toHalfBits(rand()); return a; }
    case "int8": { const a = new Int8Array(count); if (r) for (let i = 0; i < count; i++) a[i] = bit(); return a; }
    case "uint8": case "bool": { const a = new Uint8Array(count); if (r) for (let i = 0; i < count; i++) a[i] = bit(); return a; }
    case "int16": { const a = new Int16Array(count); if (r) for (let i = 0; i < count; i++) a[i] = bit(); return a; }
    case "uint16": { const a = new Uint16Array(count); if (r) for (let i = 0; i < count; i++) a[i] = bit(); return a; }
    case "int32": { const a = new Int32Array(count); if (r) for (let i = 0; i < count; i++) a[i] = bit(); return a; }
    case "uint32": { const a = new Uint32Array(count); if (r) for (let i = 0; i < count; i++) a[i] = bit(); return a; }
    case "int64": { const a = new BigInt64Array(count); if (r) for (let i = 0; i < count; i++) a[i] = BigInt(bit()); return a; }
    case "uint64": { const a = new BigUint64Array(count); if (r) for (let i = 0; i < count; i++) a[i] = BigInt(bit()); return a; }
    case "string": return new Array<string>(count).fill(r ? "a" : "");
    default: throw new Error(`Unsupported input dtype: ${dtype}`);
  }
}
