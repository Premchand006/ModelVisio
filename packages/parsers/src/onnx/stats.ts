import { DT_BYTES, num, numArr, type RawTensor } from "./raw";

export type WeightStat = {
  shape: number[];
  size: number;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
  sparse?: number;
};

const MAX_SAMPLES = 100_000;
const ZERO_EPS = 1e-9;

/** Decode an IEEE half-precision (float16) value from its 16-bit pattern. */
function halfToFloat(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  let val: number;
  if (exp === 0) {
    val = frac * Math.pow(2, -24);
  } else if (exp === 0x1f) {
    val = frac ? NaN : Infinity;
  } else {
    val = (1 + frac / 1024) * Math.pow(2, exp - 15);
  }
  return sign ? -val : val;
}

/** Read tensor element values (sampled) for statistics, or null if unreadable. */
function sampleValues(t: RawTensor): number[] | null {
  const dt = t.data_type ?? 0;

  // Prefer the explicit typed-data fields when present.
  const typed =
    (t.float_data && t.float_data.length && t.float_data) ||
    (t.double_data && t.double_data.length && t.double_data) ||
    (t.int32_data && t.int32_data.length && t.int32_data) ||
    null;
  if (typed) return strideSample(typed.map(Number));
  if (t.int64_data && t.int64_data.length) return strideSample(numArr(t.int64_data));

  const raw = t.raw_data;
  if (!raw || raw.length === 0) return null;
  const bpe = DT_BYTES[dt];
  if (!bpe) return null;

  const count = Math.floor(raw.length / bpe);
  const step = Math.max(1, Math.floor(count / MAX_SAMPLES));
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const out: number[] = [];
  for (let i = 0; i < count; i += step) {
    const off = i * bpe;
    switch (dt) {
      case 1: out.push(view.getFloat32(off, true)); break; // float32
      case 11: out.push(view.getFloat64(off, true)); break; // float64
      case 10: out.push(halfToFloat(view.getUint16(off, true))); break; // float16
      case 6: out.push(view.getInt32(off, true)); break; // int32
      case 3: out.push(view.getInt8(off)); break; // int8
      case 2: out.push(view.getUint8(off)); break; // uint8
      case 5: out.push(view.getInt16(off, true)); break; // int16
      case 7: out.push(Number(view.getBigInt64(off, true))); break; // int64
      default: return null;
    }
  }
  return out;
}

function strideSample(arr: number[]): number[] {
  if (arr.length <= MAX_SAMPLES) return arr;
  const step = Math.ceil(arr.length / MAX_SAMPLES);
  const out: number[] = [];
  for (let i = 0; i < arr.length; i += step) out.push(arr[i]);
  return out;
}

const round = (n: number) => Math.round(n * 1e4) / 1e4;

/** Compute weight statistics for an initializer tensor. */
export function weightStat(t: RawTensor): WeightStat {
  const shape = numArr(t.dims);
  const size = shape.reduce((a, b) => a * b, shape.length ? 1 : 0);
  const vals = sampleValues(t);
  if (!vals || vals.length === 0) return { shape, size };

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let zeros = 0;
  for (const v of vals) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    if (Math.abs(v) < ZERO_EPS) zeros++;
  }
  const mean = sum / vals.length;
  let variance = 0;
  for (const v of vals) variance += (v - mean) * (v - mean);
  variance /= vals.length;

  return {
    shape,
    size,
    min: round(min),
    max: round(max),
    mean: round(mean),
    std: round(Math.sqrt(variance)),
    sparse: round(zeros / vals.length),
  };
}

/** Number of scalar parameters in a tensor (product of dims). */
export function tensorParams(t: RawTensor): number {
  const dims = numArr(t.dims);
  return dims.reduce((a, b) => a * b, dims.length ? 1 : 0);
}

export { num };
