// IEEE-754 float32 → float16 (binary16), exact round-to-nearest-even, handling
// normals, subnormals, overflow→±Inf, underflow→±0, and NaN. Uses a guard+sticky
// integer conversion (not the lossy 11-bit-truncation trick). The decoder lives
// in stats.ts (halfToFloat).

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** Encode a JS number as a 16-bit half-float bit pattern (0..0xFFFF). */
export function toHalf(value: number): number {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;

  if (exp === 0xff) return sign | (mant ? 0x7e00 : 0x7c00); // NaN (quiet) / Inf

  const unbiased = exp - 127;
  if (unbiased > 15) return sign | 0x7c00; // overflow → ±Inf

  if (unbiased >= -14) {
    // Normalized half: round the 23-bit mantissa to 10 bits, round-to-nearest-even.
    const m = mant >>> 13;
    const r = mant & 0x1fff;
    let h = ((unbiased + 15) << 10) | m;
    if (r > 0x1000 || (r === 0x1000 && (m & 1))) h++; // carry into exponent is intentional (may reach Inf)
    return sign | h;
  }

  if (unbiased >= -25) {
    // Subnormal half: shift the full 24-bit significand right with guard+sticky RNE.
    const full = mant | 0x800000;
    const shift = -1 - unbiased; // 14..24
    const m = full >>> shift;
    const half = 1 << (shift - 1);
    const r = full & ((1 << shift) - 1);
    let h = m;
    if (r > half || (r === half && (m & 1))) h++; // m==0x3FF→0x400 becomes smallest normal: correct
    return sign | h;
  }

  return sign; // magnitude below the round-to-even boundary → ±0
}

/** Decode a 16-bit half-float bit pattern to a JS number. */
export function fromHalf(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * Math.pow(2, -24);
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * Math.pow(2, exp - 15);
}

/** Convert a float16 Uint8Array (LE) to a float32 Uint8Array (LE). */
export function f16BytesToF32Bytes(src: Uint8Array): Uint8Array {
  if (src.byteLength % 2 !== 0) throw new Error(`f16BytesToF32Bytes: ${src.byteLength} bytes is not a multiple of 2.`);
  const count = src.byteLength / 2;
  const inView = new DataView(src.buffer, src.byteOffset, src.byteLength);
  const out = new Uint8Array(count * 4);
  const outView = new DataView(out.buffer);
  for (let i = 0; i < count; i++) outView.setFloat32(i * 4, fromHalf(inView.getUint16(i * 2, true)), true);
  return out;
}

/** Convert a float32 Uint8Array (LE) to a float16 Uint8Array (LE). */
export function f32BytesToF16Bytes(src: Uint8Array): Uint8Array {
  if (src.byteLength % 4 !== 0) {
    throw new Error(`f32BytesToF16Bytes: ${src.byteLength} bytes is not a multiple of 4 (corrupt float32 data?).`);
  }
  const count = src.byteLength / 4;
  const inView = new DataView(src.buffer, src.byteOffset, src.byteLength);
  const out = new Uint8Array(count * 2);
  const outView = new DataView(out.buffer);
  for (let i = 0; i < count; i++) {
    outView.setUint16(i * 2, toHalf(inView.getFloat32(i * 4, true)), true);
  }
  return out;
}
