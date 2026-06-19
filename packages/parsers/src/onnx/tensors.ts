import { modelProtoType } from "./proto";
import { numArr, type I64, type Long, type RawModel, type RawTensor } from "./raw";

export type OnnxTensor = { name: string; dataType: number; dims: number[]; bytes: Uint8Array };

const ONNX_DATA_LOCATION_EXTERNAL = 1;

/** Exact BigInt from a protobufjs int64 (number or Long) without a double round-trip. */
function i64ToBig(v: I64): bigint {
  if (typeof v === "number") return BigInt(Math.trunc(v));
  return BigInt((v as Long).toString());
}

// Serialize a typed-data array to little-endian bytes (ONNX raw_data is already LE).
// Sub-32-bit dtypes (FLOAT16/BFLOAT16/UINT8/INT8/UINT16/INT16/BOOL) are packed into
// int32_data one value per slot — they must be written at their true width.
function typedToBytes(t: RawTensor): Uint8Array | null {
  const dt = t.data_type ?? 0;
  const set = (count: number, bpe: number, fn: (v: DataView, off: number, i: number) => void) => {
    const out = new Uint8Array(count * bpe);
    const view = new DataView(out.buffer);
    for (let i = 0; i < count; i++) fn(view, i * bpe, i);
    return out;
  };
  if (t.float_data && t.float_data.length) return set(t.float_data.length, 4, (v, o, i) => v.setFloat32(o, t.float_data![i], true));
  if (t.double_data && t.double_data.length) return set(t.double_data.length, 8, (v, o, i) => v.setFloat64(o, t.double_data![i], true));
  if (t.int64_data && t.int64_data.length) {
    const a = t.int64_data;
    return set(a.length, 8, (v, o, i) => v.setBigInt64(o, i64ToBig(a[i]), true));
  }
  if (t.int32_data && t.int32_data.length) {
    const a = t.int32_data;
    switch (dt) {
      case 10: case 16: return set(a.length, 2, (v, o, i) => v.setUint16(o, a[i] & 0xffff, true)); // float16 / bfloat16 bit patterns
      case 4: return set(a.length, 2, (v, o, i) => v.setUint16(o, a[i] & 0xffff, true));           // uint16
      case 5: return set(a.length, 2, (v, o, i) => v.setInt16(o, a[i], true));                     // int16
      case 2: case 9: return set(a.length, 1, (v, o, i) => v.setUint8(o, a[i] & 0xff));             // uint8 / bool
      case 3: return set(a.length, 1, (v, o, i) => v.setInt8(o, a[i]));                             // int8
      default: return set(a.length, 4, (v, o, i) => v.setInt32(o, a[i], true));                     // int32 + fallback
    }
  }
  return null;
}

/**
 * Extract initializer tensors with their full data bytes. Returns
 * { skippedExternal } for tensors whose data lives outside the file
 * (data_location = EXTERNAL, or a populated external_data with no inline data).
 */
export function extractOnnxTensors(input: ArrayBuffer | Uint8Array): { tensors: OnnxTensor[]; skippedExternal: number } {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const decoded = modelProtoType().decode(buf) as unknown as RawModel;
  const inits = decoded.graph?.initializer ?? [];
  const tensors: OnnxTensor[] = [];
  let skippedExternal = 0;
  for (const t of inits) {
    if (!t.name) continue;
    if (t.data_location === ONNX_DATA_LOCATION_EXTERNAL || (t.external_data && t.external_data.length)) {
      skippedExternal++;
      continue;
    }
    const bytes = t.raw_data && t.raw_data.length ? t.raw_data.slice() : typedToBytes(t);
    if (!bytes) continue;
    tensors.push({ name: t.name, dataType: t.data_type ?? 1, dims: numArr(t.dims), bytes });
  }
  return { tensors, skippedExternal };
}
