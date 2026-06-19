// Write a valid .safetensors file: [u64 LE header length][JSON header][data].
// JSON maps name → { dtype, shape, data_offsets:[start,end] } over the data
// block. We pad the header with spaces so the data block starts 8-byte aligned
// (HuggingFace does the same; the spec only requires the JSON + offsets).

export type OutTensor = { name: string; dtype: string; shape: number[]; data: Uint8Array };

export function writeSafetensors(tensors: OutTensor[], metadata?: Record<string, string>): Uint8Array {
  const header: Record<string, unknown> = {};
  if (metadata) header.__metadata__ = metadata;
  let offset = 0;
  for (const t of tensors) {
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + t.data.byteLength] };
    offset += t.data.byteLength;
  }
  const totalData = offset;

  let json = JSON.stringify(header);
  // Pad so (8 + jsonLen) % 8 === 0 → data starts 8-byte aligned.
  let jsonBytes = new TextEncoder().encode(json);
  const pad = (8 - ((8 + jsonBytes.length) % 8)) % 8;
  if (pad) { json += " ".repeat(pad); jsonBytes = new TextEncoder().encode(json); }

  const out = new Uint8Array(8 + jsonBytes.length + totalData);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(jsonBytes.length), true);
  out.set(jsonBytes, 8);
  let p = 8 + jsonBytes.length;
  for (const t of tensors) { out.set(t.data, p); p += t.data.byteLength; }
  return out;
}
