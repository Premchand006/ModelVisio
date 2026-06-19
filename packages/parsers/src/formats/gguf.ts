import type { Model } from "../types";
import { Reader } from "../common/reader";
import { weightsToModel, type TensorInfo } from "../common/weights";

// GGUF (llama.cpp): magic "GGUF", version, tensor & metadata counts, typed
// metadata KV pairs, then tensor infos (name, dims, ggml type, offset).
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

const GGML_TYPE: Record<number, { name: string; bpe: number }> = {
  0: { name: "F32", bpe: 4 }, 1: { name: "F16", bpe: 2 },
  2: { name: "Q4_0", bpe: 0.5 }, 3: { name: "Q4_1", bpe: 0.5 },
  6: { name: "Q5_0", bpe: 0.625 }, 7: { name: "Q5_1", bpe: 0.625 },
  8: { name: "Q8_0", bpe: 1 }, 9: { name: "Q8_1", bpe: 1 },
  10: { name: "Q2_K", bpe: 0.34 }, 11: { name: "Q3_K", bpe: 0.43 },
  12: { name: "Q4_K", bpe: 0.56 }, 13: { name: "Q5_K", bpe: 0.69 },
  14: { name: "Q6_K", bpe: 0.82 }, 15: { name: "Q8_K", bpe: 1.06 },
  24: { name: "I8", bpe: 1 }, 25: { name: "I16", bpe: 2 }, 26: { name: "I32", bpe: 4 },
  28: { name: "I64", bpe: 8 }, 29: { name: "F64", bpe: 8 }, 30: { name: "BF16", bpe: 2 },
};

function ggufString(r: Reader): string {
  const len = r.u64();
  return r.str(len);
}

// Advance past (and optionally capture) a typed metadata value.
function readValue(r: Reader, type: number): unknown {
  switch (type) {
    case 0: return r.u8();
    case 1: return r.view.getInt8(r.off++);
    case 2: return r.u16();
    case 3: return r.i16();
    case 4: return r.u32();
    case 5: return r.i32();
    case 6: return r.f32();
    case 7: return r.u8() !== 0;
    case 8: return ggufString(r);
    case 9: { // array
      const elemType = r.u32();
      const count = r.u64();
      const out: unknown[] = [];
      for (let i = 0; i < count; i++) out.push(readValue(r, elemType));
      return out;
    }
    case 10: return r.u64();
    case 11: return r.i64();
    case 12: return r.f64();
    default: throw new Error(`Unknown GGUF metadata value type ${type}`);
  }
}

type GgufHeader = { version: number; meta: Map<string, unknown>; tensors: TensorInfo[] };

function readGgufHeader(buf: Uint8Array): GgufHeader {
  const r = new Reader(buf);
  if (r.str(4) !== "GGUF") throw new Error("Not a GGUF file (bad magic).");
  const version = r.u32();
  const tensorCount = r.u64();
  const kvCount = r.u64();

  const meta = new Map<string, unknown>();
  for (let i = 0; i < kvCount; i++) {
    const key = ggufString(r);
    const type = r.u32();
    meta.set(key, readValue(r, type));
  }

  const tensors: TensorInfo[] = [];
  for (let i = 0; i < tensorCount; i++) {
    const tname = ggufString(r);
    const nDims = r.u32();
    const dims: number[] = [];
    for (let d = 0; d < nDims; d++) dims.push(r.u64());
    const ggmlType = r.u32();
    r.u64(); // offset (unused)
    const t = GGML_TYPE[ggmlType] ?? { name: `ggml_${ggmlType}`, bpe: 4 };
    const count = dims.reduce((a, b) => a * b, dims.length ? 1 : 1);
    tensors.push({ name: tname, shape: dims, dtype: t.name, bytes: Math.round(count * t.bpe) });
  }
  return { version, meta, tensors };
}

export function parseGguf(input: ArrayBuffer | Uint8Array, name = "model.gguf"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const { version, meta, tensors } = readGgufHeader(buf);

  const arch = String(meta.get("general.architecture") ?? "unknown");
  const modelName = String(meta.get("general.name") ?? name);
  const fileType = meta.get("general.file_type");
  return weightsToModel(
    {
      name: modelName,
      format: "GGUF",
      framework: `GGUF v${version} · ${arch}`,
      sizeBytes: buf.byteLength,
      note: `GGUF ${arch} model${fileType != null ? `, file_type ${fileType}` : ""}. Stored weight tensor.`,
    },
    tensors,
  );
}

/** Export GGUF metadata + tensor manifest as pretty JSON. */
export function ggufMetadataJson(input: ArrayBuffer | Uint8Array): string {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const { version, meta, tensors } = readGgufHeader(buf);
  const obj = {
    version,
    metadata: Object.fromEntries(meta),
    tensors: tensors.map((t) => ({ name: t.name, shape: t.shape, dtype: t.dtype, bytes: t.bytes })),
  };
  return JSON.stringify(obj, null, 2);
}
