import type { Model } from "../types";
import { Reader } from "../common/reader";
import { weightsToModel, type TensorInfo } from "../common/weights";
import { sliceModel, type ModelStream } from "../common/chunks";

// safetensors layout: [u64 header length][JSON header][raw tensor data].
// The JSON header maps tensor name → { dtype, shape, data_offsets:[start,end] }.
// https://github.com/huggingface/safetensors

const DT: Record<string, string> = {
  F64: "float64", F32: "float32", F16: "float16", BF16: "bfloat16",
  I64: "int64", I32: "int32", I16: "int16", I8: "int8",
  U64: "uint64", U32: "uint32", U16: "uint16", U8: "uint8",
  BOOL: "bool", F8_E4M3: "float8_e4m3", F8_E5M2: "float8_e5m2",
};

type Entry = { dtype: string; shape: number[]; data_offsets: [number, number] };

export function parseSafetensors(input: ArrayBuffer | Uint8Array, name = "model.safetensors"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const r = new Reader(buf);
  const headerLen = r.u64();
  if (headerLen <= 0 || headerLen + 8 > buf.byteLength) throw new Error("Invalid safetensors header length.");
  const header = JSON.parse(r.str(headerLen)) as Record<string, Entry | Record<string, string>>;

  const tensors: TensorInfo[] = [];
  for (const [key, val] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    const e = val as Entry;
    if (!Array.isArray(e.shape) || !Array.isArray(e.data_offsets)) continue;
    tensors.push({
      name: key,
      shape: e.shape,
      dtype: DT[e.dtype] ?? e.dtype,
      bytes: e.data_offsets[1] - e.data_offsets[0],
    });
  }
  tensors.sort((a, b) => a.name.localeCompare(b.name));

  return weightsToModel(
    { name, format: "Safetensors", framework: "safetensors", sizeBytes: buf.byteLength },
    tensors,
  );
}

/**
 * Streaming form of parseSafetensors. The model is fully described by the JSON
 * header (tensor data is never read), so after the header decode the tensor
 * layers are streamed in `batchSize` slices so a large checkpoint renders
 * progressively. Returns exactly what parseSafetensors returns.
 */
export function* safetensorsStream(input: ArrayBuffer | Uint8Array, name = "model.safetensors", batchSize = 256): ModelStream {
  const model = parseSafetensors(input, name);
  yield* sliceModel(model, batchSize);
  return model;
}
