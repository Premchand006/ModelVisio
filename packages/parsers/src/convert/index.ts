import type { Model } from "../types";
import { Reader } from "../common/reader";
import { f16BytesToF32Bytes, f32BytesToF16Bytes } from "../common/float16";
import { writeSafetensors, type OutTensor } from "../common/safetensorsWriter";
import { readNumpy, writeNpz } from "../common/numpy";
import { extractOnnxTensors } from "../onnx/tensors";

// ONNX TensorProto.DataType → safetensors dtype label.
const ONNX_TO_ST: Record<number, string> = {
  1: "F32", 2: "U8", 3: "I8", 4: "U16", 5: "I16", 6: "I32", 7: "I64",
  9: "BOOL", 10: "F16", 11: "F64", 12: "U32", 13: "U64", 16: "BF16",
};
const ST_BYTES: Record<string, number> = {
  F64: 8, F32: 4, F16: 2, BF16: 2, F8_E4M3: 1, F8_E5M2: 1,
  I64: 8, I32: 4, I16: 2, I8: 1, U64: 8, U32: 4, U16: 2, U8: 1, BOOL: 1,
};

export type ConvertResult = { filename: string; mime: string; data: Uint8Array };

/** Export the normalized graph as pretty JSON. */
export function toGraphJson(model: Model): string {
  return JSON.stringify(model, null, 2);
}

/** Export a per-layer report as CSV. */
export function toLayerCsv(model: Model): string {
  const head = ["id", "name", "op", "type", "group", "shape", "dtype", "params", "flops", "macs", "memBytes"];
  const esc = (v: unknown) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const rows = model.layers.map((l) =>
    [l.id, l.name, l.op, l.type, l.group, l.shape, l.dt, l.params, l.flops, l.macs, l.mem].map(esc).join(","),
  );
  return [head.join(","), ...rows].join("\n");
}

const baseName = (name: string) => name.replace(/\.[^.]+$/, "");

/** Read safetensors tensors with their raw data slices, validating offsets. */
function readSafetensors(buf: Uint8Array): OutTensor[] {
  const r = new Reader(buf);
  const headerLen = r.u64();
  const header = JSON.parse(r.str(headerLen)) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const dataStart = 8 + headerLen;
  const dataEnd = buf.byteLength - dataStart;
  const out: OutTensor[] = [];
  const ranges: [number, number][] = [];
  for (const [name, e] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (!e || !Array.isArray(e.data_offsets)) throw new Error(`Safetensors: tensor ${name} has no data_offsets.`);
    const [start, end] = e.data_offsets;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end > dataEnd) {
      throw new Error(`Safetensors: tensor ${name} has invalid data_offsets [${start}, ${end}] (data block is ${dataEnd} bytes).`);
    }
    ranges.push([start, end]);
    out.push({ name, dtype: e.dtype, shape: e.shape, data: buf.subarray(dataStart + start, dataStart + end) });
  }
  // Offsets must be contiguous and non-overlapping (safetensors invariant).
  ranges.sort((a, b) => a[0] - b[0]);
  let prev = 0;
  for (const [start, end] of ranges) {
    if (start !== prev) throw new Error(`Safetensors: tensor data is not contiguous (gap/overlap at byte ${prev}).`);
    prev = end;
  }
  return out;
}

export type ConvTarget = "safetensors" | "npz";
export type Precision = "keep" | "f16" | "f32";

function bf16ToF32(src: Uint8Array): Uint8Array {
  const n = src.byteLength / 2;
  const iv = new DataView(src.buffer, src.byteOffset, src.byteLength);
  const out = new Uint8Array(n * 4);
  const ov = new DataView(out.buffer);
  for (let i = 0; i < n; i++) ov.setUint32(i * 4, (iv.getUint16(i * 2, true) << 16) >>> 0, true);
  return out;
}
function f64ToF32(src: Uint8Array): Uint8Array {
  const n = src.byteLength / 8;
  const iv = new DataView(src.buffer, src.byteOffset, src.byteLength);
  const out = new Uint8Array(n * 4);
  const ov = new DataView(out.buffer);
  for (let i = 0; i < n; i++) ov.setFloat32(i * 4, iv.getFloat64(i * 8, true), true);
  return out;
}

/** Recast tensors to a target float precision; integers/others pass through. */
function recast(tensors: OutTensor[], p: Precision): OutTensor[] {
  if (p === "keep") return tensors;
  return tensors.map((t) => {
    if (p === "f16") {
      if (t.dtype === "F32") return { ...t, dtype: "F16", data: f32BytesToF16Bytes(t.data) };
      if (t.dtype === "F64") return { ...t, dtype: "F16", data: f32BytesToF16Bytes(f64ToF32(t.data)) };
      if (t.dtype === "BF16") return { ...t, dtype: "F16", data: f32BytesToF16Bytes(bf16ToF32(t.data)) };
      return t;
    }
    if (t.dtype === "F16") return { ...t, dtype: "F32", data: f16BytesToF32Bytes(t.data) };
    if (t.dtype === "BF16") return { ...t, dtype: "F32", data: bf16ToF32(t.data) };
    if (t.dtype === "F64") return { ...t, dtype: "F32", data: f64ToF32(t.data) };
    return t;
  });
}

/** Read a source model's weight tensors into the common OutTensor shape. */
function readSource(buf: Uint8Array, format: string, name: string): OutTensor[] {
  if (/onnx/i.test(format)) {
    const { tensors: onnx, skippedExternal } = extractOnnxTensors(buf);
    if (onnx.length === 0) {
      throw new Error(
        skippedExternal > 0
          ? `All ${skippedExternal} weights use external data (stored outside the file) — can't repack in-browser.`
          : "No weight initializers found in this ONNX model.",
      );
    }
    return onnx.map((t) => {
      const dtype = ONNX_TO_ST[t.dataType];
      if (!dtype) throw new Error(`Unsupported ONNX dtype ${t.dataType} for tensor ${t.name}.`);
      return { name: t.name, dtype, shape: t.dims, data: t.bytes };
    });
  }
  if (/safetensors/i.test(format)) return readSafetensors(buf);
  if (/numpy/i.test(format)) return readNumpy(buf, name);
  throw new Error(`Weights export isn't supported for ${format} sources (ONNX, Safetensors, NumPy are).`);
}

function validate(tensors: OutTensor[]): void {
  for (const t of tensors) {
    const bpe = ST_BYTES[t.dtype];
    if (bpe === undefined) throw new Error(`Tensor ${t.name}: unsupported dtype "${t.dtype}".`);
    const elems = t.shape.reduce((a, b) => a * b, t.shape.length ? 1 : 1);
    if (t.data.byteLength !== elems * bpe) {
      throw new Error(`Tensor ${t.name}: ${t.data.byteLength} bytes but shape ${t.shape.join("×")} × ${t.dtype} expects ${elems * bpe}.`);
    }
  }
}

/**
 * Convert a source model's WEIGHTS to safetensors or NumPy .npz, optionally
 * recasting precision. Sources: ONNX, Safetensors, NumPy. Pure function — safe
 * to call from a Worker.
 */
export function convertWeights(
  input: ArrayBuffer | Uint8Array,
  name: string,
  opts: { format: string; target?: ConvTarget; precision?: Precision },
): ConvertResult {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const target = opts.target ?? "safetensors";
  const precision = opts.precision ?? "keep";

  let tensors = recast(readSource(buf, opts.format, name), precision);
  validate(tensors);

  const suffix = precision === "f16" ? ".fp16" : precision === "f32" ? ".fp32" : "";

  if (target === "npz") {
    // NumPy has no bf16/f8 — upcast those to f32 so the .npz is loadable.
    tensors = tensors.map((t) =>
      t.dtype === "BF16" ? { ...t, dtype: "F32", data: bf16ToF32(t.data) } : t,
    );
    return { filename: `${baseName(name)}${suffix}.npz`, mime: "application/octet-stream", data: writeNpz(tensors) };
  }

  const data = writeSafetensors(tensors, { format: "modelvisio", source: opts.format, precision });
  return { filename: `${baseName(name)}${suffix}.safetensors`, mime: "application/octet-stream", data };
}
