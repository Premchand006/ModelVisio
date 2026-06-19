import type { Model } from "../types";
import { weightsToModel, type TensorInfo } from "../common/weights";
import { extractStored, isZip, readZipEntries } from "../common/zip";

// .npy: \x93NUMPY magic, version, header length, then a Python-dict header
// string with descr / fortran_order / shape, then raw data.
// .npz: a ZIP of .npy members (np.savez is uncompressed → STORED).

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

const DT_BYTES: Record<string, number> = {
  f2: 2, f4: 4, f8: 8, i1: 1, i2: 2, i4: 4, i8: 8,
  u1: 1, u2: 2, u4: 4, u8: 8, b1: 1, c8: 8, c16: 16,
};
const DT_NAME: Record<string, string> = {
  f2: "float16", f4: "float32", f8: "float64", i1: "int8", i2: "int16",
  i4: "int32", i8: "int64", u1: "uint8", u2: "uint16", u4: "uint32",
  u8: "uint64", b1: "bool", c8: "complex64", c16: "complex128",
};

function isNpy(buf: Uint8Array): boolean {
  return NPY_MAGIC.every((b, i) => buf[i] === b);
}

function parseNpyHeader(buf: Uint8Array): { dtype: string; shape: number[]; bytes: number } {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const major = buf[6];
  let headerLen: number, dataStart: number;
  if (major <= 1) { headerLen = view.getUint16(8, true); dataStart = 10 + headerLen; }
  else { headerLen = view.getUint32(8, true); dataStart = 12 + headerLen; }
  const headerStr = new TextDecoder().decode(buf.subarray(major <= 1 ? 10 : 12, dataStart));

  const descrM = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  const shapeM = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  const descr = descrM ? descrM[1] : "|u1";
  const key = descr.replace(/^[<>|=]/, "");
  const shape = shapeM
    ? shapeM[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number)
    : [];
  const count = shape.reduce((a, b) => a * b, shape.length ? 1 : 1);
  return { dtype: DT_NAME[key] ?? descr, shape, bytes: count * (DT_BYTES[key] ?? 1) };
}

export function parseNumpy(input: ArrayBuffer | Uint8Array, name = "array.npy"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const tensors: TensorInfo[] = [];

  if (isZip(buf)) {
    for (const e of readZipEntries(buf)) {
      if (!e.name.endsWith(".npy")) continue;
      const member = extractStored(buf, e);
      const arrName = e.name.replace(/\.npy$/, "");
      if (member && isNpy(member)) {
        const h = parseNpyHeader(member);
        tensors.push({ name: arrName, shape: h.shape, dtype: h.dtype, bytes: h.bytes });
      } else {
        // Compressed (np.savez_compressed) — list it without a decoded header.
        tensors.push({ name: arrName, shape: [], dtype: "compressed", bytes: e.size });
      }
    }
    return weightsToModel(
      { name, format: "NumPy (.npz)", framework: "numpy", sizeBytes: buf.byteLength },
      tensors,
    );
  }

  if (!isNpy(buf)) throw new Error("Not a NumPy .npy/.npz file.");
  const h = parseNpyHeader(buf);
  tensors.push({ name: name.replace(/\.npy$/, ""), shape: h.shape, dtype: h.dtype, bytes: h.bytes });
  return weightsToModel(
    { name, format: "NumPy (.npy)", framework: "numpy", sizeBytes: buf.byteLength },
    tensors,
  );
}
