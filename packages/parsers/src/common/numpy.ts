import { isZip, readZipEntries, extractStored } from "./zip";
import { writeZip, type ZipInput } from "./zipwriter";

// Low-level NumPy .npy / .npz read+write with FULL data, for conversion. The
// header-only parser used for the graph view lives in formats/numpy.ts.

export type NumpyTensor = { name: string; dtype: string; shape: number[]; data: Uint8Array };

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];

// safetensors dtype label → numpy descr (little-endian).
const ST_TO_DESCR: Record<string, string> = {
  F64: "<f8", F32: "<f4", F16: "<f2",
  I64: "<i8", I32: "<i4", I16: "<i2", I8: "|i1",
  U64: "<u8", U32: "<u4", U16: "<u2", U8: "|u1", BOOL: "|b1",
};
// numpy descr key (no endianness prefix) → safetensors label.
const DESCR_TO_ST: Record<string, string> = {
  f8: "F64", f4: "F32", f2: "F16", i1: "I8", i2: "I16", i4: "I32", i8: "I64",
  u1: "U8", u2: "U16", u4: "U32", u8: "U64", b1: "BOOL",
};
const ST_BYTES: Record<string, number> = {
  F64: 8, F32: 4, F16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U64: 8, U32: 4, U16: 2, U8: 1, BOOL: 1,
};

const enc = (s: string) => new TextEncoder().encode(s);

function isNpy(buf: Uint8Array): boolean {
  return NPY_MAGIC.every((b, i) => buf[i] === b);
}

/** Serialize one tensor to .npy v1.0 bytes (C-order, little-endian). */
export function writeNpy(t: NumpyTensor): Uint8Array {
  const descr = ST_TO_DESCR[t.dtype];
  if (!descr) throw new Error(`NumPy export: dtype ${t.dtype} has no numpy equivalent.`);
  const shapeStr =
    t.shape.length === 0 ? "()" : t.shape.length === 1 ? `(${t.shape[0]},)` : `(${t.shape.join(", ")})`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
  // Pad with spaces so (10 + headerLen) is a multiple of 64, header ends with \n.
  const pad = (64 - ((10 + header.length + 1) % 64)) % 64;
  header = header + " ".repeat(pad) + "\n";
  const hb = enc(header);
  const out = new Uint8Array(10 + hb.length + t.data.byteLength);
  out.set(NPY_MAGIC, 0);
  out[6] = 1; out[7] = 0; // version 1.0
  new DataView(out.buffer).setUint16(8, hb.length, true);
  out.set(hb, 10);
  out.set(t.data, 10 + hb.length);
  return out;
}

/** Pack tensors into an uncompressed .npz (ZIP of .npy members). */
export function writeNpz(tensors: NumpyTensor[]): Uint8Array {
  const entries: ZipInput[] = tensors.map((t) => ({ name: `${t.name}.npy`, data: writeNpy(t) }));
  return writeZip(entries);
}

/** Read a single .npy (full data). */
export function readNpy(buf: Uint8Array, name: string): NumpyTensor {
  if (!isNpy(buf)) throw new Error("Not a NumPy .npy file.");
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const major = buf[6];
  const headerLen = major <= 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const dataStart = (major <= 1 ? 10 : 12) + headerLen;
  const header = new TextDecoder().decode(buf.subarray(major <= 1 ? 10 : 12, dataStart));
  const descrM = header.match(/'descr'\s*:\s*'([^']+)'/);
  const shapeM = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
  const fortran = /'fortran_order'\s*:\s*True/.test(header);
  if (fortran) throw new Error(`NumPy: ${name} is Fortran-ordered (column-major) — not supported.`);
  const descr = descrM ? descrM[1] : "|u1";
  if (/^>/.test(descr) && !/^>[ui]1$/.test(descr)) throw new Error(`NumPy: ${name} is big-endian (${descr}) — not supported.`);
  const key = descr.replace(/^[<>|=]/, "");
  const dtype = DESCR_TO_ST[key];
  if (!dtype) throw new Error(`NumPy: unsupported dtype ${descr} in ${name}.`);
  const shape = shapeM ? shapeM[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number) : [];
  return { name, dtype, shape, data: buf.subarray(dataStart) };
}

/** Read all arrays from a .npy or (uncompressed) .npz, with full data. */
export function readNumpy(buf: Uint8Array, name: string): NumpyTensor[] {
  if (isZip(buf)) {
    const out: NumpyTensor[] = [];
    let skippedCompressed = 0;
    for (const e of readZipEntries(buf)) {
      if (!e.name.endsWith(".npy")) continue;
      const member = extractStored(buf, e);
      if (!member) { skippedCompressed++; continue; }
      out.push(readNpy(member, e.name.replace(/\.npy$/, "")));
    }
    if (out.length === 0) {
      throw new Error(skippedCompressed > 0 ? "This .npz is compressed (np.savez_compressed) — only uncompressed np.savez is supported." : "No arrays found in this .npz.");
    }
    return out;
  }
  return [readNpy(buf, name.replace(/\.npy$/, ""))];
}

export { ST_BYTES as NUMPY_ST_BYTES };
