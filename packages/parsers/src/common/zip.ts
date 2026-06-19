// Minimal ZIP reader: lists central-directory entries and extracts STORED
// (uncompressed) members. Enough for torch `.pt` archives and `np.savez`
// (which defaults to ZIP_STORED). DEFLATE members are listed but not inflated.

export type ZipEntry = {
  name: string;
  method: number; // 0 = stored, 8 = deflate
  compressedSize: number;
  size: number;
  dataOffset: number; // absolute offset of the member's data
};

const EOCD_SIG = 0x06054b50;
const CD_SIG = 0x02014b50;

export function isZip(buf: Uint8Array): boolean {
  return buf.length > 4 && buf[0] === 0x50 && buf[1] === 0x4b && buf[2] === 0x03 && buf[3] === 0x04;
}

export function readZipEntries(buf: Uint8Array): ZipEntry[] {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // Find End Of Central Directory by scanning backwards.
  let eocd = -1;
  const minPos = Math.max(0, buf.length - 22 - 65535);
  for (let i = buf.length - 22; i >= minPos; i--) {
    if (view.getUint32(i, true) === EOCD_SIG) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("Not a valid ZIP (no EOCD record).");

  const total = view.getUint16(eocd + 10, true);
  let p = view.getUint32(eocd + 16, true); // central directory offset
  const entries: ZipEntry[] = [];
  for (let i = 0; i < total; i++) {
    if (view.getUint32(p, true) !== CD_SIG) break;
    const method = view.getUint16(p + 10, true);
    const compressedSize = view.getUint32(p + 20, true);
    const size = view.getUint32(p + 24, true);
    const nameLen = view.getUint16(p + 28, true);
    const extraLen = view.getUint16(p + 30, true);
    const commentLen = view.getUint16(p + 32, true);
    const localHeader = view.getUint32(p + 42, true);
    const name = new TextDecoder().decode(buf.subarray(p + 46, p + 46 + nameLen));
    // Local header tells us the true data start (its name/extra lengths differ).
    const lhNameLen = view.getUint16(localHeader + 26, true);
    const lhExtraLen = view.getUint16(localHeader + 28, true);
    const dataOffset = localHeader + 30 + lhNameLen + lhExtraLen;
    entries.push({ name, method, compressedSize, size, dataOffset });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

/** Raw bytes of a STORED entry (returns null for compressed members). */
export function extractStored(buf: Uint8Array, e: ZipEntry): Uint8Array | null {
  if (e.method !== 0) return null;
  return buf.subarray(e.dataOffset, e.dataOffset + e.size);
}
