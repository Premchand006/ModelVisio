import { crc32 } from "./crc32";

// Minimal STORED (uncompressed) ZIP writer with correct CRC-32, so produced
// archives open in any OS/unzip tool. Pairs with the reader in zip.ts.

export type ZipInput = { name: string; data: Uint8Array };

const enc = new TextEncoder();

export function writeZip(files: ZipInput[]): Uint8Array {
  const local: number[] = [];
  const central: number[] = [];
  const u16 = (a: number[], v: number) => a.push(v & 0xff, (v >>> 8) & 0xff);
  const u32 = (a: number[], v: number) => a.push(v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff);
  const pushBytes = (a: number[], b: Uint8Array) => { for (let i = 0; i < b.length; i++) a.push(b[i]); };

  const records: { name: Uint8Array; crc: number; size: number; offset: number }[] = [];

  for (const f of files) {
    const name = enc.encode(f.name);
    const crc = crc32(f.data);
    const offset = local.length;
    u32(local, 0x04034b50);          // local file header signature
    u16(local, 20);                  // version needed
    u16(local, 0);                   // flags
    u16(local, 0);                   // method: stored
    u16(local, 0); u16(local, 0x0021); // mod time 00:00:00, mod date 1980-01-01 (valid DOS date)
    u32(local, crc);
    u32(local, f.data.length);       // compressed size
    u32(local, f.data.length);       // uncompressed size
    u16(local, name.length);
    u16(local, 0);                   // extra length
    pushBytes(local, name);
    pushBytes(local, f.data);
    records.push({ name, crc, size: f.data.length, offset });
  }

  const cdStart = local.length;
  for (const r of records) {
    u32(central, 0x02014b50);        // central directory header signature
    u16(central, 20); u16(central, 20); // version made by, needed
    u16(central, 0); u16(central, 0);   // flags, method
    u16(central, 0); u16(central, 0x0021); // time 00:00:00, date 1980-01-01
    u32(central, r.crc);
    u32(central, r.size); u32(central, r.size);
    u16(central, r.name.length);
    u16(central, 0); u16(central, 0);   // extra, comment
    u16(central, 0); u16(central, 0);   // disk start, internal attrs
    u32(central, 0);                    // external attrs
    u32(central, r.offset);             // local header offset
    pushBytes(central, r.name);
  }

  const eocd: number[] = [];
  u32(eocd, 0x06054b50);
  u16(eocd, 0); u16(eocd, 0);          // disk numbers
  u16(eocd, records.length); u16(eocd, records.length);
  u32(eocd, central.length);           // central directory size
  u32(eocd, cdStart);                  // central directory offset
  u16(eocd, 0);                        // comment length

  const out = new Uint8Array(local.length + central.length + eocd.length);
  out.set(local, 0);
  out.set(central, local.length);
  out.set(eocd, local.length + central.length);
  return out;
}
