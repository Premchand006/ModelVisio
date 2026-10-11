import { ProtoReader } from "./protobufWire";

// TensorFlow checkpoint ("tensor bundle") reader: the `variables/` folder of a
// SavedModel, or any `*.index` + `*.data-NNNNN-of-MMMMM` checkpoint.
//
// `.index` is a LevelDB-format SSTable (tensorflow/core/lib/io/table*):
//   data blocks · metaindex block · index block · 48-byte footer
//   footer  = metaindex BlockHandle · index BlockHandle · padding · magic
//   handle  = varint64 offset · varint64 size, then a 5-byte trailer
//             (1 byte compression: 0 none / 1 snappy, 4 byte crc)
//   block   = entries (shared, non_shared, value_len varints, key delta, value)
//             · u32 restart offsets · u32 restart count
// Keys are tensor names (the empty key holds the BundleHeaderProto); values
// are BundleEntryProto { dtype=1, shape=2, shard_id=3, offset=4, size=5 }
// pointing into the `.data-*` shard files, which hold raw little-endian bytes.

export type BundleTensor = { dt: string; shape: number[]; bytes: Uint8Array };

const TABLE_MAGIC = 0xdb4775248b80fb57n;
const FOOTER_LEN = 48;

// tensorflow/core/framework/types.proto DataType → dtype names used elsewhere.
const TF_DTYPE: Record<number, string> = {
  1: "float32", 2: "float64", 3: "int32", 4: "uint8", 5: "int16", 6: "int8", 7: "string", 9: "int64",
  10: "bool", 14: "bfloat16", 17: "uint16", 19: "float16", 22: "uint32", 23: "uint64",
};

function varint(buf: Uint8Array, pos: { o: number }): number {
  let r = 0;
  let mul = 1;
  for (;;) {
    if (pos.o >= buf.length) throw new Error("tensor bundle: truncated varint");
    const b = buf[pos.o++];
    r += (b & 0x7f) * mul;
    if (b < 0x80) return r;
    mul *= 128;
  }
}

/** Decompress a raw Snappy block (the format LevelDB tables use). */
export function snappyDecompress(src: Uint8Array): Uint8Array {
  const pos = { o: 0 };
  const len = varint(src, pos);
  const out = new Uint8Array(len);
  let o = 0;
  while (pos.o < src.length) {
    const tag = src[pos.o++];
    const kind = tag & 3;
    if (kind === 0) {
      let n = (tag >> 2) + 1;
      if (n > 60) {
        const bytes = n - 60;
        n = 0;
        for (let i = 0; i < bytes; i++) n |= src[pos.o++] << (8 * i);
        n += 1;
      }
      out.set(src.subarray(pos.o, pos.o + n), o);
      pos.o += n;
      o += n;
      continue;
    }
    let n: number;
    let off: number;
    if (kind === 1) {
      n = 4 + ((tag >> 2) & 7);
      off = ((tag >> 5) << 8) | src[pos.o++];
    } else if (kind === 2) {
      n = (tag >> 2) + 1;
      off = src[pos.o] | (src[pos.o + 1] << 8);
      pos.o += 2;
    } else {
      n = (tag >> 2) + 1;
      off = (src[pos.o] | (src[pos.o + 1] << 8) | (src[pos.o + 2] << 16) | (src[pos.o + 3] << 24)) >>> 0;
      pos.o += 4;
    }
    if (off === 0 || off > o) throw new Error("tensor bundle: bad snappy copy");
    for (let i = 0; i < n; i++, o++) out[o] = out[o - off]; // may overlap: byte by byte
  }
  if (o !== len) throw new Error("tensor bundle: snappy length mismatch");
  return out;
}

function readBlock(file: Uint8Array, handle: { offset: number; size: number }): Uint8Array {
  const end = handle.offset + handle.size;
  if (end + 5 > file.length) throw new Error("tensor bundle: block out of range");
  const raw = file.subarray(handle.offset, end);
  const type = file[end];
  if (type === 0) return raw;
  if (type === 1) return snappyDecompress(raw);
  throw new Error(`tensor bundle: unsupported block compression ${type}`);
}

function* blockEntries(block: Uint8Array): Generator<{ key: string; value: Uint8Array }> {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const restarts = view.getUint32(block.length - 4, true);
  const limit = block.length - 4 * (restarts + 1);
  const td = new TextDecoder();
  const pos = { o: 0 };
  let key = new Uint8Array(0);
  while (pos.o < limit) {
    const shared = varint(block, pos);
    const nonShared = varint(block, pos);
    const valueLen = varint(block, pos);
    const next = new Uint8Array(shared + nonShared);
    next.set(key.subarray(0, shared));
    next.set(block.subarray(pos.o, pos.o + nonShared), shared);
    pos.o += nonShared;
    key = next;
    yield { key: td.decode(key), value: block.subarray(pos.o, pos.o + valueLen) };
    pos.o += valueLen;
  }
}

function readHandle(buf: Uint8Array, pos: { o: number }) {
  return { offset: varint(buf, pos), size: varint(buf, pos) };
}

type Entry = { dt: string; shape: number[]; shard: number; offset: number; size: number };

function readEntry(value: Uint8Array): Entry {
  const r = new ProtoReader(value);
  const e: Entry = { dt: "float32", shape: [], shard: 0, offset: 0, size: 0 };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) e.dt = TF_DTYPE[r.varint()] ?? "unknown";
    else if (t.field === 2) {
      const s = r.sub(); // TensorShapeProto { dim=2 { size=1 } }
      for (let u; (u = s.next()); ) {
        if (u.field === 2) {
          const d = s.sub();
          let size = 0;
          for (let w; (w = d.next()); ) {
            if (w.field === 1) size = d.varint();
            else d.skip(w.wireType);
          }
          e.shape.push(size);
        } else s.skip(u.wireType);
      }
    } else if (t.field === 3) e.shard = r.varint();
    else if (t.field === 4) e.offset = r.varint();
    else if (t.field === 5) e.size = r.varint();
    else r.skip(t.wireType);
  }
  return e;
}

/**
 * Read every numeric tensor of a checkpoint. `shards[i]` is the bytes of
 * `*.data-0000i-of-*`; a missing shard only drops the tensors stored in it.
 * Returns null when `index` is not a tensor-bundle table.
 */
export function readTensorBundle(index: Uint8Array, shards: (Uint8Array | undefined)[]): Map<string, BundleTensor> | null {
  if (index.length < FOOTER_LEN) return null;
  const view = new DataView(index.buffer, index.byteOffset, index.byteLength);
  if (view.getBigUint64(index.length - 8, true) !== TABLE_MAGIC) return null;
  const out = new Map<string, BundleTensor>();
  try {
    const pos = { o: index.length - FOOTER_LEN };
    readHandle(index, pos); // metaindex (unused)
    const indexBlock = readBlock(index, readHandle(index, pos));
    for (const { value } of blockEntries(indexBlock)) {
      const data = readBlock(index, readHandle(value, { o: 0 }));
      for (const { key, value: v } of blockEntries(data)) {
        if (key === "") continue; // BundleHeaderProto
        const e = readEntry(v);
        const shard = shards[e.shard];
        if (!shard || e.dt === "string" || e.offset + e.size > shard.length) continue;
        out.set(key, { dt: e.dt, shape: e.shape, bytes: shard.subarray(e.offset, e.offset + e.size) });
      }
    }
  } catch {
    return null;
  }
  return out;
}

/** "variables.data-00002-of-00004" → 2 (shard index), else undefined. */
export function shardIndex(name: string): number | undefined {
  const m = /\.data-(\d{5})-of-\d{5}$/.exec(name);
  return m ? Number(m[1]) : undefined;
}
