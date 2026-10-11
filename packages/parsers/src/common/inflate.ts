// Synchronous raw DEFLATE (RFC 1951) decoder for ZIP members. Parsers run
// synchronously in a Worker, so the async DecompressionStream API doesn't fit.
// Straightforward canonical-Huffman implementation; fine for config-sized
// members (JSON, small protobufs), not tuned for multi-GB payloads.

const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

type Huff = { counts: Uint16Array; symbols: Uint16Array };

function buildHuff(lengths: ArrayLike<number>, n: number): Huff {
  const counts = new Uint16Array(16);
  for (let i = 0; i < n; i++) counts[lengths[i]]++;
  counts[0] = 0;
  const offs = new Uint16Array(16);
  for (let i = 1; i < 16; i++) offs[i] = offs[i - 1] + counts[i - 1];
  const symbols = new Uint16Array(n);
  for (let i = 0; i < n; i++) if (lengths[i]) symbols[offs[lengths[i]]++] = i;
  return { counts, symbols };
}

class Bits {
  pos = 0;
  bit = 0;
  constructor(readonly src: Uint8Array) {}
  get(n: number): number {
    let v = 0;
    for (let i = 0; i < n; i++) {
      if (this.pos >= this.src.length) throw new Error("inflate: unexpected end of data");
      v |= ((this.src[this.pos] >> this.bit) & 1) << i;
      if (++this.bit === 8) { this.bit = 0; this.pos++; }
    }
    return v;
  }
  decode(h: Huff): number {
    let code = 0, first = 0, index = 0;
    for (let len = 1; len < 16; len++) {
      code |= this.get(1);
      const count = h.counts[len];
      if (code - first < count) return h.symbols[index + (code - first)];
      index += count;
      first = (first + count) << 1;
      code <<= 1;
    }
    throw new Error("inflate: invalid Huffman code");
  }
}

let FIXED: { lit: Huff; dist: Huff } | null = null;
function fixedTables() {
  if (!FIXED) {
    const l = new Uint8Array(288);
    l.fill(8, 0, 144); l.fill(9, 144, 256); l.fill(7, 256, 280); l.fill(8, 280, 288);
    FIXED = { lit: buildHuff(l, 288), dist: buildHuff(new Uint8Array(30).fill(5), 30) };
  }
  return FIXED;
}

/** Inflate a raw DEFLATE stream. `sizeHint` (e.g. ZIP uncompressed size) pre-sizes the output. */
export function inflateRaw(src: Uint8Array, sizeHint = 0): Uint8Array {
  let out = new Uint8Array(Math.max(sizeHint, src.length * 4, 1024));
  let op = 0;
  const ensure = (n: number) => {
    if (op + n <= out.length) return;
    const next = new Uint8Array(Math.max(out.length * 2, op + n));
    next.set(out.subarray(0, op));
    out = next;
  };
  const br = new Bits(src);
  let final = 0;
  while (!final) {
    final = br.get(1);
    const type = br.get(2);
    if (type === 0) {
      if (br.bit) { br.bit = 0; br.pos++; }
      const len = src[br.pos] | (src[br.pos + 1] << 8);
      br.pos += 4;
      ensure(len);
      out.set(src.subarray(br.pos, br.pos + len), op);
      op += len;
      br.pos += len;
      continue;
    }
    let lit: Huff, dist: Huff;
    if (type === 1) ({ lit, dist } = fixedTables());
    else if (type === 2) {
      const hlit = br.get(5) + 257, hdist = br.get(5) + 1, hclen = br.get(4) + 4;
      const cl = new Uint8Array(19);
      for (let i = 0; i < hclen; i++) cl[CL_ORDER[i]] = br.get(3);
      const clh = buildHuff(cl, 19);
      const lens = new Uint8Array(hlit + hdist);
      for (let i = 0; i < hlit + hdist; ) {
        const sym = br.decode(clh);
        if (sym < 16) lens[i++] = sym;
        else {
          let rep = 0, val = 0;
          if (sym === 16) { val = lens[i - 1]; rep = 3 + br.get(2); }
          else if (sym === 17) rep = 3 + br.get(3);
          else rep = 11 + br.get(7);
          while (rep--) lens[i++] = val;
        }
      }
      lit = buildHuff(lens.subarray(0, hlit), hlit);
      dist = buildHuff(lens.subarray(hlit), hdist);
    } else throw new Error("inflate: invalid block type");

    for (;;) {
      const sym = br.decode(lit);
      if (sym < 256) { ensure(1); out[op++] = sym; continue; }
      if (sym === 256) break;
      const li = sym - 257;
      const len = LEN_BASE[li] + br.get(LEN_EXTRA[li]);
      const di = br.decode(dist);
      const d = DIST_BASE[di] + br.get(DIST_EXTRA[di]);
      if (d > op) throw new Error("inflate: distance too far back");
      ensure(len);
      for (let k = 0; k < len; k++, op++) out[op] = out[op - d];
    }
  }
  return out.subarray(0, op);
}
