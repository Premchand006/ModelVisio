// Minimal, bounds-checked FlatBuffers table reader shared by the schema-less
// FlatBuffer parsers (MNN, RKNN). Field indices are schema declaration order;
// a union field occupies two slots (the `_type` byte, then the value).

const td = new TextDecoder();

export class FlatBuf {
  readonly view: DataView;
  constructor(readonly buf: Uint8Array) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  private check(p: number, n: number): void {
    if (p < 0 || p + n > this.buf.byteLength) throw new Error(`FlatBuffer: offset ${p} out of range`);
  }
  u8(p: number) { this.check(p, 1); return this.view.getUint8(p); }
  i8(p: number) { this.check(p, 1); return this.view.getInt8(p); }
  u16(p: number) { this.check(p, 2); return this.view.getUint16(p, true); }
  i32(p: number) { this.check(p, 4); return this.view.getInt32(p, true); }
  u32(p: number) { this.check(p, 4); return this.view.getUint32(p, true); }
  f32(p: number) { this.check(p, 4); return this.view.getFloat32(p, true); }
  i64(p: number) { this.check(p, 8); return Number(this.view.getBigInt64(p, true)); }

  /** Root table position. */
  root(): number { return this.u32(0); }
  /** 4-char file identifier at bytes 4..8 ("" if not printable). */
  identifier(): string {
    if (this.buf.byteLength < 8) return "";
    const s = String.fromCharCode(...this.buf.subarray(4, 8));
    return /^[\x20-\x7e]{4}$/.test(s) ? s : "";
  }

  /** Absolute position of a table field, or null if absent. */
  field(table: number, index: number): number | null {
    const vtable = table - this.i32(table);
    const vtSize = this.u16(vtable);
    const slot = 4 + index * 2;
    if (slot + 2 > vtSize) return null;
    const voff = this.u16(vtable + slot);
    return voff === 0 ? null : table + voff;
  }
  has(table: number, index: number): boolean { return this.field(table, index) != null; }
  private indirect(p: number): number { return p + this.u32(p); }

  bool(t: number, i: number, d = false): boolean { const f = this.field(t, i); return f == null ? d : this.u8(f) !== 0; }
  int8(t: number, i: number, d = 0): number { const f = this.field(t, i); return f == null ? d : this.i8(f); }
  uint8(t: number, i: number, d = 0): number { const f = this.field(t, i); return f == null ? d : this.u8(f); }
  int32(t: number, i: number, d = 0): number { const f = this.field(t, i); return f == null ? d : this.i32(f); }
  float32(t: number, i: number, d = 0): number { const f = this.field(t, i); return f == null ? d : this.f32(f); }

  str(t: number, i: number): string {
    const f = this.field(t, i);
    if (f == null) return "";
    const p = this.indirect(f);
    const len = this.u32(p);
    this.check(p + 4, len);
    return td.decode(this.buf.subarray(p + 4, p + 4 + len));
  }
  /** Sub-table position, or null. */
  table(t: number, i: number): number | null {
    const f = this.field(t, i);
    return f == null ? null : this.indirect(f);
  }
  /** Vector base (first element) + length, or null. */
  vec(t: number, i: number): { base: number; len: number } | null {
    const f = this.field(t, i);
    if (f == null) return null;
    const p = this.indirect(f);
    return { base: p + 4, len: this.u32(p) };
  }
  vecLen(t: number, i: number): number { return this.vec(t, i)?.len ?? 0; }
  /** Table positions of a vector of tables. */
  tables(t: number, i: number): number[] {
    const v = this.vec(t, i);
    if (!v) return [];
    const out: number[] = [];
    for (let k = 0; k < v.len; k++) out.push(this.indirect(v.base + k * 4));
    return out;
  }
  ints(t: number, i: number): number[] {
    const v = this.vec(t, i);
    if (!v) return [];
    this.check(v.base, v.len * 4);
    const out: number[] = [];
    for (let k = 0; k < v.len; k++) out.push(this.i32(v.base + k * 4));
    return out;
  }
  longs(t: number, i: number): number[] {
    const v = this.vec(t, i);
    if (!v) return [];
    const out: number[] = [];
    for (let k = 0; k < v.len; k++) out.push(this.i64(v.base + k * 8));
    return out;
  }
  strings(t: number, i: number): string[] {
    const v = this.vec(t, i);
    if (!v) return [];
    const out: string[] = [];
    for (let k = 0; k < v.len; k++) {
      const p = this.indirect(v.base + k * 4);
      const len = this.u32(p);
      out.push(td.decode(this.buf.subarray(p + 4, p + 4 + len)));
    }
    return out;
  }
  /** Raw bytes of a scalar vector (`elemSize` bytes per element). */
  bytes(t: number, i: number, elemSize = 1): Uint8Array | null {
    const v = this.vec(t, i);
    if (!v) return null;
    this.check(v.base, v.len * elemSize);
    return this.buf.subarray(v.base, v.base + v.len * elemSize);
  }
  /** Float32 vector values (copied; handles unaligned storage). */
  floats(t: number, i: number): Float32Array | null {
    const b = this.bytes(t, i, 4);
    if (!b) return null;
    const out = new Float32Array(b.byteLength / 4);
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    for (let k = 0; k < out.length; k++) out[k] = dv.getFloat32(k * 4, true);
    return out;
  }
}
