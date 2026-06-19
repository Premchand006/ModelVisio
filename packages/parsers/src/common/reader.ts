// Little-endian-by-default binary cursor used by the binary-format parsers.
export class Reader {
  readonly view: DataView;
  off = 0;
  constructor(public readonly buf: Uint8Array, public le = true) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  get length(): number { return this.buf.byteLength; }
  get eof(): boolean { return this.off >= this.buf.byteLength; }
  seek(o: number): void { this.off = o; }
  u8(): number { const v = this.view.getUint8(this.off); this.off += 1; return v; }
  u16(): number { const v = this.view.getUint16(this.off, this.le); this.off += 2; return v; }
  i16(): number { const v = this.view.getInt16(this.off, this.le); this.off += 2; return v; }
  u32(): number { const v = this.view.getUint32(this.off, this.le); this.off += 4; return v; }
  i32(): number { const v = this.view.getInt32(this.off, this.le); this.off += 4; return v; }
  u64(): number { const v = this.view.getBigUint64(this.off, this.le); this.off += 8; return Number(v); }
  i64(): number { const v = this.view.getBigInt64(this.off, this.le); this.off += 8; return Number(v); }
  f32(): number { const v = this.view.getFloat32(this.off, this.le); this.off += 4; return v; }
  f64(): number { const v = this.view.getFloat64(this.off, this.le); this.off += 8; return v; }
  bytes(n: number): Uint8Array { const b = this.buf.subarray(this.off, this.off + n); this.off += n; return b; }
  str(n: number): string { return new TextDecoder().decode(this.bytes(n)); }
}

/** Product of a shape, treating empty as scalar (1). */
export function product(shape: number[]): number {
  return shape.reduce((a, b) => a * b, shape.length ? 1 : 1);
}
