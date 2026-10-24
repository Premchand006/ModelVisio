// Minimal protobuf wire-format reader. Schema-less: callers walk fields by
// number. Works on a Uint8Array view with offsets (sub-messages are views, not
// copies), so it is cheap enough for multi-hundred-MB weight files and safe in
// a Web Worker. Used by the Caffe / Core ML (and other protobuf) parsers.

export const WT_VARINT = 0;
export const WT_I64 = 1;
export const WT_LEN = 2;
export const WT_SGROUP = 3;
export const WT_EGROUP = 4;
export const WT_I32 = 5;

const utf8 = new TextDecoder();

export class ProtoReader {
  readonly view: DataView;
  pos: number;
  readonly end: number;

  constructor(public readonly buf: Uint8Array, start = 0, end = buf.byteLength) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    this.pos = start;
    this.end = end;
  }

  get eof(): boolean { return this.pos >= this.end; }

  /** Next field tag, or null at the end of this message. */
  next(): { field: number; wireType: number } | null {
    if (this.pos >= this.end) return null;
    const tag = this.varint();
    const field = Math.floor(tag / 8);
    const wireType = tag & 7;
    if (field === 0) throw new Error(`protobuf: invalid field 0 at offset ${this.pos}`);
    return { field, wireType };
  }

  /** Unsigned varint as a JS number (exact up to 2^53). */
  varint(): number {
    let result = 0;
    let mul = 1;
    for (let i = 0; i < 10; i++) {
      if (this.pos >= this.end) throw new Error("protobuf: truncated varint");
      const b = this.buf[this.pos++];
      result += (b & 0x7f) * mul;
      if (b < 0x80) return result;
      mul *= 128;
    }
    throw new Error("protobuf: varint too long");
  }

  /** Signed int64/int32 varint (two's complement, e.g. -1 encodes as 10 bytes). */
  int64(): number {
    const start = this.pos;
    const v = this.varint();
    if (v < 2 ** 53) return v;
    // Beyond 2^53 the float sum is inexact (e.g. -1 = 2^64-1 rounds to 2^64 →
    // 0), so re-decode the varint exactly with BigInt and sign-convert.
    let r = 0n;
    for (let i = this.pos - 1; i >= start; i--) r = (r << 7n) | BigInt(this.buf[i] & 0x7f);
    return Number(BigInt.asIntN(64, r));
  }

  /** int32 field (negative values are sign-extended to 64 bits on the wire). */
  int32(): number { return this.int64() | 0; }

  /** zigzag sint32/sint64. */
  sint(): number {
    const v = this.varint();
    return v % 2 === 0 ? v / 2 : -(v + 1) / 2;
  }

  bool(): boolean { return this.varint() !== 0; }

  fixed32(): number { const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  fixed64(): number { const v = this.view.getBigUint64(this.pos, true); this.pos += 8; return Number(v); }
  float(): number { const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
  double(): number { const v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; }

  /** Length-delimited payload as a zero-copy view. */
  bytes(): Uint8Array {
    const len = this.varint();
    const start = this.pos;
    if (start + len > this.end) throw new Error("protobuf: length-delimited field overruns message");
    this.pos += len;
    return this.buf.subarray(start, start + len);
  }

  string(): string { return utf8.decode(this.bytes()); }

  /** Reader over an embedded message (shares the underlying buffer). */
  sub(): ProtoReader {
    const len = this.varint();
    const start = this.pos;
    if (start + len > this.end) throw new Error("protobuf: sub-message overruns parent");
    this.pos += len;
    return new ProtoReader(this.buf, start, start + len);
  }

  /** Packed repeated float (wire type 2), or a single unpacked float (wire type 5). */
  packedFloats(wireType = WT_LEN): number[] {
    if (wireType === WT_I32) return [this.float()];
    const len = this.varint();
    const end = this.pos + len;
    const out: number[] = new Array(len >> 2);
    for (let i = 0; this.pos + 4 <= end; i++) out[i] = this.float();
    this.pos = end;
    return out;
  }

  /** Packed float payload as a byte view (no decode), for sampled statistics. */
  packedFloatBytes(): Uint8Array { return this.bytes(); }

  /** Packed repeated varints (wire type 2), or one unpacked varint (wire type 0). Signed int64 semantics. */
  packedVarints(wireType = WT_LEN): number[] {
    if (wireType === WT_VARINT) return [this.int64()];
    const len = this.varint();
    const end = this.pos + len;
    const out: number[] = [];
    while (this.pos < end) out.push(this.int64());
    this.pos = end;
    return out;
  }

  /** Skip a field's payload of the given wire type. */
  skip(wireType: number): void {
    switch (wireType) {
      case WT_VARINT: this.varint(); break;
      case WT_I64: this.pos += 8; break;
      case WT_LEN: { const len = this.varint(); this.pos += len; break; }
      case WT_I32: this.pos += 4; break;
      case WT_SGROUP: {
        // Legacy groups: skip nested fields until the matching end-group.
        for (;;) {
          const t = this.next();
          if (!t) throw new Error("protobuf: unterminated group");
          if (t.wireType === WT_EGROUP) break;
          this.skip(t.wireType);
        }
        break;
      }
      case WT_EGROUP: break;
      default: throw new Error(`protobuf: unknown wire type ${wireType}`);
    }
    if (this.pos > this.end) throw new Error("protobuf: field overruns message");
  }
}

/**
 * Cheap structural check: does the whole buffer (or its first `limit` bytes)
 * decode as a sequence of well-formed protobuf fields? Used for format sniffing.
 */
export function looksLikeProtobuf(buf: Uint8Array, limit = 4096): boolean {
  if (buf.length === 0) return false;
  try {
    const r = new ProtoReader(buf, 0, buf.length);
    let n = 0;
    while (!r.eof && r.pos < limit) {
      const t = r.next();
      if (!t || t.wireType === WT_EGROUP || t.wireType > 5) return false;
      if (t.field > 536870911) return false;
      r.skip(t.wireType);
      n++;
    }
    return n > 0;
  } catch {
    return false;
  }
}
