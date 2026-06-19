// Minimal Python unpickler — enough opcodes to read PyTorch `.pt` archives
// (protocol 2–5). It does not execute Python; unknown classes become PyObj
// holders whose state (`__dict__`) we can walk. Inspired by Netron's approach.

const MARK = Symbol("mark");

export type GlobalRef = { __global__: true; module: string; name: string };
export class PyObj {
  attrs: Map<unknown, unknown> = new Map();
  constructor(public cls: GlobalRef | null) {}
}

export type Handlers = {
  persistentLoad: (pid: unknown) => unknown;
  findClass: (module: string, name: string) => unknown;
  reduce: (func: unknown, args: unknown[]) => unknown;
  build?: (obj: unknown, state: unknown) => unknown;
};

const isGlobal = (x: unknown): x is GlobalRef =>
  typeof x === "object" && x !== null && (x as GlobalRef).__global__ === true;

export function unpickle(buf: Uint8Array, h: Handlers): unknown {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 0;
  const stack: unknown[] = [];
  const memo = new Map<number, unknown>();

  const u8 = () => buf[pos++];
  const u16 = () => { const v = view.getUint16(pos, true); pos += 2; return v; };
  const i32 = () => { const v = view.getInt32(pos, true); pos += 4; return v; };
  const u32 = () => { const v = view.getUint32(pos, true); pos += 4; return v; };
  const u64 = () => { const v = Number(view.getBigUint64(pos, true)); pos += 8; return v; };
  const f64be = () => { const v = view.getFloat64(pos, false); pos += 8; return v; };
  const bytes = (n: number) => { const b = buf.subarray(pos, pos + n); pos += n; return b; };
  const str = (n: number) => new TextDecoder().decode(bytes(n));
  const line = () => { let s = ""; while (buf[pos] !== 0x0a) s += String.fromCharCode(buf[pos++]); pos++; return s; };
  const longN = (n: number) => { // little-endian two's-complement → number
    if (n === 0) return 0;
    let v = 0; for (let i = 0; i < n; i++) v += buf[pos + i] * 2 ** (8 * i);
    if (buf[pos + n - 1] & 0x80) v -= 2 ** (8 * n);
    pos += n; return v;
  };
  const popMark = () => {
    const items: unknown[] = [];
    while (stack.length) { const x = stack.pop(); if (x === MARK) break; items.push(x); }
    return items.reverse();
  };
  const top = () => stack[stack.length - 1];

  for (let guard = 0; pos < buf.length; guard++) {
    const op = buf[pos++];
    switch (op) {
      case 0x80: u8(); break;                                  // PROTO
      case 0x95: u64(); break;                                 // FRAME
      case 0x28: stack.push(MARK); break;                      // MARK
      case 0x2e: return stack.pop();                            // STOP
      case 0x4e: stack.push(null); break;                      // NONE
      case 0x88: stack.push(true); break;                      // NEWTRUE
      case 0x89: stack.push(false); break;                     // NEWFALSE
      case 0x4a: stack.push(i32()); break;                     // BININT
      case 0x4b: stack.push(u8()); break;                      // BININT1
      case 0x4d: stack.push(u16()); break;                     // BININT2
      case 0x8a: stack.push(longN(u8())); break;               // LONG1
      case 0x8b: stack.push(longN(u32())); break;              // LONG4
      case 0x47: stack.push(f64be()); break;                   // BINFLOAT
      case 0x58: stack.push(str(u32())); break;                // BINUNICODE
      case 0x8c: stack.push(str(u8())); break;                 // SHORT_BINUNICODE
      case 0x8d: stack.push(str(u64())); break;                // BINUNICODE8
      case 0x54: stack.push(str(u32())); break;                // BINSTRING
      case 0x55: stack.push(str(u8())); break;                 // SHORT_BINSTRING
      case 0x42: stack.push(bytes(u32())); break;              // BINBYTES
      case 0x43: stack.push(bytes(u8())); break;               // SHORT_BINBYTES
      case 0x8e: stack.push(bytes(u64())); break;              // BINBYTES8
      case 0x29: stack.push([]); break;                        // EMPTY_TUPLE
      case 0x5d: stack.push([]); break;                        // EMPTY_LIST
      case 0x7d: stack.push(new Map()); break;                 // EMPTY_DICT
      case 0x8f: stack.push(new Set()); break;                 // EMPTY_SET
      case 0x74: stack.push(popMark()); break;                 // TUPLE
      case 0x85: stack.push([stack.pop()]); break;             // TUPLE1
      case 0x86: { const b = stack.pop(); const a = stack.pop(); stack.push([a, b]); break; } // TUPLE2
      case 0x87: { const c = stack.pop(); const b = stack.pop(); const a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
      case 0x61: { const v = stack.pop(); (top() as unknown[]).push(v); break; }          // APPEND
      case 0x65: { const items = popMark(); (top() as unknown[]).push(...items); break; } // APPENDS
      case 0x73: { const v = stack.pop(); const k = stack.pop(); (top() as Map<unknown, unknown>).set(k, v); break; } // SETITEM
      case 0x75: { const items = popMark(); const d = top() as Map<unknown, unknown>; for (let i = 0; i < items.length; i += 2) d.set(items[i], items[i + 1]); break; } // SETITEMS
      case 0x64: { const items = popMark(); const d = new Map(); for (let i = 0; i < items.length; i += 2) d.set(items[i], items[i + 1]); stack.push(d); break; } // DICT
      case 0x63: { const m = line(); const n = line(); stack.push(h.findClass(m, n)); break; } // GLOBAL
      case 0x93: { const n = stack.pop() as string; const m = stack.pop() as string; stack.push(h.findClass(m, n)); break; } // STACK_GLOBAL
      case 0x52: { const args = stack.pop() as unknown[]; const func = stack.pop(); stack.push(h.reduce(func, args)); break; } // REDUCE
      case 0x81: { const args = stack.pop() as unknown[]; const cls = stack.pop(); stack.push(newObj(cls, args, h)); break; } // NEWOBJ
      case 0x92: { stack.pop(); const args = stack.pop() as unknown[]; const cls = stack.pop(); stack.push(newObj(cls, args, h)); break; } // NEWOBJ_EX
      case 0x62: { const state = stack.pop(); const obj = stack.pop(); stack.push(doBuild(obj, state, h)); break; } // BUILD
      case 0x51: { const pid = stack.pop(); stack.push(h.persistentLoad(pid)); break; } // BINPERSID
      case 0x50: { const pid = line(); stack.push(h.persistentLoad(pid)); break; }      // PERSID
      case 0x68: stack.push(memo.get(u8())); break;            // BINGET
      case 0x6a: stack.push(memo.get(u32())); break;           // LONG_BINGET
      case 0x67: stack.push(memo.get(parseInt(line(), 10))); break; // GET
      case 0x71: memo.set(u8(), top()); break;                 // BINPUT
      case 0x72: memo.set(u32(), top()); break;                // LONG_BINPUT
      case 0x70: memo.set(parseInt(line(), 10), top()); break; // PUT
      case 0x94: memo.set(memo.size, top()); break;            // MEMOIZE
      case 0x30: stack.pop(); break;                           // POP
      case 0x31: popMark(); break;                             // POP_MARK
      case 0x32: stack.push(top()); break;                     // DUP
      case 0x49: { const v = line(); stack.push(v === "01" ? true : v === "00" ? false : parseInt(v, 10)); break; } // INT
      case 0x4c: stack.push(parseInt(line().replace(/L$/, ""), 10)); break; // LONG
      case 0x46: stack.push(parseFloat(line())); break;        // FLOAT
      case 0x53: stack.push(line().replace(/^'|'$/g, "")); break; // STRING
      case 0x56: stack.push(line()); break;                    // UNICODE
      default:
        throw new Error(`Unsupported pickle opcode 0x${op.toString(16)} at ${pos - 1}`);
    }
    if (guard > 50_000_000) throw new Error("Pickle stream too long.");
  }
  return stack.pop();
}

function newObj(cls: unknown, args: unknown[], h: Handlers): unknown {
  if (isGlobal(cls)) {
    const r = h.reduce(cls, args);
    if (r !== undefined) return r;
  }
  return new PyObj(isGlobal(cls) ? cls : null);
}

function doBuild(obj: unknown, state: unknown, h: Handlers): unknown {
  if (h.build) { const r = h.build(obj, state); if (r !== undefined) return r; }
  if (obj instanceof PyObj) {
    let s = state;
    if (Array.isArray(s) && s.length === 2 && (s[0] instanceof Map || s[0] === null)) s = s[0];
    if (s instanceof Map) obj.attrs = s;
  }
  return obj;
}

export { isGlobal };
