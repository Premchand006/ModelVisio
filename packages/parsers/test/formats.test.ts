import { describe, expect, it } from "vitest";
import {
  detectFormat,
  parseModel,
  parseSafetensors,
  parseNumpy,
  parseGguf,
  parseDarknet,
} from "../src/index";

// Little-endian byte writer for constructing real format buffers.
class W {
  bytes: number[] = [];
  u8(v: number) { this.bytes.push(v & 0xff); }
  u16(v: number) { this.u8(v); this.u8(v >>> 8); }
  u32(v: number) { this.u8(v); this.u8(v >>> 8); this.u8(v >>> 16); this.u8(v >>> 24); }
  u64(v: number) { this.u32(v >>> 0); this.u32(Math.floor(v / 2 ** 32)); }
  str(s: string) { for (const c of new TextEncoder().encode(s)) this.u8(c); }
  ggufStr(s: string) { const b = new TextEncoder().encode(s); this.u64(b.length); for (const c of b) this.u8(c); }
  done() { return new Uint8Array(this.bytes); }
}

describe("safetensors", () => {
  it("parses tensor headers", () => {
    const json = JSON.stringify({
      __metadata__: { format: "pt" },
      "layer.weight": { dtype: "F32", shape: [2, 3], data_offsets: [0, 24] },
    });
    const jb = new TextEncoder().encode(json);
    const w = new W();
    w.u64(jb.length);
    for (const c of jb) w.u8(c);
    for (let i = 0; i < 24; i++) w.u8(0);
    const m = parseSafetensors(w.done(), "t.safetensors");
    expect(m.format).toBe("Safetensors");
    expect(m.layers).toHaveLength(1);
    expect(m.layers[0].shape).toBe("2×3");
    expect(m.layers[0].dt).toBe("float32");
    expect(m.layers[0].params).toBe(6);
  });
});

function npyBytes(): Uint8Array {
  const header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }\n";
  const w = new W();
  for (const b of [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]) w.u8(b);
  w.u8(1); w.u8(0); // version 1.0
  w.u16(header.length);
  w.str(header);
  for (let i = 0; i < 24; i++) w.u8(0); // 2*3*4 bytes of float32 data
  return w.done();
}

describe("numpy", () => {
  it("parses a .npy header", () => {
    const m = parseNumpy(npyBytes(), "a.npy");
    expect(m.format).toBe("NumPy (.npy)");
    expect(m.layers[0].shape).toBe("2×3");
    expect(m.layers[0].dt).toBe("float32");
  });

  it("parses a .npz (stored zip of .npy)", () => {
    const npz = makeZip([{ name: "arr_0.npy", data: npyBytes() }, { name: "arr_1.npy", data: npyBytes() }]);
    const m = parseNumpy(npz, "bundle.npz");
    expect(m.format).toBe("NumPy (.npz)");
    expect(m.layers).toHaveLength(2);
    expect(m.layers[0].shape).toBe("2×3");
  });
});

describe("gguf", () => {
  it("parses metadata and tensor infos", () => {
    const w = new W();
    w.str("GGUF");
    w.u32(3); // version
    w.u64(1); // tensor count
    w.u64(1); // metadata kv count
    w.ggufStr("general.architecture");
    w.u32(8); // value type: string
    w.ggufStr("llama");
    // one tensor
    w.ggufStr("token_embd.weight");
    w.u32(2); // n_dims
    w.u64(4096); w.u64(32000); // dims
    w.u32(0); // ggml type F32
    w.u64(0); // offset
    const m = parseGguf(w.done(), "x.gguf");
    expect(m.format).toBe("GGUF");
    expect(m.framework).toContain("llama");
    expect(m.layers).toHaveLength(1);
    expect(m.layers[0].shape).toBe("4096×32000");
    expect(m.layers[0].dt).toBe("F32");
    expect(m.layers[0].group).toBe("input"); // "embd" → embedding → input
  });
});

describe("darknet", () => {
  const cfg = `
[net]
width=416
height=416
channels=3

[convolutional]
filters=16
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
filters=32
size=3
stride=1
pad=1

[yolo]
`;
  it("builds a real graph with shapes and edges", () => {
    const m = parseDarknet(cfg, "tiny.cfg");
    expect(m.format).toBe("Darknet");
    // input + conv + maxpool + conv + yolo = 5
    expect(m.layers).toHaveLength(5);
    const conv1 = m.layers[1];
    expect(conv1.op).toBe("convolutional");
    expect(conv1.shape).toBe("1×16×416×416"); // pad=1, stride 1 keeps size
    expect(conv1.params).toBe(16 * 3 * 3 * 3 + 16); // no BN
    const pool = m.layers[2];
    expect(pool.shape).toBe("1×16×208×208"); // stride 2 halves
    expect(m.edges.length).toBeGreaterThanOrEqual(4);
  });
});

describe("registry", () => {
  it("detects formats by extension and magic", () => {
    expect(detectFormat("m.onnx", new Uint8Array([0, 0]))).toBe("onnx");
    expect(detectFormat("m.safetensors", new Uint8Array([0, 0]))).toBe("safetensors");
    expect(detectFormat("m.pt", new Uint8Array([0x50, 0x4b, 0x03, 0x04]))).toBe("pytorch");
    // GGUF magic overrides a misleading extension.
    expect(detectFormat("weights.bin", new Uint8Array([0x47, 0x47, 0x55, 0x46]))).toBe("gguf");
  });

  it("routes unknown/heavy formats to honest metadata", () => {
    const m = parseModel(new Uint8Array([1, 2, 3, 4]), "model.mlmodel");
    expect(m.format).toBe("Core ML");
    expect(m.layers).toHaveLength(1);
    expect(m.layers[0].op).toBe("Detected");
  });
});

// --- minimal STORED-only ZIP writer for the .npz test ---
function makeZip(files: { name: string; data: Uint8Array }[]): Uint8Array {
  const enc = new TextEncoder();
  const local: number[] = [];
  const pushU16 = (a: number[], v: number) => a.push(v & 0xff, (v >>> 8) & 0xff);
  const pushU32 = (a: number[], v: number) => a.push(v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff);
  const records: { name: Uint8Array; size: number; off: number }[] = [];
  for (const f of files) {
    const name = enc.encode(f.name);
    const off = local.length;
    pushU32(local, 0x04034b50);
    pushU16(local, 20); pushU16(local, 0); pushU16(local, 0); // ver, flags, method=stored
    pushU16(local, 0); pushU16(local, 0); // time, date
    pushU32(local, 0); // crc
    pushU32(local, f.data.length); pushU32(local, f.data.length);
    pushU16(local, name.length); pushU16(local, 0);
    for (const c of name) local.push(c);
    for (const c of f.data) local.push(c);
    records.push({ name, size: f.data.length, off });
  }
  const cdStart = local.length;
  const cd: number[] = [];
  for (const r of records) {
    pushU32(cd, 0x02014b50);
    pushU16(cd, 20); pushU16(cd, 20); pushU16(cd, 0); pushU16(cd, 0);
    pushU16(cd, 0); pushU16(cd, 0); pushU32(cd, 0);
    pushU32(cd, r.size); pushU32(cd, r.size);
    pushU16(cd, r.name.length); pushU16(cd, 0); pushU16(cd, 0);
    pushU16(cd, 0); pushU16(cd, 0); pushU32(cd, 0);
    pushU32(cd, r.off);
    for (const c of r.name) cd.push(c);
  }
  const eocd: number[] = [];
  pushU32(eocd, 0x06054b50);
  pushU16(eocd, 0); pushU16(eocd, 0);
  pushU16(eocd, records.length); pushU16(eocd, records.length);
  pushU32(eocd, cd.length); pushU32(eocd, cdStart);
  pushU16(eocd, 0);
  return new Uint8Array([...local, ...cd, ...eocd]);
}
