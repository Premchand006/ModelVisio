import { describe, expect, it } from "vitest";
import { modelProtoType } from "../src/onnx/proto";
import { parseModel } from "../src/registry";
import { parseModelProgressive, parseModelStream, type ParseChunk } from "../src/stream";
import type { Model } from "../src/types";

// ─── Synthetic fixtures ──────────────────────────────────────────────────────

/**
 * A real ONNX model with `blocks` residual blocks:
 *   x → MatMul(W_i) → Add(b_i) → Relu → Add(skip from block input) → …
 * so it has fan-out/fan-in edges, initializers and inferred shapes.
 */
function buildOnnx(blocks: number, dim = 8): Uint8Array {
  const ModelProto = modelProtoType();
  const tensor = (name: string, dims: number[]) => ({
    name, data_type: 1, dims,
    float_data: Array.from({ length: dims.reduce((a, b) => a * b, 1) }, (_, i) => (i % 7) * 0.01),
  });
  const vi = (name: string, dims: number[]) => ({
    name, type: { tensor_type: { elem_type: 1, shape: { dim: dims.map((d) => ({ dim_value: d })) } } },
  });
  const initializer = [];
  const node = [];
  let cur = "X";
  for (let i = 0; i < blocks; i++) {
    initializer.push(tensor(`W${i}`, [dim, dim]), tensor(`b${i}`, [dim]));
    node.push(
      { op_type: "MatMul", name: `mm${i}`, input: [cur, `W${i}`], output: [`m${i}`] },
      { op_type: "Add", name: `bias${i}`, input: [`m${i}`, `b${i}`], output: [`a${i}`] },
      { op_type: "Relu", name: `relu${i}`, input: [`a${i}`], output: [`r${i}`] },
      { op_type: "Add", name: `res${i}`, input: [`r${i}`, cur], output: [`y${i}`] },
    );
    cur = `y${i}`;
  }
  node.push({ op_type: "Identity", name: "out", input: [cur], output: ["out"] });
  const obj = {
    ir_version: 8, producer_name: "stream-test",
    opset_import: [{ domain: "", version: 17 }],
    graph: { name: "chain", input: [vi("X", [1, dim])], output: [vi("out", [1, dim])], initializer, node },
  };
  return ModelProto.encode(ModelProto.create(obj)).finish();
}

/**
 * Minimal FlatBuffer writer: tables laid out front to back (vtable before its
 * table, children after), which is all the TFLite reader needs.
 */
type FbField = { u8: number } | { i32: number } | { off: (w: FbWriter) => number } | null;
class FbWriter {
  b: number[] = [];
  get pos() { return this.b.length; }
  align() { while (this.b.length % 4) this.b.push(0); }
  u16(v: number) { this.b.push(v & 255, (v >>> 8) & 255); }
  u32(v: number) { this.b.push(v & 255, (v >>> 8) & 255, (v >>> 16) & 255, (v >>> 24) & 255); }
  patch(at: number, v: number) { for (let i = 0; i < 4; i++) this.b[at + i] = (v >>> (8 * i)) & 255; }
  table(fields: FbField[]): number {
    this.align();
    const vt = this.pos;
    this.u16(4 + fields.length * 2);
    this.u16(4 + fields.length * 4);
    fields.forEach((f, i) => this.u16(f ? 4 + i * 4 : 0));
    this.align();
    const t = this.pos;
    this.u32(t - vt);
    const offs: [number, (w: FbWriter) => number][] = [];
    for (const f of fields) {
      const at = this.pos;
      if (f && "off" in f) offs.push([at, f.off]);
      this.u32(f && "i32" in f ? f.i32 : f && "u8" in f ? f.u8 : 0);
    }
    for (const [at, write] of offs) this.patch(at, write(this) - at);
    return t;
  }
  str(s: string): number {
    this.align();
    const p = this.pos;
    const bytes = new TextEncoder().encode(s);
    this.u32(bytes.length);
    this.b.push(...bytes, 0);
    return p;
  }
  ints(v: number[]): number {
    this.align();
    const p = this.pos;
    this.u32(v.length);
    for (const x of v) this.u32(x);
    return p;
  }
  tables(items: FbField[][]): number {
    this.align();
    const p = this.pos;
    this.u32(items.length);
    const slots = items.map(() => { const at = this.pos; this.u32(0); return at; });
    items.forEach((fields, i) => this.patch(slots[i], this.table(fields) - slots[i]));
    return p;
  }
}

/** TFLite: input → `n` FullyConnected/Relu operators chained → output. */
function buildTflite(n: number): Uint8Array {
  const w = new FbWriter();
  w.u32(0); // root offset, patched below
  const tensors: FbField[][] = [];
  for (let i = 0; i <= n; i++) {
    tensors.push([{ off: (x) => x.ints([1, 16]) }, { u8: 0 }, null, { off: (x) => x.str(`t${i}`) }]);
  }
  const ops: FbField[][] = [];
  for (let i = 0; i < n; i++) {
    ops.push([{ i32: i % 2 }, { off: (x) => x.ints([i]) }, { off: (x) => x.ints([i + 1]) }]);
  }
  const root = w.table([
    { i32: 3 }, // version
    { off: (x) => x.tables([[{ u8: 9 }, null, null, { i32: 9 }], [{ u8: 19 }, null, null, { i32: 19 }]]) },
    { off: (x) => x.tables([[
      { off: (x2) => x2.tables(tensors) },
      { off: (x2) => x2.ints([0]) },
      { off: (x2) => x2.ints([n]) },
      { off: (x2) => x2.tables(ops) },
    ]]) },
  ]);
  w.patch(0, root);
  return new Uint8Array(w.b);
}

class LE {
  b: number[] = [];
  u8(v: number) { this.b.push(v & 255); }
  u32(v: number) { for (let i = 0; i < 4; i++) this.u8(v >>> (8 * i)); }
  u64(v: number) { this.u32(v >>> 0); this.u32(Math.floor(v / 2 ** 32)); }
  bytes(s: string) { for (const c of new TextEncoder().encode(s)) this.u8(c); }
  gstr(s: string) { const e = new TextEncoder().encode(s); this.u64(e.length); for (const c of e) this.u8(c); }
}

function buildGguf(n: number): Uint8Array {
  const w = new LE();
  w.bytes("GGUF"); w.u32(3); w.u64(n); w.u64(1);
  w.gstr("general.architecture"); w.u32(8); w.gstr("llama");
  for (let i = 0; i < n; i++) {
    w.gstr(`blk.${i}.ffn.weight`); w.u32(2); w.u64(64); w.u64(64); w.u32(1); w.u64(0);
  }
  return new Uint8Array(w.b);
}

function buildSafetensors(n: number): Uint8Array {
  const header: Record<string, unknown> = { __metadata__: { format: "pt" } };
  for (let i = 0; i < n; i++) header[`layers.${i}.weight`] = { dtype: "F16", shape: [4, 4], data_offsets: [i * 32, (i + 1) * 32] };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const w = new LE();
  w.u64(json.length);
  for (const c of json) w.u8(c);
  return new Uint8Array([...w.b, ...new Uint8Array(n * 32)]);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function collect(buf: Uint8Array, name: string, batchSize: number) {
  const chunks: ParseChunk[] = [];
  const model = parseModelProgressive(buf, name, (c) => chunks.push(c), { batchSize });
  return { chunks, model };
}

/** Concatenated chunks must rebuild the one-shot model exactly. */
function expectChunksRebuild(chunks: ParseChunk[], reference: Model) {
  const layers = chunks.flatMap((c) => c.layers);
  const edges = chunks.flatMap((c) => c.edges);
  expect(layers.map((l) => l.id)).toEqual(reference.layers.map((l) => l.id));
  expect(layers).toEqual(reference.layers);
  expect(edges).toEqual(reference.edges);
  // No chunk references a layer that hasn't been emitted yet.
  const seen = new Set<number>();
  for (const c of chunks) {
    for (const l of c.layers) seen.add(l.id);
    for (const [a, b] of c.edges) {
      expect(seen.has(a)).toBe(true);
      expect(seen.has(b)).toBe(true);
    }
  }
  // Progress is monotonic and ends at 1.
  const p = chunks.map((c) => c.progress);
  expect([...p].sort((a, b) => a - b)).toEqual(p);
  expect(p[p.length - 1]).toBe(1);
  // Merged headers equal the final header.
  const header = Object.assign({}, ...chunks.map((c) => c.header ?? {}));
  const { layers: _l, edges: _e, ...ref } = reference;
  expect(header).toEqual(ref);
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("progressive parsing (parseModelStream)", () => {
  it("ONNX: header first, many node batches, and chunks rebuild parseModel exactly", () => {
    const bytes = buildOnnx(400); // 1 input + 1601 node layers
    const reference = parseModel(bytes, "chain.onnx");
    const { chunks, model } = collect(bytes, "chain.onnx", 200);

    expect(model).toEqual(reference);
    expect(chunks.length).toBeGreaterThanOrEqual(9);
    // The first chunk is the early header (no layers yet).
    expect(chunks[0].layers).toHaveLength(0);
    expect(chunks[0].header).toMatchObject({ name: "chain.onnx", format: "ONNX", opset: 17, inputShape: [1, 8] });
    // Batches are bounded by the batch size.
    for (const c of chunks) expect(c.layers.length).toBeLessThanOrEqual(201);
    expectChunksRebuild(chunks, reference);
    expect(reference.layers).toHaveLength(1602);
  });

  it("ONNX: the generator can be paused between chunks", () => {
    const it = parseModelStream(buildOnnx(50), "chain.onnx", { batchSize: 32 });
    const first = it.next();
    expect(first.done).toBe(false);
    let n = 1, r = it.next();
    while (!r.done) { n++; r = it.next(); }
    expect(n).toBeGreaterThan(3);
    expect(r.value.layers).toHaveLength(1 + 50 * 4 + 1);
  });

  it("TFLite: streams operator batches that rebuild parseModel exactly", () => {
    const bytes = buildTflite(600);
    const reference = parseModel(bytes, "chain.tflite");
    expect(reference.layers).toHaveLength(601);
    expect(reference.edges).toHaveLength(600);
    expect(reference.layers[1].op).toBe("FullyConnected");
    expect(reference.layers[2].op).toBe("Relu");
    const { chunks, model } = collect(bytes, "chain.tflite", 128);
    expect(model).toEqual(reference);
    expect(chunks.length).toBeGreaterThanOrEqual(5);
    expectChunksRebuild(chunks, reference);
  });

  it("GGUF: streams tensor-layer slices that rebuild parseModel exactly", () => {
    const bytes = buildGguf(700);
    const reference = parseModel(bytes, "big.gguf");
    const { chunks, model } = collect(bytes, "big.gguf", 256);
    expect(model).toEqual(reference);
    expect(chunks).toHaveLength(3);
    expectChunksRebuild(chunks, reference);
  });

  it("Safetensors: streams tensor-layer slices that rebuild parseModel exactly", () => {
    const bytes = buildSafetensors(500);
    const reference = parseModel(bytes, "big.safetensors");
    const { chunks, model } = collect(bytes, "big.safetensors", 100);
    expect(model).toEqual(reference);
    expect(chunks).toHaveLength(5);
    expectChunksRebuild(chunks, reference);
  });

  it("formats without an incremental path arrive as one final chunk", () => {
    const cfg = "[net]\nwidth=32\nheight=32\nchannels=3\n\n[convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\n";
    const bytes = new TextEncoder().encode(cfg);
    const reference = parseModel(bytes, "tiny.cfg");
    const { chunks, model } = collect(bytes, "tiny.cfg", 1);
    expect(model).toEqual(reference);
    expect(chunks).toHaveLength(1);
    expectChunksRebuild(chunks, reference);
  });

  it("parse errors propagate from the stream", () => {
    expect(() => parseModelProgressive(new Uint8Array([1, 2, 3]), "bad.gguf", () => {})).toThrow();
  });
});
