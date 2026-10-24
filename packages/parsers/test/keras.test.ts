import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { deflateRawSync } from "node:zlib";
import { parseModel } from "../src/index";
import { describe, expect, it } from "vitest";
import { isHdf5, isKerasJson, parseKeras, parseKerasArchive, parseKerasH5, parseKerasJson } from "../src/formats/keras";
import { extractStored, readZipEntries } from "../src/common/zip";
import { inflateRaw } from "../src/common/inflate";

// Fixtures: test/fixtures/scripts/gen_tensorflow_keras.py (Keras 3.12.4, tf_keras 2.21.0).
// expected.json carries each layer's count_params() and output shape from Keras itself.
const dir = fileURLToPath(new URL("./fixtures/keras/", import.meta.url));
const read = (f: string) => new Uint8Array(readFileSync(dir + f));
const expected = JSON.parse(readFileSync(dir + "expected.json", "utf8"));
const total = (m: { layers: { params: number }[] }) => m.layers.reduce((a, l) => a + l.params, 0);
const fmt = (s: (number | null)[]) => s.map((d) => (d == null ? "?" : String(d))).join("×");

describe("Keras 3 functional (.keras)", () => {
  const m = parseKerasArchive(read("tiny_functional.keras"), "tiny_functional.keras");
  const exp = expected.tiny_functional;

  it("builds one node per layer in dependency order", () => {
    expect(m.format).toBe("Keras (.keras)");
    expect(m.producer).toBe("Keras 3.12.4");
    expect(m.layers.map((l) => l.name).sort()).toEqual([...exp.layers].sort());
    expect(m.layers[0].op).toBe("InputLayer");
    expect(m.layers[0].group).toBe("input");
    expect(m.layers[m.layers.length - 1].name).toBe("logits");
    expect(m.layers[m.layers.length - 1].group).toBe("output");
  });

  it("matches Keras count_params per layer and in total", () => {
    for (const l of m.layers) expect([l.name, l.params]).toEqual([l.name, exp.per_layer[l.name]]);
    expect(total(m)).toBe(exp.params);
  });

  it("propagates output shapes like Keras", () => {
    for (const l of m.layers) expect([l.name, l.shape]).toEqual([l.name, fmt(exp.output_shapes[l.name])]);
    expect(m.inputShape).toEqual([0, 32, 32, 3]);
    expect(m.outputShape).toEqual([0, 10]);
  });

  it("decodes Keras 3 inbound_nodes into edges (residual Add, Concatenate)", () => {
    const id = (n: string) => m.layers.find((l) => l.name === n)!.id;
    const has = (a: string, b: string) => m.edges.some(([x, y]) => x === id(a) && y === id(b));
    expect(has("stem_relu", "residual")).toBe(true);
    expect(has("pw", "residual")).toBe(true);
    expect(has("sep", "concat")).toBe(true);
    expect(has("zpad", "concat")).toBe(true);
    expect(m.edges).toHaveLength(17);
  });

  it("estimates MACs (conv, depthwise, separable, dense)", () => {
    const L = (n: string) => m.layers.find((l) => l.name === n)!;
    expect(L("stem_conv").macs).toBe(32 * 32 * 8 * 27);
    expect(L("dw").macs).toBe(32 * 32 * 8 * 9);
    expect(L("sep").macs).toBe(16 * 16 * 8 * 9 + 16 * 16 * 8 * 8);
    expect(L("logits").macs).toBe(12 * 10);
    expect(L("logits").math).toContain("softmax");
    expect(L("stem_conv").w?.shape).toEqual([3, 3, 3, 8]);
  });
});

describe("Keras legacy HDF5 (.h5)", () => {
  it("finds the Keras 3 model_config attribute and matches the .keras graph", () => {
    const buf = read("tiny_functional.h5");
    expect(isHdf5(buf)).toBe(true);
    const h = parseKerasH5(buf, "tiny_functional.h5");
    const k = parseKerasArchive(read("tiny_functional.keras"));
    expect(h.format).toBe("Keras (HDF5)");
    expect(h.layers.map((l) => [l.name, l.op, l.shape, l.params])).toEqual(k.layers.map((l) => [l.name, l.op, l.shape, l.params]));
    expect(h.edges).toEqual(k.edges);
  });

  it("handles the Keras 2 (tf_keras) inbound_nodes dialect", () => {
    const m = parseKeras(read("tiny_k2.h5"), "tiny_k2.h5");
    const exp = expected.tiny_k2;
    for (const l of m.layers) expect([l.name, l.params]).toEqual([l.name, exp.per_layer[l.name]]);
    expect(total(m)).toBe(exp.params);
    const id = (n: string) => m.layers.find((l) => l.name === n)!.id;
    expect(m.edges).toContainEqual([id("flat"), id("d1")]);
    expect(m.edges).toContainEqual([id("flat"), id("d2")]);
    expect(m.edges).toContainEqual([id("d1"), id("sum")]);
    expect(m.edges).toContainEqual([id("d2"), id("sum")]);
    expect(m.layers.find((l) => l.name === "flat")!.shape).toBe("?×144");
  });

  it("errors clearly on HDF5 without a model config", () => {
    const fake = new Uint8Array([0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0]);
    expect(() => parseKerasH5(fake)).toThrow(/model_config/);
  });
});

describe("Keras 3 Sequential (.keras)", () => {
  const m = parseKeras(read("tiny_sequential.keras"), "tiny_sequential.keras");
  const exp = expected.tiny_sequential;

  it("chains layers and matches params (Embedding, LSTM, Bidirectional GRU, Dense)", () => {
    expect(m.layers.map((l) => l.op)).toEqual(["InputLayer", "Embedding", "LSTM", "Bidirectional", "Dense"]);
    expect(m.edges).toEqual([[0, 1], [1, 2], [2, 3], [3, 4]]);
    for (const l of m.layers.slice(1)) expect([l.name, l.params]).toEqual([l.name, exp.per_layer[l.name]]);
    expect(total(m)).toBe(exp.params);
    for (const l of m.layers.slice(1)) expect([l.name, l.shape]).toEqual([l.name, fmt(exp.output_shapes[l.name])]);
  });
});

describe("Keras JSON + DEFLATE support", () => {
  const entries = readZipEntries(read("tiny_functional.keras"));
  const cfg = extractStored(read("tiny_functional.keras"), entries.find((e) => e.name === "config.json")!)!;

  it("inflates raw DEFLATE streams (stored, fixed and dynamic Huffman blocks)", () => {
    for (const level of [0, 1, 9]) {
      const z = deflateRawSync(cfg, { level });
      expect(Buffer.from(inflateRaw(new Uint8Array(z), cfg.length)).equals(Buffer.from(cfg))).toBe(true);
    }
    const tiny = new TextEncoder().encode("aaaaaaaaaabbbbbbbbbbaaaaaaaaaa");
    expect(new TextDecoder().decode(inflateRaw(new Uint8Array(deflateRawSync(tiny))))).toBe("aaaaaaaaaabbbbbbbbbbaaaaaaaaaa");
  });

  it("parses a bare Keras config JSON and a TF.js layers-model wrapper", () => {
    const text = new TextDecoder().decode(cfg);
    expect(isKerasJson(text)).toBe(true);
    const a = parseKerasJson(text, "model.json");
    expect(a.format).toBe("Keras (JSON)");
    expect(total(a)).toBe(expected.tiny_functional.params);
    const tfjs = JSON.stringify({ format: "layers-model", modelTopology: { keras_version: "tfjs-layers 4.0.0", model_config: JSON.parse(text) }, weightsManifest: [] });
    const b = parseKerasJson(tfjs, "model.json");
    expect(b.format).toBe("TensorFlow.js (layers)");
    expect(b.layers).toHaveLength(a.layers.length);
    expect(isKerasJson(JSON.stringify({ format: "graph-model", modelTopology: { node: [] } }))).toBe(false);
  });
});

describe("TF.js layers model weights (official tensorflowjs converter output)", () => {
  const d = fileURLToPath(new URL("./fixtures/keras/tfjs/", import.meta.url));
  const r = (f: string) => new Uint8Array(readFileSync(d + f));
  // TF-Keras's own values for each weight, written when the fixture was generated.
  const truth = JSON.parse(readFileSync(d + "weights.expected.json", "utf8")) as Record<string, { mean: number; std: number; min: number; max: number }>;
  const kernelOf = (m: ReturnType<typeof parseModel>, layer: string) => m.layers.find((l) => l.name === layer)!.w!;

  it("decodes float32 shards", () => {
    const m = parseModel(r("model.json"), "model.json", { "group1-shard1of1.bin": r("group1-shard1of1.bin") });
    for (const [layer, key] of [["conv", "conv/kernel"], ["dw", "dw/depthwise_kernel"], ["fc", "fc/kernel"]]) {
      const w = kernelOf(m, layer);
      expect(w.mean, layer).toBeCloseTo(truth[key].mean, 3);
      expect(w.std, layer).toBeCloseTo(truth[key].std, 3);
      expect(w.max, layer).toBeCloseTo(truth[key].max, 3);
    }
    expect(m.notes).toBeUndefined();
  });

  it("dequantizes uint8 shards (within one quantization step)", () => {
    const m = parseModel(r("quant/model.json"), "model.json", { "group1-shard1of1.bin": r("quant/group1-shard1of1.bin") });
    const w = kernelOf(m, "conv");
    const t = truth["conv/kernel"];
    const step = (t.max - t.min) / 255;
    expect(Math.abs(w.mean! - t.mean)).toBeLessThan(step);
    expect(Math.abs(w.min! - t.min)).toBeLessThan(step);
    expect(Math.abs(w.max! - t.max)).toBeLessThan(step);
  });

  it("notes the missing shard", () => {
    const m = parseModel(r("model.json"), "model.json");
    expect(kernelOf(m, "conv").std).toBeUndefined();
    expect(m.notes?.[0]).toMatch(/group1-shard1of1\.bin/);
  });
});
