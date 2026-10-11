import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isNcnnParam, parseNcnn } from "../src/formats/ncnn";

// Fixtures: TinyNet converted by pnnx (see fixtures/scripts/make_ncnn.py).
const fx = (f: string) => readFileSync(fileURLToPath(new URL(`./fixtures/ncnn/${f}`, import.meta.url)));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

// conv1 3→8 3×3 (216+8) + conv2 8→8 3×3 (576+8) + conv3 8→16 1×1 (128+16) + fc 16→10 (160+10)
const PARAMS = 1122;
// MACs: 8·32·32·27 + 8·16·16·72 + 16·8·8·8 + 160
const MACS = 221184 + 147456 + 8192 + 160;

describe("ncnn .param (pnnx TinyNet)", () => {
  it("detects the magic", () => {
    expect(isNcnnParam(fx("tinynet.ncnn.param"))).toBe(true);
    expect(isNcnnParam(fx("tinynet_shaped.ncnn.param.bin"))).toBe(true);
    expect(isNcnnParam(fx("tinynet.ncnn.bin"))).toBe(false);
  });

  it("builds the layer graph with edges, params and shapes", () => {
    const m = parseNcnn(fx("tinynet_shaped.ncnn.param"), "tinynet.param");
    expect(m.format).toBe("ncnn");
    expect(m.layers).toHaveLength(12);
    expect(m.layers.map((l) => l.type)).toEqual([
      "Input", "Convolution", "Pooling", "Split", "Convolution", "BinaryOp", "Convolution",
      "Pooling", "Reshape", "Flatten", "InnerProduct", "Softmax",
    ]);
    // 11 sequential edges + the residual branch Split→BinaryOp.
    expect(m.edges).toHaveLength(12);
    expect(m.edges).toContainEqual([3, 5]);
    expect(m.edges).toContainEqual([4, 5]);

    expect(m.inputShape).toEqual([3, 32, 32]);
    expect(m.outputShape).toEqual([10]);
    const shape = (n: string) => m.layers.find((l) => l.name === n)?.shape;
    expect(shape("convrelu_0")).toBe("8×32×32");
    expect(shape("maxpool2d_4")).toBe("8×16×16");
    expect(shape("convrelu_2")).toBe("16×8×8");
    expect(shape("gap_5")).toBe("16");
    expect(shape("reshape_10")).toBe("16×1×1");
    expect(shape("linear_3")).toBe("10");

    expect(sum(m.layers.map((l) => l.params))).toBe(PARAMS);
    expect(sum(m.layers.map((l) => l.macs))).toBe(MACS);
    const conv = m.layers[1];
    expect(conv.attr.num_output).toBe(8);
    expect(conv.attr.kernel_h).toBe(3);
    expect(conv.attr.activation).toBe("ReLU");
    expect(conv.w?.shape).toEqual([8, 3, 3, 3]);
    expect(m.layers[0].group).toBe("input");
    expect(m.layers[11].group).toBe("output");
  });

  it("uses opts.inputShape when the Input layer has no dims", () => {
    const bare = parseNcnn(fx("tinynet.ncnn.param"), "tinynet.param");
    expect(bare.layers[1].shape).toBe("?");
    const m = parseNcnn(fx("tinynet.ncnn.param"), "tinynet.param", { inputShape: [3, 32, 32] });
    expect(m.layers[1].shape).toBe("8×32×32");
    expect(sum(m.layers.map((l) => l.macs))).toBe(MACS);
  });

  it("reads weight statistics from the .bin", () => {
    const m = parseNcnn(fx("tinynet_shaped.ncnn.param"), "tinynet.param", { bin: fx("tinynet.ncnn.bin") });
    const w = m.layers[1].w!;
    // Independent read of the first blob: 4-byte fp32 tag (0) + 216 floats (BN folded by pnnx).
    const bin = fx("tinynet.ncnn.bin");
    const dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
    expect(dv.getUint32(0, true)).toBe(0);
    const vals = Array.from({ length: 216 }, (_, i) => dv.getFloat32(4 + i * 4, true));
    const mean = sum(vals) / vals.length;
    expect(w.size).toBe(216);
    expect(w.min).toBeCloseTo(Math.min(...vals), 3);
    expect(w.max).toBeCloseTo(Math.max(...vals), 3);
    expect(w.mean).toBeCloseTo(mean, 3);
    expect(w.std).toBeCloseTo(Math.sqrt(sum(vals.map((v) => (v - mean) ** 2)) / vals.length), 3);
    expect(m.layers[10].w?.size).toBe(160);
    expect(m.layers[10].w?.mean).toBeDefined();
  });

  it("parses the binary .param.bin variant identically", () => {
    const m = parseNcnn(fx("tinynet_shaped.ncnn.param.bin"), "tinynet.param.bin");
    expect(m.format).toBe("ncnn (.param.bin)");
    expect(m.layers.map((l) => l.type)[10]).toBe("InnerProduct");
    expect(m.edges).toHaveLength(12);
    expect(m.outputShape).toEqual([10]);
    expect(sum(m.layers.map((l) => l.params))).toBe(PARAMS);
    expect(sum(m.layers.map((l) => l.macs))).toBe(MACS);
  });

  it("accepts text input and rejects non-ncnn data", () => {
    const m = parseNcnn("7767517\n2 2\nInput data 0 1 data 0=4 1=4 2=1\nReLU relu 1 1 data out\n", "t.param");
    expect(m.layers).toHaveLength(2);
    expect(m.layers[1].shape).toBe("1×4×4");
    expect(m.layers[1].flops).toBe(16);
    expect(() => parseNcnn("hello", "x.param")).toThrow();
  });
});
