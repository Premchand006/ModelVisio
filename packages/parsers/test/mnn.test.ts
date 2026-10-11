import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isMnn, parseMnn } from "../src/formats/mnn";

// Fixture: TinyNet → ONNX → mnnconvert 3.6.1 (see fixtures/scripts/make_mnn.py).
const fx = (f: string) => readFileSync(fileURLToPath(new URL(`./fixtures/${f}`, import.meta.url)));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

describe("MNN (.mnn FlatBuffer, TinyNet)", () => {
  const buf = fx("mnn/tinynet.mnn");
  const m = parseMnn(buf, "tinynet.mnn");

  it("detects MNN structurally and rejects other FlatBuffers/data", () => {
    expect(isMnn(buf)).toBe(true);
    expect(isMnn(fx("ncnn/tinynet.ncnn.bin"))).toBe(false);
    expect(isMnn(new Uint8Array(64))).toBe(false);
  });

  it("reads net metadata", () => {
    expect(m.format).toBe("MNN");
    expect(m.producer).toContain("source ONNX");
    expect(m.producer).toContain("MNN 3.6.1");
    expect(m.sizeBytes).toBe(buf.byteLength);
  });

  it("builds operators and tensor-index edges (Const ops folded)", () => {
    // 26 ops in the file, 5 of them Const → 21 graph layers.
    expect(m.layers).toHaveLength(21);
    expect(m.layers.some((l) => l.type === "Const")).toBe(false);
    const types = m.layers.map((l) => l.type);
    expect(types.slice(0, 11)).toEqual([
      "Input", "ConvertTensor", "Convolution", "Pooling", "Convolution", "BinaryOp",
      "Convolution", "Pooling3D", "Flatten", "Reshape", "Convolution",
    ]);
    expect(types[types.length - 1]).toBe("Softmax");
    // Residual add consumes conv2 (id 4) and the pool output (id 3).
    expect(m.edges).toContainEqual([4, 5]);
    expect(m.edges).toContainEqual([3, 5]);
    expect(m.edges).toHaveLength(25);
  });

  it("propagates shapes, including the folded Shape→Slice→Concat→Reshape chain", () => {
    expect(m.inputShape).toEqual([1, 3, 32, 32]);
    expect(m.outputShape).toEqual([1, 10]);
    const byName = (n: string) => m.layers.find((l) => l.name === n)!;
    expect(byName("/conv1/Conv_output_0").shape).toBe("1×8×32×32");
    expect(byName("/pool/MaxPool_output_0").shape).toBe("1×8×16×16");
    expect(byName("/conv3/Conv_output_0").shape).toBe("1×16×8×8");
    expect(byName("/GlobalAveragePool_output_0").shape).toBe("1×16×1×1");
    expect(byName("Concat32").attr.value).toEqual([1, 10]);
    expect(byName("probs").shape).toBe("1×10");
    expect(byName("probs").group).toBe("output");
  });

  it("counts params and MACs from Convolution2D weights", () => {
    expect(sum(m.layers.map((l) => l.params))).toBe(1122);
    expect(sum(m.layers.map((l) => l.macs))).toBe(221184 + 147456 + 8192 + 160);
    const conv1 = m.layers[2];
    expect(conv1.attr.kernel).toEqual([3, 3]);
    expect(conv1.attr.relu).toBe(true);
    expect(conv1.w?.shape).toEqual([8, 3, 3, 3]);
    expect(conv1.w?.size).toBe(216);
    expect(conv1.w?.std).toBeGreaterThan(0.05);
  });

  it("throws on non-MNN input", () => {
    expect(() => parseMnn(new Uint8Array(32), "x.mnn")).toThrow();
  });
});
