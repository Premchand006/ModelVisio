import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { isCoremlModel, isCoremlPackageZip, parseCoreml } from "../src/formats/coreml";

// Fixtures: TinyNet (fixtures/scripts/tinynet.py) converted with coremltools 9.0
// by fixtures/scripts/make_coreml.py: NeuralNetwork, NeuralNetworkClassifier,
// ML Program spec, and a zipped .mlpackage (deflated) with weights/weight.bin.
const dir = join(__dirname, "fixtures", "coreml");
const read = (f: string) => new Uint8Array(readFileSync(join(dir, f)));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const TOTAL_PARAMS = 1122; // BN folded into conv1 by the converter

describe("coreml NeuralNetwork", () => {
  const buf = read("tinynet_nn.mlmodel");

  it("builds the layer graph with shapes, params and costs", () => {
    expect(isCoremlModel(buf)).toBe(true);
    const m = parseCoreml(buf, "tinynet_nn.mlmodel");
    expect(m.format).toBe("Core ML");
    expect(m.framework).toBe("Core ML (NeuralNetwork)");
    expect(m.opset).toBe(4); // specificationVersion
    expect(m.producer).toMatch(/^coremltools 9\.0/);
    expect(m.layers.map((l) => l.type)).toEqual([
      "Input", "Convolution", "ReLU", "Pooling", "Convolution", "ReLU", "Add", "Convolution", "ReLU",
      "Pooling", "ReshapeStatic", "InnerProduct", "SoftmaxND",
    ]);
    expect(m.edges).toHaveLength(13);
    expect(sum(m.layers.map((l) => l.params))).toBe(TOTAL_PARAMS);
    expect(m.inputShape).toEqual([1, 3, 32, 32]);
    expect(m.outputShape).toEqual([1, 10]);

    const conv1 = m.layers[1];
    expect(conv1.shape).toBe("1×8×32×32");
    expect(conv1.params).toBe(224);
    expect(conv1.macs).toBe(8 * 32 * 32 * 27);
    expect(conv1.w?.shape).toEqual([8, 3, 3, 3]);
    expect(conv1.w?.std).toBeGreaterThan(0);
    expect(m.layers[3].shape).toBe("1×8×16×16");
    expect(m.layers[7].shape).toBe("1×16×8×8");
    expect(m.layers[9].shape).toBe("1×16×1×1");
    const ip = m.layers[11];
    expect(ip.shape).toBe("1×10");
    expect(ip.macs).toBe(160);
    expect(ip.group).toBe("head");
    expect(m.layers[0].group).toBe("input");
    expect(m.layers[12].group).toBe("output");
    // Residual: pool → add as well as relu → add.
    expect(m.edges).toContainEqual([3, 6]);
    expect(m.edges).toContainEqual([5, 6]);
  });

  it("adds the classifier head for NeuralNetworkClassifier", () => {
    const m = parseCoreml(read("tinynet_classifier.mlmodel"), "tinynet_classifier.mlmodel");
    expect(m.layers).toHaveLength(14);
    const clf = m.layers[13];
    expect(clf.type).toBe("Classifier");
    expect(clf.attr.classes).toBe(10);
    expect(clf.outs[0].n).toBe("classLabel");
    expect(clf.group).toBe("output");
    expect(m.edges).toContainEqual([12, 13]);
  });
});

describe("coreml ML Program", () => {
  it("parses MIL ops, folds const ops, reads explicit shapes", () => {
    const m = parseCoreml(read("tinynet_mlprogram.mlmodel"), "tinynet_mlprogram.mlmodel");
    expect(m.framework).toBe("Core ML (ML Program)");
    expect(m.opset).toBe(6);
    expect(m.layers.map((l) => l.type)).toEqual([
      "Input", "cast", "conv", "relu", "max_pool", "conv", "relu", "add", "conv", "relu",
      "reduce_mean", "reshape", "linear", "softmax", "cast",
    ]);
    expect(m.layers.some((l) => l.type === "const")).toBe(false);
    expect(m.edges).toHaveLength(15);
    expect(sum(m.layers.map((l) => l.params))).toBe(TOTAL_PARAMS);
    const conv1 = m.layers[2];
    expect(conv1.shape).toBe("1×8×32×32");
    expect(conv1.dt).toBe("float16");
    expect(conv1.params).toBe(224);
    expect(conv1.macs).toBe(8 * 32 * 32 * 27);
    expect(conv1.attr.strides).toEqual([1, 1]);
    expect(conv1.attr.pad_type).toBe("custom");
    expect(conv1.w?.shape).toEqual([8, 3, 3, 3]);
    expect(conv1.w?.std).toBeUndefined(); // weights live in weight.bin, not provided
    expect(m.layers[4].macs).toBe(0);
    expect(m.layers[4].flops).toBe(8 * 16 * 16 * 4);
    expect(m.layers[10].shape).toBe("1×16×1×1");
    expect(m.layers[12].macs).toBe(160);
    expect(m.layers[14].group).toBe("output");
    expect(m.outputShape).toEqual([1, 10]);
  });

  it("reads a zipped .mlpackage including weight.bin statistics", () => {
    const zip = read("tinynet.mlpackage.zip");
    expect(isCoremlPackageZip(zip)).toBe(true);
    expect(isCoremlPackageZip(read("tinynet_nn.mlmodel"))).toBe(false);
    const m = parseCoreml(zip, "tinynet.mlpackage.zip");
    expect(m.format).toBe("Core ML package");
    expect(m.layers).toHaveLength(15);
    expect(sum(m.layers.map((l) => l.params))).toBe(TOTAL_PARAMS);
    const conv1 = m.layers[2];
    expect(conv1.w?.std).toBeGreaterThan(0);
    expect(conv1.w?.min).toBeLessThan(0);
    const fc = m.layers[12];
    expect(fc.w?.shape).toEqual([10, 16]);
    expect(fc.w?.std).toBeGreaterThan(0);
  });

  it("falls back to metadata for non-Core ML bytes", () => {
    const m = parseCoreml(new Uint8Array([0xff, 0xfe, 0x00, 0x01, 0x02]), "junk.mlmodel");
    expect(m.layers).toHaveLength(1);
    expect(m.layers[0].op).toBe("Detected");
  });
});
