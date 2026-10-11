import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { decodeTensorflow, isTensorflowGraph, parseTensorflow } from "../src/formats/tensorflow";
import { readTensorBundle, snappyDecompress } from "../src/common/tensorBundle";
import { parseModel } from "../src/index";

// Fixtures: test/fixtures/scripts/gen_tensorflow_keras.py (TF 2.21.0).
//   tiny_frozen.pb/.pbtxt  convert_variables_to_constants_v2 of a Keras functional model
//   saved_model.pb         tf.saved_model.save of a tf.Module (conv → mean → dense)
const dir = fileURLToPath(new URL("./fixtures/tensorflow/", import.meta.url));
const read = (f: string) => new Uint8Array(readFileSync(dir + f));
const expected = JSON.parse(readFileSync(dir + "expected.json", "utf8"));
const total = (m: { layers: { params: number }[] }) => m.layers.reduce((a, l) => a + l.params, 0);

describe("TensorFlow frozen GraphDef (.pb)", () => {
  const buf = read("tiny_frozen.pb");
  const m = parseTensorflow(buf, "tiny_frozen.pb");
  const byName = (suffix: string) => m.layers.find((l) => l.name.endsWith(suffix))!;

  it("classifies the protobuf as a GraphDef", () => {
    expect(decodeTensorflow(buf).kind).toBe("GraphDef");
    expect(isTensorflowGraph(buf)).toBe(true);
    expect(m.format).toBe("TensorFlow GraphDef");
  });

  it("folds Const / ReadVariableOp / bias-reshape nodes and hides NoOp + Identity", () => {
    const ops = m.layers.map((l) => l.op);
    expect(ops).not.toContain("Const");
    expect(ops).not.toContain("NoOp");
    expect(ops).not.toContain("Identity");
    expect(ops).not.toContain("Squeeze"); // Keras 3 bias Reshape→Squeeze is weight-only
    expect(m.layers).toHaveLength(30);
    expect(ops.filter((o) => o === "Conv2D")).toHaveLength(4);
    expect(ops.filter((o) => o === "DepthwiseConv2dNative")).toHaveLength(2);
    expect(m.layers[0].op).toBe("Placeholder");
    expect(m.layers[0].group).toBe("input");
    expect(m.layers[m.layers.length - 1].op).toBe("Softmax");
    expect(m.layers[m.layers.length - 1].group).toBe("output");
  });

  it("params equal Keras count_params (float consts of rank ≥ 1)", () => {
    expect(total(m)).toBe(expected.tiny_frozen.params);
    expect(byName("stem_conv_1/convolution").params).toBe(3 * 3 * 3 * 8);
    expect(byName("stem_conv_1/BiasAdd").params).toBe(8);
    expect(byName("logits_1/MatMul").params).toBe(12 * 10);
  });

  it("propagates NHWC shapes, incl. dynamic-shape plumbing (UpSampling Tile/Reshape)", () => {
    expect(m.inputShape).toEqual([1, 32, 32, 3]);
    expect(m.outputShape).toEqual(expected.tiny_frozen.output_shape);
    expect(byName("pool_1/MaxPool2d").shape).toBe("1×16×16×8");
    expect(byName("strided_1/convolution").shape).toBe("1×7×7×4");
    expect(byName("up_1/Repeat_1/Reshape_1").shape).toBe("1×14×14×4");
    expect(byName("zpad_1/Pad").shape).toBe("1×16×16×4");
    expect(byName("concat_1/concat").shape).toBe("1×16×16×12");
    expect(byName("gap_1/Mean").shape).toBe("1×12");
    expect(byName("gap_1/Mean").group).toBe("neck"); // recognised as a global average pool
  });

  it("wires data-flow edges including the residual and concat branches", () => {
    const id = (s: string) => byName(s).id;
    const has = (a: string, b: string) => m.edges.some(([x, y]) => x === id(a) && y === id(b));
    expect(has("stem_relu_1/Relu", "residual_1/Add")).toBe(true);
    expect(has("pw_1/Relu", "residual_1/Add")).toBe(true);
    expect(has("sep_1/BiasAdd", "concat_1/concat")).toBe(true);
    expect(has("zpad_1/Pad", "concat_1/concat")).toBe(true);
    expect(m.edges).toHaveLength(31);
  });

  it("estimates MACs and reads weight statistics from tensor_content", () => {
    expect(byName("stem_conv_1/convolution").macs).toBe(32 * 32 * 8 * 27);
    expect(byName("dw_1/depthwise").macs).toBe(32 * 32 * 8 * 9);
    expect(byName("logits_1/MatMul").macs).toBe(120);
    const w = byName("stem_conv_1/convolution").w!;
    expect(w.shape).toEqual([3, 3, 3, 8]);
    expect(w.size).toBe(216);
    expect(typeof w.std).toBe("number");
    expect(w.std!).toBeGreaterThan(0);
    expect(byName("stem_conv_1/convolution").attr.padding).toBe("SAME");
    expect(byName("stem_conv_1/convolution").attr.strides).toEqual([1, 1, 1, 1]);
  });

  it("parses the text-format GraphDef (.pbtxt) to the same graph", () => {
    const t = parseTensorflow(read("tiny_frozen.pbtxt"), "tiny_frozen.pbtxt");
    expect(t.format).toBe("TensorFlow GraphDef (text)");
    expect(t.layers.map((l) => [l.op, l.shape, l.params, l.macs])).toEqual(m.layers.map((l) => [l.op, l.shape, l.params, l.macs]));
    expect(t.edges).toEqual(m.edges);
    expect(t.layers[1].w?.mean).toBe(m.layers[1].w?.mean);
  });
});

describe("TensorFlow SavedModel (saved_model.pb)", () => {
  const buf = read("saved_model.pb");
  const m = parseTensorflow(buf, "saved_model.pb");

  it("detects SavedModel and follows the serving signature into library functions", () => {
    expect(decodeTensorflow(buf).kind).toBe("SavedModel");
    expect(m.format).toBe("TensorFlow SavedModel");
    expect(m.producer).toBe("TensorFlow 2.21.0");
    expect(m.doc).toContain("serving_default");
    expect(m.layers.map((l) => l.op)).toEqual(["Placeholder", "Conv2D", "BiasAdd", "Relu", "Mean", "MatMul", "Softmax"]);
    expect(m.edges).toEqual([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]);
  });

  it("resolves captured variables (VarHandleOp shapes) as params", () => {
    expect(total(m)).toBe(expected.saved_model.params);
    expect(m.layers[1].params).toBe(108);
    expect(m.layers[1].w?.shape).toEqual([3, 3, 3, 4]);
  });

  it("propagates shapes from the signature placeholder", () => {
    expect(m.inputShape).toEqual([0, 16, 16, 3]);
    expect(m.outputShape).toEqual([0, 5]);
    expect(m.layers[1].shape).toBe("?×16×16×4");
    expect(m.layers[1].macs).toBe(16 * 16 * 4 * 27);
  });

  it("rejects non-TF bytes", () => {
    expect(isTensorflowGraph(new TextEncoder().encode("hello world, not a graph"))).toBe(false);
  });
});

describe("TensorFlow robustness", () => {
  it("handles very deep graphs in any file order without recursion overflow", () => {
    const N = 20000;
    const nodes = ['node { name: "x" op: "Placeholder" attr { key: "shape" value { shape { dim { size: 1 } dim { size: 4 } } } } }'];
    for (let i = 1; i <= N; i++) nodes.push(`node { name: "r${i}" op: "Relu" input: "${i === 1 ? "x" : `r${i - 1}`}" }`);
    const m = parseTensorflow(new TextEncoder().encode(nodes.reverse().join("\n")), "deep.pbtxt");
    expect(m.layers).toHaveLength(N + 1);
    expect(m.layers[0].op).toBe("Placeholder"); // topologically sorted despite reversed file order
    expect(m.layers[N].shape).toBe("1×4");
    expect(m.edges).toHaveLength(N);
    expect(m.layers[N].group).toBe("output");
  });
});

describe("SavedModel variables/ checkpoint", () => {
  // Generated with TF 2.21 (tf.Module, tf.saved_model.save); weights.expected.json
  // holds TensorFlow's own min/max/mean/std for each variable.
  const d = fileURLToPath(new URL("./fixtures/tensorflow/savedmodel_vars/", import.meta.url));
  const r = (f: string) => new Uint8Array(readFileSync(d + f));
  const truth = JSON.parse(readFileSync(d + "weights.expected.json", "utf8")) as Record<string, { shape: number[]; min: number; max: number; mean: number; std: number }>;
  const vars = {
    "variables/variables.index": r("variables/variables.index"),
    "variables/variables.data-00000-of-00001": r("variables/variables.data-00000-of-00001"),
  };

  it("reads every variable's values through the object graph", () => {
    const m = parseModel(r("saved_model.pb"), "saved_model.pb", vars);
    const byShape = (s: number[]) => m.layers.find((l) => JSON.stringify(l.w?.shape) === JSON.stringify(s))!;
    for (const t of Object.values(truth)) {
      const w = byShape(t.shape).w!;
      expect(w.mean).toBeCloseTo(t.mean, 3);
      expect(w.std).toBeCloseTo(t.std, 3);
      expect(w.min).toBeCloseTo(t.min, 3);
      expect(w.max).toBeCloseTo(t.max, 3);
    }
    expect(m.notes).toBeUndefined();
  });

  it("reads the table directly", () => {
    const b = readTensorBundle(vars["variables/variables.index"], [vars["variables/variables.data-00000-of-00001"]])!;
    expect([...b.keys()].sort()).toEqual(["b/.ATTRIBUTES/VARIABLE_VALUE", "k/.ATTRIBUTES/VARIABLE_VALUE", "w/.ATTRIBUTES/VARIABLE_VALUE"]);
    expect(b.get("k/.ATTRIBUTES/VARIABLE_VALUE")!.shape).toEqual([3, 3, 3, 4]);
    expect(readTensorBundle(new Uint8Array(64), [])).toBeNull();
  });

  it("notes the missing variables/ folder", () => {
    expect(parseModel(r("saved_model.pb"), "saved_model.pb").notes?.[0]).toMatch(/variables\/variables\.index/);
  });
});

describe("snappy", () => {
  it("decodes literals and overlapping copies", () => {
    // "abcabcabcabcX": literal "abc", copy(len 9, off 3), literal "X".
    const src = new Uint8Array([13, 0b1000, 97, 98, 99, (9 - 1) << 2 | 2, 3, 0, 0b0000, 88]);
    expect(new TextDecoder().decode(snappyDecompress(src))).toBe("abcabcabcabcX");
  });
});
