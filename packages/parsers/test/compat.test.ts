import { describe, expect, it } from "vitest";
import { annotateCompat, layerCompat, normOp, parseOnnx, type ModelLayer } from "../src/index";
import { modelProtoType } from "../src/onnx/proto";

const layer = (op: string, over: Partial<ModelLayer> = {}): ModelLayer => ({
  id: 0, name: op.toLowerCase(), type: op, op, shape: "1×64×80×80", dt: "float32",
  params: 0, flops: 0, macs: 0, mem: 0, group: "backbone",
  ins: [{ n: "x" }], outs: [{ n: "y", s: [1, 64, 80, 80] }], attr: {}, w: null,
  math: "", insight: "", qSens: 0, compIssues: [], ...over,
});
const targets = (l: ModelLayer) => layerCompat(l).map((c) => `${c.target}:${c.severity}`);

describe("normOp", () => {
  it("folds format spellings to one name", () => {
    expect(normOp("stablehlo.dot_general")).toBe("dotgeneral");
    expect(normOp("NonMaxSuppressionV5")).toBe("nonmaxsuppressionv5");
    expect(normOp("RealDiv")).toBe("realdiv");
  });
});

describe("layerCompat", () => {
  it("passes the ops every NPU here supports", () => {
    for (const op of ["Conv", "Relu", "Add", "MaxPool", "Concat", "Sigmoid", "Mul", "Conv2D", "Convolution"]) {
      expect(layerCompat(layer(op))).toEqual([]);
    }
  });

  it("flags ops missing from the Edge TPU list, in any format's spelling", () => {
    for (const op of ["Div", "RealDiv", "Erf", "Gather", "LayerNormalization", "Sqrt"]) {
      expect(targets(layer(op))).toEqual(["coral_tpu:error"]);
    }
  });

  it("flags data-dependent ops for every NPU target", () => {
    expect(targets(layer("NonMaxSuppression"))).toEqual(["coral_tpu:error", "rk3588:warn", "hailo:error", "kneron_kl720:error"]);
  });

  it("applies the Edge TPU Softmax limit (1-D, at most 16,000 elements)", () => {
    expect(layerCompat(layer("Softmax", { outs: [{ n: "y", s: [1, 1000] }] }))).toEqual([]);
    expect(targets(layer("Softmax", { outs: [{ n: "y", s: [1, 16, 4, 8400] }] }))).toEqual(["coral_tpu:error"]);
  });

  it("applies the Edge TPU rule of at most 3 non-unit dims", () => {
    expect(layerCompat(layer("Transpose", { outs: [{ n: "y", s: [1, 4, 16, 8400] }] }))).toEqual([]);
    expect(targets(layer("Transpose", { outs: [{ n: "y", s: [197, 1, 12, 64] }] }))).toEqual(["coral_tpu:error"]);
  });

  it("separates activation × activation matmul from a fully-connected layer", () => {
    expect(targets(layer("MatMul", { ins: [{ n: "q" }, { n: "k" }] }))).toEqual(["coral_tpu:error"]);
    expect(layerCompat(layer("MatMul", { params: 4096 }))).toEqual([]);
  });

  it("ignores shape arithmetic that converters fold away", () => {
    expect(layerCompat(layer("Gather", { constFolded: true }))).toEqual([]);
    expect(layerCompat(layer("Cast", { dt: "int64" }))).toEqual([]);
  });

  it("flags dynamic input dims beyond the batch", () => {
    const input = (shape: string) => layer("Input", { group: "input", shape, outs: [] });
    expect(layerCompat(input("?×3×640×640"))).toEqual([]);
    expect(targets(input("1×3×?×?"))).toEqual(["coral_tpu:error", "rk3588:warn", "hailo:error", "kneron_kl720:error"]);
    expect(targets(layer("Input", { group: "input", attr: { declaredShape: "?×3×?×?" } }))).toHaveLength(4);
  });

  it("does not report anything for op names it does not recognize", () => {
    expect(layerCompat(layer("MyCustomPlugin"))).toEqual([]);
  });
});

describe("annotateCompat", () => {
  it("leaves curated issues alone and fills empty layers", () => {
    const curated = layer("Div", { compIssues: [{ target: "x", severity: "warn", msg: "curated" }] });
    const m = annotateCompat({ layers: [curated, layer("Div")] });
    expect(m.layers[0].compIssues).toEqual([{ target: "x", severity: "warn", msg: "curated" }]);
    expect(m.layers[1].compIssues).toHaveLength(1);
  });

  it("does not flag an ONNX model's shape subgraph", () => {
    // X → Shape → Gather → Reshape(X, ·) → Div: only the Div is real compute.
    const M = modelProtoType();
    const vi = (name: string, dims: number[]) => ({ name, type: { tensor_type: { elem_type: 1, shape: { dim: dims.map((d) => ({ dim_value: d })) } } } });
    const bytes = M.encode(M.create({
      ir_version: 8, opset_import: [{ domain: "", version: 17 }],
      graph: {
        name: "g", input: [vi("X", [1, 8, 4, 4])], output: [{ name: "Y" }],
        initializer: [{ name: "idx", data_type: 7, dims: [2], int64_data: [0, 1] }, { name: "two", data_type: 1, dims: [], float_data: [2] }],
        node: [
          { op_type: "Shape", input: ["X"], output: ["s"] },
          { op_type: "Gather", input: ["s", "idx"], output: ["g"] },
          { op_type: "Reshape", input: ["X", "g"], output: ["r"] },
          { op_type: "Div", input: ["r", "two"], output: ["Y"] },
        ],
      },
    })).finish();
    const flagged = annotateCompat(parseOnnx(bytes)).layers.filter((l) => l.compIssues.length).map((l) => l.op);
    expect(flagged).toEqual(["Div"]);
  });
});

describe("weight shapes are not activations", () => {
  it("skips shape rules for checkpoint layers whose shape is the weight's", () => {
    // PyTorch state-dict parsing: a Conv2d module with only its weight shape known.
    const conv = layer("Conv2d", { shape: "16×3×3×3", outs: [], w: { shape: [16, 3, 3, 3], size: 432 } });
    expect(layerCompat(conv)).toEqual([]);
    const tensor = layer("Weight", { outs: [{ n: "w", s: [64, 32, 3, 3] }], w: { shape: [64, 32, 3, 3] } });
    expect(layerCompat(tensor)).toEqual([]);
  });
});
