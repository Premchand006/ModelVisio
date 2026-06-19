import { describe, expect, it } from "vitest";
import { modelProtoType } from "../src/onnx/proto";
import { parseOnnx } from "../src/index";

// Build a small but real ONNX model by encoding through the same protobuf
// schema the parser decodes. This exercises the genuine wire format end-to-end:
//   X[1,3,8,8] → Conv(4×3×3×3) → Relu → GlobalAveragePool → Flatten → Gemm(10×4) → out[1,10]
function buildOnnxBytes(): Uint8Array {
  const ModelProto = modelProtoType();

  const tensor = (name: string, dims: number[]) => ({
    name,
    data_type: 1, // FLOAT
    dims,
    float_data: new Array(dims.reduce((a, b) => a * b, 1)).fill(0.1),
  });

  const valueInfo = (name: string, dims: number[]) => ({
    name,
    type: { tensor_type: { elem_type: 1, shape: { dim: dims.map((d) => ({ dim_value: d })) } } },
  });

  const obj = {
    ir_version: 8,
    producer_name: "test-suite",
    opset_import: [{ domain: "", version: 17 }],
    graph: {
      name: "tiny",
      input: [valueInfo("X", [1, 3, 8, 8])],
      output: [valueInfo("out", [1, 10])],
      initializer: [
        tensor("convW", [4, 3, 3, 3]),
        tensor("fcW", [10, 4]),
        tensor("fcB", [10]),
      ],
      node: [
        {
          op_type: "Conv",
          name: "conv1",
          input: ["X", "convW"],
          output: ["c1"],
          attribute: [
            { name: "kernel_shape", type: 7, ints: [3, 3] },
            { name: "strides", type: 7, ints: [1, 1] },
            { name: "pads", type: 7, ints: [1, 1, 1, 1] },
            { name: "group", type: 2, i: 1 },
          ],
        },
        { op_type: "Relu", name: "relu1", input: ["c1"], output: ["r1"] },
        { op_type: "GlobalAveragePool", name: "gap", input: ["r1"], output: ["g1"] },
        {
          op_type: "Flatten",
          name: "flatten",
          input: ["g1"],
          output: ["f1"],
          attribute: [{ name: "axis", type: 2, i: 1 }],
        },
        {
          op_type: "Gemm",
          name: "fc",
          input: ["f1", "fcW", "fcB"],
          output: ["out"],
          attribute: [{ name: "transB", type: 2, i: 1 }],
        },
      ],
    },
  };

  return ModelProto.encode(ModelProto.create(obj)).finish();
}

describe("parseOnnx", () => {
  const model = parseOnnx(buildOnnxBytes(), { name: "tiny.onnx" });

  it("reads model-level metadata", () => {
    expect(model.format).toBe("ONNX");
    expect(model.opset).toBe(17);
    expect(model.framework).toBe("test-suite");
    expect(model.inputShape).toEqual([1, 3, 8, 8]);
    expect(model.outputShape).toEqual([1, 10]);
  });

  it("creates one layer per input + node", () => {
    // 1 synthetic input + 5 nodes
    expect(model.layers).toHaveLength(6);
    expect(model.layers[0].op).toBe("Input");
    expect(model.layers.map((l) => l.op)).toEqual([
      "Input",
      "Conv",
      "Relu",
      "GlobalAveragePool",
      "Flatten",
      "Gemm",
    ]);
  });

  it("infers the conv output shape and FLOPs", () => {
    const conv = model.layers.find((l) => l.op === "Conv")!;
    expect(conv.shape).toBe("1×4×8×8");
    expect(conv.params).toBe(4 * 3 * 3 * 3); // 108, no bias initializer
    // macs = outN(256) * (Cin/group = 3) * kArea(9) = 6912
    expect(conv.macs).toBe(6912);
    expect(conv.flops).toBe(13824);
    expect(conv.w?.shape).toEqual([4, 3, 3, 3]);
    expect(conv.w?.size).toBe(108);
  });

  it("infers the gemm head", () => {
    const fc = model.layers.find((l) => l.op === "Gemm")!;
    expect(fc.shape).toBe("1×10");
    expect(fc.params).toBe(40 + 10); // weight + bias
    expect(fc.macs).toBe(40); // M=1, N=10, K=4
    expect(fc.group).toBe("output"); // produces the graph output
  });

  it("builds data-flow edges excluding weights", () => {
    // X→conv, conv→relu, relu→gap, gap→flatten, flatten→gemm
    expect(model.edges).toHaveLength(5);
  });
});
