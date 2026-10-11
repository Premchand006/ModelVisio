import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isMlir, parseMlir, parseMlirType } from "../src/formats/mlir";

// Fixtures: a tiny JAX CNN lowered to StableHLO (fixtures/scripts/make_mlir.py),
// in both pretty and generic printing.
const fx = (f: string) => readFileSync(fileURLToPath(new URL(`./fixtures/mlir/${f}`, import.meta.url)));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

describe("MLIR StableHLO (JAX tiny CNN)", () => {
  for (const file of ["tiny_cnn.stablehlo.mlir", "tiny_cnn.generic.mlir"]) {
    it(`builds the @main graph from ${file}`, () => {
      const buf = fx(file);
      expect(isMlir(new TextDecoder().decode(buf))).toBe(true);
      const m = parseMlir(buf, file);
      expect(m.format).toBe("MLIR (StableHLO)");
      expect(m.doc).toContain("@relu");
      // Input + 9 ops (4 constants folded into their consumers).
      expect(m.layers).toHaveLength(10);
      expect(m.layers.map((l) => l.type)).toEqual([
        "Input", "stablehlo.convolution", "stablehlo.broadcast_in_dim", "stablehlo.broadcast_in_dim", "stablehlo.add",
        "func.call", "stablehlo.reshape", "stablehlo.dot_general", "stablehlo.broadcast_in_dim", "stablehlo.add",
      ]);
      expect(m.edges).toEqual([[0, 1], [2, 3], [1, 4], [3, 4], [4, 5], [5, 6], [6, 7], [7, 9], [8, 9]]);
      expect(m.inputShape).toEqual([1, 3, 8, 8]);
      expect(m.outputShape).toEqual([1, 10]);
      expect(m.layers[1].shape).toBe("1×4×8×8");
      expect(m.layers[6].shape).toBe("1×256");
      expect(m.layers[9].group).toBe("output");
      expect(m.layers[5].attr.callee).toBe("relu");

      // W1 4·3·3·3 + B1 4 + W2 256·10 + B2 10; the relu zero splat is not a parameter.
      expect(sum(m.layers.map((l) => l.params))).toBe(108 + 4 + 2560 + 10);
      expect(m.layers[1].macs).toBe(4 * 8 * 8 * 27); // conv: out · (Cin·kh·kw)
      expect(m.layers[7].macs).toBe(10 * 256); // dot_general contracting dim 256
      // Weight stats decoded from the hex dense attribute.
      const w = m.layers[1].w!;
      expect(w.shape).toEqual([4, 3, 3, 3]);
      expect(w.size).toBe(108);
      expect(w.std).toBeGreaterThan(0.05);
      expect(w.min).toBeLessThan(0);
      expect(m.layers[2].w?.min).toBeCloseTo(-0.225, 3); // B1 from dense<[…]>
    });
  }
});

const TOSA = `// hand-written TOSA/linalg sample: regions, multi-result ops, resources
#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {tf.versions = {producer = 1 : i32}} {
  func.func private @helper(%a: tensor<4xf32>) -> tensor<4xf32> {
    return %a : tensor<4xf32>
  }
  func.func @main(%arg0: tensor<1x8x8x3xf32>) -> (tensor<1x10xf32>, tensor<1xi32>) {
    %w = "tosa.const"() <{values = dense_resource<conv_w> : tensor<4x3x3x3xf32>}> : () -> tensor<4x3x3x3xf32>
    %b = "tosa.const"() <{values = dense<[0.1, -0.2, 0.3, 0.0]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %0 = tosa.conv2d %arg0, %w, %b {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x3xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x8x8x4xf32>
    %1 = tosa.clamp %0 {max_val = 6.0 : f32, min_val = 0.0 : f32} : (tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32> // relu6
    %2 = tosa.max_pool2d %1 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x8x8x4xf32>) -> tensor<1x4x4x4xf32>
    %3:2 = "test.split"(%2) : (tensor<1x4x4x4xf32>) -> (tensor<1x4x4x2xf32>, tensor<1x4x4x2xf32>)
    %4 = tosa.add %3#0, %3#1 : (tensor<1x4x4x2xf32>, tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32>
    %5 = tosa.reshape %4 {new_shape = array<i64: 1, 32>} : (tensor<1x4x4x2xf32>) -> tensor<1x32xf32>
    %fcw = arith.constant dense_resource<fc_w> : tensor<32x10xf32>
    %init = tensor.empty() : tensor<1x10xf32>
    %6 = linalg.matmul ins(%5, %fcw : tensor<1x32xf32>, tensor<32x10xf32>)
                       outs(%init : tensor<1x10xf32>) -> tensor<1x10xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1x10xf32>) outs(%init : tensor<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %e = math.exp %in : f32
      linalg.yield %e : f32
    } -> tensor<1x10xf32>
    %8 = "tosa.argmax"(%7) <{axis = 1 : i32}> : (tensor<1x10xf32>) -> tensor<1xi32>
    return %7, %8 : tensor<1x10xf32>, tensor<1xi32>
  }
}
{-#
  dialect_resources: {
    builtin: {
      conv_w: "0x04000000AABBCCDD",
      fc_w: "0x0400000011223344"
    }
  }
#-}
`;

describe("MLIR TOSA / linalg (hand-written)", () => {
  const m = parseMlir(TOSA, "sample.mlir");

  it("selects @main, folds constants and collapses region bodies", () => {
    expect(m.format).toBe("MLIR (TOSA)");
    expect(m.layers.map((l) => l.type)).toEqual([
      "Input", "tosa.conv2d", "tosa.clamp", "tosa.max_pool2d", "test.split", "tosa.add",
      "tosa.reshape", "linalg.matmul", "linalg.generic", "tosa.argmax",
    ]);
    expect(m.edges).toHaveLength(9);
    expect(m.edges).toContainEqual([4, 5]); // %3#0 / %3#1 both from the split
    expect(m.layers[4].outs).toHaveLength(2);
    expect(m.layers[8].shape).toBe("1×10"); // type from the region-closing line
    expect(m.layers[8].group).toBe("output");
    expect(m.layers[9].group).toBe("output");
    expect(m.outputShape).toEqual([1, 10]);
    expect(m.doc).toContain("@helper");
  });

  it("counts params (dense_resource + dense lists) and NHWC conv / matmul / pool cost", () => {
    expect(m.layers[1].params).toBe(108 + 4);
    expect(m.layers[7].params).toBe(320);
    expect(m.layers[1].macs).toBe(8 * 8 * 4 * 27);
    expect(m.layers[1].attr.stride).toBe("array<i64: 1, 1>");
    expect(m.layers[3].flops).toBe(4 * 4 * 4 * 4); // out · kernel area
    expect(m.layers[7].macs).toBe(10 * 32);
    expect(m.layers[5].flops).toBe(32);
  });

  it("parses types", () => {
    expect(parseMlirType("tensor<1x?x224xf16>")).toEqual({ shape: [1, -1, 224], dt: "float16" });
    expect(parseMlirType("tensor<f32>")).toEqual({ shape: [], dt: "float32" });
    expect(parseMlirType("!torch.vtensor<[1,3,224,224],f32>")).toEqual({ shape: [1, 3, 224, 224], dt: "float32" });
    expect(parseMlirType("tensor<1x8x!quant.uniform<i8:f32, 0.1:-128>>").dt).toBe("int8");
    expect(parseMlirType("tensor<*xf32>").shape).toBeUndefined();
  });

  it("collapses a pretty stablehlo.reduce body into one node", () => {
    const r = parseMlir(`func.func @main(%arg0: tensor<1x10xf32>) -> tensor<1xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %cst) across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
   reducer(%a: tensor<f32>, %b: tensor<f32>)  {
    %1 = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }
  return %0 : tensor<1xf32>
}`, "reduce.mlir");
    expect(r.layers.map((l) => l.type)).toEqual(["Input", "stablehlo.reduce"]);
    expect(r.edges).toEqual([[0, 1]]);
    expect(r.layers[1].shape).toBe("1");
    expect(r.layers[1].flops).toBe(10);
    expect(r.layers[1].group).toBe("output");
  });

  it("rejects text without functions", () => {
    expect(isMlir("hello world")).toBe(false);
    expect(() => parseMlir("module {}", "e.mlir")).toThrow();
  });
});
