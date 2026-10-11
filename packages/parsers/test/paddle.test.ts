import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { isPaddleProgram, parsePaddle, readPdiparams } from "../src/formats/paddle";
import { parseModel } from "../src/index";

// Fixture: test/fixtures/scripts/gen_paddle.py (PaddlePaddle 3.3.1, legacy
// ProgramDesc via FLAGS_enable_pir_api=0). expected.json holds Paddle's own
// parameter count for the same network.
const dir = fileURLToPath(new URL("./fixtures/paddle/", import.meta.url));
const buf = new Uint8Array(readFileSync(dir + "tiny_cnn.pdmodel"));
const expected = JSON.parse(readFileSync(dir + "tiny_cnn.expected.json", "utf8"));

describe("PaddlePaddle .pdmodel", () => {
  const m = parsePaddle(buf, "tiny_cnn.pdmodel");

  it("sniffs as a ProgramDesc", () => {
    expect(isPaddleProgram(buf)).toBe(true);
    expect(isPaddleProgram(new TextEncoder().encode("not a model"))).toBe(false);
  });

  it("builds one layer per op, skipping feed/fetch and folding bias reshapes", () => {
    expect(m.format).toBe("PaddlePaddle");
    expect(m.layers.map((l) => l.op)).toEqual([
      "feed", "conv2d", "elementwise_add", "batch_norm", "relu", "pool2d",
      "depthwise_conv2d", "elementwise_add", "conv2d", "elementwise_add", "relu", "pool2d",
      "flatten_contiguous_range", "matmul_v2", "elementwise_add", "softmax",
    ]);
    expect(m.producer).toBe("Paddle 3.3.1");
  });

  it("chains data-flow edges through shared var names", () => {
    expect(m.edges).toHaveLength(m.layers.length - 1);
    for (const [a, b] of m.edges) expect(b).toBe(a + 1);
  });

  it("reads shapes from VarDescs (dynamic batch as ?)", () => {
    expect(m.inputShape).toEqual([0, 3, 32, 32]);
    expect(m.outputShape).toEqual([0, 10]);
    expect(m.layers[0].shape).toBe("?×3×32×32");
    expect(m.layers[1].shape).toBe("?×8×32×32");
    expect(m.layers[5].shape).toBe("?×8×16×16");
    expect(m.layers[15].group).toBe("output");
    expect(m.layers[0].group).toBe("input");
  });

  it("totals persistable params to the framework count", () => {
    const total = m.layers.reduce((a, l) => a + l.params, 0);
    expect(total).toBe(expected.total_params);
    expect(m.layers[1].params).toBe(8 * 3 * 3 * 3); // conv weight
    expect(m.layers[2].params).toBe(8); // bias via folded reshape2
    expect(m.layers[3].params).toBe(4 * 8); // BN scale/bias/mean/var
    expect(m.layers[1].w?.shape).toEqual([8, 3, 3, 3]);
  });

  it("estimates MACs for conv / depthwise / matmul", () => {
    expect(m.layers[1].macs).toBe(8 * 32 * 32 * 27);
    expect(m.layers[6].macs).toBe(8 * 16 * 16 * 9);
    expect(m.layers[8].macs).toBe(16 * 16 * 16 * 8);
    expect(m.layers[13].macs).toBe(10 * 16);
    expect(m.layers[13].flops).toBe(2 * 160);
  });

  it("decodes op attributes", () => {
    expect(m.layers[1].attr.strides).toEqual([1, 1]);
    expect(m.layers[1].attr.paddings).toEqual([1, 1]);
    expect(m.layers[5].attr.pooling_type).toBe("max");
    expect(m.layers[11].attr.adaptive).toBe(true);
    expect(m.layers[1].attr.op_callstack).toBeUndefined();
  });
});

describe("Paddle .pdiparams weight statistics", () => {
  const dir = fileURLToPath(new URL("./fixtures/paddle/with_params/", import.meta.url));
  const model = new Uint8Array(readFileSync(join(dir, "tiny_cnn.pdmodel")));
  const params = new Uint8Array(readFileSync(join(dir, "tiny_cnn.pdiparams")));
  // Written by paddle itself (load_inference_model + numpy) when the fixture was generated.
  const truth = JSON.parse(readFileSync(join(dir, "weights.expected.json"), "utf8")) as Record<string, { min: number; max: number; mean: number; std: number }>;

  it("matches the framework's own values for every layer weight", () => {
    const m = parseModel(model, "tiny_cnn.pdmodel", { "tiny_cnn.pdiparams": params });
    const withW = m.layers.filter((l) => typeof l.attr.weight === "string");
    expect(withW.length).toBeGreaterThan(0);
    for (const l of withW) {
      const t = truth[l.attr.weight as string];
      expect(l.w?.mean, l.name).toBeCloseTo(t.mean, 3);
      expect(l.w?.std, l.name).toBeCloseTo(t.std, 3);
      expect(l.w?.min, l.name).toBeCloseTo(t.min, 3);
      expect(l.w?.max, l.name).toBeCloseTo(t.max, 3);
    }
    expect(m.notes).toBeUndefined();
  });

  it("refuses a params file that does not belong to the program", () => {
    expect(readPdiparams(params, [{ name: "x", dims: [3, 3] }])).toBeNull();
    expect(readPdiparams(params.subarray(0, 100), [{ name: "conv2d_0.w_0", dims: [8, 3, 3, 3] }])).toBeNull();
  });

  it("notes the missing .pdiparams", () => {
    expect(parseModel(model, "tiny_cnn.pdmodel").notes?.[0]).toMatch(/tiny_cnn\.pdiparams/);
  });
});
