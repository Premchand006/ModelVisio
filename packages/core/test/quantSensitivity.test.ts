import { describe, expect, it } from "vitest";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { demoModel } from "../src/demo/demoModel";
import { analyzeQuantization, quantSqnrDb, sqnrRisk, isDepthwise } from "../src/quant/sensitivity";
import { ortQuantizeScript, planToJson, tfliteQuantizeScript } from "../src/quant/exportPlan";
import { computeNodeColors, textOn, viridis } from "../src/components/Graph/colorBy";

function layer(p: Partial<ModelLayer>): ModelLayer {
  return {
    id: 0, name: "l", type: "Conv", op: "Conv", shape: "", dt: "float32",
    params: 0, flops: 0, macs: 0, mem: 0, group: "backbone",
    ins: [], outs: [], attr: {}, w: null, math: "", insight: "", qSens: 0, compIssues: [],
    ...p,
  };
}

const input = (id = 0) => layer({ id, name: "images", type: "Tensor", op: "Input", outs: [{ n: "images", s: [1, 3, 64, 64] }] });
const output = (id: number) => layer({ id, name: "out", type: "Output", op: "Identity" });
const conv = (id: number, w: ModelLayer["w"], extra: Partial<ModelLayer> = {}) =>
  layer({ id, name: `conv${id}`, params: 64 * 64 * 9, w, outs: [{ n: `t${id}`, s: [1, 64, 32, 32] }], ...extra });

/** input → relu(pad) → [middle layers] → relu → output, so the tested layer sits in the middle. */
function chain(mid: ModelLayer[]): Model {
  const wFine = { shape: [64, 64, 3, 3], size: 36864, min: -0.2, max: 0.2, mean: 0, std: 0.05, sparse: 0 };
  const layers = [input(0), conv(1, wFine), ...mid.map((l, i) => ({ ...l, id: i + 2 })), conv(mid.length + 2, wFine), output(mid.length + 3)];
  const edges: [number, number][] = layers.slice(1).map((l) => [l.id - 1, l.id]);
  return { name: "t.onnx", format: "ONNX", framework: "test", opset: 17, sizeBytes: 0, inputShape: [1, 3, 64, 64], outputShape: [1, 10], layers, edges };
}

describe("quantizer SQNR math", () => {
  it("matches the uniform-quantizer formula 12·L²/c²", () => {
    // crest 4: INT8 → 10·log10(12·127²/16) ≈ 40.8 dB; INT4 → 10·log10(12·7²/16) ≈ 15.6 dB
    expect(quantSqnrDb(8, 4)).toBeCloseTo(40.8, 1);
    expect(quantSqnrDb(4, 4)).toBeCloseTo(15.65, 1);
  });
  it("each extra bit adds ~6 dB, sparsity and smaller crest raise SQNR", () => {
    expect(quantSqnrDb(8, 4) - quantSqnrDb(7, 4)).toBeGreaterThan(5.9);
    expect(quantSqnrDb(8, 4, 0.5)).toBeGreaterThan(quantSqnrDb(8, 4));
    expect(quantSqnrDb(8, 3)).toBeGreaterThan(quantSqnrDb(8, 6));
  });
  it("risk is monotone in SQNR and centred at 20 dB", () => {
    expect(sqnrRisk(20)).toBeCloseTo(0.5, 5);
    expect(sqnrRisk(40)).toBeLessThan(0.01);
    expect(sqnrRisk(10)).toBeGreaterThan(0.95);
  });
});

describe("analyzeQuantization: per-layer", () => {
  it("well-behaved conv weights are safe for INT8 per-tensor", () => {
    const a = analyzeQuantization(chain([conv(0, { shape: [64, 64, 3, 3], size: 36864, min: -0.3, max: 0.3, mean: 0, std: 0.02, sparse: 0 })]));
    const s = a.byId.get(2)!;
    expect(s.kind).toBe("weighted");
    expect(s.statsSource).toBe("measured");
    expect(s.position).toBe("middle");
    expect(s.int8.sqnrDb!).toBeGreaterThan(28);
    expect(s.recommendation).toBe("int8");
    expect(s.int4.perChannelRisk).toBeGreaterThan(s.int8.perChannelRisk);
  });

  it("outlier-heavy weights need per-channel scales", () => {
    // crest 40 → per-tensor ≈ 20.8 dB; per-channel estimate recovers ~10 dB
    const a = analyzeQuantization(chain([conv(0, { shape: [64, 64, 3, 3], size: 36864, min: -0.4, max: 0.4, mean: 0, std: 0.01, sparse: 0 })]));
    const s = a.byId.get(2)!;
    expect(s.outlierRatio).toBeCloseTo(40, 0);
    expect(s.int8.perChannelSqnrDb!).toBeGreaterThan(s.int8.sqnrDb! + 5);
    expect(s.recommendation).toBe("int8-per-channel");
    expect(a.summary.perChannel.map((l) => l.id)).toContain(2);
  });

  it("Softmax stays in float (keep-fp32) and LayerNorm is flagged", () => {
    const a = analyzeQuantization(chain([layer({ op: "Softmax", type: "Softmax" }), layer({ op: "LayerNormalization", type: "LayerNormalization" })]));
    expect(a.byId.get(2)!.recommendation).toBe("keep-fp32");
    expect(["fp16", "keep-fp32"]).toContain(a.byId.get(3)!.recommendation);
    expect(a.summary.excluded.length).toBe(2);
  });

  it("layout ops are passthrough and not quantizable; ReLU is low risk", () => {
    const a = analyzeQuantization(chain([layer({ op: "Reshape", type: "Reshape" }), layer({ op: "Relu", type: "Relu" })]));
    expect(a.byId.get(2)!.kind).toBe("passthrough");
    expect(a.byId.get(2)!.quantizable).toBe(false);
    expect(a.byId.get(3)!.int8.risk).toBeLessThan(0.1);
    expect(a.byId.get(0)!.kind).toBe("io");
  });

  it("first and last weighted layers carry a position penalty, larger for INT4", () => {
    const a = analyzeQuantization(chain([conv(0, { shape: [64, 64, 3, 3], size: 36864, min: -0.2, max: 0.2, mean: 0, std: 0.05, sparse: 0 })]));
    const first = a.byId.get(1)!, mid = a.byId.get(2)!, last = a.byId.get(3)!;
    expect(first.position).toBe("first");
    expect(last.position).toBe("last");
    expect(first.int8.risk).toBeGreaterThan(mid.int8.risk);
    expect(first.int4.risk).toBeGreaterThan(mid.int4.risk);
    expect(first.int4.risk).toBeGreaterThanOrEqual(0.6); // 4-bit first/last layers: high risk
  });

  it("weights without statistics fall back to a Gaussian assumption", () => {
    const a = analyzeQuantization(chain([conv(0, { shape: [64, 64, 3, 3], size: 36864 })]));
    const s = a.byId.get(2)!;
    expect(s.statsSource).toBe("assumed");
    expect(Number.isFinite(s.int8.sqnrDb!)).toBe(true);
    expect(s.factors.join(" ")).toMatch(/assumed/i);
  });

  it("depthwise convs are detected and per-tensor is riskier than per-channel", () => {
    const dw = conv(0, { shape: [64, 1, 3, 3], size: 576 }, { params: 576, attr: { groups: 64 }, ins: [{ n: "x", s: [1, 64, 32, 32] }] });
    expect(isDepthwise(dw)).toBe(true);
    const s = analyzeQuantization(chain([dw])).byId.get(2)!;
    expect(s.int8.risk).toBeGreaterThan(s.int8.perChannelRisk);
    expect(s.factors[0]).toMatch(/Depthwise/);
  });

  it("weights beyond the FP16 range must stay FP32", () => {
    const a = analyzeQuantization(chain([conv(0, { shape: [64, 64, 3, 3], size: 36864, min: -1e5, max: 1e5, mean: 0, std: 1, sparse: 0 })]));
    expect(a.byId.get(2)!.recommendation).toBe("keep-fp32");
  });
});

describe("analyzeQuantization: model summary", () => {
  it("handles a model with zero weights/stats gracefully", () => {
    const m: Model = {
      name: "empty", format: "ONNX", framework: "t", opset: 1, sizeBytes: 0, inputShape: [], outputShape: [],
      layers: [input(0), layer({ id: 1, op: "Add", type: "Add" }), output(2)], edges: [[0, 1], [1, 2]],
    };
    const a = analyzeQuantization(m);
    expect(a.summary.weighted).toBe(0);
    expect(a.summary.totalParams).toBe(0);
    expect(a.summary.sizeBytes.mixed).toBe(0);
    expect(a.summary.confidence).toBe("low");
    expect(a.summary.safeInt4Pct).toBe(100);
    const empty: Model = { ...m, layers: [], edges: [] };
    expect(analyzeQuantization(empty).summary.quantizable).toBe(0);
  });

  it("summarizes the demo model consistently", () => {
    const a = analyzeQuantization(demoModel);
    const s = a.summary;
    expect(a.layers).toHaveLength(demoModel.layers.length);
    const params = demoModel.layers.reduce((n, l) => n + l.params, 0);
    expect(s.totalParams).toBe(params);
    expect(s.sizeBytes.fp32).toBe(params * 4);
    expect(s.sizeBytes.int8).toBe(params);
    expect(s.sizeBytes.mixed).toBeGreaterThanOrEqual(s.sizeBytes.int8);
    expect(s.sizeBytes.mixed).toBeLessThanOrEqual(s.sizeBytes.fp32);
    expect(s.safeInt8Pct).toBeGreaterThan(0);
    expect(s.safeInt8Pct).toBeLessThanOrEqual(100);
    for (const l of a.layers) {
      expect(l.int8.risk).toBeGreaterThanOrEqual(0);
      expect(l.int8.risk).toBeLessThanOrEqual(1);
      expect(l.reason.length).toBeGreaterThan(0);
    }
    // stem conv is the first weighted layer
    expect(a.byId.get(1)!.position).toBe("first");
  });

  it("target int4 never assigns INT4 to an unsafe layer", () => {
    const a = analyzeQuantization(demoModel, { target: "int4" });
    for (const l of a.layers) if (l.recommendation === "int4") expect(l.int4.perChannelRisk).toBeLessThan(0.3);
    expect(a.summary.target).toBe("int4");
  });
});

describe("plan export", () => {
  const m = chain([layer({ op: "Softmax", type: "Softmax", outs: [{ n: "probs" }], name: "softmax_0" })]);
  const a = analyzeQuantization(m);

  it("JSON lists exclusions and quantizable layers only", () => {
    const j = planToJson(m, a);
    expect(j.nodesToExclude).toEqual(["softmax_0"]);
    expect(j.layers.every((l) => l.name !== "images")).toBe(true);
    expect(j.disclaimer).toMatch(/Not measured accuracy/);
  });

  it("ONNX Runtime script passes nodes_to_exclude to quantize_static", () => {
    const py = ortQuantizeScript(m, a);
    expect(py).toContain("quantize_static(");
    expect(py).toContain("nodes_to_exclude=NODES_TO_EXCLUDE");
    expect(py).toContain('"softmax_0",');
    expect(py).toContain('INPUT_NAME = "images"');
    expect(py).toContain("INPUT_SHAPE = (1, 3, 64, 64)");
  });

  it("TFLite script denylists output tensor names via the Quantization Debugger", () => {
    const py = tfliteQuantizeScript(m, a);
    expect(py).toContain("QuantizationDebugOptions(denylisted_nodes=DENYLISTED_NODES)");
    expect(py).toContain('"probs",');
    expect(py).toContain("get_nondebug_quantized_model()");
    expect(py).toContain("INPUT_SHAPE = (1, 64, 64, 3)");
  });
});

describe("graph color-by scale", () => {
  it("viridis endpoints and clamping", () => {
    expect(viridis(0)).toBe("#440154");
    expect(viridis(1)).toBe("#fde725");
    expect(viridis(-1)).toBe("#440154");
    expect(viridis(NaN)).toBe("#440154");
  });
  it("text contrast flips between dark and light swatches", () => {
    expect(textOn("#440154")).toBe("#FFFFFF");
    expect(textOn("#fde725")).toBe("#111111");
  });
  it("colors every layer; none mode keeps the default look", () => {
    expect(computeNodeColors(demoModel, "none", "#999999")).toBeNull();
    for (const mode of ["flops", "params", "mem"] as const) {
      const c = computeNodeColors(demoModel, mode, "#999999")!;
      expect(c.colors.size).toBe(demoModel.layers.length);
      expect(c.legend.scale).toBe("log");
    }
    const q = computeNodeColors(demoModel, "q8", "#999999", analyzeQuantization(demoModel))!;
    expect(q.colors.get(0)).toBe("#999999"); // input: no arithmetic
    expect(q.colors.get(1)).not.toBe("#999999");
  });
});
