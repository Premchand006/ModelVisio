import { describe, expect, it, vi } from "vitest";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { demoModel } from "../src/demo/demoModel";
import { HW, type DeviceSpec } from "../src/data/hardware";
import {
  runSearch, paretoFront, dominates, runtimesFor, supportedPrecisions, applicableFixes, fixSubsets,
  scaleResolution, canRescale, scriptKeyFor,
} from "../src/optsearch/search";
import { heuristicRisk, sensitivityRisk } from "../src/optsearch/risk";
import { buildRecipe } from "../src/optsearch/recipe";
import { layerRoofline } from "../src/investigate/layerRoofline";

const dev = (id: string): DeviceSpec => {
  const d = HW.find((h) => h.id === id);
  if (!d) throw new Error(`no device ${id}`);
  return d;
};

/** Deterministic PRNG for property tests. */
function rng(seed: number) {
  return () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
}

describe("Pareto frontier", () => {
  it("matches a brute-force non-dominated filter on random points", () => {
    const r = rng(42);
    for (let trial = 0; trial < 20; trial++) {
      const pts = Array.from({ length: 80 }, (_, i) => ({ id: i, latencyMs: Math.round(r() * 20), sizeMB: Math.round(r() * 20), risk: Math.round(r() * 5) / 5 }));
      const brute = pts.filter((p) => !pts.some((q) => dominates(q, p))).map((p) => p.id).sort((a, b) => a - b);
      const fast = paretoFront(pts).map((p) => p.id).sort((a, b) => a - b);
      expect(fast).toEqual(brute);
    }
  });
  it("keeps exact duplicates (neither dominates) and drops dominated points", () => {
    const a = { latencyMs: 1, sizeMB: 1, risk: 0 };
    const b = { latencyMs: 1, sizeMB: 1, risk: 0 };
    const c = { latencyMs: 2, sizeMB: 1, risk: 0 };
    const d = { latencyMs: 3, sizeMB: 0.5, risk: 0 };
    expect(paretoFront([c, a, b, d])).toEqual([a, b, d]);
    expect(dominates(a, b)).toBe(false);
    expect(dominates(a, c)).toBe(true);
  });
});

describe("search space derivation", () => {
  it("derives runtimes from device toolchains, scripted ones first, no kernel libraries", () => {
    expect(runtimesFor(dev("agx_orin"))).toEqual(["TensorRT"]);
    expect(runtimesFor(dev("intel_npu"))[0]).toBe("OpenVINO");
    expect(runtimesFor(dev("intel_npu"))).toContain("ONNX RT");
    expect(scriptKeyFor("TensorRT", "int8")).toBe("onnx→tensorrt");
    expect(scriptKeyFor("ONNX RT", "fp16")).toBe("onnx→onnx-fp16");
    expect(scriptKeyFor("HailoRT", "int8")).toBeNull();
  });
  it("derives precisions from peaks and quant capability", () => {
    expect(supportedPrecisions(dev("coral")).map((p) => p.prec)).toEqual(["int8"]);
    const orin = supportedPrecisions(dev("agx_orin"));
    expect(orin.map((p) => p.prec)).toEqual(["int8", "fp16", "fp32"]);
    expect(orin.find((p) => p.prec === "fp16")!.estimated).toBe(false);
    expect(orin.find((p) => p.prec === "fp32")!.estimated).toBe(true);
    const rk = supportedPrecisions(dev("rk3588"));
    expect(rk.find((p) => p.prec === "fp16")).toMatchObject({ peakTops: 3, estimated: true });
  });
  it("enumerates every subset of the model's applicable fixes", () => {
    const fx = applicableFixes(demoModel);
    expect(fx.length).toBeGreaterThan(0);
    const subs = fixSubsets(fx);
    expect(subs).toHaveLength(2 ** fx.length);
    expect(subs[0]).toEqual([]);
    expect(new Set(subs.map((s) => s.join("+"))).size).toBe(subs.length);
  });
  it("rescaling the input shrinks FLOPs ~s² and latency, never touches weights", () => {
    expect(canRescale(demoModel)).toBe(true);
    const half = scaleResolution(demoModel, 0.5);
    expect(scaleResolution(demoModel, 1)).toBe(demoModel);
    expect(half.inputShape).toEqual([1, 3, 320, 320]);
    const f0 = demoModel.layers.reduce((s, l) => s + l.flops, 0);
    const f1 = half.layers.reduce((s, l) => s + l.flops, 0);
    expect(f1).toBeLessThan(f0 * 0.4);
    expect(half.layers.reduce((s, l) => s + l.params, 0)).toBe(demoModel.layers.reduce((s, l) => s + l.params, 0));
    expect(layerRoofline(half, dev("agx_orin")).latencyMs).toBeLessThan(layerRoofline(demoModel, dev("agx_orin")).latencyMs);
  });
});

describe("risk estimators", () => {
  it("heuristic risk grows as precision drops and with accuracy-affecting fixes", () => {
    const ctx = { model: demoModel, device: dev("agx_orin"), fixes: [], scale: 1 };
    const r = (p: "fp32" | "fp16" | "int8" | "int4") => heuristicRisk({ ...ctx, precision: p });
    expect(r("fp32")).toBeLessThan(r("fp16"));
    expect(r("fp16")).toBeLessThan(r("int8"));
    expect(r("int8")).toBeLessThan(r("int4"));
    expect(heuristicRisk({ ...ctx, precision: "int8", fixes: ["channel_prune"] })).toBeGreaterThan(r("int8"));
    expect(heuristicRisk({ ...ctx, precision: "int8", scale: 0.5 })).toBeGreaterThan(r("int8"));
  });
  it("sensitivity-based risk is in range and orders FP16 below INT8", () => {
    const ctx = { model: demoModel, device: dev("agx_orin"), fixes: [], scale: 1 };
    const i8 = sensitivityRisk({ ...ctx, precision: "int8" });
    const f16 = sensitivityRisk({ ...ctx, precision: "fp16" });
    expect(i8).toBeGreaterThanOrEqual(0);
    expect(i8).toBeLessThanOrEqual(1);
    expect(f16).toBeLessThan(i8);
  });
});

describe("runSearch", () => {
  it("returns a ranked feasible list whose head is Pareto-optimal", () => {
    const r = runSearch(demoModel, { devices: [dev("agx_orin"), dev("orin_nano"), dev("rk3588"), dev("coral")] });
    expect(r.evaluated).toBeGreaterThan(0);
    expect(r.ranked[0].pareto).toBe(true);
    expect(r.ranked.map((c) => c.rank)).toEqual(r.ranked.map((_, i) => i + 1));
    for (const p of r.pareto) for (const q of r.ranked) expect(dominates(q, p)).toBe(false);
    for (const c of r.all) {
      expect(Number.isFinite(c.latencyMs) && c.latencyMs > 0).toBe(true);
      expect(c.sizeMB).toBeGreaterThan(0);
      expect(c.risk).toBeGreaterThanOrEqual(0);
    }
  });
  it("on the demo + Coral, fixes reduce errors; an unfixable earlier error keeps the CPU partition", () => {
    const r = runSearch(demoModel, { devices: [dev("coral")] });
    const none = r.all.find((c) => c.fixes.length === 0)!;
    const sppf = r.all.find((c) => c.fixes.length === 1 && c.fixes[0] === "sppf_parallel")!;
    expect(none.fallbackLayers).toBeGreaterThan(0);
    expect(sppf.errors).toBeLessThan(none.errors);
    expect(sppf.latencyMs).toBeLessThanOrEqual(none.latencyMs);
  });
  it("a fix that removes the only unsupported op lifts the single-partition CPU fallback", () => {
    const L = (id: number, p: Partial<ModelLayer>): ModelLayer => ({
      id, name: `l${id}`, type: "Conv2d", op: "Conv", shape: "", dt: "float32", params: 2304, flops: 3e7, macs: 1.5e7, mem: 4e5, group: "backbone",
      ins: [{ n: "a", s: [1, 16, 80, 80] }], outs: [{ n: "b", s: [1, 16, 80, 80] }], attr: {}, w: null, math: "", insight: "", qSens: 0.01, compIssues: [], ...p,
    });
    const m: Model = {
      name: "m.onnx", format: "ONNX", framework: "t", opset: 17, sizeBytes: 1e4, inputShape: [1, 3, 160, 160], outputShape: [1, 16, 80, 80],
      layers: [L(0, {}), L(1, { type: "SiLU", op: "Mul(Sigmoid)", params: 0, flops: 1e6, compIssues: [{ target: "tpu_x", severity: "error", msg: "SiLU unsupported" }] }), L(2, {}), L(3, {})],
      edges: [[0, 1], [1, 2], [2, 3]],
    };
    const tpu: DeviceSpec = { ...dev("coral"), id: "tpu", compatId: "tpu_x" };
    const r = runSearch(m, { devices: [tpu] });
    const none = r.all.find((c) => c.fixes.length === 0)!;
    const fixed = r.all.find((c) => c.fixes.includes("silu_hardswish"))!;
    expect(none.fallbackLayers).toBe(3);
    expect(fixed.fallbackLayers).toBe(0);
    expect(fixed.latencyMs).toBeLessThan(none.latencyMs);
  });
  it("prunes configs that cannot compile on a no-fallback device before estimating them", () => {
    const silu = (id: number): ModelLayer => ({
      id, name: `act${id}`, type: "SiLU", op: "Mul(Sigmoid)", shape: "", dt: "float32", params: 0, flops: 1e6, macs: 5e5, mem: 4e5, group: "backbone",
      ins: [{ n: "a", s: [1, 16, 80, 80] }], outs: [{ n: "b", s: [1, 16, 80, 80] }], attr: {}, w: null, math: "", insight: "", qSens: 0.01,
      compIssues: [{ target: "strict_npu", severity: "error", msg: "SiLU unsupported" }],
    });
    const conv: ModelLayer = { ...silu(0), name: "conv0", type: "Conv2d", op: "Conv", params: 2304, flops: 3e7, compIssues: [] };
    const m: Model = { name: "m.onnx", format: "ONNX", framework: "t", opset: 17, sizeBytes: 1e4, inputShape: [1, 3, 160, 160], outputShape: [1, 16, 80, 80], layers: [conv, silu(1)], edges: [[0, 1]] };
    const strict: DeviceSpec = { ...dev("hailo8"), id: "strict", compatId: "strict_npu" };
    const r = runSearch(m, { devices: [strict] });
    expect(r.pruned["no-compile"]).toBeGreaterThan(0);
    expect(r.all.length).toBeGreaterThan(0);
    for (const c of r.all) expect(c.fixes).toContain("silu_hardswish");
  });
  it("prunes configs that don't fit memory", () => {
    const tiny: DeviceSpec = { ...dev("orin_nano"), id: "tiny", ramGB: 1e-6 };
    const r = runSearch(demoModel, { devices: [tiny] });
    expect(r.all).toHaveLength(0);
    expect(r.pruned["no-memory"]).toBeGreaterThan(0);
  });
  it("applies constraints to the ranked list and the frontier", () => {
    const base = runSearch(demoModel, { devices: [dev("agx_orin"), dev("rk3588")] });
    const lats = base.all.map((c) => c.latencyMs).sort((a, b) => a - b);
    const cap = lats[Math.floor(lats.length / 2)];
    const r = runSearch(demoModel, { devices: [dev("agx_orin"), dev("rk3588")], constraints: { maxLatencyMs: cap, maxRisk: 0.9 } });
    expect(r.ranked.length).toBeGreaterThan(0);
    expect(r.ranked.length).toBeLessThan(r.all.length);
    for (const c of r.ranked) expect(c.latencyMs).toBeLessThanOrEqual(cap);
    for (const c of r.pareto) expect(c.meetsConstraints).toBe(true);
  });
  it("uses a custom risk estimator when given", () => {
    const risk = vi.fn(() => 0.5);
    const r = runSearch(demoModel, { devices: [dev("hailo8")], risk });
    expect(risk).toHaveBeenCalled();
    for (const c of r.all) expect(c.risk).toBe(0.5);
  });
  it("searches every device × scale quickly with memoization", () => {
    const r = runSearch(demoModel, { devices: HW, scales: [1, 0.75, 0.5] });
    expect(r.scalesConsidered).toEqual([1, 0.75, 0.5]);
    expect(r.spaceSize).toBeGreaterThan(r.evaluated);
    expect(r.elapsedMs).toBeLessThan(5000);
    expect(r.all.some((c) => c.scale === 0.5)).toBe(true);
  });
});

describe("buildRecipe", () => {
  it("reuses the Converter's TensorRT script with the config's precision", () => {
    const r = runSearch(demoModel, { devices: [dev("orin_nano")], precisions: ["int8"] });
    const c = r.all.find((x) => x.fixes.length === 0)!;
    const rec = buildRecipe(demoModel, c);
    expect(c.scriptKey).toBe("onnx→tensorrt");
    expect(rec.text).toContain("trtexec");
    expect(rec.text).toContain("--int8");
  });
  it("lists fixes and falls back to vendor guidance without a script", () => {
    const r = runSearch(demoModel, { devices: [dev("hailo8")] });
    const c = r.all.find((x) => x.fixes.length > 0)!;
    const rec = buildRecipe(demoModel, c);
    expect(c.scriptKey).toBeNull();
    expect(rec.steps[0].title).toMatch(/graph fix/);
    expect(rec.text).toMatch(/vendor toolchain/);
  });
});
