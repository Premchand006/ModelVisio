import { describe, expect, it } from "vitest";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { demoModel } from "../src/demo/demoModel";
import { HW } from "../src/data/hardware";
import { advise, scaleResolution, simulate } from "../src/advisor/advise";

const byId = (id: string) => {
  const d = HW.find((h) => h.id === id);
  if (!d) throw new Error(`no device ${id}`);
  return d;
};

function layer(p: Partial<ModelLayer>): ModelLayer {
  return {
    id: 0, name: "l", type: "Conv", op: "Conv", shape: "", dt: "float32",
    params: 0, flops: 0, macs: 0, mem: 0, group: "backbone",
    ins: [], outs: [], attr: {}, w: null, math: "", insight: "", qSens: 0, compIssues: [],
    ...p,
  };
}

const finiteImpact = (m: Model, devId: string) => {
  const r = advise(m, byId(devId));
  for (const a of r.advice) {
    for (const v of [a.impact.speedup, a.impact.latencyPct, a.impact.sizePct, a.impact.flopsPct]) {
      if (v != null) expect(Number.isFinite(v)).toBe(true);
    }
    expect(Number.isFinite(a.score)).toBe(true);
  }
  expect(Number.isFinite(r.plan.latencyMs)).toBe(true);
  return r;
};

describe("simulate (per-precision roofline)", () => {
  const dev = byId("orin_nano");
  const m: Model = {
    name: "s", format: "ONNX", framework: "t", opset: 17, sizeBytes: 0, inputShape: [1, 3, 8, 8], outputShape: [],
    layers: [
      // 1 GFLOP on a tiny tensor → compute-bound
      layer({ id: 0, name: "heavy", flops: 1e9, outs: [{ n: "a", s: [1, 8, 8, 8] }] }),
      // 1 KFLOP over 16M elements → memory-bound
      layer({ id: 1, name: "light", op: "Add", type: "Add", flops: 1e3, outs: [{ n: "b", s: [1, 16, 1024, 1024] }] }),
    ],
    edges: [[0, 1]],
  };

  it("classifies compute- vs memory-bound layers against the ridge point", () => {
    const s = simulate(m, dev, () => "int8");
    expect(s.layers[0].bound).toBe("compute");
    expect(s.layers[1].bound).toBe("memory");
    expect(s.layers[1].intensity).toBeLessThan(1);
    const share = s.layers.reduce((a, l) => a + l.share, 0);
    expect(share).toBeGreaterThan(0.99);
    expect(share).toBeLessThanOrEqual(1.0001);
  });

  it("INT8 is faster than FP16 and Q/DQ boundaries cost time", () => {
    const i8 = simulate(m, dev, () => "int8");
    const f16 = simulate(m, dev, () => "fp16");
    const mixed = simulate(m, dev, (l) => (l.id === 0 ? "int8" : "fp16"));
    expect(i8.ms).toBeLessThan(f16.ms);
    expect(mixed.idealMs).toBeGreaterThan(i8.idealMs);
  });
});

describe("advise", () => {
  it("Coral: conversion + full INT8 are required, op fixes are proposed, no float baseline", () => {
    const r = finiteImpact(demoModel, "coral");
    expect(r.baseline).toBeNull();
    const ids = r.advice.map((a) => a.id);
    expect(ids).toContain("format");
    expect(ids).toContain("quantize");
    expect(ids).toContain("fix-silu_hardswish");
    expect(ids).toContain("fix-sppf_parallel");
    expect(r.advice.find((a) => a.id === "quantize")!.required).toBe(true);
    expect(r.advice.find((a) => a.id === "quantize")!.title).toMatch(/integer only/);
    expect(r.advice.find((a) => a.id === "format")!.title).toMatch(/TFLite/);
    // ranked
    for (let i = 1; i < r.advice.length; i++) expect(r.advice[i - 1].score).toBeGreaterThanOrEqual(r.advice[i].score);
  });

  it("Jetson: mixed INT8 plan beats the FP16 baseline and shrinks the model", () => {
    const r = finiteImpact(demoModel, "agx_orin");
    expect(r.baseline).not.toBeNull();
    const q = r.advice.find((a) => a.id === "quantize")!;
    expect(q.impact.speedup!).toBeGreaterThan(1);
    expect(q.impact.sizePct!).toBeLessThan(-50);
    expect(r.plan.latencyMs).toBeLessThan(r.baseline!.latencyMs);
    expect(r.fullInt8.latencyMs).toBeLessThanOrEqual(r.plan.latencyMs + 1e-9);
  });

  it("proposes 640→480 input resolution with about −44% FLOPs", () => {
    const r = advise(demoModel, byId("orin_nano"));
    const res = r.advice.find((a) => a.id === "resolution")!;
    expect(res.title).toContain("640→480");
    expect(res.impact.flopsPct!).toBeCloseTo(-43.75, 0);
    expect(res.impact.speedup!).toBeGreaterThan(1);
  });

  it("scaleResolution scales spatial shapes and work by r²", () => {
    const s = scaleResolution(demoModel, 0.5);
    expect(s.inputShape).toEqual([1, 3, 320, 320]);
    const stem = s.layers.find((l) => l.id === 1)!;
    expect(stem.flops).toBe(Math.round(demoModel.layers[1].flops * 0.25));
    expect(stem.outs[0].s).toEqual([1, 16, 160, 160]);
  });

  it("reports per-layer bottlenecks for the target", () => {
    const r = advise(demoModel, byId("rk3588"));
    expect(r.layers).toHaveLength(demoModel.layers.length);
    expect(r.layers.some((l) => l.bound === "memory")).toBe(true);
    expect(r.ridgeAI).toBeGreaterThan(0);
  });

  it("never produces NaN for any device in the database", () => {
    for (const d of HW) finiteImpact(demoModel, d.id);
  });

  it("handles models without weights or FLOPs", () => {
    const m: Model = {
      name: "w", format: "Safetensors", framework: "t", opset: 0, sizeBytes: 0, inputShape: [], outputShape: [],
      layers: [layer({ id: 0, op: "Add", type: "Add" }), layer({ id: 1, op: "Relu", type: "Relu" })], edges: [[0, 1]],
    };
    const r = finiteImpact(m, "hailo8");
    expect(r.quant.summary.weighted).toBe(0);
    expect(r.notes.join(" ")).toMatch(/No weighted layers/);
    const empty = finiteImpact({ ...m, layers: [], edges: [] }, "agx_orin");
    expect(empty.layers).toHaveLength(0);
  });
});
