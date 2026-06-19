import { describe, expect, it } from "vitest";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { demoModel } from "../src/demo/demoModel";
import { applyFix, countApplicable, applicableLayers, FIXES } from "../src/fixes/transforms";

function layer(p: Partial<ModelLayer>): ModelLayer {
  return {
    id: 0, name: "l", type: "X", op: "X", shape: "", dt: "float32",
    params: 0, flops: 0, macs: 0, mem: 0, group: "backbone",
    ins: [], outs: [], attr: {}, w: null, math: "", insight: "", qSens: 0, compIssues: [], ...p,
  };
}
function model(layers: ModelLayer[]): Model {
  return { name: "t.onnx", format: "ONNX", framework: "x", opset: 18, sizeBytes: 0, inputShape: [1, 3, 8, 8], outputShape: [1, 8], layers, edges: [] };
}

const totalParams = (m: typeof demoModel) => m.layers.reduce((s, l) => s + l.params, 0);

describe("apply-fix transforms (real graph rewrites)", () => {
  it("is immutable — the original model is never mutated", () => {
    const before = JSON.stringify(demoModel);
    for (const f of FIXES) {
      const out = applyFix(demoModel, f.id);
      expect(out).not.toBe(demoModel);
      expect(out.layers.length).toBe(demoModel.layers.length);
    }
    expect(JSON.stringify(demoModel)).toBe(before);
  });

  it("SiLU → HardSwish rewrites activations and clears their compat issues", () => {
    expect(countApplicable(demoModel, "silu_hardswish")).toBeGreaterThan(0);
    const out = applyFix(demoModel, "silu_hardswish");
    expect(demoModel.layers.some((l) => l.type === "SiLU")).toBe(true); // original intact
    expect(out.layers.some((l) => l.type === "SiLU")).toBe(false);
    expect(out.layers.some((l) => l.type === "Hardswish")).toBe(true);
    const siluIssues = out.layers.flatMap((l) => l.compIssues).filter((c) => /silu/i.test(c.msg));
    expect(siluIssues).toHaveLength(0);
  });

  it("channel prune cuts heavy layers to 75% and leaves light layers untouched", () => {
    const out = applyFix(demoModel, "channel_prune");
    expect(totalParams(out)).toBeLessThan(totalParams(demoModel));
    const heavy = demoModel.layers.filter((l) => l.params > 200_000);
    expect(heavy.length).toBeGreaterThan(0);
    for (const h of heavy) {
      expect(out.layers.find((l) => l.id === h.id)!.params).toBe(Math.round(h.params * 0.75));
    }
    const light = demoModel.layers.find((l) => l.params > 0 && l.params <= 200_000)!;
    expect(out.layers.find((l) => l.id === light.id)!.params).toBe(light.params);
  });

  it("Resize → ConvTranspose removes Resize CPU-fallback warnings and adds params", () => {
    expect(countApplicable(demoModel, "resize_convtranspose")).toBeGreaterThan(0);
    const out = applyFix(demoModel, "resize_convtranspose");
    expect(out.layers.flatMap((l) => l.compIssues).filter((c) => /resize/i.test(c.msg))).toHaveLength(0);
    expect(out.layers.some((l) => l.type === "ConvTranspose2d")).toBe(true);
  });

  it("SPPF → parallel SPP removes the cascaded-MaxPool error", () => {
    expect(countApplicable(demoModel, "sppf_parallel")).toBeGreaterThan(0);
    const out = applyFix(demoModel, "sppf_parallel");
    expect(out.layers.flatMap((l) => l.compIssues).filter((c) => /maxpool|cascad/i.test(c.msg))).toHaveLength(0);
  });

  it("applicableLayers lists the layers a fix would touch", () => {
    const names = applicableLayers(demoModel, "silu_hardswish");
    expect(names.length).toBe(countApplicable(demoModel, "silu_hardswish"));
    expect(names.every((n) => typeof n === "string")).toBe(true);
  });
});

describe("apply-fix edge cases (regression guards)", () => {
  it("does NOT rewrite a standalone Sigmoid — only SiLU / Mul(Sigmoid)", () => {
    const m = model([
      layer({ id: 0, type: "Sigmoid", op: "Sigmoid" }),
      layer({ id: 1, type: "SiLU", op: "Mul(Sigmoid)" }),
    ]);
    expect(countApplicable(m, "silu_hardswish")).toBe(1);
    const out = applyFix(m, "silu_hardswish");
    expect(out.layers[0].type).toBe("Sigmoid"); // untouched
    expect(out.layers[1].type).toBe("Hardswish");
  });

  it("Resize → ConvTranspose never decreases FLOPs/MACs on dynamic (-1) spatial dims", () => {
    const m = model([
      layer({ id: 0, type: "Upsample", op: "Resize", flops: 1000, macs: 500, outs: [{ n: "y", s: [1, 128, -1, -1] }] }),
    ]);
    const out = applyFix(m, "resize_convtranspose");
    expect(out.layers[0].flops).toBeGreaterThanOrEqual(1000);
    expect(out.layers[0].macs).toBeGreaterThanOrEqual(500);
    expect(Number.isNaN(out.layers[0].flops)).toBe(false);
  });

  it("channel prune shrinks the output-channel dim in the shape (footprint follows)", () => {
    const m = model([
      layer({ id: 0, type: "Conv", op: "Conv", params: 300_000, macs: 1000, flops: 2000, outs: [{ n: "y", s: [1, 256, 20, 20] }] }),
    ]);
    const out = applyFix(m, "channel_prune");
    expect(out.layers[0].outs[0].s![1]).toBe(192); // 256 × 0.75
    expect(m.layers[0].outs[0].s![1]).toBe(256); // original untouched
  });
});
