import { describe, expect, it } from "vitest";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { demoModel } from "../src/demo/demoModel";
import { HW } from "../src/data/hardware";
import { scoreAll, scoreDevice } from "../src/scoring/scoreDevice";
import { classifyWorkload, peakOpsPerSec, estimateRoofline } from "../src/scoring/roofline";

const byId = (id: string) => {
  const d = HW.find((h) => h.id === id);
  if (!d) throw new Error(`no device ${id}`);
  return d;
};

// Minimal valid layer factory for synthetic models.
function layer(p: Partial<ModelLayer>): ModelLayer {
  return {
    id: 0, name: "l", type: "Conv2d", op: "Conv", shape: "", dt: "float32",
    params: 0, flops: 0, macs: 0, mem: 0, group: "backbone",
    ins: [], outs: [], attr: {}, w: null, math: "", insight: "", qSens: 0, compIssues: [],
    ...p,
  };
}

describe("roofline fundamentals", () => {
  it("uses DENSE peak compute, never the sparse marketing figure", () => {
    const agx = byId("agx_orin");
    expect(agx.peak.sparseInt8).toBe(275);
    // 137 dense TOPS → 137e12 ops/s, NOT 275e12.
    expect(peakOpsPerSec(agx, "int8")).toBeCloseTo(137e12, -10);
  });

  it("classifies a YOLO detector as cnnDet", () => {
    expect(classifyWorkload(demoModel)).toBe("cnnDet");
  });

  it("produces a finite, positive, non-degenerate FPS (no 999 clamp)", () => {
    const rOrin = estimateRoofline(demoModel, byId("agx_orin"));
    const rNxp = estimateRoofline(demoModel, byId("nxp"));
    expect(rOrin.fps).toBeGreaterThan(0);
    expect(Number.isFinite(rOrin.fps)).toBe(true);
    // A 137-TOPS Orin must estimate far more FPS than a 0.5-TOPS i.MX 93.
    expect(rOrin.fps).toBeGreaterThan(rNxp.fps * 10);
  });
});

describe("memory-fit hard-fail guard (the headline fix)", () => {
  // ~8B params → 8 GB at INT8. Fits on a 64GB AGX Orin, not on an 8GB Orin Nano.
  const huge: Model = {
    name: "huge.onnx", format: "ONNX", framework: "x", opset: 18, sizeBytes: 8e9,
    inputShape: [1, 3, 224, 224], outputShape: [1, 1000],
    layers: [
      layer({ id: 0, op: "Placeholder", type: "Input", group: "input", outs: [{ n: "x", s: [1, 3, 224, 224] }] }),
      layer({ id: 1, params: 8e9, macs: 8e9, flops: 16e9, ins: [{ n: "x", s: [1, 3, 224, 224] }], outs: [{ n: "y", s: [1, 1000] }] }),
    ],
    edges: [[0, 1]],
  };

  it("hard-fails (≤15, never green) when the model exceeds device memory", () => {
    const r = scoreDevice(huge, byId("orin_nano"));
    expect(r.memory.hardFail).toBe(true);
    expect(r.overall).toBeLessThanOrEqual(15);
    expect(r.banners.some((b) => b.level === "error" && /cannot deploy/i.test(b.msg))).toBe(true);
  });

  it("does NOT fail the same model on a device with enough memory", () => {
    const r = scoreDevice(huge, byId("agx_orin"));
    expect(r.memory.hardFail).toBe(false);
    expect(r.overall).toBeGreaterThan(15);
  });

  it("the demo model fits on a small device and is not memory-failed", () => {
    const r = scoreDevice(demoModel, byId("orin_nano"));
    expect(r.memory.hardFail).toBe(false);
  });
});

describe("op-support & CPU-fallback modeling", () => {
  it("penalizes Coral's single-partition fallback on the demo model", () => {
    const r = scoreDevice(demoModel, byId("coral"));
    // demoModel annotates coral_tpu errors (SPPF, conv4) → tail runs on CPU.
    expect(r.opSupport.coverage).toBeLessThan(1);
    expect(r.opSupport.fallbackOps).toBeGreaterThan(0);
    expect(r.banners.some((b) => /fall back to host CPU/i.test(b.msg))).toBe(true);
  });

  it("a graceful-fallback NVIDIA part keeps full coverage on the demo model", () => {
    const r = scoreDevice(demoModel, byId("agx_orin"));
    expect(r.opSupport.coverage).toBe(1);
  });
});

describe("scoreAll ranking", () => {
  const ranked = scoreAll(demoModel, HW);

  it("returns one result per device, sorted descending by overall", () => {
    expect(ranked).toHaveLength(HW.length);
    for (let i = 1; i < ranked.length; i++) {
      expect(ranked[i - 1].overall).toBeGreaterThanOrEqual(ranked[i].overall);
    }
  });

  it("keeps every score within 0..100", () => {
    for (const r of ranked) {
      expect(r.overall).toBeGreaterThanOrEqual(0);
      expect(r.overall).toBeLessThanOrEqual(100);
    }
  });

  it("ranks a capable Jetson above a 0.5-TOPS i.MX 93 for the demo", () => {
    const agx = ranked.find((r) => r.device.id === "agx_orin")!;
    const nxp = ranked.find((r) => r.device.id === "nxp")!;
    expect(agx.overall).toBeGreaterThan(nxp.overall);
  });
});
