import { describe, expect, it } from "vitest";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { ruleBasedReport } from "@modelvisio/ai";
import { demoModel } from "../src/demo/demoModel";
import { HW, type DeviceSpec } from "../src/data/hardware";
import { estimateRoofline } from "../src/scoring/roofline";
import { layerRoofline } from "../src/investigate/layerRoofline";
import { buildEvidence, fixesResolving } from "../src/investigate/evidence";

const dev = (id: string): DeviceSpec => {
  const d = HW.find((h) => h.id === id);
  if (!d) throw new Error(`no device ${id}`);
  return d;
};

function layer(p: Partial<ModelLayer>): ModelLayer {
  return {
    id: 0, name: "l", type: "Conv2d", op: "Conv", shape: "", dt: "float32",
    params: 0, flops: 0, macs: 0, mem: 0, group: "backbone",
    ins: [], outs: [], attr: {}, w: null, math: "", insight: "", qSens: 0, compIssues: [],
    ...p,
  };
}
function model(layers: ModelLayer[], over: Partial<Model> = {}): Model {
  const ls = layers.map((l, i) => ({ ...l, id: i, name: l.name === "l" ? `l${i}` : l.name }));
  return {
    name: "synthetic.onnx", format: "ONNX", framework: "test", opset: 17, sizeBytes: 1e6,
    inputShape: [1, 3, 224, 224], outputShape: [1, 1000], layers: ls,
    edges: ls.slice(1).map((l, i) => [i, l.id] as [number, number]), ...over,
  };
}

describe("layerRoofline", () => {
  it("sums to exactly the scoring engine's latency at the default precision", () => {
    for (const id of ["agx_orin", "coral", "hailo8", "rk3588", "nxp"]) {
      const r = layerRoofline(demoModel, dev(id));
      const ref = estimateRoofline(demoModel, dev(id));
      expect(r.latencyMs).toBeCloseTo(ref.latencyMs, 9);
      expect(r.layers).toHaveLength(demoModel.layers.length);
      const sum = r.layers.reduce((s, l) => s + l.timeMs, 0) + r.dispatchMs;
      expect(sum).toBeCloseTo(r.latencyMs, 9);
    }
  });
  it("classifies layers against the ridge point", () => {
    const r = layerRoofline(demoModel, dev("agx_orin"));
    for (const l of r.layers) if (l.bytes > 0 && l.flops > 0) expect(l.bound).toBe(l.ai >= r.ridgeAI ? "compute" : "memory");
  });
});

describe("buildEvidence", () => {
  it("produces sequential, unique ids and only real layer names", () => {
    const p = buildEvidence(demoModel, dev("coral"));
    const names = new Set(demoModel.layers.map((l) => l.name));
    p.evidence.forEach((e, i) => expect(e.id).toBe(`E${i + 1}`));
    for (const e of p.evidence) for (const l of e.layers) expect(names.has(l)).toBe(true);
    for (const b of p.bottlenecks) {
      expect(names.has(b.layer)).toBe(true);
      expect(p.evidence.find((e) => e.id === b.id)?.kind).toBe("bottleneck");
    }
    expect(p.evidence[0].kind).toBe("summary");
  });
  it("ranks bottlenecks by time share and keeps shares sane", () => {
    const p = buildEvidence(demoModel, dev("agx_orin"), { topK: 5 });
    expect(p.bottlenecks).toHaveLength(5);
    for (let i = 1; i < p.bottlenecks.length; i++) expect(p.bottlenecks[i - 1].timeMs).toBeGreaterThanOrEqual(p.bottlenecks[i].timeMs);
    const total = p.bottlenecks.reduce((s, b) => s + b.share, 0);
    expect(total).toBeGreaterThan(0);
    expect(total).toBeLessThanOrEqual(1);
    expect(p.totals.computeBoundShare + p.totals.memoryBoundShare).toBeLessThanOrEqual(1 + 1e-9);
  });
  it("surfaces device compiler errors with the auto-fix that resolves them", () => {
    const p = buildEvidence(demoModel, dev("coral"));
    const comp = p.evidence.filter((e) => e.kind === "compiler" && e.severity === "error");
    expect(comp.length).toBeGreaterThan(0);
    expect(comp.some((e) => (e.fixIds ?? []).includes("sppf_parallel"))).toBe(true);
    // No compatId → no per-op compiler evidence.
    expect(buildEvidence(demoModel, dev("agx_orin")).evidence.some((e) => e.kind === "compiler")).toBe(false);
  });
  it("fixesResolving only lists fixes that drop an issue for that target", () => {
    const sppf = demoModel.layers.find((l) => /sppf/i.test(l.type))!;
    expect(fixesResolving(sppf, "coral_tpu")).toEqual(["sppf_parallel"]);
    expect(fixesResolving(sppf, "nonexistent")).toEqual([]);
  });
  it("the rule-based report on the demo + Coral leads with the unsupported ops", () => {
    const r = ruleBasedReport(buildEvidence(demoModel, dev("coral")));
    expect(r.hypotheses[0].title).toMatch(/Unsupported ops/);
    expect(r.hypotheses[0].fixIds).toContain("sppf_parallel");
  });

  it("detects unusual patterns", () => {
    const m = model([
      layer({ name: "in", type: "Input", op: "Placeholder", outs: [{ n: "x", s: [1, 3, 224, 224] }], group: "input" }),
      layer({ name: "conv", op: "Conv", params: 864, flops: 2 * 864 * 112 * 112, macs: 864 * 112 * 112, ins: [{ n: "x", s: [1, 3, 224, 224] }], outs: [{ n: "c", s: [1, 32, 112, 112] }] }),
      layer({ name: "bn", op: "BatchNormalization", type: "BatchNorm2d", params: 64, flops: 4 * 32 * 112 * 112, ins: [{ n: "c", s: [1, 32, 112, 112] }], outs: [{ n: "b", s: [1, 32, 112, 112] }] }),
      layer({ name: "dw", op: "Conv", attr: { group: 32 }, params: 288, flops: 2 * 288 * 112 * 112, ins: [{ n: "b", s: [1, 32, 112, 112] }], outs: [{ n: "d", s: [1, 32, 112, 112] }] }),
      layer({ name: "t1", op: "Transpose", ins: [{ n: "d", s: [1, 32, 112, 112] }], outs: [{ n: "t1", s: [1, 112, 112, 32] }] }),
      layer({ name: "r1", op: "Reshape", ins: [{ n: "t1", s: [1, 112, 112, 32] }], outs: [{ n: "r1", s: [1, 12544, 32] }] }),
      layer({ name: "t2", op: "Transpose", ins: [{ n: "r1", s: [1, 12544, 32] }], outs: [{ n: "t2", s: [1, 32, 12544] }] }),
      layer({ name: "sm", op: "Softmax", ins: [{ n: "t2", s: [1, 4, 1024, 1024] }], outs: [{ n: "sm", s: [1, 4, 1024, 1024] }], flops: 5 * 4 * 1024 * 1024 }),
      layer({ name: "dyn", op: "Add", ins: [{ n: "sm", s: [-1, 4, 1024, 1024] }], outs: [{ n: "dy", s: [-1, 4, 1024, 1024] }] }),
      layer({ name: "fc", op: "Gemm", params: 4_000_000, flops: 8_000_000, ins: [{ n: "dy", s: [1, 4000] }], outs: [{ n: "y", s: [1, 1000] }] }),
    ]);
    const p = buildEvidence(m, dev("agx_orin"));
    const tag = (t: string) => p.evidence.find((e) => e.kind === "pattern" && e.tag === t);
    expect(tag("huge-fc")?.layers).toEqual(["fc"]);
    expect(tag("unfused-bn")?.layers).toEqual(["bn"]);
    expect(tag("depthwise")?.layers).toEqual(["dw"]);
    expect(tag("layout")?.layers).toEqual(["t1", "r1", "t2"]);
    expect(tag("attention")?.layers).toEqual(["sm"]);
    expect(tag("dynamic")?.layers).toEqual(["dyn"]);
    // And the fallback turns them into hypotheses citing those items.
    const r = ruleBasedReport(p);
    const cited = new Set(r.hypotheses.flatMap((h) => h.evidence));
    expect(cited.has(tag("huge-fc")!.id) || cited.has(tag("layout")!.id)).toBe(true);
  });
  it("does not flag patterns on a clean model", () => {
    const m = model([
      layer({ name: "in", type: "Input", op: "Placeholder", outs: [{ n: "x", s: [1, 3, 224, 224] }] }),
      layer({ name: "conv", op: "Conv", params: 864, flops: 2e7, ins: [{ n: "x", s: [1, 3, 224, 224] }], outs: [{ n: "c", s: [1, 32, 112, 112] }] }),
    ]);
    expect(buildEvidence(m, dev("agx_orin")).evidence.filter((e) => e.kind === "pattern")).toEqual([]);
  });
});
