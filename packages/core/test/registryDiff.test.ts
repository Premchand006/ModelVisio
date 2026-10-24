import { describe, expect, it } from "vitest";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { demoModel } from "../src/demo/demoModel";
import { applyFix } from "../src/fixes/transforms";
import { diffLayers, diffSnapshots, snapshotModel, type LayerFingerprint, type ModelSnapshot } from "../src/registry";

const fp = (name: string, op: string, params = 10, flops = 100, shape = "1×8"): LayerFingerprint => ({ name, op, shape, params, flops });

function layer(id: number, name: string, op: string, params: number, flops: number, compIssues: ModelLayer["compIssues"] = []): ModelLayer {
  return {
    id, name, type: op, op, shape: "1×16×32×32", dt: "float32", params, flops, macs: flops / 2, mem: 65536,
    group: id === 0 ? "input" : "backbone",
    ins: [{ n: `t${id}`, s: [1, 16, 32, 32] }], outs: [{ n: `t${id + 1}`, s: [1, 16, 32, 32] }],
    attr: {}, w: null, math: "", insight: "", qSens: 0.1, compIssues,
  };
}

function tiny(layers: ModelLayer[], name = "tiny.onnx"): Model {
  return {
    name, format: "ONNX", framework: "test", opset: 17, sizeBytes: 1000,
    inputShape: [1, 16, 32, 32], outputShape: [1, 16, 32, 32],
    layers, edges: layers.slice(1).map((l) => [l.id - 1, l.id] as [number, number]),
  };
}

describe("diffLayers", () => {
  it("matches by name+op and reports added / removed / changed", () => {
    const a = [fp("conv1", "Conv"), fp("act1", "Relu"), fp("fc", "Gemm", 40)];
    const b = [fp("conv1", "Conv"), fp("act1", "HardSwish"), fp("fc", "Gemm", 80), fp("extra", "Add")];
    const d = diffLayers(a, b);
    expect(d.unchanged).toBe(1);
    expect(d.added.map((l) => l.name).sort()).toEqual(["act1", "extra"]); // op change = remove + add
    expect(d.removed.map((l) => `${l.name}:${l.op}`)).toEqual(["act1:Relu"]);
    expect(d.changed).toHaveLength(1);
    expect(d.changed[0]).toMatchObject({ name: "fc", fields: ["params"] });
  });

  it("pairs duplicate names in order", () => {
    const d = diffLayers([fp("x", "Add"), fp("x", "Add")], [fp("x", "Add")]);
    expect(d.unchanged).toBe(1);
    expect(d.removed).toHaveLength(1);
  });
});

describe("diffSnapshots", () => {
  it("identical models: no regressions, zero deltas", async () => {
    const m = tiny([layer(0, "in", "Input", 0, 0), layer(1, "conv", "Conv", 2304, 4_718_592)]);
    const bytes = new Uint8Array([1, 2, 3]);
    const a = await snapshotModel(m, { bytes });
    const b = await snapshotModel(m, { bytes });
    const d = diffSnapshots(a, b);
    expect(d.identical).toBe(true);
    expect(d.regressions).toEqual([]);
    expect(d.stats.params.delta).toBe(0);
    expect(d.layers.unchanged).toBe(2);
    expect(d.compat.every((c) => c.status === "unchanged")).toBe(true);
  });

  it("flags new compat errors per target as regressions and fixes as improvements", async () => {
    const base = [layer(0, "in", "Input", 0, 0), layer(1, "conv", "Conv", 2304, 4_718_592)];
    const a = await snapshotModel(tiny(base));
    const b = await snapshotModel(tiny([
      ...base,
      layer(2, "gelu", "Gelu", 0, 16384, [{ target: "coral_tpu", severity: "error", msg: "GELU unsupported on Edge TPU" }]),
    ]));
    const d = diffSnapshots(a, b);
    const coral = d.compat.find((c) => c.target === "coral")!;
    expect(coral.status).toBe("regression");
    expect(coral.newIssues).toContain("gelu: GELU unsupported on Edge TPU");
    expect(d.compat.find((c) => c.target === "hailo")!.status).toBe("unchanged");
    expect(d.regressions.some((r) => r.kind === "compat" && r.target === "coral" && r.severity === "error")).toBe(true);
    expect(d.layers.added.map((l) => l.name)).toEqual(["gelu"]);

    const back = diffSnapshots(b, a);
    expect(back.compat.find((c) => c.target === "coral")!.status).toBe("improvement");
    expect(back.regressions.filter((r) => r.kind === "compat")).toEqual([]);
  });

  it("reports latency regressions beyond the tolerance and param/FLOP deltas", async () => {
    const small = tiny([layer(0, "in", "Input", 0, 0), layer(1, "conv", "Conv", 2304, 4_718_592)]);
    const big = tiny([layer(0, "in", "Input", 0, 0), layer(1, "conv", "Conv", 2304 * 50, 4_718_592 * 400)]);
    const a = await snapshotModel(small);
    const b = await snapshotModel(big);
    const d = diffSnapshots(a, b);
    expect(d.stats.params.delta).toBe(2304 * 49);
    expect(d.stats.params.pct).toBeCloseTo(4900);
    expect(d.layers.changed[0].fields).toEqual(["params", "flops"]);
    expect(d.latency.length).toBeGreaterThan(0);
    expect(d.latency.some((l) => l.status === "regression")).toBe(true);
    expect(d.regressions.some((r) => r.kind === "latency")).toBe(true);
    // A huge tolerance suppresses latency regressions.
    expect(diffSnapshots(a, b, { latencyTolerancePct: 1e9 }).regressions.filter((r) => r.kind === "latency")).toEqual([]);
  });

  it("size regression only when a tolerance is given", () => {
    const s = (sizeBytes: number): ModelSnapshot => ({
      fileName: "m.onnx", sizeBytes, sha256: "", format: "ONNX", framework: "x", opset: 17, inputShape: [], outputShape: [],
      stats: { layers: 0, params: 0, flops: 0, macs: 0, memBytes: 0 }, layers: [], compat: [], estimates: [],
    });
    expect(diffSnapshots(s(100), s(200)).regressions).toEqual([]);
    expect(diffSnapshots(s(100), s(200), { sizeTolerancePct: 10 }).regressions[0].kind).toBe("size");
    expect(diffSnapshots(s(100), s(200)).identical).toBe(false); // empty hashes never count as identical
  });

  it("demo model: applying the SiLU fix shows as a compat improvement", async () => {
    const before = await snapshotModel(demoModel);
    const after = await snapshotModel(applyFix(demoModel, "silu_hardswish"));
    const d = diffSnapshots(before, after);
    const rk = d.compat.find((c) => c.target === "rk3588")!;
    expect(rk.status).toBe("improvement");
    expect(rk.resolvedIssues.length).toBeGreaterThan(0);
    expect(d.layers.removed.length + d.layers.changed.length).toBeGreaterThan(0);
  });
});
