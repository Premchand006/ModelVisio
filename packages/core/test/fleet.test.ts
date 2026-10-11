import { describe, expect, it } from "vitest";
import { demoModel } from "../src/demo/demoModel";
import { HW } from "../src/data/hardware";
import {
  queueStats, responseCdf, sloAttainment, mixtureQuantile, mulberry32, histogram,
  simulateFleet, fleetResultToCsv, fleetResultToJson, type FleetEntry,
} from "../src/fleet/simulate";
import { estimateDevice, peakAt, pinPrecision } from "../src/fleet/estimate";
import { FLEET_PRESETS, presetEntries } from "../src/fleet/presets";
import { runtimesFor, precisionsFor } from "../src/deploy/runtimes";

const dev = (id: string) => HW.find((d) => d.id === id)!;

describe("queueing math", () => {
  it("matches the closed-form M/M/1 and M/D/1 mean waits", () => {
    // S = 10 ms, λ = 50/s → ρ = 0.5
    const mm1 = queueStats(10, 50, "mm1");
    const md1 = queueStats(10, 50, "md1");
    expect(mm1.rho).toBeCloseTo(0.5);
    expect(mm1.meanWaitMs).toBeCloseTo((0.5 * 10) / (1 - 0.5)); // ρS/(1-ρ) = 10
    expect(md1.meanWaitMs).toBeCloseTo((0.5 * 10) / (2 * (1 - 0.5))); // ρS/(2(1-ρ)) = 5
    expect(md1.meanMs).toBeCloseTo(15);
  });

  it("p95 is consistent with the response CDF", () => {
    for (const [s, l] of [[10, 50], [5, 150], [20, 10]] as const) {
      const q = queueStats(s, l, "md1");
      expect(responseCdf(q.p95Ms, s, l, "md1")).toBeCloseTo(0.95, 6);
    }
  });

  it("has no wait at negligible load and grows monotonically with utilization", () => {
    expect(queueStats(10, 1).p95WaitMs).toBe(0); // ρ = 0.01
    const p = [10, 40, 70, 90, 99].map((l) => queueStats(10, l).p95Ms);
    for (let i = 1; i < p.length; i++) expect(p[i]).toBeGreaterThan(p[i - 1]);
  });

  it("is unstable at or beyond saturation", () => {
    const q = queueStats(10, 100);
    expect(q.stable).toBe(false);
    expect(q.p95Ms).toBe(Infinity);
    expect(sloAttainment(1000, 10, 120)).toBe(0);
  });

  it("SLO attainment: 0 below the service time, → 1 for generous SLOs", () => {
    expect(sloAttainment(5, 10, 10)).toBe(0);
    expect(sloAttainment(1000, 10, 10)).toBeCloseTo(1, 6);
  });

  it("mixture quantile of identical components equals the component p95", () => {
    const q = queueStats(10, 50);
    const m = mixtureQuantile([{ sMs: 10, lambda: 50, weight: 3 }, { sMs: 10, lambda: 50, weight: 7 }], 0.95);
    expect(m).toBeCloseTo(q.p95Ms, 4);
    expect(mixtureQuantile([{ sMs: 10, lambda: 200, weight: 1 }], 0.95)).toBe(Infinity);
  });
});

describe("seeded RNG + histogram", () => {
  it("mulberry32 is deterministic per seed and in [0,1)", () => {
    const a = mulberry32(7), b = mulberry32(7), c = mulberry32(8);
    const xa = Array.from({ length: 5 }, a), xb = Array.from({ length: 5 }, b), xc = Array.from({ length: 5 }, c);
    expect(xa).toEqual(xb);
    expect(xa).not.toEqual(xc);
    expect(xa.every((x) => x >= 0 && x < 1)).toBe(true);
  });

  it("histogram counts every finite value", () => {
    const h = histogram([1, 2, 3, 4, Infinity, 5], 4);
    expect(h).toHaveLength(4);
    expect(h.reduce((s, b) => s + b.count, 0)).toBe(5);
    expect(histogram([2, 2, 2])).toEqual([{ lo: 2, hi: 2, count: 3 }]);
  });
});

describe("precision-aware estimate", () => {
  it("pins the scorer to the requested precision", () => {
    expect(pinPrecision(dev("orin_nano"), "fp16").peak).toEqual({ fp16: 1.3 });
    expect(peakAt(dev("rk3588"), "fp16")).toEqual({ value: 3, approx: true });
    const i8 = estimateDevice(demoModel, dev("orin_nano"), "int8");
    const f16 = estimateDevice(demoModel, dev("orin_nano"), "fp16");
    expect(i8.fps).toBeGreaterThan(f16.fps);
    expect(f16.footprintBytes).toBeGreaterThan(i8.footprintBytes);
    expect(i8.energyMj).toBeCloseTo(i8.powerW * i8.latencyMs);
  });

  it("rejects precisions/runtimes a device can't run", () => {
    expect(estimateDevice(demoModel, dev("coral"), "fp16").supported).toBe(false);
    expect(estimateDevice(demoModel, dev("hailo8"), "int8", "tensorrt").supported).toBe(false);
    expect(precisionsFor(dev("hailo8"))).toEqual(["int8"]);
    expect(runtimesFor(dev("orin_nano"))[0]).toBe("tensorrt");
  });

  it("surfaces runtime compat issues from the model graph", () => {
    const e = estimateDevice(demoModel, dev("coral"), "int8");
    expect(e.issues.length).toBeGreaterThan(0);
  });
});

describe("simulateFleet", () => {
  const fleet: FleetEntry[] = [
    { id: "a", deviceId: "orin_nano", count: 100, runtime: "tensorrt", precision: "int8", targetFps: 30, thermal: "passive" },
    { id: "b", deviceId: "coral", count: 50, runtime: "tflite-edgetpu", precision: "int8", targetFps: 30 },
    { id: "c", deviceId: "nope", count: 5, runtime: "vendor", precision: "int8" },
  ];

  it("computes per-class and fleet aggregates", () => {
    const r = simulateFleet(demoModel, fleet);
    expect(r.classes).toHaveLength(3);
    expect(r.aggregate.totalDevices).toBe(155);
    expect(r.classes[2].canRun).toBe(false);
    expect(r.classes[2].reasons[0]).toMatch(/Unknown device/);
    expect(r.aggregate.cantRunDevices).toBeGreaterThanOrEqual(5);
    const a = r.classes[0];
    expect(a.utilization).toBeCloseTo((30 * a.latencyMs) / 1000);
    expect(a.sloMs).toBeCloseTo(1000 / 30);
    expect(r.aggregate.pctMeetingSlo).toBeGreaterThanOrEqual(0);
    expect(r.aggregate.pctMeetingSlo).toBeLessThanOrEqual(100);
    expect(r.aggregate.bottleneck).not.toBeNull();
    expect(r.monteCarlo).toBeNull();
  });

  it("thermal and power profiles slow a class down", () => {
    const base: FleetEntry = { id: "x", deviceId: "orin_nano", count: 1, runtime: "tensorrt", precision: "int8" };
    const cool = simulateFleet(demoModel, [base]).classes[0];
    const hot = simulateFleet(demoModel, [{ ...base, thermal: "hot", powerMode: "low" }]).classes[0];
    expect(hot.latencyMs).toBeGreaterThan(cool.latencyMs * 1.5);
    expect(hot.powerW).toBeLessThan(cool.powerW);
  });

  it("Monte-Carlo is deterministic for a seed and differs across seeds", () => {
    const opts = { monteCarlo: { trials: 60, seed: 123, samplesPerClass: 16 } };
    const r1 = simulateFleet(demoModel, fleet, opts).monteCarlo!;
    const r2 = simulateFleet(demoModel, fleet, opts).monteCarlo!;
    const r3 = simulateFleet(demoModel, fleet, { monteCarlo: { ...opts.monteCarlo, seed: 124 } }).monteCarlo!;
    expect(r1.fleetP95).toEqual(r2.fleetP95);
    expect(r1.fleetP95).not.toEqual(r3.fleetP95);
    expect(r1.trials).toBe(60);
    expect(r1.p95Quantiles.p5).toBeLessThanOrEqual(r1.p95Quantiles.p95);
    expect(r1.histogram.reduce((s, b) => s + b.count, 0)).toBe(60 - r1.unstableTrials);
  });

  it("an overloaded class is unstable and misses its SLO", () => {
    const r = simulateFleet(demoModel, [{ id: "o", deviceId: "kneron", count: 10, runtime: "vendor", precision: "int8", requestRate: 10_000 }]);
    expect(r.classes[0].stable).toBe(false);
    expect(r.classes[0].meetsSlo).toBe(false);
  });

  it("exports JSON (Infinity-safe) and CSV", () => {
    const r = simulateFleet(demoModel, fleet, { monteCarlo: { trials: 10, seed: 1 } });
    const j = JSON.parse(fleetResultToJson(r, "m.onnx"));
    expect(j.classes).toHaveLength(3);
    expect(j.classes[2].p95Ms).toBeNull();
    const csv = fleetResultToCsv(r).trim().split("\n");
    expect(csv).toHaveLength(4);
    expect(csv[0]).toMatch(/^id,device,deviceId,count/);
  });
});

describe("fleet presets", () => {
  it("only use devices and runtimes/precisions that exist for them", () => {
    for (const p of FLEET_PRESETS) {
      for (const e of presetEntries(p)) {
        const d = HW.find((x) => x.id === e.deviceId);
        expect(d, `${p.id}: ${e.deviceId}`).toBeDefined();
        expect(runtimesFor(d!)).toContain(e.runtime);
        expect(precisionsFor(d!)).toContain(e.precision);
      }
      expect(simulateFleet(demoModel, presetEntries(p)).aggregate.totalDevices).toBeGreaterThan(0);
    }
  });
});
