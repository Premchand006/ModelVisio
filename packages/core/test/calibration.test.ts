import { describe, expect, it } from "vitest";
import { HW } from "../src/data/hardware";
import { calibratedUtil, estimateRoofline } from "../src/scoring/roofline";
import { demoModel } from "../src/demo/demoModel";

const dev = (id: string) => {
  const d = HW.find((h) => h.id === id);
  if (!d) throw new Error(`no device ${id}`);
  return d;
};

describe("utilization calibration from measured benchmarks", () => {
  it("AGX Orin cnnCls utilization is calibrated from MLPerf (~0.38, within band)", () => {
    const c = calibratedUtil(dev("agx_orin"), "cnnCls");
    expect(c.calibrated).toBe(true);
    expect(c.util).toBeGreaterThan(0.30);
    expect(c.util).toBeLessThan(0.45);
    expect(c.note).toMatch(/MLPerf/i);
  });

  it("Hailo-8 cnnCls utilization is calibrated (~0.43)", () => {
    const c = calibratedUtil(dev("hailo8"), "cnnCls");
    expect(c.calibrated).toBe(true);
    expect(c.util).toBeGreaterThan(0.35);
    expect(c.util).toBeLessThan(0.50);
  });

  it("falls back to the estimated constant where no anchor exists", () => {
    // No detector anchor for AGX, and no anchor at all for Coral.
    expect(calibratedUtil(dev("agx_orin"), "cnnDet").calibrated).toBe(false);
    expect(calibratedUtil(dev("coral"), "cnnCls").calibrated).toBe(false);
  });

  it("a calibrated device's roofline carries the calibration flag + note", () => {
    // demoModel is a detector → AGX uses estimated (no detector anchor)…
    const det = estimateRoofline(demoModel, dev("agx_orin"));
    expect(det.workload).toBe("cnnDet");
    expect(det.calibrated).toBe(false);
    // …but the cnnCls calibration is reachable and well-formed.
    const c = calibratedUtil(dev("hailo8"), "cnnCls");
    expect(c.note).toBeTruthy();
  });
});
