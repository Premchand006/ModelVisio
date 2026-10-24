// Precision- and runtime-aware single-device estimate, built on the existing
// roofline scorer (scoring/scoreDevice). scoreDevice always runs a device at its
// preferred precision (pickPrecision → INT8 where available); here we pin the
// precision by presenting the scorer a device whose peak table only lists that
// precision, so memory-fit (bytes/elem), roofline compute ceiling and
// calibration all follow the requested precision. Pure module.

import type { Model } from "@modelvisio/parsers";
import type { DeviceSpec } from "../data/hardware";
import { scoreDevice, type Banner } from "../scoring";
import {
  RUNTIMES, precisionsFor, runtimeEff, runtimesFor,
  type DeployPrecision, type RuntimeId,
} from "../deploy/runtimes";

export type CompatIssue = { layer: string; target: string; severity: "warn" | "error"; msg: string };

export type DeviceEstimate = {
  device: DeviceSpec;
  precision: DeployPrecision;
  runtime: RuntimeId;
  /** False when the device can't run this precision/runtime at all. */
  supported: boolean;
  reason?: string;
  /** Peak compute at this precision is derived (not on the datasheet). */
  approx: boolean;
  latencyMs: number;
  fps: number;
  bound: "compute" | "memory";
  fits: boolean;
  footprintBytes: number;
  budgetBytes: number;
  headroom: number;
  compileFail: boolean;
  coverage: number;
  fallbackOps: number;
  conversionNeeded: boolean;
  issues: CompatIssue[];
  errors: number;
  warns: number;
  powerW: number;
  /** Energy per inference in millijoules (typical power × latency). */
  energyMj: number;
  fpsPerW: number;
  score: number;
  confidence: "high" | "medium" | "low";
  banners: Banner[];
};

/** Peak compute (TOPS/TFLOPS) at a precision, deriving it when not listed:
 *  fp16 ≈ int8/2, fp32 ≈ fp16/2 (typical NPU/GPU ratios). */
export function peakAt(dev: DeviceSpec, prec: DeployPrecision): { value: number; approx: boolean } {
  const p = dev.peak;
  if (prec === "int8") {
    if (p.int8 != null) return { value: p.int8, approx: false };
    return { value: (p.fp16 ?? p.fp32 ?? 1) * 2, approx: true };
  }
  if (prec === "fp16") {
    if (p.fp16 != null) return { value: p.fp16, approx: false };
    return { value: (p.int8 ?? (p.fp32 ?? 0.5) * 4) / 2, approx: true };
  }
  if (p.fp32 != null) return { value: p.fp32, approx: false };
  return { value: (p.fp16 ?? (p.int8 ?? 1) / 2) / 2, approx: true };
}

/** A copy of `dev` whose peak table only lists `prec`, so the scorer runs it there. */
export function pinPrecision(dev: DeviceSpec, prec: DeployPrecision): DeviceSpec {
  const { value } = peakAt(dev, prec);
  return {
    ...dev,
    peak: { [prec]: value },
    // Benchmark anchors are INT8; don't calibrate other precisions against them.
    measured: prec === "int8" ? dev.measured : undefined,
  };
}

function compatIssues(model: Model, dev: DeviceSpec, runtime: RuntimeId): CompatIssue[] {
  const targets = RUNTIMES[runtime]?.compatTargets ?? [];
  const out: CompatIssue[] = [];
  for (const l of model.layers) {
    for (const c of l.compIssues ?? []) {
      const t = c.target.toLowerCase();
      if ((dev.compatId && c.target === dev.compatId) || targets.some((x) => t.includes(x))) {
        out.push({ layer: l.name, target: c.target, severity: c.severity, msg: c.msg });
      }
    }
  }
  return out;
}

export function estimateDevice(model: Model, dev: DeviceSpec, prec: DeployPrecision, runtime?: RuntimeId): DeviceEstimate {
  const rt = runtime ?? runtimesFor(dev)[0];
  let reason: string | undefined;
  if (!precisionsFor(dev).includes(prec)) reason = `${dev.n} is INT8-only; ${prec.toUpperCase()} is not supported`;
  else if (!runtimesFor(dev).includes(rt)) reason = `${RUNTIMES[rt]?.label ?? rt} does not target ${dev.n} (use ${runtimesFor(dev).map((r) => RUNTIMES[r].label).join(" / ")})`;

  // Numbers are still computed for an unsupported combo (at the device's best
  // precision) so the UI can show what the device would do; `supported` gates it.
  const usePrec = reason && !precisionsFor(dev).includes(prec) ? precisionsFor(dev)[0] : prec;
  const { approx } = peakAt(dev, usePrec);
  const r = scoreDevice(model, pinPrecision(dev, usePrec));
  const eff = runtimeEff(dev, rt);
  const fps = r.fps * eff;
  const latencyMs = fps > 0 ? 1000 / fps : Infinity;
  const issues = compatIssues(model, dev, rt);
  const powerW = dev.power.typ;

  return {
    device: dev, precision: prec, runtime: rt,
    supported: !reason, reason, approx,
    latencyMs, fps, bound: r.roofline.bound,
    fits: r.memory.fits,
    footprintBytes: r.memory.footprintBytes,
    budgetBytes: r.memory.budgetBytes,
    headroom: r.memory.headroom,
    compileFail: r.opSupport.compileFail,
    coverage: r.opSupport.coverage,
    fallbackOps: r.opSupport.fallbackOps,
    conversionNeeded: r.opSupport.conversionNeeded,
    issues,
    errors: issues.filter((i) => i.severity === "error").length,
    warns: issues.filter((i) => i.severity === "warn").length,
    powerW,
    energyMj: isFinite(latencyMs) ? powerW * latencyMs : Infinity,
    fpsPerW: fps / powerW,
    score: r.overall,
    confidence: approx && r.confidence === "high" ? "medium" : r.confidence,
    banners: r.banners,
  };
}

/** True when the estimate is deployable at all (supported, fits, compiles). */
export function canRun(e: DeviceEstimate): boolean {
  return e.supported && e.fits && !e.compileFail && isFinite(e.latencyMs);
}
