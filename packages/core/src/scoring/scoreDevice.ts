// scoreDevice(model, device) — a defensible 0-100 model→hardware fit score.
//
// The number is COMPUTED from the loaded model, not hand-assigned. It folds four
// sub-scores under a deployment profile, with two hard overrides:
//   • memory-fit hard-fail — a model that doesn't fit in device memory can never
//     show green, regardless of compute headroom (the highest-value guard);
//   • compile-fail — an unsupported core op on a no-fallback part caps the score.
//
// Sub-scores: latency (roofline FPS), memoryFit, format/op-support (incl. the
// Coral single-partition CPU-fallback rule), and efficiency (FPS/W, FPS/$).
//
// Pure module — no React.

import type { Model } from "@modelvisio/parsers";
import { BYTES, type DeviceSpec } from "../data/hardware";
import { estimateRoofline, modelCompute, peakActivationBytes, pickPrecision, type Roofline } from "./roofline";

export type Profile = "throughput" | "power" | "cost";

export type SubScores = { latency: number; memory: number; opSupport: number; efficiency: number };

export type Banner = { level: "error" | "warn" | "info"; msg: string };

export type ScoreResult = {
  device: DeviceSpec;
  overall: number;
  sub: SubScores;
  weights: SubScores;          // the profile weights actually used (for the UI breakdown)
  roofline: Roofline;
  fps: number;
  latencyMs: number;
  fpsPerW: number;
  fpsPerDollar?: number;
  memory: { footprintBytes: number; budgetBytes: number; fits: boolean; headroom: number; hardFail: boolean };
  opSupport: { coverage: number; fallbackOps: number; conversionNeeded: boolean; compileFail: boolean; quantRisk: number };
  banners: Banner[];
  confidence: "high" | "medium" | "low";
  precision: string;
};

const PROFILES: Record<Profile, SubScores> = {
  throughput: { latency: 0.40, memory: 0.25, opSupport: 0.20, efficiency: 0.15 },
  power: { latency: 0.30, memory: 0.25, opSupport: 0.15, efficiency: 0.30 },
  cost: { latency: 0.30, memory: 0.25, opSupport: 0.20, efficiency: 0.25 },
};

const clamp = (n: number, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, n));
const log10 = (n: number) => Math.log(Math.max(n, 1e-9)) / Math.LN10;
const log2 = (n: number) => Math.log(Math.max(n, 1e-9)) / Math.LN2;

/** FPS → 0-100. Log curve anchored on edge real-time targets, deliberately
 *  non-saturating so capable devices still spread apart on easy models:
 *  ~1 fps→35, ~10→55, ~30 (real-time)→65, ~100→75, ~1000→95, ≥~3000→100. */
function latencyScore(fps: number): number {
  return clamp(Math.round(35 + 20 * log10(fps)), 2, 100);
}

/** FPS/W → 0-100. Anchored on Hailo-class efficiency (~50 FPS/W ≈ excellent). */
function efficiencyScore(fpsPerW: number, fpsPerDollar?: number): number {
  let s = 55 + 30 * log10(fpsPerW);
  if (fpsPerDollar != null) s = 0.7 * s + 0.3 * (55 + 30 * log10(fpsPerDollar * 100)); // $ in cents-scale
  return clamp(Math.round(s));
}

export function scoreDevice(model: Model, dev: DeviceSpec, profile: Profile = "throughput"): ScoreResult {
  const prec = pickPrecision(dev);
  const bpe = BYTES[prec];
  const mc = modelCompute(model);
  const roofline = estimateRoofline(model, dev);
  const banners: Banner[] = [];

  // ── Memory fit ──────────────────────────────────────────────────────────────
  const weightBytes = mc.totalParams * bpe;
  const actBytes = peakActivationBytes(model, prec);
  const footprintBytes = (weightBytes + actBytes) * 1.1; // +10% runtime overhead
  const onchip = dev.memType === "onchip";
  const budgetBytes = onchip
    ? (dev.sramMB ?? 0) * 1e6
    : dev.ramGB * 1e9;

  let memScore: number;
  let hardFail = false;
  if (budgetBytes <= 0) {
    memScore = 40; // unknown budget — neutral, low confidence handled below
  } else {
    const headroom = budgetBytes / footprintBytes;
    if (headroom < 1) {
      if (onchip && dev.externalMem) {
        // Spills to external/host memory: heavy throughput penalty, not a hard fail.
        memScore = clamp(Math.round(25 * headroom));
        banners.push({ level: "warn", msg: `Model (${fmtB(footprintBytes)}) exceeds on-chip budget (${fmtB(budgetBytes)}) — spills to host, severe slowdown` });
      } else {
        hardFail = true;
        memScore = clamp(Math.round(15 * headroom));
        banners.push({ level: "error", msg: `Model footprint ${fmtB(footprintBytes)} > device memory ${fmtB(budgetBytes)} — cannot deploy` });
      }
    } else {
      memScore = clamp(Math.round(50 + 40 * log2(headroom)));
    }
  }
  const headroom = budgetBytes > 0 ? budgetBytes / footprintBytes : 0;

  // ── Format & op support (+ CPU-fallback class) ───────────────────────────────
  const fmtSupported = dev.fmts.includes(model.format);
  // Reachable from a mainstream trainable/exchange format via the device toolchain?
  const convertible = /onnx|tflite|pytorch|tensorflow|keras|torchscript|safetensors/i.test(model.format);
  const conversionNeeded = !fmtSupported;

  const issues = dev.compatId
    ? model.layers.flatMap((l) => l.compIssues.filter((c) => c.target === dev.compatId).map((c) => ({ ...c, layer: l })))
    : [];
  const errLayers = issues.filter((i) => i.severity === "error").map((i) => i.layer);
  const warnCount = issues.filter((i) => i.severity === "warn").length;
  const totalOps = mc.totalOps || 1;

  let coverage = 1;
  let fallbackOps = 0;
  let compileFail = false;

  if (errLayers.length > 0) {
    if (dev.cpuFallback === "none") {
      compileFail = true;
      coverage = 0;
      banners.push({ level: "error", msg: `${errLayers.length} unsupported op(s); no CPU fallback — model will not compile` });
    } else if (dev.cpuFallback === "single-partition") {
      // Everything from the FIRST unsupported op to the end runs on the host CPU.
      const firstErrId = Math.min(...errLayers.map((l) => l.id));
      const tail = model.layers.filter((l) => l.id >= firstErrId);
      const tailOps = tail.reduce((s, l) => s + (l.flops || l.macs * 2), 0);
      coverage = clamp(1 - tailOps / totalOps, 0, 1);
      fallbackOps = tail.length;
      banners.push({ level: "warn", msg: `${tail.length} ops fall back to host CPU after the first unsupported op (~${(1 / Math.max(coverage, 0.05)).toFixed(0)}× slowdown)` });
    } else {
      // graceful: only the unsupported ops fall back, weighted by their compute.
      const errOps = errLayers.reduce((s, l) => s + (l.flops || l.macs * 2), 0);
      coverage = clamp(1 - errOps / totalOps, 0, 1);
      fallbackOps = errLayers.length;
      banners.push({ level: "warn", msg: `${errLayers.length} op(s) fall back to CPU on this target` });
    }
  }

  let opScore = 100 * coverage;
  opScore -= warnCount * 3;                          // soft per-warning penalty
  if (conversionNeeded) {
    opScore -= convertible ? 12 : 40;
    banners.push({ level: convertible ? "info" : "warn", msg: convertible ? `Needs conversion from ${model.format} to ${dev.toolchain[0]}` : `${model.format} not supported by ${dev.toolchain[0]}` });
  }

  // Quantization gate: INT8-only parts risk accuracy on sensitive layers.
  const avgQSens = model.layers.length ? model.layers.reduce((s, l) => s + l.qSens, 0) / model.layers.length : 0;
  let quantRisk = 0;
  if (dev.quant.int8Only && avgQSens > 0.06) {
    quantRisk = clamp(Math.round(avgQSens * 100));
    const pen = Math.min(15, avgQSens * 120);
    opScore -= pen;
    const sensCount = model.layers.filter((l) => l.qSens > 0.1).length;
    if (sensCount > 0) banners.push({ level: "info", msg: `INT8-only: ${sensCount} quantization-sensitive layer(s) — validate accuracy` });
  }
  opScore = clamp(Math.round(opScore));

  // ── Latency & efficiency ─────────────────────────────────────────────────────
  // If the single-partition tail runs on CPU, the device FPS estimate must drop.
  const effFps = roofline.fps * (dev.cpuFallback === "single-partition" && errLayers.length ? coverage : 1);
  const latScore = latencyScore(effFps);
  const fpsPerW = effFps / dev.power.typ;
  const fpsPerDollar = dev.priceUSD ? effFps / dev.priceUSD : undefined;
  const effScore = efficiencyScore(fpsPerW, fpsPerDollar);

  // ── Weighted overall + hard overrides ────────────────────────────────────────
  const w = PROFILES[profile];
  const sub: SubScores = { latency: latScore, memory: memScore, opSupport: opScore, efficiency: effScore };
  let overall = Math.round(sub.latency * w.latency + sub.memory * w.memory + sub.opSupport * w.opSupport + sub.efficiency * w.efficiency);
  if (hardFail) overall = Math.min(overall, 15);
  if (compileFail) overall = Math.min(overall, 10);
  overall = clamp(overall);

  // ── Confidence ───────────────────────────────────────────────────────────────
  let confidence: ScoreResult["confidence"];
  if (dev.conf === "M" || dev.conf === "V") confidence = "high";
  else if (dev.conf === "B") confidence = "medium";
  else confidence = "low";
  // The roofline leans on bandwidth; an estimated bandwidth on a memory-bound
  // workload demotes confidence (per nn-Meter: NPU/VPU prediction is least accurate).
  if (dev.bwConf === "?" && roofline.bound === "memory" && confidence === "high") confidence = "medium";
  if (dev.bwConf === "?" && roofline.bound === "memory" && confidence === "medium") confidence = "low";

  return {
    device: dev, overall, sub, weights: w, roofline,
    fps: effFps, latencyMs: effFps > 0 ? 1000 / effFps : Infinity,
    fpsPerW, fpsPerDollar,
    memory: { footprintBytes, budgetBytes, fits: !hardFail, headroom, hardFail },
    opSupport: { coverage, fallbackOps, conversionNeeded, compileFail, quantRisk },
    banners, confidence, precision: prec,
  };
}

export function scoreAll(model: Model, devices: DeviceSpec[], profile: Profile = "throughput"): ScoreResult[] {
  return devices.map((d) => scoreDevice(model, d, profile)).sort((a, b) => b.overall - a.overall);
}

function fmtB(b: number): string {
  if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
  return Math.round(b) + " B";
}
