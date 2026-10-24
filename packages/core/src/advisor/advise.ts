// Hardware-aware model advisor (pure, no React).
//
// For one target device it combines:
//   • a per-layer, per-precision roofline (same model as scoring/roofline.ts:
//     time = max(ops / peak_p, bytes_p / bandwidth), de-rated by the calibrated
//     utilization, plus the fixed dispatch overhead),
//   • the quantization-sensitivity analysis (quant/sensitivity.ts),
//   • the compiler pre-flight issues and auto-fixes (fixes/transforms.ts),
//   • the device-fit score (scoring/scoreDevice.ts),
// and returns concrete, ranked recommendations with estimated latency / size /
// accuracy-risk deltas. Every number is an analytical estimate, not a benchmark.

import type { Model, ModelLayer } from "@modelvisio/parsers";
import { BYTES, type DeviceSpec, type Precision } from "../data/hardware";
import { calibratedUtil, classifyWorkload, pickPrecision, type WorkloadClass } from "../scoring/roofline";
import { scoreDevice, type ScoreResult } from "../scoring/scoreDevice";
import { FIXES, applyFix, type FixId } from "../fixes/transforms";
import { analyzeQuantization, type LayerSensitivity, type QuantAnalysis } from "../quant/sensitivity";

export type Confidence = "high" | "medium" | "low";
export type AccuracyRisk = "none" | "low" | "medium" | "high";
export type AdviceKind =
  | "format" | "memory-fit" | "quantize" | "op-support" | "resolution"
  | "memory-bound" | "fusion" | "prune" | "int4";

export type Advice = {
  id: string;
  kind: AdviceKind;
  title: string;
  rationale: string;
  impact: {
    /** old latency / new latency (> 1 is faster). */
    speedup?: number;
    /** % change in latency (negative is faster). */
    latencyPct?: number;
    /** % change in weight storage vs the FP32 model (negative is smaller). */
    sizePct?: number;
    /** % change in FLOPs. */
    flopsPct?: number;
    accuracyRisk: AccuracyRisk;
  };
  confidence: Confidence;
  /** Must be done before the model can run on this device at all. */
  required?: boolean;
  layers?: string[];
  layerIds?: number[];
  /** Ranking key (higher first). */
  score: number;
};

export type LayerBottleneck = {
  id: number;
  name: string;
  op: string;
  precision: Precision;
  ops: number;
  bytes: number;
  /** Arithmetic intensity, FLOP/byte (Infinity when no bytes move). */
  intensity: number;
  bound: "compute" | "memory" | "idle";
  /** Estimated time on the device (ms, after utilization de-rating). */
  ms: number;
  /** Fraction of the model's estimated time. */
  share: number;
};

export type Perf = {
  label: string;
  latencyMs: number;
  fps: number;
  /** Weight storage at this plan's precisions (bytes). */
  sizeBytes: number;
  /** Fraction of time spent in memory-bound layers. */
  memBoundShare: number;
};

export type AdvisorResult = {
  device: DeviceSpec;
  workload: WorkloadClass;
  /** Precision the device's integer path runs (int8 for every device in the DB). */
  deployPrecision: Precision;
  /** Float baseline (FP16/FP32), or null when the accelerator has no float path. */
  baseline: Perf | null;
  /** The sensitivity-driven mixed-precision plan (all-INT8 on INT8-only parts). */
  plan: Perf;
  /** Everything INT8 regardless of sensitivity. */
  fullInt8: Perf;
  /** Ridge point at the deploy precision, FLOP/byte. */
  ridgeAI: number;
  layers: LayerBottleneck[];
  advice: Advice[];
  quant: QuantAnalysis;
  score: ScoreResult;
  notes: string[];
};

// ── Per-precision roofline ────────────────────────────────────────────────────

type FloatPath = { prec: Precision; opsPerSec: number; assumed: boolean };

/** The device's float path, if any. When an NPU that is not INT8-only lists no
 *  float peak, FP16 is assumed at a quarter of the INT8 rate (flagged). */
function floatPath(dev: DeviceSpec): FloatPath | null {
  if (dev.peak.fp16 != null) return { prec: "fp16", opsPerSec: dev.peak.fp16 * 1e12, assumed: false };
  if (dev.peak.bf16 != null) return { prec: "bf16", opsPerSec: dev.peak.bf16 * 1e12, assumed: false };
  if (dev.peak.fp32 != null) return { prec: "fp32", opsPerSec: dev.peak.fp32 * 1e12, assumed: false };
  if (!dev.quant.int8Only && dev.peak.int8 != null) return { prec: "fp16", opsPerSec: (dev.peak.int8 * 1e12) / 4, assumed: true };
  return null;
}

function rate(dev: DeviceSpec, p: Precision, fp: FloatPath | null): number {
  const int8 = (dev.peak.int8 ?? 0) * 1e12;
  if (p === "int8") return int8 || fp?.opsPerSec || 1e9;
  // W4A8 without INT4 hardware runs INT8 math; only the weight bytes shrink.
  if (p === "int4") return dev.peak.int4 != null ? dev.peak.int4 * 1e12 : int8 || 1e9;
  if (p === "fp32" && dev.peak.fp32 != null) return dev.peak.fp32 * 1e12;
  return fp?.opsPerSec ?? int8 ?? 1e9;
}

const isFloat = (p: Precision) => p === "fp16" || p === "bf16" || p === "fp32";

function elems(s?: number[]): number {
  if (!s || s.length === 0) return 0;
  return s.reduce((a, b) => a * (b > 0 ? b : 1), 1);
}

type Sim = { ms: number; idealMs: number; layers: LayerBottleneck[]; sizeBytes: number; memBoundShare: number };

/** Estimate latency for a per-layer precision assignment. */
export function simulate(
  model: Model,
  dev: DeviceSpec,
  precOf: (l: ModelLayer) => Precision,
  workload: WorkloadClass = classifyWorkload(model),
): Sim {
  const fp = floatPath(dev);
  const bw = dev.bandwidthGBs * 1e9;
  const onchip = dev.memType === "onchip";
  const u = calibratedUtil(dev, workload).util;
  const precById = new Map<number, Precision>();
  const rows: LayerBottleneck[] = [];
  let idealSec = 0;
  let size = 0;

  for (const l of model.layers) {
    const p = precOf(l);
    precById.set(l.id, p);
    const ops = l.flops || l.macs * 2;
    let act = 0;
    for (const i of l.ins) act += elems(i.s);
    for (const o of l.outs) act += elems(o.s);
    if (act === 0 && l.mem > 0) act = l.mem / 4;
    const actBpe = p === "int4" ? 1 : BYTES[p]; // W4A8: activations stay 8-bit
    const wBytes = l.params * BYTES[p];
    size += wBytes;
    const bytes = act * actBpe + (onchip ? 0 : wBytes);
    const tc = ops / rate(dev, p, fp);
    const tm = bw > 0 ? bytes / bw : 0;
    const t = Math.max(tc, tm);
    idealSec += t;
    rows.push({
      id: l.id, name: l.name, op: l.op, precision: p, ops, bytes,
      intensity: bytes > 0 ? ops / bytes : Infinity,
      bound: ops === 0 && bytes === 0 ? "idle" : tc >= tm ? "compute" : "memory",
      ms: t * 1000, share: 0,
    });
  }

  // Quantize/Dequantize at every int↔float boundary: one extra read + write.
  const byId = new Map(model.layers.map((l) => [l.id, l]));
  for (const [a, b] of model.edges) {
    const pa = precById.get(a), pb = precById.get(b);
    if (!pa || !pb || isFloat(pa) === isFloat(pb)) continue;
    const e = elems(byId.get(a)?.outs[0]?.s);
    if (bw > 0) idealSec += (e * (1 + BYTES[isFloat(pa) ? pa : pb])) / bw;
  }

  const idealMs = idealSec * 1000;
  const ms = idealMs / u + dev.dispatchMs;
  let memMs = 0;
  for (const r of rows) {
    r.ms = r.ms / u;
    r.share = idealMs > 0 ? (r.ms * u) / idealMs : 0;
    if (r.bound === "memory") memMs += r.ms * u;
  }
  return { ms, idealMs, layers: rows, sizeBytes: size, memBoundShare: idealMs > 0 ? memMs / idealMs : 0 };
}

const perf = (label: string, s: Sim): Perf => ({
  label, latencyMs: s.ms, fps: s.ms > 0 ? 1000 / s.ms : 0, sizeBytes: s.sizeBytes, memBoundShare: s.memBoundShare,
});

// ── Helpers ───────────────────────────────────────────────────────────────────

const CONF_W: Record<Confidence, number> = { high: 1, medium: 0.75, low: 0.5 };
const RISK_W: Record<AccuracyRisk, number> = { none: 1, low: 0.9, medium: 0.7, high: 0.45 };
const pctChange = (from: number, to: number) => (from > 0 ? ((to - from) / from) * 100 : 0);
const round1 = (n: number) => Math.round(n * 10) / 10;
const round2 = (n: number) => Math.round(n * 100) / 100;

function listNames(names: string[], max = 3): string {
  if (names.length <= max) return names.join(", ");
  return `${names.slice(0, max).join(", ")} +${names.length - max} more`;
}

function fmtBytes(b: number): string {
  if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
  return Math.round(b) + " B";
}

function gainScore(speedup: number, conf: Confidence, risk: AccuracyRisk): number {
  const gain = Math.max(0, 1 - 1 / Math.max(speedup, 1e-9)); // fraction of latency removed
  return 100 * gain * CONF_W[conf] * RISK_W[risk];
}

const FIX_RISK: Record<FixId, AccuracyRisk> = {
  silu_hardswish: "low", sppf_parallel: "none", resize_convtranspose: "low", channel_prune: "medium",
};

/** Image-like input → { h, w, hAxis }. NCHW (C ≤ 4) or NHWC. */
function imageInput(model: Model): { h: number; w: number } | null {
  const s = model.inputShape;
  if (s.length !== 4) return null;
  if (s[1] > 0 && s[1] <= 4 && s[2] >= 64 && s[3] >= 64) return { h: s[2], w: s[3] };
  if (s[3] > 0 && s[3] <= 4 && s[1] >= 64 && s[2] >= 64) return { h: s[1], w: s[2] };
  return null;
}

/** Scale spatial work by r² for layers with a 4-D (feature-map) output. */
export function scaleResolution(model: Model, r: number): Model {
  const scaleShape = (s?: number[]) =>
    s && s.length === 4 && s[2] > 1 && s[3] > 1 ? [s[0], s[1], Math.max(1, Math.round(s[2] * r)), Math.max(1, Math.round(s[3] * r))] : s;
  return {
    ...model,
    inputShape: scaleShape(model.inputShape) ?? model.inputShape,
    layers: model.layers.map((l) => {
      const s = l.outs[0]?.s;
      const spatial = !!s && s.length === 4 && s[2] > 1 && s[3] > 1;
      const f = spatial ? r * r : 1;
      return {
        ...l,
        flops: Math.round(l.flops * f), macs: Math.round(l.macs * f), mem: Math.round(l.mem * f),
        ins: l.ins.map((i) => ({ ...i, s: scaleShape(i.s) })),
        outs: l.outs.map((o) => ({ ...o, s: scaleShape(o.s) })),
      };
    }),
  };
}

const FUSIBLE = /^(relu|leakyrelu|prelu|clip|relu6|sigmoid|silu|swish|hardswish|hardsigmoid|mul\(sigmoid\)|batchnormalization|batchnorm|batchnorm2d|add)$/i;

// ── Main ──────────────────────────────────────────────────────────────────────

export type AdviseOptions = { quant?: QuantAnalysis };

export function advise(model: Model, dev: DeviceSpec, opts: AdviseOptions = {}): AdvisorResult {
  const workload = classifyWorkload(model);
  const quant = opts.quant ?? analyzeQuantization(model);
  const score = scoreDevice(model, dev);
  const deployPrecision = pickPrecision(dev);
  const fp = floatPath(dev);
  const notes: string[] = [];
  const advice: Advice[] = [];
  const s = quant.summary;
  const fp32Size = s.totalParams * 4;

  const sens = (l: ModelLayer): LayerSensitivity | undefined => quant.byId.get(l.id);
  const intPrec: Precision = deployPrecision === "int8" || deployPrecision === "int4" ? deployPrecision : "int8";
  const floatPrec: Precision = fp?.prec ?? "fp16";
  const mixedOk = dev.quant.mixedPrecision && !dev.quant.int8Only && fp != null;

  const planPrec = (l: ModelLayer): Precision => {
    if (!dev.peak.int8) return floatPrec;
    const q = sens(l);
    if (!mixedOk || !q || !q.quantizable) return intPrec;
    if (q.recommendation === "fp16") return floatPrec;
    if (q.recommendation === "keep-fp32") return dev.peak.fp32 != null ? "fp32" : floatPrec;
    if (q.recommendation === "int4") return "int4";
    return intPrec;
  };

  const planSim = simulate(model, dev, planPrec, workload);
  const fullSim = simulate(model, dev, () => (dev.peak.int8 ? intPrec : floatPrec), workload);
  const baseSim = fp ? simulate(model, dev, () => fp.prec, workload) : null;
  const plan = perf(mixedOk && s.excluded.length ? "Mixed INT8 plan" : "INT8", planSim);
  const fullInt8 = perf("Full INT8", fullSim);
  const baseline = baseSim ? perf(fp!.prec.toUpperCase() + " baseline", baseSim) : null;
  const ridgeAI = dev.bandwidthGBs > 0 ? rate(dev, deployPrecision, fp) / (dev.bandwidthGBs * 1e9) : Infinity;

  if (fp?.assumed) notes.push(`${dev.n} lists no float throughput; FP16 is assumed at 1/4 of the INT8 rate.`);
  if (!fp) notes.push(`${dev.n} has no float path on the accelerator: every layer must be quantized.`);
  if (s.statsCoverage < 0.5 && s.weighted > 0) notes.push("Few layers carry weight statistics, so quantization risks are mostly op-type priors (low confidence).");
  if (s.weighted === 0) notes.push("No weighted layers found: quantization advice is based on op types only.");
  if (!dev.compatId) notes.push(`No per-op compatibility data for ${dev.n}: operator coverage is not verified.`);
  notes.push("All figures are analytical roofline and weight-statistics estimates, not on-device measurements.");

  // ── Required: format conversion ──
  if (score.opSupport.conversionNeeded) {
    advice.push({
      id: "format", kind: "format", required: true,
      title: `Convert ${model.format} → ${dev.fmts[0] ?? dev.toolchain[0]} (${dev.toolchain[0]})`,
      rationale: `${dev.toolchain[0]} ingests ${dev.fmts.join(", ") || "its own format"}, not ${model.format}. Convert before compiling; the quantization plan below applies during that conversion.`,
      impact: { accuracyRisk: "none" }, confidence: "high", score: 900,
    });
  }

  // ── Required / critical: memory fit ──
  if (score.memory.hardFail || (score.memory.budgetBytes > 0 && score.memory.headroom < 1)) {
    const int4Size = s.totalParams * 0.5;
    advice.push({
      id: "memory-fit", kind: "memory-fit", required: score.memory.hardFail,
      title: `Model needs ${fmtBytes(score.memory.footprintBytes)} but ${dev.n} has ${fmtBytes(score.memory.budgetBytes)}`,
      rationale: `Footprint at ${deployPrecision.toUpperCase()} (weights + peak activations + 10% runtime overhead) exceeds the ${dev.memType === "onchip" ? "on-chip" : "device"} budget. Shrink weights (INT4 weight-only ≈ ${fmtBytes(int4Size)}), prune channels, or pick a smaller variant.`,
      impact: { sizePct: round1(pctChange(fp32Size, int4Size)), accuracyRisk: "high" },
      confidence: "medium", score: score.memory.hardFail ? 950 : 400,
    });
  }

  // ── Quantization ──
  const excludedNames = s.excluded.map((l) => l.name);
  const pcNeeded = s.perChannel.length > 0;
  const highNames = quant.layers.filter((l) => l.quantizable && l.int8.perChannelRisk >= 0.6).map((l) => l.name);
  const qConf: Confidence = s.confidence === "medium" ? "medium" : "low";
  if (!dev.peak.int8) {
    notes.push(`${dev.n} has no INT8 path in the hardware database; quantization advice is skipped.`);
  } else if (!mixedOk) {
    const risk: AccuracyRisk = highNames.length > 0 ? "high" : s.excluded.length > 0 ? "medium" : "low";
    const pcText = pcNeeded
      ? dev.quant.perChannel
        ? ` Use per-channel weight scales (${s.perChannel.length} layer(s) need them).`
        : ` ${s.perChannel.length} layer(s) need per-channel scales, which this toolchain lacks: apply cross-layer equalization (Nagel et al. 2019) or QAT.`
      : "";
    advice.push({
      id: "quantize", kind: "quantize", required: dev.quant.int8Only,
      title: dev.quant.int8Only ? `Full INT8 quantization (${dev.n} runs integer only)` : "Full INT8 quantization",
      rationale: `Mixed precision is not available on ${dev.toolchain[0]}, so every layer runs INT8.${pcText}${s.excluded.length ? ` Layers at risk: ${listNames(excludedNames)}; calibrate with representative data and consider quantization-aware training for them.` : " No layer is flagged as high-risk."}`,
      impact: {
        speedup: baseline ? round2(baseline.latencyMs / fullInt8.latencyMs) : undefined,
        latencyPct: baseline ? round1(pctChange(baseline.latencyMs, fullInt8.latencyMs)) : undefined,
        sizePct: round1(pctChange(fp32Size, s.totalParams)), accuracyRisk: risk,
      },
      confidence: qConf, layers: excludedNames, layerIds: s.excluded.map((l) => l.id),
      score: dev.quant.int8Only ? 800 : gainScore(baseline ? baseline.latencyMs / fullInt8.latencyMs : 1, qConf, risk),
    });
  } else {
    const speed = baseline ? baseline.latencyMs / plan.latencyMs : 1;
    const risk: AccuracyRisk = s.confidence === "low" ? "medium" : "low";
    const title = `Quantize to INT8${pcNeeded ? " (per-channel)" : ""}${s.excluded.length ? ` except ${s.excluded.length} layer(s): ${listNames(excludedNames, 2)}` : ""}`;
    advice.push({
      id: "quantize", kind: "quantize", title,
      rationale: `Sensitivity analysis keeps ${s.excluded.length} layer(s) in ${floatPrec.toUpperCase()} (estimated INT8 risk ≥ 30%) and runs the rest in INT8${pcNeeded ? `, ${s.perChannel.length} with per-channel scales` : ""}. Est. ${round2(speed)}× vs the ${baseline?.label ?? "float"} path${s.excluded.length ? `, including Q/DQ boundaries` : ""}.`,
      impact: {
        speedup: round2(speed), latencyPct: baseline ? round1(pctChange(baseline.latencyMs, plan.latencyMs)) : undefined,
        sizePct: round1(pctChange(fp32Size, plan.sizeBytes)), accuracyRisk: risk,
      },
      confidence: qConf, layers: excludedNames, layerIds: s.excluded.map((l) => l.id),
      score: 300 + gainScore(speed, qConf, risk),
    });
    if (s.excluded.length > 0 && baseline) {
      const sp = baseline.latencyMs / fullInt8.latencyMs;
      advice.push({
        id: "quantize-full", kind: "quantize",
        title: "Alternative: full INT8 (no float exclusions)",
        rationale: `Fastest option (${round2(fullInt8.latencyMs / plan.latencyMs)}× the mixed plan's latency), but quantizes the ${s.excluded.length} flagged layer(s) too: ${listNames(excludedNames)}. Only with QAT or a validated calibration set.`,
        impact: { speedup: round2(sp), latencyPct: round1(pctChange(baseline.latencyMs, fullInt8.latencyMs)), sizePct: round1(pctChange(fp32Size, s.totalParams)), accuracyRisk: highNames.length ? "high" : "medium" },
        confidence: qConf, layers: excludedNames, layerIds: s.excluded.map((l) => l.id),
        score: gainScore(sp, qConf, highNames.length ? "high" : "medium") * 0.5,
      });
    }
    if (pcNeeded && !dev.quant.perChannel) {
      notes.push(`${dev.toolchain[0]} supports per-tensor scales only; ${s.perChannel.length} layer(s) that need per-channel scales should use cross-layer equalization or stay in float.`);
    }
  }

  // ── INT4 weights (only where the device has INT4 or the model does not fit) ──
  if (mixedOk && (dev.peak.int4 != null || score.memory.hardFail)) {
    const int4Ok = new Set(quant.layers.filter((l) => l.kind === "weighted" && l.int4.perChannelRisk < 0.3).map((l) => l.id));
    if (int4Ok.size > 0) {
      const sim = simulate(model, dev, (l) => (int4Ok.has(l.id) ? "int4" : planPrec(l)), workload);
      const sp = plan.latencyMs / sim.ms;
      advice.push({
        id: "int4", kind: "int4",
        title: `INT4 weights on ${int4Ok.size} low-risk layer(s)`,
        rationale: `W4A8 on layers whose estimated INT4 risk is < 30%. Weight storage ${fmtBytes(plan.sizeBytes)} → ${fmtBytes(sim.sizeBytes)}. Use GPTQ/AWQ-style calibration and validate accuracy.`,
        impact: { speedup: round2(sp), latencyPct: round1(pctChange(plan.latencyMs, sim.ms)), sizePct: round1(pctChange(fp32Size, sim.sizeBytes)), accuracyRisk: "medium" },
        confidence: "low", layerIds: [...int4Ok], score: 50 + gainScore(sp, "low", "medium"),
      });
    }
  }

  // ── Operator support (compiler pre-flight issues for this target) ──
  if (dev.compatId) {
    const issueLayers = model.layers.filter((l) => l.compIssues.some((c) => c.target === dev.compatId));
    const covered = new Set<number>();
    // Pruning changes capacity, not op support: it is proposed separately below.
    for (const fix of FIXES.filter((f) => f.id !== "channel_prune")) {
      const hit = issueLayers.filter((l) => fix.applies(l));
      if (hit.length === 0) continue;
      hit.forEach((l) => covered.add(l.id));
      const after = scoreDevice(applyFix(model, fix.id), dev);
      const sp = score.fps > 0 ? after.fps / score.fps : 1;
      const unblocks = score.opSupport.compileFail && !after.opSupport.compileFail;
      const msgs = [...new Set(hit.flatMap((l) => l.compIssues.filter((c) => c.target === dev.compatId).map((c) => c.msg)))];
      advice.push({
        id: `fix-${fix.id}`, kind: "op-support", required: unblocks,
        title: `${fix.title} on ${hit.length} layer(s)`,
        rationale: `${msgs.slice(0, 2).join(" ")} ${fix.change} ${fix.impact}`.trim(),
        impact: { speedup: round2(sp), latencyPct: round1(pctChange(1, 1 / Math.max(sp, 1e-9))), accuracyRisk: FIX_RISK[fix.id] },
        confidence: "medium", layers: hit.map((l) => l.name), layerIds: hit.map((l) => l.id),
        score: unblocks ? 850 : 150 + gainScore(sp, "medium", FIX_RISK[fix.id]),
      });
    }
    // Issues no auto-fix covers, grouped by op.
    const rest = new Map<string, { layers: ModelLayer[]; msgs: Set<string>; error: boolean }>();
    for (const l of issueLayers) {
      if (covered.has(l.id)) continue;
      const g = rest.get(l.op) ?? { layers: [], msgs: new Set<string>(), error: false };
      g.layers.push(l);
      for (const c of l.compIssues) if (c.target === dev.compatId) { g.msgs.add(c.msg); if (c.severity === "error") g.error = true; }
      rest.set(l.op, g);
    }
    for (const [op, g] of rest) {
      const fallback = dev.cpuFallback === "none" ? "will not compile" : dev.cpuFallback === "single-partition" ? "moves it and everything after it to the host CPU" : "falls back to the CPU";
      advice.push({
        id: `op-${op}`, kind: "op-support", required: g.error && dev.cpuFallback === "none",
        title: `${g.error ? "Unsupported" : "Limited"} op ${op} (${g.layers.length} layer(s))`,
        rationale: `${[...g.msgs].slice(0, 2).join(" ")} On ${dev.n} this ${fallback}. No automatic fix: replace it in the source model (or restructure) before export.`,
        impact: { accuracyRisk: "low" }, confidence: "medium",
        layers: g.layers.map((l) => l.name), layerIds: g.layers.map((l) => l.id),
        score: g.error ? (dev.cpuFallback === "none" ? 840 : 120) : 20,
      });
    }
  }

  // ── Input resolution ──
  const img = imageInput(model);
  if (img && img.h >= 256 && planSim.idealMs > 0) {
    const newH = Math.max(32, Math.floor((img.h * 0.75) / 32) * 32);
    const r = newH / img.h;
    const scaled = scaleResolution(model, r);
    const sim = simulate(scaled, dev, planPrec, workload);
    const oldF = model.layers.reduce((a, l) => a + (l.flops || l.macs * 2), 0);
    const newF = scaled.layers.reduce((a, l) => a + (l.flops || l.macs * 2), 0);
    const sp = plan.latencyMs / sim.ms;
    const risk: AccuracyRisk = workload === "cnnDet" ? "medium" : "low";
    const newW = Math.round((img.w * r) / 32) * 32 || newH;
    advice.push({
      id: "resolution", kind: "resolution",
      title: `Input resolution ${img.h}→${newH}${img.w !== img.h ? ` (${img.w}→${newW})` : ""}`,
      rationale: `Feature-map work scales with H×W, so ${(r * r * 100).toFixed(0)}% of the spatial FLOPs remain.${workload === "cnnDet" ? " Small-object recall drops first; retrain or fine-tune at the new size." : " Fine-tune at the new size for best accuracy."}`,
      impact: { speedup: round2(sp), latencyPct: round1(pctChange(plan.latencyMs, sim.ms)), flopsPct: round1(pctChange(oldF, newF)), accuracyRisk: risk },
      confidence: "medium", score: gainScore(sp, "medium", risk) * (plan.fps >= 60 ? 0.5 : 1),
    });
  }

  // ── Memory-bound layers & fusion ──
  const memRows = planSim.layers.filter((r) => r.bound === "memory" && r.ops > 0);
  const memShare = memRows.reduce((a, r) => a + r.share, 0);
  if (memRows.length > 0 && memShare >= 0.2) {
    const top = [...memRows].sort((a, b) => b.share - a.share).slice(0, 4);
    advice.push({
      id: "memory-bound", kind: "memory-bound",
      title: `${memRows.length} layer(s) are memory-bound on ${dev.n} (${Math.round(memShare * 100)}% of time)`,
      rationale: `Their arithmetic intensity is below the device ridge point (${round1(ridgeAI)} FLOP/B), so bandwidth, not compute, sets their speed. Top: ${top.map((r) => `${r.name} (${round1(r.intensity)} FLOP/B, ${Math.round(r.share * 100)}%)`).join(", ")}. Keep activations INT8 end to end, fuse elementwise ops into their producers, avoid layout transposes, and shrink the early high-resolution stages.`,
      impact: { accuracyRisk: "none" }, confidence: dev.bwConf === "?" ? "low" : "medium",
      layers: top.map((r) => r.name), layerIds: memRows.map((r) => r.id),
      score: 30 + memShare * 40,
    });
  }

  const pred = new Map<number, number[]>();
  for (const [a, b] of model.edges) (pred.get(b) ?? pred.set(b, []).get(b)!).push(a);
  const byId = new Map(model.layers.map((l) => [l.id, l]));
  const rowById = new Map(planSim.layers.map((r) => [r.id, r]));
  const fusible = model.layers.filter((l) => {
    if (l.params > 0 && !/batchnorm/i.test(l.op)) return false;
    if (!FUSIBLE.test(l.op) && !FUSIBLE.test(l.type)) return false;
    const ps = pred.get(l.id) ?? [];
    return ps.some((p) => /conv|gemm|matmul|linear/i.test(byId.get(p)?.op ?? "")) && rowById.get(l.id)?.bound === "memory";
  });
  if (fusible.length > 0) {
    const savedIdeal = fusible.reduce((a, l) => a + (rowById.get(l.id)?.ms ?? 0), 0) * calibratedUtil(dev, workload).util;
    const u = calibratedUtil(dev, workload).util;
    const newMs = Math.max(dev.dispatchMs, (planSim.idealMs - savedIdeal) / u + dev.dispatchMs);
    const sp = plan.latencyMs / newMs;
    if (sp > 1.03) {
      advice.push({
        id: "fusion", kind: "fusion",
        title: `Fuse ${fusible.length} activation/BN/add layer(s) into their producer conv`,
        rationale: `These elementwise layers only move memory on ${dev.n}. Conv+BN+activation fusion removes their round-trip. TensorRT, OpenVINO and most NPU compilers do this automatically for standard patterns; check the compiled graph, since decomposed activations (Sigmoid+Mul) may not fuse.`,
        impact: { speedup: round2(sp), latencyPct: round1(pctChange(plan.latencyMs, newMs)), accuracyRisk: "none" },
        confidence: "medium", layers: fusible.map((l) => l.name), layerIds: fusible.map((l) => l.id),
        score: gainScore(sp, "medium", "none") * 0.8,
      });
    }
  }

  // ── Channel pruning (compute-bound models with heavy layers) ──
  const pruneable = model.layers.filter((l) => FIXES.find((f) => f.id === "channel_prune")!.applies(l));
  if (pruneable.length > 0 && planSim.memBoundShare < 0.7) {
    const pruned = applyFix(model, "channel_prune");
    const sim = simulate(pruned, dev, planPrec, workload);
    const sp = plan.latencyMs / sim.ms;
    if (sp > 1.02) {
      advice.push({
        id: "prune", kind: "prune",
        title: `Structured channel pruning (keep 75%) on ${pruneable.length} heavy layer(s)`,
        rationale: "Removes whole output channels from the largest layers; needs a short fine-tune to recover accuracy.",
        impact: { speedup: round2(sp), latencyPct: round1(pctChange(plan.latencyMs, sim.ms)), sizePct: round1(pctChange(plan.sizeBytes, sim.sizeBytes)), accuracyRisk: "medium" },
        confidence: "low", layers: pruneable.map((l) => l.name), layerIds: pruneable.map((l) => l.id),
        score: gainScore(sp, "low", "medium"),
      });
    }
  }

  advice.sort((a, b) => b.score - a.score);
  return {
    device: dev, workload, deployPrecision, baseline, plan, fullInt8, ridgeAI,
    layers: planSim.layers, advice, quant, score, notes,
  };
}
