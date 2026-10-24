// Cross-compiler optimization search.
//
// Enumerates deployment configurations
//     device × runtime/compiler × precision × auto-fix set × input-resolution scale
// and scores each on estimated latency (per-layer roofline, the same model as the
// Hardware tab), model size, and an accuracy-risk proxy (optsearch/risk.ts). It
// returns the Pareto frontier over (latency, size, risk) and a ranked list.
//
// Everything is derived from existing data, not a parallel catalog:
//   • runtimes/compilers come from each device's `toolchain` (data/hardware.ts);
//   • precisions from the device's listed peaks and quant capabilities;
//   • fix sets are subsets of the auto-fixes that apply to THIS model
//     (fixes/transforms.ts), applied to a copy of the model;
//   • compatibility comes from ModelLayer.compIssues for the device's compatId,
//     re-evaluated after the fixes;
//   • the deploy recipe for a config reuses the Converter's script templates
//     (data/convScripts.ts), see optsearch/recipe.ts.
//
// Speed: fixed and rescaled model variants and their prepared per-layer arrays
// are memoized; compatibility is checked once per (device, fix set) and combos
// that cannot compile are pruned before any latency work; memory-infeasible
// precisions are pruned before the roofline runs.
//
// Pure module: no React.

import type { Model, ModelLayer } from "@modelvisio/parsers";
import { BYTES, type DeviceSpec, type Precision } from "../data/hardware";
import { CONV_SCRIPTS } from "../data/convScripts";
import { FIXES, applyFix, countApplicable, type FixId } from "../fixes/transforms";
import { classifyWorkload, peakActivationBytes } from "../scoring/roofline";
import { prepareLayers, rooflineFromPrepared, type PreparedLayers } from "../investigate/layerRoofline";
import { defaultRisk, type RiskEstimator } from "./risk";

// ─── Space derivation ────────────────────────────────────────────────────────

/** Kernel libraries listed in toolchains that are not deployment compilers/runtimes. */
const NOT_A_RUNTIME = /^(cuDNN|CUDA)$/i;

/** Device toolchain → Converter script that produces its input (null: vendor tool only). */
const SCRIPT_FOR: [RegExp, (p: Precision) => string | null][] = [
  [/tensorrt/i, () => "onnx→tensorrt"],
  [/openvino/i, () => "onnx→openvino"],
  [/rknn/i, () => "onnx→rknn"],
  [/edgetpu|eiq|tflite/i, () => "pt→tflite"],
  [/onnx ?rt|onnx ?runtime/i, (p) => (p === "fp16" ? "onnx→onnx-fp16" : null)],
  [/core ?ml/i, () => "pt→coreml"],
  [/ncnn/i, () => "onnx→ncnn"],
];

/** Converter script key for a runtime at a precision, if the Converter has one. */
export function scriptKeyFor(runtime: string, prec: Precision): string | null {
  for (const [re, fn] of SCRIPT_FOR) {
    if (re.test(runtime)) {
      const k = fn(prec);
      return k && k in CONV_SCRIPTS ? k : null;
    }
  }
  return null;
}

/** Deployment runtimes/compilers for a device, from its toolchain; ones with a generated script first. */
export function runtimesFor(dev: DeviceSpec): string[] {
  const rts = dev.toolchain.filter((t) => !NOT_A_RUNTIME.test(t));
  const list = rts.length ? rts : dev.toolchain.slice(0, 1);
  const hasScript = (r: string) => SCRIPT_FOR.some(([re]) => re.test(r));
  return [...list.filter(hasScript), ...list.filter((r) => !hasScript(r))];
}

export type PrecisionSupport = { prec: Precision; peakTops: number; estimated: boolean };

/**
 * Precisions a device can run, with the peak used for each. Listed peaks are
 * used as-is; float precisions missing from the datasheet on a part that is not
 * INT8-only are derived (FP16 ≈ INT8/2, FP32 ≈ FP16/2) and flagged `estimated`.
 */
export function supportedPrecisions(dev: DeviceSpec): PrecisionSupport[] {
  const out: PrecisionSupport[] = [];
  const pk = dev.peak;
  if (pk.int4 != null) out.push({ prec: "int4", peakTops: pk.int4, estimated: false });
  if (pk.int8 != null) out.push({ prec: "int8", peakTops: pk.int8, estimated: false });
  if (dev.quant.int8Only) return out;
  let fp16: PrecisionSupport | null = null;
  if (pk.fp16 != null) fp16 = { prec: "fp16", peakTops: pk.fp16, estimated: false };
  else if (pk.int8 != null) fp16 = { prec: "fp16", peakTops: pk.int8 / 2, estimated: true };
  if (fp16) out.push(fp16);
  if (pk.bf16 != null) out.push({ prec: "bf16", peakTops: pk.bf16, estimated: false });
  if (pk.fp32 != null) out.push({ prec: "fp32", peakTops: pk.fp32, estimated: false });
  else if (fp16) out.push({ prec: "fp32", peakTops: fp16.peakTops / 2, estimated: true });
  return out;
}

/** Auto-fixes that touch at least one layer of this model, in FIXES order. */
export function applicableFixes(model: Model): FixId[] {
  return FIXES.filter((f) => countApplicable(model, f.id) > 0).map((f) => f.id);
}

/** Every subset of `fixes` (2^n, n ≤ 4 today), empty set first. */
export function fixSubsets(fixes: FixId[]): FixId[][] {
  const out: FixId[][] = [];
  for (let mask = 0; mask < 1 << fixes.length; mask++) out.push(fixes.filter((_, i) => mask & (1 << i)));
  return out;
}

/** Can input resolution be rescaled meaningfully (4-D image input, not a transformer)? */
export function canRescale(model: Model): boolean {
  const s = model.inputShape;
  return s.length === 4 && s[2] > 0 && s[3] > 0 && classifyWorkload(model) !== "transformer";
}

/**
 * Model at a different input resolution: spatial dims of 4-D tensors scale by
 * `s`, and FLOPs/MACs/activation memory of spatial layers by s². Weights are
 * unchanged. Returns the same object for s = 1.
 */
export function scaleResolution(model: Model, s: number): Model {
  if (s === 1) return model;
  const sc = (shape?: number[]) =>
    shape && shape.length === 4 ? shape.map((d, i) => (i >= 2 && d > 0 ? Math.max(1, Math.round(d * s)) : d)) : shape;
  const spatial = (l: ModelLayer) => [...l.ins, ...l.outs].some((x) => x.s && x.s.length === 4 && x.s[2] > 0 && x.s[3] > 0);
  const f = s * s;
  return {
    ...model,
    inputShape: sc(model.inputShape) ?? model.inputShape,
    layers: model.layers.map((l) =>
      spatial(l)
        ? {
            ...l,
            flops: Math.round(l.flops * f), macs: Math.round(l.macs * f), mem: Math.round(l.mem * f),
            ins: l.ins.map((x) => ({ ...x, s: sc(x.s) })), outs: l.outs.map((x) => ({ ...x, s: sc(x.s) })),
          }
        : l),
  };
}

// ─── Types ───────────────────────────────────────────────────────────────────

export type Constraints = { maxLatencyMs?: number; maxSizeMB?: number; maxRisk?: number };

export type SearchOptions = {
  devices: DeviceSpec[];
  /** Input-resolution scales to try (default [1]); ignored when the model can't be rescaled. */
  scales?: number[];
  /** Restrict precisions (default: every precision each device supports). */
  precisions?: Precision[];
  /** Restrict fix ids considered (default: all applicable to the model). */
  fixes?: FixId[];
  constraints?: Constraints;
  risk?: RiskEstimator;
  /** Ranking weights over normalized (log-latency, size, risk). Default 0.5/0.2/0.3. */
  weights?: { latency: number; size: number; risk: number };
};

export type ConfigResult = {
  key: string;
  device: DeviceSpec;
  /** Primary runtime/compiler (one with a generated script when available). */
  runtime: string;
  /** All toolchain runtimes this config applies to (identical estimates). */
  runtimes: string[];
  precision: Precision;
  fixes: FixId[];
  scale: number;
  latencyMs: number;
  fps: number;
  /** Weight storage at this precision (MB). */
  sizeMB: number;
  /** Weights + peak activations + 10% overhead (MB). */
  footprintMB: number;
  risk: number;
  errors: number;
  warnings: number;
  /** Layers that run on the host CPU because of unsupported ops. */
  fallbackLayers: number;
  /** The precision's peak was derived, not from the datasheet. */
  peakEstimated: boolean;
  /** Converter script key for the primary runtime, if any. */
  scriptKey: string | null;
  notes: string[];
  meetsConstraints: boolean;
  pareto: boolean;
  /** 0..1, higher is better (feasible configs only). */
  score: number;
  rank: number;
};

export type PruneReason = "no-compile" | "no-memory" | "precision-filter";

export type SearchResult = {
  /** Feasible configs (meet constraints), best first. */
  ranked: ConfigResult[];
  /** Pareto frontier of the feasible configs over (latency, size, risk). */
  pareto: ConfigResult[];
  /** Every compatible config evaluated (feasible or not). */
  all: ConfigResult[];
  pruned: Record<PruneReason, number>;
  /** Size of the full space before pruning. */
  spaceSize: number;
  evaluated: number;
  fixesConsidered: FixId[];
  scalesConsidered: number[];
  elapsedMs: number;
};

/** Host-CPU slowdown applied to layers that fall back off a graceful-fallback accelerator. */
export const CPU_FALLBACK_SLOWDOWN = 10;
/** Per fallback layer accelerator↔CPU transfer/sync cost (ms). */
export const PARTITION_SWITCH_MS = 0.3;

// ─── Pareto ──────────────────────────────────────────────────────────────────

type Pt = { latencyMs: number; sizeMB: number; risk: number };

/** a dominates b: no worse on all three objectives, strictly better on one. */
export function dominates(a: Pt, b: Pt): boolean {
  return a.latencyMs <= b.latencyMs && a.sizeMB <= b.sizeMB && a.risk <= b.risk
    && (a.latencyMs < b.latencyMs || a.sizeMB < b.sizeMB || a.risk < b.risk);
}

/**
 * Non-dominated subset (minimize all three). Lexicographic sort means any
 * dominator precedes what it dominates, so each point is only checked against
 * the frontier built so far: O(n·f) instead of O(n²).
 */
export function paretoFront<T extends Pt>(pts: T[]): T[] {
  const sorted = [...pts].sort((a, b) => a.latencyMs - b.latencyMs || a.sizeMB - b.sizeMB || a.risk - b.risk);
  const front: T[] = [];
  for (const p of sorted) if (!front.some((f) => dominates(f, p))) front.push(p);
  return front;
}

// ─── Search ──────────────────────────────────────────────────────────────────

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export function runSearch(model: Model, opts: SearchOptions): SearchResult {
  const t0 = now();
  const risk = opts.risk ?? defaultRisk;
  const cons = opts.constraints ?? {};
  const w = opts.weights ?? { latency: 0.5, size: 0.2, risk: 0.3 };
  const fixesConsidered = applicableFixes(model).filter((f) => !opts.fixes || opts.fixes.includes(f));
  const subsets = fixSubsets(fixesConsidered);
  const scalesConsidered = canRescale(model)
    ? [...new Set((opts.scales?.length ? opts.scales : [1]).filter((s) => s > 0 && s <= 1))].sort((a, b) => b - a)
    : [1];
  if (scalesConsidered.length === 0) scalesConsidered.push(1);

  // Memoized model variants.
  const fixedMemo = new Map<string, Model>();
  const fixed = (fx: FixId[]): Model => {
    const k = fx.join("+");
    let m = fixedMemo.get(k);
    if (!m) {
      m = fx.length === 0 ? model : applyFix(fixed(fx.slice(0, -1)), fx[fx.length - 1]);
      fixedMemo.set(k, m);
    }
    return m;
  };
  const variantMemo = new Map<string, { model: Model; prep: PreparedLayers; params: number }>();
  const variant = (fx: FixId[], s: number) => {
    const k = `${fx.join("+")}@${s}`;
    let v = variantMemo.get(k);
    if (!v) {
      const m = scaleResolution(fixed(fx), s);
      v = { model: m, prep: prepareLayers(m), params: m.layers.reduce((a, l) => a + l.params, 0) };
      variantMemo.set(k, v);
    }
    return v;
  };
  const actMemo = new Map<string, number>();

  const pruned: Record<PruneReason, number> = { "no-compile": 0, "no-memory": 0, "precision-filter": 0 };
  const all: ConfigResult[] = [];
  let spaceSize = 0;

  for (const dev of opts.devices) {
    const runtimes = runtimesFor(dev);
    const precs = supportedPrecisions(dev);
    const perFix = precs.length * scalesConsidered.length;
    spaceSize += subsets.length * perFix * runtimes.length;

    for (const fx of subsets) {
      // Compatibility once per (device, fix set): prune before any latency work.
      const fm = fixed(fx);
      let errors = 0, warnings = 0;
      const errIds = new Set<number>();
      if (dev.compatId) {
        for (const l of fm.layers) for (const c of l.compIssues) {
          if (c.target !== dev.compatId) continue;
          if (c.severity === "error") { errors++; errIds.add(l.id); } else warnings++;
        }
      }
      if (errIds.size > 0 && dev.cpuFallback === "none") { pruned["no-compile"] += perFix * runtimes.length; continue; }

      for (const ps of precs) {
        if (opts.precisions && !opts.precisions.includes(ps.prec)) { pruned["precision-filter"] += scalesConsidered.length * runtimes.length; continue; }
        for (const s of scalesConsidered) {
          const v = variant(fx, s);
          // Memory fit (same footprint model as scoreDevice), before the roofline.
          const ak = `${fx.join("+")}@${s}:${ps.prec}`;
          let act = actMemo.get(ak);
          if (act == null) { act = peakActivationBytes(v.model, ps.prec); actMemo.set(ak, act); }
          const weightBytes = v.params * BYTES[ps.prec];
          const footprint = (weightBytes + act) * 1.1;
          const onchip = dev.memType === "onchip";
          const budget = onchip ? (dev.sramMB ?? 0) * 1e6 : dev.ramGB * 1e9;
          const notes: string[] = [];
          if (budget > 0 && footprint > budget) {
            if (onchip && dev.externalMem) notes.push("Exceeds on-chip memory; spills to host memory (slower than estimated)");
            else { pruned["no-memory"] += runtimes.length; continue; }
          }

          const needLayers = errIds.size > 0 && dev.cpuFallback === "graceful";
          const rf = rooflineFromPrepared(v.prep, dev, ps.prec, { peakOps: ps.peakTops * 1e12, totalsOnly: !needLayers });
          let latencyMs = rf.latencyMs;
          let fallbackLayers = 0;
          if (errIds.size > 0) {
            if (dev.cpuFallback === "single-partition") {
              // Everything from the first unsupported op on runs on the host (as scoreDevice).
              const first = Math.min(...errIds);
              let tail = 0, tot = 0;
              v.model.layers.forEach((l, i) => { tot += v.prep.flops[i]; if (l.id >= first) { tail += v.prep.flops[i]; fallbackLayers++; } });
              const coverage = tot > 0 ? Math.max(0.05, 1 - tail / tot) : 1;
              latencyMs = latencyMs / coverage;
              notes.push(`${fallbackLayers} layers after the first unsupported op run on the host CPU`);
            } else {
              let errMs = 0;
              for (const lt of rf.layers) if (errIds.has(lt.id)) errMs += lt.timeMs;
              fallbackLayers = errIds.size;
              latencyMs += errMs * (CPU_FALLBACK_SLOWDOWN - 1) + PARTITION_SWITCH_MS * fallbackLayers;
              notes.push(`${fallbackLayers} unsupported layer(s) fall back to the host CPU`);
            }
          }
          if (ps.estimated) notes.push(`${ps.prec.toUpperCase()} peak not in datasheet; derived`);
          if (!dev.fmts.includes(model.format)) notes.push(`Needs conversion from ${model.format}`);

          const r = risk({ model: v.model, device: dev, precision: ps.prec, fixes: fx, scale: s });
          const runtime = runtimes.find((rt) => scriptKeyFor(rt, ps.prec)) ?? runtimes[0];
          all.push({
            key: `${dev.id}|${ps.prec}|${fx.join("+") || "none"}|${s}`,
            device: dev, runtime, runtimes, precision: ps.prec, fixes: fx, scale: s,
            latencyMs, fps: latencyMs > 0 ? 1000 / latencyMs : 0,
            sizeMB: weightBytes / 1e6, footprintMB: footprint / 1e6, risk: r,
            errors, warnings, fallbackLayers, peakEstimated: ps.estimated,
            scriptKey: scriptKeyFor(runtime, ps.prec), notes,
            meetsConstraints: true, pareto: false, score: 0, rank: 0,
          });
        }
      }
    }
  }

  // Constraints → feasible set.
  for (const c of all) {
    c.meetsConstraints = (cons.maxLatencyMs == null || c.latencyMs <= cons.maxLatencyMs)
      && (cons.maxSizeMB == null || c.sizeMB <= cons.maxSizeMB)
      && (cons.maxRisk == null || c.risk <= cons.maxRisk);
  }
  const feasible = all.filter((c) => c.meetsConstraints);
  const pareto = paretoFront(feasible);
  for (const p of pareto) p.pareto = true;

  // Scalarized rank over normalized objectives (log latency: spans orders of magnitude).
  const norm = (vals: number[]) => {
    const lo = Math.min(...vals), hi = Math.max(...vals);
    return (v: number) => (hi > lo ? (v - lo) / (hi - lo) : 0);
  };
  if (feasible.length) {
    const nl = norm(feasible.map((c) => Math.log(Math.max(c.latencyMs, 1e-6))));
    const ns = norm(feasible.map((c) => c.sizeMB));
    const nr = norm(feasible.map((c) => c.risk));
    const tw = w.latency + w.size + w.risk || 1;
    for (const c of feasible) {
      c.score = 1 - (w.latency * nl(Math.log(Math.max(c.latencyMs, 1e-6))) + w.size * ns(c.sizeMB) + w.risk * nr(c.risk)) / tw;
    }
  }
  const ranked = [...feasible].sort((a, b) =>
    b.score - a.score
    || Number(b.pareto) - Number(a.pareto)
    || a.fixes.length - b.fixes.length
    || Number(!!b.scriptKey) - Number(!!a.scriptKey)
    || a.latencyMs - b.latencyMs);
  ranked.forEach((c, i) => { c.rank = i + 1; });

  return {
    ranked, pareto: pareto.sort((a, b) => a.latencyMs - b.latencyMs), all, pruned,
    spaceSize, evaluated: all.length, fixesConsidered, scalesConsidered, elapsedMs: now() - t0,
  };
}
