// Accuracy-risk proxies for the optimization search. A config's risk is a 0..1
// number: 0 = no expected accuracy change, 1 = very likely broken. It is an
// ESTIMATE, not measured accuracy; the search uses it only to trade off against
// latency and size.
//
// Two estimators share one interface:
//   • sensitivityRisk: uses the per-layer quantization-sensitivity analysis
//     (quant/sensitivity.ts: weight SQNR + op-type priors + position) and
//     aggregates the per-layer INT8/INT4 risks for the model;
//   • heuristicRisk: a small precision-based heuristic over ModelLayer.qSens,
//     used as the fallback when the analysis is unavailable or throws.
// Both add the same penalties for accuracy-affecting auto-fixes and for
// reduced input resolution.
//
// Pure module: no React.

import type { Model } from "@modelvisio/parsers";
import type { DeviceSpec, Precision } from "../data/hardware";
import type { FixId } from "../fixes/transforms";
import { analyzeQuantization, type QuantAnalysis } from "../quant/sensitivity";

export type RiskContext = {
  /** The model AFTER fixes and resolution scaling. */
  model: Model;
  device: DeviceSpec;
  precision: Precision;
  fixes: FixId[];
  /** Input resolution scale (1 = native). */
  scale: number;
};

export type RiskEstimator = (ctx: RiskContext) => number;

/** Accuracy cost of each auto-fix (activation swap / learned upsampler / pruning need a fine-tune). */
export const FIX_RISK: Record<FixId, number> = {
  silu_hardswish: 0.01,
  sppf_parallel: 0,
  resize_convtranspose: 0.015,
  channel_prune: 0.04,
};

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));

/** Penalties shared by every estimator: fixes + resolution. */
export function structuralRisk(fixes: FixId[], scale: number): number {
  const fx = fixes.reduce((s, f) => s + (FIX_RISK[f] ?? 0), 0);
  const res = scale < 1 ? (1 - scale) * 0.2 : 0;
  return fx + res;
}

/** Params-weighted mean qSens over weighted layers (falls back to plain mean). */
function meanQSens(model: Model): number {
  let w = 0, s = 0;
  for (const l of model.layers) if (l.params > 0) { w += l.params; s += l.qSens * l.params; }
  if (w > 0) return s / w;
  const n = model.layers.length;
  return n ? model.layers.reduce((a, l) => a + l.qSens, 0) / n : 0;
}

const PREC_BASE: Record<Precision, number> = { fp32: 0, bf16: 0.01, fp16: 0.005, int8: 0.05, int4: 0.18 };
const PREC_SENS: Record<Precision, number> = { fp32: 0, bf16: 0.02, fp16: 0.01, int8: 0.6, int4: 1.5 };

/** Precision heuristic over the parser's per-layer qSens. */
export const heuristicRisk: RiskEstimator = ({ model, device, precision, fixes, scale }) => {
  let r = PREC_BASE[precision] + PREC_SENS[precision] * meanQSens(model);
  if ((precision === "int8" || precision === "int4") && !device.quant.perChannel) r += 0.05;
  return clamp01(r + structuralRisk(fixes, scale));
};

const analysisCache = new WeakMap<Model, QuantAnalysis | null>();

function analysis(model: Model): QuantAnalysis | null {
  if (analysisCache.has(model)) return analysisCache.get(model)!;
  let a: QuantAnalysis | null = null;
  try { a = analyzeQuantization(model); } catch { a = null; }
  analysisCache.set(model, a);
  return a;
}

/**
 * Model-level risk from the per-layer sensitivity analysis: 60% the mean
 * quantizable-layer risk, 40% the mean of the worst decile (accuracy loss is
 * driven by the few most sensitive layers). Per-channel risks are used when the
 * device supports per-channel scales. Falls back to heuristicRisk on failure.
 */
export const sensitivityRisk: RiskEstimator = (ctx) => {
  const { model, device, precision, fixes, scale } = ctx;
  if (precision !== "int8" && precision !== "int4") {
    return clamp01(PREC_BASE[precision] + structuralRisk(fixes, scale));
  }
  const a = analysis(model);
  if (!a) return heuristicRisk(ctx);
  const per = device.quant.perChannel;
  const rs = a.layers
    .filter((l) => l.quantizable)
    .map((l) => (precision === "int8" ? (per ? l.int8.perChannelRisk : l.int8.risk) : (per ? l.int4.perChannelRisk : l.int4.risk)))
    .filter((r) => Number.isFinite(r))
    .sort((x, y) => y - x);
  if (rs.length === 0) return heuristicRisk(ctx);
  const mean = rs.reduce((s, r) => s + r, 0) / rs.length;
  const k = Math.max(1, Math.ceil(rs.length * 0.1));
  const worst = rs.slice(0, k).reduce((s, r) => s + r, 0) / k;
  return clamp01(0.6 * mean + 0.4 * worst + structuralRisk(fixes, scale));
};

/** Default estimator for the search. */
export const defaultRisk: RiskEstimator = sensitivityRisk;
