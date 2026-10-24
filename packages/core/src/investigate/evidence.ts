// Deterministic evidence for the AI performance investigator.
//
// For one model on one device this computes, without any AI:
//   • per-layer roofline time estimates (investigate/layerRoofline.ts, the same
//     model as the Hardware tab's scoring), the top-k bottleneck layers by time
//     share and each layer's compute- vs memory-bound regime;
//   • an op-type time breakdown;
//   • compiler issues for the device (ModelLayer.compIssues for its compatId)
//     and the auto-fixes that would resolve them;
//   • memory fit, precision state, dispatch overhead;
//   • unusual patterns: huge FC layers, unfolded BatchNorm, low-intensity
//     depthwise convs, long-sequence attention, Transpose/Reshape chains,
//     dynamic shapes.
// The result is a compact, citable EvidencePack (ids E1…En) that the AI layer
// (@modelvisio/ai investigate) reasons over, and that the rule-based fallback
// can diagnose on its own.
//
// Pure module: no React, no side effects.

import type { Model, ModelLayer } from "@modelvisio/parsers";
import type { EvidenceItem, EvidencePack } from "@modelvisio/ai";
import type { DeviceSpec } from "../data/hardware";
import { FIXES } from "../fixes/transforms";
import { scoreDevice } from "../scoring/scoreDevice";
import { layerRoofline, type LayerTime } from "./layerRoofline";

export type EvidenceOptions = {
  /** Number of bottleneck layers to report (default 8). */
  topK?: number;
  /** Number of op types in the breakdown (default 6). */
  topOps?: number;
};

const LAYOUT_RE = /^(transpose|reshape|permute|flatten|squeeze|unsqueeze|view|expand)$/i;
const FC_RE = /gemm|matmul|linear|fullyconnected|innerproduct|dense/i;
const BN_RE = /batchnorm/i;
const CONV_RE = /conv/i;
const ATTN_RE = /attention|softmax/i;

const pct = (x: number) => `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
const ms = (x: number) => (x >= 10 ? x.toFixed(1) : x >= 0.1 ? x.toFixed(2) : x.toFixed(3));
const mb = (b: number) => b / 1e6;

function num(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (Array.isArray(v) && typeof v[0] === "number") return v[0];
  return undefined;
}

/** Fixes whose transform removes at least one of this layer's issues for `target`. */
export function fixesResolving(l: ModelLayer, target: string): string[] {
  const before = l.compIssues.filter((c) => c.target === target).length;
  if (before === 0) return [];
  return FIXES.filter((f) => f.applies(l) && f.apply(l).compIssues.filter((c) => c.target === target).length < before).map((f) => f.id);
}

export function buildEvidence(model: Model, dev: DeviceSpec, opts: EvidenceOptions = {}): EvidencePack {
  const topK = opts.topK ?? 8;
  const topOps = opts.topOps ?? 6;
  const rf = layerRoofline(model, dev);
  const score = scoreDevice(model, dev);
  const total = rf.latencyMs > 0 ? rf.latencyMs : 1;
  const byId = new Map<number, LayerTime>(rf.layers.map((t) => [t.id, t]));
  const layerById = new Map<number, ModelLayer>(model.layers.map((l) => [l.id, l]));
  const share = (ids: Iterable<number>) => {
    let s = 0;
    for (const id of ids) s += byId.get(id)?.timeMs ?? 0;
    return s / total;
  };

  const evidence: EvidenceItem[] = [];
  const add = (e: Omit<EvidenceItem, "id">): EvidenceItem => {
    const item = { id: `E${evidence.length + 1}`, ...e };
    evidence.push(item);
    return item;
  };

  // ── Regime split ────────────────────────────────────────────────────────────
  let cMs = 0, mMs = 0;
  for (const t of rf.layers) (t.bound === "compute" ? (cMs += t.timeMs) : (mMs += t.timeMs));
  const computeBoundShare = cMs / total;
  const memoryBoundShare = mMs / total;
  const totalParams = model.layers.reduce((s, l) => s + l.params, 0);
  const totalFlops = model.layers.reduce((s, l) => s + (l.flops || l.macs * 2), 0);
  const modelBytes = rf.layers.reduce((s, t) => s + t.bytes, 0);
  const modelAI = modelBytes > 0 ? totalFlops / modelBytes : Infinity;

  // E1: roofline summary (always present, always citable).
  add({
    kind: "summary", severity: "info", title: "Roofline summary", layers: [],
    detail: `Est. ${ms(score.latencyMs)} ms/inference (~${Math.round(score.fps)} FPS) at ${rf.prec.toUpperCase()}; `
      + `utilization ${pct(rf.util)} (${rf.calibrated ? "calibrated" : "estimated"}, ${rf.workload}); `
      + `model intensity ${Number.isFinite(modelAI) ? modelAI.toFixed(1) : "∞"} vs ridge ${rf.ridgeAI.toFixed(1)} FLOP/B; `
      + `compute-bound layers ${pct(computeBoundShare)}, memory-bound ${pct(memoryBoundShare)} of time.`,
  });

  // ── Bottlenecks ─────────────────────────────────────────────────────────────
  const sorted = [...rf.layers].filter((t) => t.timeMs > 0).sort((a, b) => b.timeMs - a.timeMs);
  const bottlenecks: EvidencePack["bottlenecks"] = [];
  for (const t of sorted.slice(0, topK)) {
    const l = layerById.get(t.id);
    const sh = t.timeMs / total;
    const item = add({
      kind: "bottleneck", severity: sh >= 0.2 ? "warn" : "info", tag: t.bound,
      title: `${t.name} (${t.op})`, layers: [t.name], share: sh,
      fixIds: l && dev.compatId ? fixesResolving(l, dev.compatId) : [],
      detail: `${ms(t.timeMs)} ms = ${pct(sh)} of latency; ${t.bound}-bound; intensity ${Number.isFinite(t.ai) ? t.ai.toFixed(1) : "∞"} FLOP/B; `
        + `${(t.flops / 1e6).toFixed(1)} MFLOP, ${mb(t.bytes).toFixed(2)} MB moved`,
    });
    bottlenecks.push({ id: item.id, layer: t.name, op: t.op, timeMs: t.timeMs, share: sh, bound: t.bound, ai: t.ai });
  }

  // ── Op-type breakdown ───────────────────────────────────────────────────────
  const ops = new Map<string, { count: number; timeMs: number }>();
  for (const t of rf.layers) {
    const o = ops.get(t.op) ?? { count: 0, timeMs: 0 };
    o.count++; o.timeMs += t.timeMs;
    ops.set(t.op, o);
  }
  const opBreakdown = [...ops.entries()]
    .map(([op, v]) => ({ op, count: v.count, timeMs: v.timeMs, share: v.timeMs / total }))
    .filter((o) => o.timeMs > 0)
    .sort((a, b) => b.timeMs - a.timeMs)
    .slice(0, topOps);
  for (const o of opBreakdown) {
    add({
      kind: "op", severity: "info", title: `Op type ${o.op}`, layers: [], share: o.share,
      detail: `${o.count} layer(s), ${ms(o.timeMs)} ms = ${pct(o.share)} of latency`,
    });
  }

  // ── Compiler issues for this device ─────────────────────────────────────────
  if (dev.compatId) {
    const issueLayers = model.layers.filter((l) => l.compIssues.some((c) => c.target === dev.compatId));
    for (const l of issueLayers.slice(0, 12)) {
      const iss = l.compIssues.filter((c) => c.target === dev.compatId);
      const err = iss.some((c) => c.severity === "error");
      add({
        kind: "compiler", severity: err ? "error" : "warn", title: `${l.name}: ${err ? "unsupported" : "degraded"} on ${dev.n}`,
        layers: [l.name], share: share([l.id]), fixIds: fixesResolving(l, dev.compatId),
        detail: `${iss.map((c) => c.msg).join(" | ")}${err ? ` (CPU fallback mode: ${dev.cpuFallback})` : ""}`,
      });
    }
    if (score.opSupport.fallbackOps > 0 || score.opSupport.compileFail) {
      add({
        kind: "compiler", severity: "error", title: "Op coverage on accelerator", layers: [],
        share: 1 - score.opSupport.coverage,
        detail: score.opSupport.compileFail
          ? "Model will not compile: unsupported ops and no CPU fallback."
          : `${score.opSupport.fallbackOps} layer(s) run on the host CPU; ${pct(score.opSupport.coverage)} of FLOPs stay on the accelerator.`,
      });
    }
  }

  // ── Memory fit ──────────────────────────────────────────────────────────────
  const m = score.memory;
  add({
    kind: "memory",
    severity: m.budgetBytes <= 0 ? "info" : m.headroom < 1 ? "error" : m.headroom < 1.5 ? "warn" : "info",
    title: "Memory fit", layers: [],
    detail: `Footprint ~${mb(m.footprintBytes).toFixed(1)} MB (weights + peak activations at ${rf.prec.toUpperCase()}) vs `
      + `${m.budgetBytes > 0 ? `${mb(m.budgetBytes).toFixed(1)} MB ${dev.memType === "onchip" ? "on-chip" : "device"} budget (${m.headroom.toFixed(2)}× headroom)` : "unknown budget"}.`,
  });

  // ── Precision state ─────────────────────────────────────────────────────────
  const dts = new Map<string, number>();
  for (const l of model.layers) if (l.params > 0) dts.set(l.dt, (dts.get(l.dt) ?? 0) + 1);
  const stored = [...dts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "unknown";
  const sensitive = model.layers.filter((l) => l.qSens > 0.1).sort((a, b) => b.qSens - a.qSens);
  const quantizing = /int/.test(rf.prec) && !/int/.test(stored);
  add({
    kind: "precision",
    severity: quantizing && sensitive.length > 0 ? "warn" : "info",
    title: "Precision state", layers: sensitive.slice(0, 5).map((l) => l.name),
    detail: `Weights stored as ${stored}; ${dev.n} deploys at ${rf.prec.toUpperCase()}${dev.quant.int8Only ? " (INT8-only part)" : ""}`
      + `${quantizing ? ", so PTQ calibration is required" : ""}. `
      + (sensitive.length
        ? `${sensitive.length} quantization-sensitive layer(s) (qSens>10%): ${sensitive.slice(0, 5).map((l) => `${l.name} ${(l.qSens * 100).toFixed(0)}%`).join(", ")}.`
        : "No layers flagged as quantization-sensitive."),
  });

  // ── Patterns ────────────────────────────────────────────────────────────────
  const preds = new Map<number, number[]>();
  const succs = new Map<number, number[]>();
  for (const [a, b] of model.edges) {
    (succs.get(a) ?? succs.set(a, []).get(a)!).push(b);
    (preds.get(b) ?? preds.set(b, []).get(b)!).push(a);
  }
  const txt = (l: ModelLayer) => `${l.op} ${l.type}`;

  // Huge FC / MatMul.
  const fcMin = Math.max(1e6, totalParams * 0.25);
  const fcs = model.layers.filter((l) => FC_RE.test(txt(l)) && l.params >= fcMin);
  if (fcs.length) {
    const sh = share(fcs.map((l) => l.id));
    add({
      kind: "pattern", tag: "huge-fc", severity: sh >= 0.1 ? "warn" : "info", title: "Large fully-connected layers",
      layers: fcs.slice(0, 8).map((l) => l.name), share: sh,
      detail: `${fcs.length} FC/MatMul layer(s) with ≥${(fcMin / 1e6).toFixed(1)}M params (${pct(fcs.reduce((s, l) => s + l.params, 0) / Math.max(totalParams, 1))} of all params) take ${pct(sh)} of latency.`,
    });
  }

  // BatchNorm not folded into the preceding Conv.
  const bns = model.layers.filter((l) => BN_RE.test(txt(l)) && (preds.get(l.id) ?? []).some((p) => CONV_RE.test(txt(layerById.get(p) ?? ({ op: "", type: "" } as ModelLayer)))));
  if (bns.length) {
    const sh = share(bns.map((l) => l.id));
    add({
      kind: "pattern", tag: "unfused-bn", severity: sh >= 0.05 ? "warn" : "info", title: "Unfolded Conv→BatchNorm",
      layers: bns.slice(0, 8).map((l) => l.name), share: sh,
      detail: `${bns.length} BatchNorm layer(s) directly after a Conv are still separate ops in the graph (${pct(sh)} of latency if not folded by the compiler).`,
    });
  }

  // Depthwise convs below the ridge point.
  const dws = model.layers.filter((l) => {
    if (!CONV_RE.test(txt(l))) return false;
    const g = num(l.attr.groups) ?? num(l.attr.group) ?? 1;
    const cin = l.ins[0]?.s?.[1] ?? 0;
    const isDw = /depthwise|dwconv/i.test(txt(l)) || (g > 1 && (cin <= 0 || g === cin));
    const t = byId.get(l.id);
    return isDw && !!t && t.ai < rf.ridgeAI;
  });
  if (dws.length) {
    const sh = share(dws.map((l) => l.id));
    add({
      kind: "pattern", tag: "depthwise", severity: sh >= 0.1 ? "warn" : "info", title: "Low-intensity depthwise convs",
      layers: dws.slice(0, 8).map((l) => l.name), share: sh,
      detail: `${dws.length} depthwise conv(s) run below the ridge point (${rf.ridgeAI.toFixed(1)} FLOP/B), ${pct(sh)} of latency.`,
    });
  }

  // Attention over long sequences.
  const attn = model.layers.filter((l) => ATTN_RE.test(txt(l)) || (FC_RE.test(txt(l)) && (l.outs[0]?.s?.length ?? 0) === 4));
  const seqOf = (l: ModelLayer) => { const s = l.outs[0]?.s; return s && s.length >= 2 ? s[s.length - 1] : 0; };
  const longAttn = attn.filter((l) => ATTN_RE.test(txt(l)) && seqOf(l) >= 512);
  if (longAttn.length) {
    const sh = share(attn.map((l) => l.id));
    add({
      kind: "pattern", tag: "attention", severity: sh >= 0.1 ? "warn" : "info", title: "Long-sequence attention",
      layers: longAttn.slice(0, 8).map((l) => l.name), share: sh,
      detail: `Sequence length up to ${Math.max(...longAttn.map(seqOf))}; attention-related layers take ${pct(sh)} of latency (S² score matrices).`,
    });
  }

  // Transpose/Reshape chains.
  const isLayout = (id: number) => { const l = layerById.get(id); return !!l && (LAYOUT_RE.test(l.op) || LAYOUT_RE.test(l.type)); };
  const layoutLayers = model.layers.filter((l) => isLayout(l.id));
  const chains: number[][] = [];
  for (const l of layoutLayers) {
    if ((preds.get(l.id) ?? []).some(isLayout)) continue; // not a chain head
    const chain = [l.id];
    let cur = l.id;
    const seen = new Set([cur]);
    for (;;) {
      const nx = (succs.get(cur) ?? []).find((s) => isLayout(s) && !seen.has(s));
      if (nx == null) break;
      chain.push(nx); seen.add(nx); cur = nx;
    }
    if (chain.length >= 3) chains.push(chain);
  }
  const layoutShare = share(layoutLayers.map((l) => l.id));
  if (chains.length || (layoutLayers.length >= 8 && layoutShare >= 0.05)) {
    const names = (chains.length ? chains.flat() : layoutLayers.map((l) => l.id)).map((id) => layerById.get(id)!.name);
    add({
      kind: "pattern", tag: "layout", severity: layoutShare >= 0.1 ? "warn" : "info", title: "Transpose/Reshape churn",
      layers: names.slice(0, 8), share: layoutShare,
      detail: `${layoutLayers.length} layout op(s)${chains.length ? `, ${chains.length} chain(s) of ≥3 consecutive (longest ${Math.max(...chains.map((c) => c.length))})` : ""}; ${pct(layoutShare)} of latency.`,
    });
  }

  // Dynamic shapes.
  const dynIn = model.inputShape.some((d) => !(d > 0));
  const dyn = model.layers.filter((l) => [...l.ins, ...l.outs].some((x) => (x.s ?? []).some((d) => d < 0)));
  if (dynIn || dyn.length) {
    add({
      kind: "pattern", tag: "dynamic", severity: "warn", title: "Dynamic shapes",
      layers: dyn.slice(0, 8).map((l) => l.name),
      detail: `${dynIn ? `Input shape [${model.inputShape.join(", ")}] has unknown dims; ` : ""}${dyn.length} layer(s) carry dynamic dimensions (time estimates treat them as 1).`,
    });
  }

  // ── Dispatch overhead ───────────────────────────────────────────────────────
  const dispShare = dev.dispatchMs / total;
  if (dispShare >= 0.25) {
    add({
      kind: "overhead", severity: "warn", title: "Dispatch overhead", layers: [], share: dispShare,
      detail: `Fixed ${dev.dispatchMs} ms per-inference host/dispatch cost is ${pct(dispShare)} of the ${ms(total)} ms estimate; the model is too small to hide it.`,
    });
  }

  return {
    version: 1,
    model: {
      name: model.name, format: model.format, framework: model.framework,
      params: totalParams, flops: totalFlops, layers: model.layers.length, inputShape: model.inputShape,
      workload: rf.workload,
    },
    device: {
      id: dev.id, name: dev.n, vendor: dev.v, toolchain: dev.toolchain,
      precision: rf.prec, int8Only: dev.quant.int8Only, cpuFallback: dev.cpuFallback,
      peakTops: rf.peakOps / 1e12, bandwidthGBs: dev.bandwidthGBs, ridgeAI: rf.ridgeAI, memType: dev.memType,
    },
    totals: {
      latencyMs: score.latencyMs, fps: score.fps, computeMs: rf.computeMs, dispatchMs: dev.dispatchMs,
      util: rf.util, calibrated: rf.calibrated, bound: score.roofline.bound,
      computeBoundShare, memoryBoundShare,
      footprintMB: mb(m.footprintBytes), budgetMB: mb(m.budgetBytes), memFits: m.fits,
      opCoverage: score.opSupport.coverage,
    },
    bottlenecks,
    opBreakdown,
    evidence,
  };
}
