// Compiler auto-fix transforms. Each fix is a PURE function over the normalized
// Model graph: it rewrites the applicable layers (op/type, params, FLOPs, and the
// per-target compatibility issues it resolves) and returns a NEW Model — the
// input is never mutated. The app swaps its in-memory model to the result, so the
// graph, stats, and compiler pre-flight all update live and visibly.
//
// Scope (honest): these edit the in-memory *analysis* model — the graph, stats,
// compiler pre-flight, and the Convert tab's Graph-JSON / Layers-CSV exports all
// reflect them. They do NOT rewrite the original model file's bytes; producing a
// fixed binary (ONNX/TFLite/…) is a training/export-pipeline step, done with the
// op-level recipe these fixes describe — not in the browser.

import type { Model, ModelLayer } from "@modelvisio/parsers";

export type FixId = "silu_hardswish" | "sppf_parallel" | "resize_convtranspose" | "channel_prune";

export type FixDef = {
  id: FixId;
  title: string;
  change: string;
  impact: string;
  /** Does this fix apply to layer `l`? Drives the count + affected-layers list. */
  applies: (l: ModelLayer) => boolean;
  /** Rewrite one applicable layer (pure — returns a new layer). */
  apply: (l: ModelLayer) => ModelLayer;
};

const PRUNE_THRESHOLD = 200_000;

const dropIssues = (l: ModelLayer, re: RegExp) =>
  l.compIssues.filter((c) => !(re.test(c.msg) || re.test(c.target)));

function outChannels(l: ModelLayer): number {
  const s = l.outs?.[0]?.s;
  return s && s.length > 1 && s[1] > 0 ? s[1] : 0;
}
function spatial(l: ModelLayer): number {
  const s = l.outs?.[0]?.s;
  if (!s || s.length < 4) return 0;
  // Dynamic dims are stored as -1; treat unknown H/W as 0 so the FLOP delta is
  // skipped rather than going negative.
  return s[2] > 0 && s[3] > 0 ? s[2] * s[3] : 0;
}
/** Scale the channel (dim 1) of the primary output tensor, so pruning also shrinks
 *  the activation working set the scorer reads from shapes (not just `mem`). */
function pruneOutChannels(outs: ModelLayer["outs"]): ModelLayer["outs"] {
  return outs.map((o, i) =>
    i === 0 && o.s && o.s.length > 1 && o.s[1] > 0
      ? { ...o, s: o.s.map((d, j) => (j === 1 ? Math.max(1, Math.round(d * 0.75)) : d)) }
      : o,
  );
}

export const FIXES: FixDef[] = [
  {
    id: "silu_hardswish",
    title: "SiLU → HardSwish",
    change: "Replace SiLU/Swish activations with HardSwish (piecewise-linear).",
    impact: "≈ same accuracy (−0.2% typical) · fuses on NPUs · no Sigmoid+Mul CPU fallback.",
    // SiLU/Swish either by type, or the decomposed "Mul(Sigmoid)" op — NOT a
    // standalone Sigmoid (which is a different function we must not rewrite).
    applies: (l) => /silu|swish/i.test(l.type) || (/sigmoid/i.test(l.op) && /mul/i.test(l.op)),
    apply: (l) => ({
      ...l,
      type: "Hardswish",
      op: "HardSwish",
      math: "HardSwish(x) = x · ReLU6(x + 3) / 6",
      insight: "Hardened activation — piecewise-linear, maps to a single op on Coral / RKNN / most NPUs.",
      qSens: +(l.qSens * 0.7).toFixed(3),
      compIssues: dropIssues(l, /silu|sigmoid|swish/i),
    }),
  },
  {
    id: "sppf_parallel",
    title: "SPPF → parallel SPP",
    change: "Restructure the cascaded MaxPool chain (SPPF) into parallel pooling branches.",
    impact: "Compiles on Coral / Kneron · ≈ +5% latency · identical 13×13 receptive field.",
    applies: (l) => /sppf/i.test(l.type) || /sppf/i.test(l.op),
    apply: (l) => ({
      ...l,
      type: "SPP",
      op: "MaxPool×3(parallel)+Concat+Conv",
      insight: "Parallel SPP — three independent pools instead of a cascade; compiler-friendly.",
      compIssues: dropIssues(l, /maxpool|sppf|cascad/i),
    }),
  },
  {
    id: "resize_convtranspose",
    title: "Resize(nearest) → ConvTranspose",
    change: "Replace nearest-neighbour Resize/Upsample with a stride-2 transposed convolution.",
    impact: "Runs fully on-chip (no Resize CPU fallback) · adds a small learned upsampler.",
    applies: (l) => /resize|upsample/i.test(l.op) || /resize|upsample/i.test(l.type),
    apply: (l) => {
      const c = outChannels(l);
      const sp = spatial(l);
      const addParams = c > 0 ? c * 4 : 0; // depthwise 2×2 transposed conv
      const addFlops = c > 0 && sp > 0 ? 2 * c * sp * 4 : 0; // never negative
      return {
        ...l,
        type: "ConvTranspose2d",
        op: "ConvTranspose",
        params: l.params + addParams,
        macs: l.macs + Math.round(addFlops / 2),
        flops: l.flops + addFlops,
        insight: "Learned 2× upsample (depthwise transposed conv) — on-chip, no Resize CPU path.",
        qSens: Math.max(l.qSens, 0.02),
        compIssues: dropIssues(l, /resize|upsample|nearest/i),
      };
    },
  },
  {
    id: "channel_prune",
    title: "Channel prune 0.75×",
    change: `Structured channel pruning (keep 75%) on the heaviest layers (> ${(PRUNE_THRESHOLD / 1000).toFixed(0)}K params).`,
    impact: "−25% params/FLOPs on pruned layers · ≈ −0.5% mAP · needs a short fine-tune.",
    applies: (l) => l.params > PRUNE_THRESHOLD,
    apply: (l) => ({
      ...l,
      params: Math.round(l.params * 0.75),
      macs: Math.round(l.macs * 0.75),
      flops: Math.round(l.flops * 0.75),
      mem: Math.round(l.mem * 0.9),
      // Shrink the output channels too, so the activation working set the scorer
      // reads from shapes reflects the prune (not just the `mem` fallback).
      outs: pruneOutChannels(l.outs),
      insight: `${l.insight} · Pruned to 75% channels.`,
      qSens: Math.min(1, +(l.qSens + 0.02).toFixed(3)),
      w: l.w ? { ...l.w, sparse: Math.min(1, (l.w.sparse ?? 0) + 0.25) } : l.w,
    }),
  },
];

export function fixById(id: FixId): FixDef | undefined {
  return FIXES.find((f) => f.id === id);
}

/** How many layers a fix would touch in this model (0 → not applicable). */
export function countApplicable(model: Model, id: FixId): number {
  const f = fixById(id);
  if (!f) return 0;
  return model.layers.filter(f.applies).length;
}

/** Names of the layers a fix would touch (for the "Affected" list). */
export function applicableLayers(model: Model, id: FixId): string[] {
  const f = fixById(id);
  if (!f) return [];
  return model.layers.filter(f.applies).map((l) => l.name);
}

/** Apply a fix to every applicable layer, returning a NEW immutable Model. */
export function applyFix(model: Model, id: FixId): Model {
  const f = fixById(id);
  if (!f) return model;
  return { ...model, layers: model.layers.map((l) => (f.applies(l) ? f.apply(l) : l)) };
}
