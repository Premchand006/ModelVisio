// AI performance investigator: prompt template, response parsing/validation,
// and a rule-based fallback.
//
// Flow: the core package computes a deterministic EVIDENCE PACK (per-layer
// roofline times, bottlenecks, op breakdown, compiler issues, patterns) for one
// model on one device. This module hands that pack to Gemini through the normal
// sendChat → /api/chat proxy path (same {system, messages} contract every shell
// already implements; the key never reaches the browser), asks for ranked
// hypotheses that CITE evidence IDs, then validates the answer: hypotheses that
// cite no real evidence are dropped, and layer names the evidence never
// mentioned are stripped (hallucination guard). If the AI is unavailable
// (no key, proxy error, unparseable answer) the rule-based report built from
// the evidence alone is returned instead, so the feature always works.
//
// Pure module, no React.

import { sendChat, type ChatResponse, type SendChatArgs, type Source } from "./index";

// ─── Evidence pack contract (produced by @modelvisio/core investigate/evidence) ──

export type EvidenceKind =
  | "summary" | "bottleneck" | "op" | "compiler" | "memory" | "precision" | "pattern" | "overhead";

export type EvidenceItem = {
  /** Stable citation id, e.g. "E3". */
  id: string;
  kind: EvidenceKind;
  severity: "info" | "warn" | "error";
  title: string;
  /** Numbers and facts, compact. */
  detail: string;
  /** Layer names this item is about (all exist in the model). */
  layers: string[];
  /** Fraction of estimated latency this item accounts for (0..1), when known. */
  share?: number;
  /** Auto-fix ids (core fixes/transforms.ts) that would address this item. */
  fixIds?: string[];
  /** Sub-type: pattern kind ("huge-fc", "depthwise", …) or a bottleneck's bound ("compute" | "memory"). */
  tag?: string;
};

export type EvidencePack = {
  version: 1;
  model: {
    name: string; format: string; framework: string;
    params: number; flops: number; layers: number; inputShape: number[];
    workload: string;
  };
  device: {
    id: string; name: string; vendor: string; toolchain: string[];
    precision: string; int8Only: boolean; cpuFallback: string;
    peakTops: number; bandwidthGBs: number; ridgeAI: number; memType: string;
  };
  totals: {
    latencyMs: number; fps: number; computeMs: number; dispatchMs: number;
    util: number; calibrated: boolean; bound: "compute" | "memory";
    computeBoundShare: number; memoryBoundShare: number;
    footprintMB: number; budgetMB: number; memFits: boolean; opCoverage: number;
  };
  /** Top layers by estimated time; `id` is the matching evidence item's id. */
  bottlenecks: { id: string; layer: string; op: string; timeMs: number; share: number; bound: "compute" | "memory"; ai: number }[];
  opBreakdown: { op: string; count: number; timeMs: number; share: number }[];
  evidence: EvidenceItem[];
};

// ─── Report shape ────────────────────────────────────────────────────────────

export type Experiment = { action: string; expectedGain: string };

export type Hypothesis = {
  rank: number;
  title: string;
  confidence: "high" | "medium" | "low";
  /** Evidence ids cited (all exist in the pack). */
  evidence: string[];
  /** Layer names (all present in the evidence). */
  layers: string[];
  explanation: string;
  experiments: Experiment[];
  /** Auto-fix ids suggested by the cited evidence. */
  fixIds: string[];
};

export type InvestigationReport = {
  source: "ai" | "rules";
  summary: string;
  hypotheses: Hypothesis[];
  sources: Source[];
  /** What validation stripped from the AI answer. */
  rejected: { layers: string[]; evidence: string[]; hypotheses: number };
  /** Why the rule-based fallback was used, when it was. */
  note?: string;
};

const MAX_HYPOTHESES = 6;
const MAX_EXPERIMENTS = 4;

// ─── Prompt ──────────────────────────────────────────────────────────────────

const pct = (x: number) => `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;

/** The evidence pack as compact prompt text (JSON, rounded numbers). */
export function formatEvidenceForPrompt(pack: EvidencePack): string {
  const round = (_k: string, v: unknown) =>
    typeof v === "number" && Number.isFinite(v) && !Number.isInteger(v) ? +v.toPrecision(4) : v;
  return JSON.stringify(pack, round);
}

/**
 * Build the two strings handed to sendChat: `modelSummary` (rides in the
 * copilot system prompt, so the evidence is the grounding context) and the user
 * turn with the task, the user's question, and the strict JSON answer schema.
 */
export function buildInvestigationPrompt(pack: EvidencePack, question?: string): { modelSummary: string; userMessage: string } {
  const modelSummary = [
    `PERFORMANCE INVESTIGATION for "${pack.model.name}" on ${pack.device.name}.`,
    `Estimated ${pack.totals.latencyMs.toFixed(2)} ms/inference (~${Math.round(pack.totals.fps)} FPS) at ${pack.device.precision.toUpperCase()}, ` +
      `${pack.totals.bound}-bound overall (${pct(pack.totals.memoryBoundShare)} of time in memory-bound layers).`,
    "Deterministic EVIDENCE PACK (roofline estimates computed by ModelVisio; ids like E1 are citable):",
    formatEvidenceForPrompt(pack),
  ].join("\n");

  const q = question?.trim() || `Why is this model slow on ${pack.device.name}, and what should I change first?`;
  const userMessage = [
    `QUESTION: ${q}`,
    "",
    "TASK: Diagnose the performance of this model on this device using ONLY the evidence pack.",
    `Return a ranked list of at most ${MAX_HYPOTHESES} hypotheses, most impactful first. Every hypothesis MUST cite`,
    "one or more evidence ids from the pack (e.g. \"E2\"). Only name layers that appear in the pack's",
    "evidence/bottlenecks; never invent layer names, numbers, or ids. For each hypothesis give concrete",
    "experiments (exact flag/tool/change) with an expected gain grounded in the cited time shares.",
    "",
    "Respond with ONLY one JSON object, no prose before or after, no markdown fences, matching:",
    '{"summary":"<1-2 sentence direct answer>","hypotheses":[{"title":"<short>","confidence":"high|medium|low",' +
      '"evidence":["E1"],"layers":["<layer name from the pack>"],"explanation":"<why, citing numbers>",' +
      '"experiments":[{"action":"<concrete change>","expectedGain":"<e.g. ~30% latency>"}]}]}',
  ].join("\n");

  return { modelSummary, userMessage };
}

// ─── Response parsing + validation ───────────────────────────────────────────

/** Pull the first balanced JSON object out of a model reply (handles fences / prose). */
export function extractJsonObject(text: string): unknown | null {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidates = fenced ? [fenced[1], text] : [text];
  for (const c of candidates) {
    let start = c.indexOf("{");
    while (start !== -1) {
      let depth = 0, inStr = false, esc = false;
      for (let i = start; i < c.length; i++) {
        const ch = c[i];
        if (inStr) {
          if (esc) esc = false;
          else if (ch === "\\") esc = true;
          else if (ch === "\"") inStr = false;
          continue;
        }
        if (ch === "\"") inStr = true;
        else if (ch === "{") depth++;
        else if (ch === "}") {
          depth--;
          if (depth === 0) {
            try { return JSON.parse(c.slice(start, i + 1)); } catch { break; }
          }
        }
      }
      start = c.indexOf("{", start + 1);
    }
  }
  return null;
}

const str = (v: unknown, max = 600): string => (typeof v === "string" ? v.trim().slice(0, max) : "");
const strArr = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string").map((s) => s.trim()) : []);

/** Every layer name the evidence mentions: the only names an answer may use. */
export function knownLayers(pack: EvidencePack): Set<string> {
  const s = new Set<string>();
  for (const b of pack.bottlenecks) s.add(b.layer);
  for (const e of pack.evidence) for (const l of e.layers) s.add(l);
  return s;
}

/**
 * Validate a parsed AI answer against the pack. Unknown evidence ids and layer
 * names are stripped (and counted in `rejected`); a hypothesis left with no
 * valid citation is dropped. Returns null when nothing usable remains.
 */
export function validateInvestigation(raw: unknown, pack: EvidencePack): Omit<InvestigationReport, "source" | "sources"> | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as { summary?: unknown; hypotheses?: unknown };
  if (!Array.isArray(obj.hypotheses)) return null;
  // Bottleneck rows are evidence items too, so this covers every citable id.
  const ids = new Map(pack.evidence.map((e) => [e.id, e]));
  const layers = knownLayers(pack);
  const rejected = { layers: [] as string[], evidence: [] as string[], hypotheses: 0 };
  const out: Hypothesis[] = [];

  for (const h of obj.hypotheses) {
    if (!h || typeof h !== "object") { rejected.hypotheses++; continue; }
    const r = h as Record<string, unknown>;
    const ev = strArr(r.evidence).map((s) => s.toUpperCase());
    const goodEv = [...new Set(ev.filter((id) => ids.has(id)))];
    for (const id of ev) if (!ids.has(id) && !rejected.evidence.includes(id)) rejected.evidence.push(id);
    const ls = strArr(r.layers);
    const goodLs = [...new Set(ls.filter((l) => layers.has(l)))];
    for (const l of ls) if (!layers.has(l) && !rejected.layers.includes(l)) rejected.layers.push(l);
    const title = str(r.title, 140);
    if (goodEv.length === 0 || !title) { rejected.hypotheses++; continue; }
    const conf = str(r.confidence).toLowerCase();
    const experiments: Experiment[] = (Array.isArray(r.experiments) ? r.experiments : [])
      .map((x) => (x && typeof x === "object" ? x as Record<string, unknown> : {}))
      .map((x) => ({ action: str(x.action, 300), expectedGain: str(x.expectedGain, 120) }))
      .filter((x) => x.action)
      .slice(0, MAX_EXPERIMENTS);
    const fixIds = [...new Set(goodEv.flatMap((id) => ids.get(id)?.fixIds ?? []))];
    out.push({
      rank: out.length + 1,
      title,
      confidence: conf === "high" || conf === "low" ? conf : "medium",
      evidence: goodEv,
      layers: goodLs,
      explanation: str(r.explanation, 1200),
      experiments,
      fixIds,
    });
    if (out.length >= MAX_HYPOTHESES) break;
  }
  if (out.length === 0) return null;
  return { summary: str(obj.summary, 600) || out[0].title, hypotheses: out, rejected };
}

/** Parse + validate a raw AI reply. Null when it can't be used. */
export function parseInvestigation(text: string, pack: EvidencePack): Omit<InvestigationReport, "source" | "sources"> | null {
  return validateInvestigation(extractJsonObject(text), pack);
}

// ─── Rule-based fallback ─────────────────────────────────────────────────────

type Draft = Omit<Hypothesis, "rank"> & { impact: number };

/**
 * Deterministic diagnosis from the evidence alone. Each rule maps an evidence
 * kind/tag to a hypothesis with experiments; hypotheses are ranked by the share
 * of latency they touch (blocking issues first).
 */
export function ruleBasedReport(pack: EvidencePack): InvestigationReport {
  const drafts: Draft[] = [];
  const dev = pack.device;
  const ev = pack.evidence;
  const of = (pred: (e: EvidenceItem) => boolean) => ev.filter(pred);
  const cite = (items: EvidenceItem[]) => items.map((e) => e.id);
  const lay = (items: EvidenceItem[]) => [...new Set(items.flatMap((e) => e.layers))].slice(0, 8);
  const fixes = (items: EvidenceItem[]) => [...new Set(items.flatMap((e) => e.fixIds ?? []))];

  const compErr = of((e) => e.kind === "compiler" && e.severity === "error");
  if (compErr.length) {
    const fx = fixes(compErr);
    drafts.push({
      title: dev.cpuFallback === "none" ? "Unsupported ops block compilation" : "Unsupported ops force CPU fallback",
      confidence: "high", evidence: cite(compErr), layers: lay(compErr), fixIds: fx,
      explanation: dev.cpuFallback === "none"
        ? `${dev.name} has no CPU fallback, so ${compErr.length} unsupported op(s) stop the model compiling at all.`
        : dev.cpuFallback === "single-partition"
          ? `${dev.name} runs everything after the first unsupported op on the host CPU (single partition), so most of the graph leaves the accelerator.`
          : `${compErr.length} op(s) run on the host CPU, adding transfers and slow CPU kernels between accelerator partitions.`,
      experiments: [
        ...(fx.length ? [{ action: `Apply auto-fix: ${fx.join(", ")} (Compiler tab)`, expectedGain: "removes the CPU fallback for the fixed ops" }] : []),
        { action: `Rewrite or replace the flagged ops with ones ${dev.toolchain[0]} supports, then re-check the Compiler tab`, expectedGain: "full on-accelerator execution" },
      ],
      impact: 2,
    });
  }

  const mem = of((e) => e.kind === "memory" && e.severity !== "info");
  if (mem.length) {
    drafts.push({
      title: pack.totals.memFits ? "Tight memory headroom" : "Model does not fit device memory",
      confidence: "high", evidence: cite(mem), layers: lay(mem), fixIds: fixes(mem),
      explanation: `Estimated footprint ${pack.totals.footprintMB.toFixed(1)} MB vs ${pack.totals.budgetMB.toFixed(1)} MB budget; spills or failure dominate any compute gains.`,
      experiments: [
        { action: "Quantize weights to INT8 (or FP16 where INT8 is unavailable)", expectedGain: "2-4× smaller weights" },
        { action: "Structured channel pruning on the heaviest layers + short fine-tune", expectedGain: "~25% fewer params on pruned layers" },
      ],
      impact: pack.totals.memFits ? 0.6 : 1.9,
    });
  }

  const memBound = of((e) => e.kind === "bottleneck" && e.tag === "memory");
  if (pack.totals.memoryBoundShare >= 0.5) {
    const items = memBound.length ? memBound : of((e) => e.kind === "summary");
    const cur = dev.precision;
    drafts.push({
      title: "Bandwidth-bound: time goes to moving bytes, not math",
      confidence: "high", evidence: cite(items), layers: lay(items), fixIds: [],
      explanation: `${pct(pack.totals.memoryBoundShare)} of estimated time is in layers below the device ridge point (${pack.device.ridgeAI.toFixed(1)} FLOP/B); more TOPS won't help, fewer bytes will.`,
      experiments: [
        ...(cur === "fp32" || cur === "fp16" ? [{ action: `Deploy at ${cur === "fp32" ? "FP16" : "INT8"} (halves activation + weight bytes)`, expectedGain: `up to ~${pct(pack.totals.memoryBoundShare / 2)} latency` }] : []),
        { action: "Make sure Conv+BN+activation are fused so intermediates stay on-chip", expectedGain: "fewer full-resolution round trips" },
        { action: "Reduce input resolution (e.g. 0.75×) if accuracy allows", expectedGain: "~44% fewer activation bytes" },
      ],
      impact: pack.totals.memoryBoundShare * 0.9,
    });
  }

  const top = of((e) => e.kind === "bottleneck").sort((a, b) => (b.share ?? 0) - (a.share ?? 0))[0];
  if (top && (top.share ?? 0) >= 0.2) {
    drafts.push({
      title: `One layer dominates: ${top.layers[0]}`,
      confidence: "medium", evidence: [top.id], layers: top.layers, fixIds: top.fixIds ?? [],
      explanation: `${top.layers[0]} alone takes ${pct(top.share ?? 0)} of estimated latency (${top.detail}).`,
      experiments: [
        { action: `Profile ${top.layers[0]} on-device (e.g. ${dev.toolchain[0]} per-layer profiler) to confirm`, expectedGain: "validates the estimate" },
        { action: "Reduce its channels/kernel or replace with a cheaper block (e.g. depthwise-separable)", expectedGain: `up to ~${pct((top.share ?? 0) / 2)} latency` },
      ],
      impact: top.share ?? 0,
    });
  }

  const PATTERN: Record<string, { title: string; why: string; exps: Experiment[] }> = {
    "huge-fc": {
      title: "Large fully-connected layer is weight-bandwidth bound",
      why: "At batch 1 a big FC/MatMul reads every weight once per inference with almost no reuse.",
      exps: [{ action: "INT8 weight quantization or low-rank factorization of the FC", expectedGain: "2-4× fewer weight bytes" }, { action: "Replace the flatten+FC head with global pooling + a small FC", expectedGain: "most of that layer's time" }],
    },
    "unfused-bn": {
      title: "BatchNorm not folded into Conv",
      why: "Standalone BN layers add a full read+write of each activation unless the compiler folds them.",
      exps: [{ action: "Export in eval mode / fold BN into Conv weights (e.g. onnxsim, torch.fx fuse)", expectedGain: "removes the BN layers' time" }],
    },
    "depthwise": {
      title: "Depthwise convolutions have low arithmetic intensity",
      why: "Depthwise convs do few FLOPs per byte, so they sit far below the ridge point and under-use the accelerator.",
      exps: [{ action: "Keep depthwise+pointwise fused, or swap for a regular conv in early stages on GPUs/NPUs", expectedGain: "better utilization on the flagged layers" }],
    },
    "attention": {
      title: "Attention cost grows quadratically with sequence length",
      why: "Softmax/attention over a long sequence moves an S×S score matrix per head.",
      exps: [{ action: "Use a fused attention kernel (FlashAttention / TensorRT MHA plugin) or shorten the sequence", expectedGain: "large cut in attention memory traffic" }],
    },
    "layout": {
      title: "Transpose/Reshape chains add layout churn",
      why: "Chains of layout ops copy whole tensors without doing math and often break fusion or force CPU fallback.",
      exps: [{ action: "Simplify the graph (onnxsim) and fold consecutive Reshape/Transpose", expectedGain: "removes the copies" }],
    },
    "dynamic": {
      title: "Dynamic shapes limit compiler optimization",
      why: "Unknown dims stop many edge compilers from static allocation and kernel selection; some reject them outright.",
      exps: [{ action: "Export with fixed input shapes (dynamic=False) for the deployment resolution", expectedGain: "enables static compilation" }],
    },
  };
  for (const e of of((x) => x.kind === "pattern")) {
    const p = PATTERN[e.tag ?? ""];
    if (!p) continue;
    drafts.push({
      title: p.title, confidence: e.severity === "info" ? "low" : "medium",
      evidence: [e.id], layers: e.layers.slice(0, 8), fixIds: e.fixIds ?? [],
      explanation: `${p.why} ${e.detail}`, experiments: p.exps, impact: (e.share ?? 0.05) * 0.8,
    });
  }

  const prec = of((e) => e.kind === "precision" && e.severity !== "info");
  if (prec.length) {
    drafts.push({
      title: "Precision / quantization needs attention",
      confidence: "medium", evidence: cite(prec), layers: lay(prec), fixIds: [],
      explanation: prec.map((e) => e.detail).join(" "),
      experiments: [
        { action: "Calibrate INT8 PTQ with ~200 representative images; keep the most sensitive layers in FP16 if the runtime allows mixed precision", expectedGain: "INT8 speed with bounded accuracy loss" },
      ],
      impact: 0.3,
    });
  }

  const over = of((e) => e.kind === "overhead");
  if (over.length) {
    drafts.push({
      title: "Per-inference dispatch overhead dominates",
      confidence: "medium", evidence: cite(over), layers: [], fixIds: [],
      explanation: over[0].detail,
      experiments: [{ action: "Batch frames or pipeline pre/post-processing with inference (async queues)", expectedGain: "amortizes host/USB overhead" }],
      impact: over[0].share ?? 0.3,
    });
  }

  if (drafts.length === 0) {
    const s = of((e) => e.kind === "summary");
    drafts.push({
      title: "No dominant bottleneck: the model is well matched to this device",
      confidence: "low", evidence: cite(s.length ? s : ev.slice(0, 1)), layers: [], fixIds: [],
      explanation: `Estimated ${pack.totals.latencyMs.toFixed(2)} ms with ${pct(pack.totals.util)} utilization; gains now come from precision and runtime tuning.`,
      experiments: [{ action: "Measure on-device to calibrate this estimate", expectedGain: "confidence" }],
      impact: 0,
    });
  }

  const hyps = drafts
    .filter((d) => d.evidence.length > 0)
    .sort((a, b) => b.impact - a.impact)
    .slice(0, MAX_HYPOTHESES)
    .map(({ impact: _impact, ...h }, i) => ({ ...h, rank: i + 1 }));
  const top1 = hyps[0];
  return {
    source: "rules",
    summary: top1
      ? `${pack.model.name} is estimated at ${pack.totals.latencyMs.toFixed(2)} ms on ${dev.name} (${pack.totals.bound}-bound). Main suspect: ${top1.title.toLowerCase()}.`
      : `No issues found.`,
    hypotheses: hyps,
    sources: [],
    rejected: { layers: [], evidence: [], hypotheses: 0 },
  };
}

// ─── Orchestration ───────────────────────────────────────────────────────────

export type InvestigateArgs = {
  evidence: EvidencePack;
  question?: string;
  /** False → skip the AI call (e.g. desktop with no key) and return the rule-based report. */
  aiEnabled?: boolean;
  endpoint?: string;
  signal?: AbortSignal;
  /** Injectable for tests; defaults to sendChat (POST /api/chat). */
  send?: (args: SendChatArgs) => Promise<ChatResponse>;
};

/**
 * Run the investigation. Never throws (except on abort): any AI failure or an
 * unusable answer yields the rule-based report with `note` saying why.
 */
export async function investigate({
  evidence, question, aiEnabled = true, endpoint, signal, send = sendChat,
}: InvestigateArgs): Promise<InvestigationReport> {
  if (!aiEnabled) return { ...ruleBasedReport(evidence), note: "AI unavailable: showing the rule-based diagnosis." };
  const { modelSummary, userMessage } = buildInvestigationPrompt(evidence, question);
  let reply: ChatResponse;
  try {
    reply = await send({ endpoint, modelSummary, messages: [{ role: "user", content: userMessage }], signal });
  } catch (e) {
    if (signal?.aborted) throw e;
    const msg = e instanceof Error ? e.message : String(e);
    return { ...ruleBasedReport(evidence), note: `AI request failed (${msg}). Showing the rule-based diagnosis.` };
  }
  const parsed = parseInvestigation(reply.text, evidence);
  if (!parsed) {
    return { ...ruleBasedReport(evidence), note: "The AI answer could not be validated against the evidence. Showing the rule-based diagnosis." };
  }
  return { source: "ai", sources: reply.sources, ...parsed };
}
