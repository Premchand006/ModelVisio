// Per-layer quantization-sensitivity analysis (pure, no React).
//
// These are STATISTICAL ESTIMATES from the weights and the graph structure, not
// measured accuracy. Nothing here runs the model or sees calibration data, so
// activation ranges are unknown and are covered by op-type priors instead.
//
// How each layer is scored, per precision (INT8 = W8A8, INT4 = W4A8):
//
// 1. Weight SQNR. For a symmetric uniform quantizer with L = 2^(b-1) - 1 levels
//    on each side, step Δ = R / L where R = max|w|. The noise is ≈ Δ²/12 on every
//    non-zero weight (exact zeros stay exact). With signal power E[w²] = σ² + μ²
//    and crest factor c = R / rms:
//        SQNR = 10·log10( 12·L² / ((1 − sparsity) · c²) )
//    The per-tensor crest comes from the parser's sampled min/max/std. The
//    per-channel crest is estimated from the Gaussian expected maximum over one
//    output channel's fan-in, √(2·ln n_c), plus part of the measured tail excess.
//    When a layer has weights but no statistics, a Gaussian with a moderate
//    tail is assumed and the result is marked `statsSource: "assumed"`.
// 2. Op-type prior for the activation side: Softmax, LayerNorm, attention
//    MatMuls, residual adds, saturating activations, depthwise convs and so on
//    are known to be quantization-sensitive (Krishnamoorthi 2018; Sheng et al.
//    2018; Wu et al. 2020; Bondarenko et al. 2021; Dettmers et al. 2022).
// 3. Position: the first layers (raw input) and the last layers (logits, boxes)
//    are conventionally kept at higher precision.
//
// The three risks are combined as a noisy-OR into one 0..1 risk.

import type { Model, ModelLayer } from "@modelvisio/parsers";

export type QuantPrecision = "int8" | "int4";
export type Recommendation = "int4" | "int8" | "int8-per-channel" | "fp16" | "keep-fp32";
export type RiskLevel = "low" | "medium" | "high";
/** weighted: has a weight tensor · activation: arithmetic without weights ·
 *  passthrough: layout/data movement only · io: graph input/output tensor. */
export type LayerKind = "weighted" | "activation" | "passthrough" | "io";
export type StatsSource = "measured" | "assumed" | "none";

export type PrecisionEstimate = {
  /** Weight SQNR with one scale per tensor (dB); null when the layer has no weights. */
  sqnrDb: number | null;
  /** Weight SQNR with one scale per output channel (dB); null when no weights. */
  perChannelSqnrDb: number | null;
  /** Combined 0..1 risk with per-tensor scales. */
  risk: number;
  /** Combined 0..1 risk with per-channel scales (the best standard PTQ setup). */
  perChannelRisk: number;
  /** Level of the per-channel risk. */
  level: RiskLevel;
};

export type LayerSensitivity = {
  id: number;
  name: string;
  op: string;
  kind: LayerKind;
  /** Has arithmetic that quantization changes (weighted or activation). */
  quantizable: boolean;
  params: number;
  statsSource: StatsSource;
  /** max|w| / σ from the measured stats (null when unavailable). */
  outlierRatio: number | null;
  int8: PrecisionEstimate;
  int4: PrecisionEstimate;
  recommendation: Recommendation;
  reason: string;
  /** Every factor that raised the risk, most important first. */
  factors: string[];
  /** First weighted layer(s) after the input, or last before the output. */
  position: "first" | "last" | "middle";
};

export type QuantSummary = {
  layers: number;
  quantizable: number;
  weighted: number;
  /** Fraction of weighted layers with measured min/max/std statistics (0..1). */
  statsCoverage: number;
  totalParams: number;
  /** Weight storage at each precision, plus the mixed-precision plan. Bytes. */
  sizeBytes: { fp32: number; fp16: number; int8: number; int4: number; mixed: number };
  /** % of quantizable layers recommended INT8 or INT8 per-channel (0..100). */
  safeInt8Pct: number;
  /** % of weighted layers whose W4A8 per-channel risk is below the safe threshold (0..100). */
  safeInt4Pct: number;
  highRisk8: number;
  highRisk4: number;
  counts: Record<Recommendation, number>;
  /** Layers the plan keeps out of integer quantization (fp16 / keep-fp32). */
  excluded: LayerSensitivity[];
  /** Layers that are only safe with per-channel scales. */
  perChannel: LayerSensitivity[];
  target: QuantPrecision;
  confidence: "medium" | "low";
  note: string;
};

export type QuantAnalysis = {
  layers: LayerSensitivity[];
  byId: Map<number, LayerSensitivity>;
  summary: QuantSummary;
};

export type QuantOptions = {
  /** Plan target: INT8 everywhere it is safe (default), or INT4 weights where safe. */
  target?: QuantPrecision;
  /** Risk below this is "safe" (default 0.3). */
  safeRisk?: number;
  /** Risk at or above this is "high" (default 0.6). */
  highRisk?: number;
};

export const DISCLAIMER =
  "Statistical estimates from weight statistics, op types and graph position. Not measured accuracy: validate the plan on a calibration/eval set.";

export const REC_LABEL: Record<Recommendation, string> = {
  int4: "INT4 (W4A8)",
  int8: "INT8",
  "int8-per-channel": "INT8 per-channel",
  fp16: "FP16",
  "keep-fp32": "Keep FP32",
};

/** Bytes per weight element for each recommendation. */
export const REC_BYTES: Record<Recommendation, number> = {
  int4: 0.5, int8: 1, "int8-per-channel": 1, fp16: 2, "keep-fp32": 4,
};

const FP16_MAX = 65504;

// ── Op rules ──────────────────────────────────────────────────────────────────

type OpRule = { re: RegExp; prior: number; note: string; fp32?: boolean; kind?: LayerKind };

// Order matters: first match wins. Tested against lower-cased "op type".
const OP_RULES: OpRule[] = [
  { re: /softmax/, prior: 0.65, fp32: true, note: "Softmax: exponentials and a normalization; INT8 outputs crush small probabilities, so most INT8 toolchains keep it in float" },
  { re: /layernorm|layer_norm|groupnorm|instancenorm|rmsnorm|simplifiedlayernorm|\bnorm\b/, prior: 0.6, fp32: true, note: "Normalization layer: per-channel outliers make one activation scale a poor fit (Bondarenko et al. 2021)" },
  { re: /attention|scaleddotproduct|sdpa/, prior: 0.55, note: "Attention block: score and softmax ranges are input-dependent" },
  { re: /\b(exp|log|pow|sqrt|reciprocal|erf)\b/, prior: 0.45, fp32: true, note: "Exponential/power op: large dynamic range; float autocast lists keep these in FP32" },
  { re: /\bdiv\b/, prior: 0.4, note: "Division: amplifies quantization error in the divisor" },
  { re: /batchnorm|batch_norm/, prior: 0.3, note: "Standalone BatchNorm: fold it into the preceding conv before quantizing" },
  { re: /hardswish|hard_swish|hardsigmoid/, prior: 0.12, note: "Piecewise-linear activation: quantizes well" },
  { re: /\b(gelu|sigmoid|tanh|silu|swish|mish|elu|selu|celu|softplus)\b/, prior: 0.22, note: "Saturating nonlinearity: needs a fine output scale near the knee" },
  { re: /\b(maxpool|globalmaxpool|relu|leakyrelu|prelu|clip|relu6)\b/, prior: 0.03, note: "Commutes with quantization (order-preserving)" },
  { re: /averagepool|globalaveragepool|reducemean|avgpool/, prior: 0.1, note: "Averaging: rounding error mostly averages out" },
  { re: /resize|upsample/, prior: 0.08, note: "Resize: interpolation keeps the input range" },
  { re: /concat/, prior: 0.2, note: "Concat: branches must share one scale, so the narrower branch loses resolution" },
  { re: /\b(add|sum|sub)\b/, prior: 0.28, note: "Residual/elementwise add: two branches with different scales are requantized" },
  { re: /\bmul\b/, prior: 0.3, note: "Elementwise multiply: relative errors of both operands add up" },
  { re: /conv|gemm|matmul|linear|dense|fullyconnected|embedding|gather/, prior: 0.12, note: "" },
  { re: /c2f|c3k?\b|bottleneck|sppf|\bspp\b|block/, prior: 0.18, note: "Composite block (convs with an internal residual/concat)" },
];

const PASSTHROUGH =
  /^(reshape|transpose|flatten|squeeze|unsqueeze|identity|shape|cast|constant|constantofshape|slice|split|expand|tile|dropout|pad|depthtospace|spacetodepth|permute|view|contiguous|nonzero|range|where)$/;

function opKey(l: ModelLayer): string {
  return `${l.op} ${l.type}`.toLowerCase();
}

function isIo(l: ModelLayer): boolean {
  const s = opKey(l);
  return /\b(input|placeholder)\b/.test(s) || (/\boutput\b/.test(l.type.toLowerCase()) && l.params === 0);
}

function kindOf(l: ModelLayer): LayerKind {
  if (l.params > 0) return "weighted";
  if (isIo(l)) return "io";
  const op = l.op.toLowerCase();
  if (PASSTHROUGH.test(op) || PASSTHROUGH.test(l.type.toLowerCase())) return "passthrough";
  return "activation";
}

function ruleFor(l: ModelLayer): OpRule | null {
  const s = opKey(l);
  for (const r of OP_RULES) if (r.re.test(s)) return r;
  return null;
}

/** Depthwise conv: groups equals the input channels (or weight dim 1 is 1 with groups > 1). */
export function isDepthwise(l: ModelLayer): boolean {
  if (!/conv/i.test(l.op + l.type)) return false;
  const g = Number((l.attr.groups ?? l.attr.group) as number | undefined);
  if (!(g > 1)) return false;
  const inC = l.ins[0]?.s?.[1];
  const ws = Array.isArray(l.w?.shape) ? l.w!.shape : null;
  return (inC != null && inC > 0 && g === inC) || (ws != null && ws[1] === 1);
}

// ── Weight SQNR ───────────────────────────────────────────────────────────────

/** Expected max |x| over n Gaussian samples, in units of σ (≈ √(2 ln n)). */
export function gaussianCrest(n: number): number {
  return Math.max(Math.sqrt(3), Math.sqrt(2 * Math.log(Math.max(n, 2))));
}

/** SQNR (dB) of a symmetric b-bit uniform quantizer for crest factor c = max|w|/rms. */
export function quantSqnrDb(bits: number, crest: number, sparsity = 0): number {
  const L = Math.pow(2, bits - 1) - 1;
  const c = Math.max(1, crest);
  const nz = Math.max(1e-6, 1 - Math.min(Math.max(sparsity, 0), 0.999999));
  return 10 * Math.log10((12 * L * L) / (nz * c * c));
}

/** Map an SQNR (dB) to a 0..1 risk. Logistic centred at 20 dB (noise at 1% of
 *  signal power), 3 dB scale: 35 dB → ~0.01, 26 dB → ~0.12, 15 dB → ~0.84. */
export function sqnrRisk(db: number): number {
  return 1 / (1 + Math.exp((db - 20) / 3));
}

function weightElems(l: ModelLayer): number {
  const w = l.w;
  if (w?.size && w.size > 0) return w.size;
  if (Array.isArray(w?.shape) && w!.shape.length) return w!.shape.reduce((a, b) => a * Math.max(1, b), 1);
  return Math.max(1, l.params);
}

/** Fan-in per output channel (elements sharing one per-channel scale). */
function perChannelElems(l: ModelLayer, n: number): number {
  const ws = Array.isArray(l.w?.shape) ? (l.w!.shape as number[]) : null;
  let outC = 0;
  if (ws && ws.length >= 2) outC = /matmul/i.test(l.op) ? ws[ws.length - 1] : ws[0];
  if (!(outC > 0)) outC = l.outs[0]?.s?.[1] ?? 0;
  if (!(outC > 0)) return n;
  return Math.max(1, Math.round(n / outC));
}

type WeightCrest = { pt: number; pc: number; source: StatsSource; outlier: number | null; sparse: number; maxAbs: number | null };

function weightCrest(l: ModelLayer, depthwise: boolean): WeightCrest | null {
  if (l.params <= 0) return null;
  const n = weightElems(l);
  const nc = perChannelElems(l, n);
  const w = l.w;
  const has = w && w.min != null && w.max != null && w.std != null;
  if (has) {
    const std = w!.std as number;
    const mean = w!.mean ?? 0;
    const maxAbs = Math.max(Math.abs(w!.min as number), Math.abs(w!.max as number));
    const rms = Math.sqrt(std * std + mean * mean);
    if (maxAbs > 0 && rms > 0) {
      const pt = Math.max(1, maxAbs / rms);
      // Tail excess beyond a Gaussian of the same size. Assume half of it (in
      // log terms) is spread between channel scales, which per-channel removes.
      const tail = pt / gaussianCrest(n);
      const pc = Math.min(pt, gaussianCrest(nc) * Math.sqrt(Math.max(1, tail)));
      return { pt, pc, source: "measured", outlier: std > 0 ? maxAbs / std : null, sparse: w!.sparse ?? 0, maxAbs };
    }
    if (maxAbs === 0) {
      // All-zero tensor: exactly representable.
      return { pt: 1, pc: 1, source: "measured", outlier: null, sparse: 1, maxAbs: 0 };
    }
  }
  // No usable stats: Gaussian with a moderate tail; depthwise channels vary widely.
  const pt = gaussianCrest(n) * (depthwise ? 3 : 1.5);
  const pc = gaussianCrest(nc) * 1.15;
  return { pt, pc, source: "assumed", outlier: null, sparse: 0, maxAbs: null };
}

// ── Graph position ────────────────────────────────────────────────────────────

/** First weighted layers reachable from the inputs, and last weighted layers
 *  that reach the outputs, without crossing another weighted layer. */
function boundaryLayers(model: Model, kinds: Map<number, LayerKind>): { first: Set<number>; last: Set<number> } {
  const first = new Set<number>();
  const last = new Set<number>();
  const layers = model.layers;
  // Weight-only formats chain tensors in file order: no real data flow.
  if (layers.length === 0 || layers.every((l) => /^weight$/i.test(l.op))) return { first, last };

  const succ = new Map<number, number[]>();
  const pred = new Map<number, number[]>();
  for (const [a, b] of model.edges) {
    (succ.get(a) ?? succ.set(a, []).get(a)!).push(b);
    (pred.get(b) ?? pred.set(b, []).get(b)!).push(a);
  }
  const walk = (starts: number[], next: Map<number, number[]>, out: Set<number>) => {
    const seen = new Set<number>(starts);
    const queue = [...starts];
    while (queue.length) {
      const id = queue.shift()!;
      for (const nb of next.get(id) ?? []) {
        if (seen.has(nb)) continue;
        seen.add(nb);
        if (kinds.get(nb) === "weighted") out.add(nb);
        else queue.push(nb);
      }
    }
  };
  const ids = layers.map((l) => l.id);
  let ins = layers.filter((l) => kinds.get(l.id) === "io" && !pred.get(l.id)?.length).map((l) => l.id);
  if (ins.length === 0) ins = ids.filter((id) => !pred.get(id)?.length);
  let outs = layers.filter((l) => kinds.get(l.id) === "io" && !succ.get(l.id)?.length).map((l) => l.id);
  if (outs.length === 0) outs = ids.filter((id) => !succ.get(id)?.length);
  // A source/sink that is itself weighted is its own boundary.
  for (const id of ins) if (kinds.get(id) === "weighted") first.add(id);
  for (const id of outs) if (kinds.get(id) === "weighted") last.add(id);
  walk(ins, succ, first);
  walk(outs, pred, last);
  return { first, last };
}

// ── Main analysis ─────────────────────────────────────────────────────────────

const noisyOr = (...rs: number[]) => 1 - rs.reduce((p, r) => p * (1 - Math.min(1, Math.max(0, r))), 1);
const r3 = (n: number) => Math.round(n * 1000) / 1000;
const r1 = (n: number) => Math.round(n * 10) / 10;

export function analyzeQuantization(model: Model, opts: QuantOptions = {}): QuantAnalysis {
  const target = opts.target ?? "int8";
  const SAFE = opts.safeRisk ?? 0.3;
  const HIGH = opts.highRisk ?? 0.6;
  const level = (r: number): RiskLevel => (r >= HIGH ? "high" : r >= SAFE ? "medium" : "low");

  const kinds = new Map<number, LayerKind>();
  for (const l of model.layers) kinds.set(l.id, kindOf(l));
  const { first, last } = boundaryLayers(model, kinds);

  const layers: LayerSensitivity[] = model.layers.map((l) => {
    const kind = kinds.get(l.id)!;
    const position: LayerSensitivity["position"] = first.has(l.id) ? "first" : last.has(l.id) ? "last" : "middle";
    const empty = (): PrecisionEstimate => ({ sqnrDb: null, perChannelSqnrDb: null, risk: 0, perChannelRisk: 0, level: "low" });

    if (kind === "io" || kind === "passthrough") {
      return {
        id: l.id, name: l.name, op: l.op, kind, quantizable: false, params: l.params,
        statsSource: "none", outlierRatio: null, int8: empty(), int4: empty(), position,
        recommendation: "int8", factors: [],
        reason: kind === "io" ? "Graph input/output tensor: no arithmetic to quantize." : "Layout/data movement only: inherits its neighbours' precision.",
      };
    }

    const factors: string[] = [];
    const rule = ruleFor(l);
    const dw = isDepthwise(l);
    let actPrior = rule?.prior ?? 0.2;
    // Activation × activation MatMul (attention scores / context).
    if (kind === "activation" && /matmul|einsum|bmm/i.test(l.op)) {
      actPrior = 0.5;
      factors.push("Activation × activation MatMul (attention): both operands are quantized from runtime ranges");
    } else if (rule?.note) {
      factors.push(rule.note);
    } else if (!rule) {
      factors.push(`Unrecognized op "${l.op}": moderate default prior`);
    }
    if (dw) {
      actPrior = Math.min(1, actPrior + 0.1);
      factors.unshift("Depthwise conv: channel ranges vary widely (Sheng et al. 2018); per-tensor INT8 is risky");
    }
    // Weight-only tensors (safetensors/GGUF/NumPy) have no activations to quantize.
    if (/^weight$/i.test(l.op)) actPrior = 0.02;

    const crest = weightCrest(l, dw);
    // INT8 first/last layers are routinely quantized (TensorRT, ORT); at 4 bits
    // they are the classic exception kept at higher precision.
    const posPrior = (p: QuantPrecision) => (position === "middle" ? 0 : p === "int4" ? 0.6 : 0.12);
    if (position === "first") factors.push("First layer after the input: raw input ranges, few channels to average out noise");
    if (position === "last") factors.push("Last layer before the output: errors land directly in logits/box coordinates");

    const est = (p: QuantPrecision): PrecisionEstimate => {
      const bits = p === "int8" ? 8 : 4;
      if (!crest) {
        const r = noisyOr(actPrior, posPrior(p));
        return { sqnrDb: null, perChannelSqnrDb: null, risk: r3(r), perChannelRisk: r3(r), level: level(r) };
      }
      const pt = quantSqnrDb(bits, crest.pt, crest.sparse);
      const pc = quantSqnrDb(bits, crest.pc, crest.sparse);
      const riskPt = noisyOr(sqnrRisk(pt), actPrior, posPrior(p));
      const riskPc = noisyOr(sqnrRisk(pc), actPrior, posPrior(p));
      return { sqnrDb: r1(pt), perChannelSqnrDb: r1(pc), risk: r3(riskPt), perChannelRisk: r3(riskPc), level: level(riskPc) };
    };
    const int8 = est("int8");
    const int4 = est("int4");

    if (crest) {
      if (crest.source === "assumed") factors.push("No weight statistics in the file: assumed a Gaussian weight distribution");
      if (crest.outlier != null && crest.outlier > 8) factors.unshift(`Weight outliers: max|w|/σ = ${crest.outlier.toFixed(1)}, so most INT8 levels go unused per-tensor`);
      if (int8.sqnrDb != null && int8.sqnrDb < 30) factors.unshift(`Low per-tensor INT8 weight SQNR (${int8.sqnrDb} dB)`);
    }

    // ── Recommendation ──
    const floatRec = (): Recommendation => {
      if (crest?.maxAbs != null && crest.maxAbs > FP16_MAX) return "keep-fp32";
      if (rule?.fp32 && int8.perChannelRisk >= HIGH) return "keep-fp32";
      return "fp16";
    };
    let recommendation: Recommendation;
    let reason: string;
    const top = factors[0] ?? "";
    if (target === "int4" && kind === "weighted" && int4.perChannelRisk < SAFE) {
      recommendation = "int4";
      reason = `W4A8 per-channel risk ${pct(int4.perChannelRisk)}${int4.perChannelSqnrDb != null ? ` (weight SQNR ≈ ${int4.perChannelSqnrDb} dB)` : ""}.`;
    } else if (int8.risk < SAFE) {
      recommendation = "int8";
      reason = `INT8 risk ${pct(int8.risk)}${int8.sqnrDb != null ? `, weight SQNR ≈ ${int8.sqnrDb} dB` : ""}.`;
    } else if (kind === "weighted" && int8.perChannelRisk < SAFE) {
      recommendation = "int8-per-channel";
      reason = `Per-tensor risk ${pct(int8.risk)} drops to ${pct(int8.perChannelRisk)} with per-channel scales${int8.perChannelSqnrDb != null ? ` (≈ ${int8.sqnrDb} → ${int8.perChannelSqnrDb} dB)` : ""}.`;
    } else if (kind === "activation" && int8.risk < HIGH) {
      // Weightless elementwise ops at medium risk stay INT8: runtimes fuse them
      // into the producer, and a float island would add two Q/DQ conversions.
      recommendation = "int8";
      reason = `Medium INT8 risk (${pct(int8.risk)}), kept INT8 because a float island costs two Q/DQ conversions. Calibrate this range carefully${top ? `. ${top}` : ""}.`;
    } else {
      recommendation = floatRec();
      reason = `INT8 risk ${pct(int8.perChannelRisk)} even per-channel${top ? `. ${top}` : ""}.`;
      if (recommendation === "keep-fp32" && crest?.maxAbs != null && crest.maxAbs > FP16_MAX) reason += " Weights exceed the FP16 range.";
    }

    return {
      id: l.id, name: l.name, op: l.op, kind, quantizable: true, params: l.params,
      statsSource: crest?.source ?? "none", outlierRatio: crest?.outlier != null ? r1(crest.outlier) : null,
      int8, int4, recommendation, reason, factors, position,
    };
  });

  const byId = new Map(layers.map((s) => [s.id, s]));
  return { layers, byId, summary: summarize(layers, target, SAFE, HIGH) };
}

function pct(r: number): string {
  return `${Math.round(r * 100)}%`;
}

function summarize(layers: LayerSensitivity[], target: QuantPrecision, SAFE: number, HIGH: number): QuantSummary {
  const q = layers.filter((s) => s.quantizable);
  const weighted = layers.filter((s) => s.kind === "weighted");
  const measured = weighted.filter((s) => s.statsSource === "measured").length;
  const totalParams = layers.reduce((s, l) => s + l.params, 0);
  const counts: Record<Recommendation, number> = { int4: 0, int8: 0, "int8-per-channel": 0, fp16: 0, "keep-fp32": 0 };
  let mixed = 0;
  for (const s of layers) {
    if (s.quantizable) counts[s.recommendation]++;
    mixed += s.params * (s.kind === "weighted" ? REC_BYTES[s.recommendation] : 0);
  }
  const safe8 = q.filter((s) => s.recommendation === "int8" || s.recommendation === "int8-per-channel" || s.recommendation === "int4").length;
  const safe4 = weighted.filter((s) => s.int4.perChannelRisk < SAFE).length;
  const statsCoverage = weighted.length ? measured / weighted.length : 0;
  return {
    layers: layers.length,
    quantizable: q.length,
    weighted: weighted.length,
    statsCoverage,
    totalParams,
    sizeBytes: { fp32: totalParams * 4, fp16: totalParams * 2, int8: totalParams, int4: totalParams / 2, mixed },
    safeInt8Pct: q.length ? Math.round((safe8 / q.length) * 1000) / 10 : 100,
    safeInt4Pct: weighted.length ? Math.round((safe4 / weighted.length) * 1000) / 10 : 100,
    highRisk8: q.filter((s) => s.int8.perChannelRisk >= HIGH).length,
    highRisk4: weighted.filter((s) => s.int4.perChannelRisk >= HIGH).length,
    counts,
    excluded: q.filter((s) => s.recommendation === "fp16" || s.recommendation === "keep-fp32"),
    perChannel: q.filter((s) => s.recommendation === "int8-per-channel"),
    target,
    confidence: weighted.length > 0 && statsCoverage >= 0.5 ? "medium" : "low",
    note: DISCLAIMER,
  };
}
