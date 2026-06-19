// Roofline performance model for DNN inference on edge accelerators.
//
// Backbone: Williams, Waterman & Patterson, "Roofline: An Insightful Visual
// Performance Model for Multicore Architectures," CACM 52(4), 2009. Attainable
// performance is bounded by min(peak_compute, bandwidth × arithmetic_intensity).
// Applied per-layer from the parsed ONNX shapes, then aggregated — the same
// per-layer-then-aggregate scheme used by PRoof (ICPP'24) and LLM-Viewer.
//
// nn-Meter (MobiSys'21) showed pure FLOP-counting predicts latency at only
// ~22% accuracy; an analytical roofline with a per-device-class utilization
// factor calibrated from benchmarks is the best *implementable* compromise when
// we have model shapes but cannot run every model on every chip.
//
// Pure module — no React, no side effects.

import type { Model, ModelLayer } from "@modelvisio/parsers";
import { BYTES, type DeviceSpec, type Precision, type Util } from "../data/hardware";

export type WorkloadClass = "cnnCls" | "cnnDet" | "transformer";

/** Element count of a tensor from its shape (1 if unknown). */
function elems(s?: number[]): number {
  if (!s || s.length === 0) return 0;
  return s.reduce((a, b) => a * (b > 0 ? b : 1), 1);
}

/** Sum of activation elements live around a single layer (its ins + outs).
 *  Concat layers list every skip input, so this captures FPN/residual peaks. */
function liveElems(l: ModelLayer): number {
  let n = 0;
  for (const i of l.ins) n += elems(i.s);
  for (const o of l.outs) n += elems(o.s);
  if (n === 0 && l.mem > 0) n = l.mem / 4; // fallback: mem is float32 bytes
  return n;
}

/** Choose the deployment precision a device would actually run this model at.
 *  Edge targets quantize to INT8 where supported; otherwise FP16, else FP32. */
export function pickPrecision(dev: DeviceSpec): Precision {
  if (dev.peak.int8 != null) return "int8";
  if (dev.peak.fp16 != null) return "fp16";
  if (dev.peak.bf16 != null) return "bf16";
  return "fp32";
}

/** Peak ops/sec for a precision (TOPS/TFLOPS → ops/s). Uses DENSE int8 only —
 *  never NVIDIA's sparse figure, which is unreachable on dense weights. */
export function peakOpsPerSec(dev: DeviceSpec, prec: Precision): number {
  const t = prec === "int8" ? dev.peak.int8
    : prec === "fp16" ? dev.peak.fp16
    : prec === "bf16" ? dev.peak.bf16
    : prec === "int4" ? dev.peak.int4
    : dev.peak.fp32;
  return (t ?? dev.peak.int8 ?? 1) * 1e12;
}

/** Classify the workload so the right utilization constant is applied.
 *  Transformer detection is op-based; detector/segmenter vs classifier uses the
 *  graph structure (FPN neck + detection head) the parser already labels. */
export function classifyWorkload(model: Model): WorkloadClass {
  const L = model.layers;
  const txt = (l: ModelLayer) => `${l.op} ${l.type}`;
  const tfOps = L.filter((l) => /attention|layernorm|softmax|matmul|gemm|einsum|embedding/i.test(txt(l))).length;
  if (tfOps >= Math.max(4, L.length * 0.15)) return "transformer";

  const meta = `${model.name} ${model.doc ?? ""} ${model.framework}`.toLowerCase();
  const hasNeck = L.some((l) => l.group === "neck") || L.some((l) => /resize|upsample/i.test(l.op));
  const hasHead = L.some((l) => l.group === "head");
  const detName = /yolo|ssd|detr|rcnn|retinanet|detect|nms|seg/.test(meta);
  if (detName || (hasNeck && hasHead)) return "cnnDet";
  return "cnnCls";
}

export function utilFor(wc: WorkloadClass, u: Util): number {
  return wc === "transformer" ? u.transformer : wc === "cnnDet" ? u.cnnDet : u.cnnCls;
}

/**
 * Utilization for a device+workload, CALIBRATED from a measured benchmark anchor
 * when one exists for that workload class (else the generic estimate). The
 * implied utilization is achieved_ops / peak_ops = fps·refOps / peakTOPS.
 *
 * Note: the public anchors are achievable-THROUGHPUT figures (MLPerf offline /
 * Hailo batched), so the implied factor reflects max sustained utilization, not
 * batch-1 latency — the FPS estimate is throughput-oriented and the anchor's
 * batched/offline nature is disclosed in the UI's "Calibration source" line. If
 * the implied value is outside a sane band [0.08, 0.6] the calibration is
 * REJECTED and we fall back to the generic estimate (calibrated:false).
 */
export function calibratedUtil(dev: DeviceSpec, wc: WorkloadClass): { util: number; calibrated: boolean; note?: string } {
  const m = dev.measured;
  if (m && m.workload === wc) {
    const peak = (dev.peak.int8 ?? dev.peak.fp16 ?? 0) * 1e12;
    if (peak > 0) {
      const implied = (m.fps * m.refOps) / peak;
      if (implied >= 0.08 && implied <= 0.6) return { util: implied, calibrated: true, note: m.note };
    }
  }
  return { util: utilFor(wc, dev.util), calibrated: false };
}

export type ModelCompute = {
  totalOps: number;     // 2 × MACs (FLOPs/ops), summed
  totalMacs: number;
  totalParams: number;
  peakActElems: number; // largest concurrently-live activation working set
};

export function modelCompute(model: Model): ModelCompute {
  let totalOps = 0, totalMacs = 0, totalParams = 0, peakActElems = 0;
  for (const l of model.layers) {
    totalOps += l.flops || l.macs * 2;
    totalMacs += l.macs;
    totalParams += l.params;
    const live = liveElems(l);
    if (live > peakActElems) peakActElems = live;
  }
  return { totalOps, totalMacs, totalParams, peakActElems };
}

export type Roofline = {
  prec: Precision;
  fps: number;
  latencyMs: number;
  /** binding regime of the model as a whole, vs the device ridge point */
  bound: "compute" | "memory";
  modelAI: number;     // FLOP/byte across the whole model
  ridgeAI: number;     // peak_compute / bandwidth (FLOP/byte)
  util: number;
  calibrated: boolean; // util came from a measured benchmark anchor, not an estimate
  utilNote?: string;   // provenance of a calibrated utilization
  workload: WorkloadClass;
  idealMs: number;     // before utilization de-rating + dispatch overhead
};

/**
 * Estimate steady-state throughput. For each layer the binding resource is
 * max(compute_time, memory_time) — equivalent to flops / min(peak, bw·AI) but
 * NaN-safe for zero-FLOP memory-move layers (Resize/Concat). On-chip dataflow
 * parts (Hailo) keep weights resident, so weight bytes are excluded and the
 * effective bandwidth is on-chip (compute-bound).
 */
export function estimateRoofline(model: Model, dev: DeviceSpec): Roofline {
  const prec = pickPrecision(dev);
  const bpe = BYTES[prec];
  const peak = peakOpsPerSec(dev, prec);          // ops/s
  const bw = dev.bandwidthGBs * 1e9;              // bytes/s
  const onchip = dev.memType === "onchip";
  const workload = classifyWorkload(model);
  const cal = calibratedUtil(dev, workload);
  const u = cal.util;

  let idealSec = 0;
  let totalOps = 0;
  let totalBytes = 0;

  for (const l of model.layers) {
    const ops = l.flops || l.macs * 2;
    // bytes moved this layer: input acts + output acts + weights (at prec)
    let actElems = 0;
    for (const i of l.ins) actElems += elems(i.s);
    for (const o of l.outs) actElems += elems(o.s);
    if (actElems === 0 && l.mem > 0) actElems = l.mem / 4;
    const weightBytes = onchip ? 0 : l.params * bpe;   // resident on dataflow parts
    const bytes = actElems * bpe + weightBytes;

    const tCompute = ops / peak;
    const tMem = bytes / bw;
    idealSec += Math.max(tCompute, tMem);

    totalOps += ops;
    totalBytes += bytes;
  }

  const idealMs = idealSec * 1000;
  const realSec = idealSec / u + dev.dispatchMs / 1000;
  const fps = realSec > 0 ? 1 / realSec : 0;
  const modelAI = totalBytes > 0 ? totalOps / totalBytes : Infinity;
  const ridgeAI = bw > 0 ? peak / bw : Infinity;

  return {
    prec, fps, latencyMs: realSec * 1000,
    bound: modelAI >= ridgeAI ? "compute" : "memory",
    modelAI, ridgeAI, util: u, calibrated: cal.calibrated, utilNote: cal.note, workload, idealMs,
  };
}

/** Largest concurrently-live activation working set, in bytes at `prec`. */
export function peakActivationBytes(model: Model, prec: Precision): number {
  let peak = 0;
  for (const l of model.layers) {
    const live = liveElems(l);
    if (live > peak) peak = live;
  }
  return peak * BYTES[prec];
}
