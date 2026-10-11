// Per-layer roofline time estimates. The scoring engine (scoring/roofline.ts)
// only returns model totals; the investigator and the optimization search need
// to know WHERE the time goes. This module runs the same per-layer
// max(compute_time, memory_time) model, de-rated by the same calibrated
// utilization plus the device's dispatch overhead, so with the device's default
// precision its total equals estimateRoofline(model, dev).latencyMs.
//
// Two-step API for speed: prepareLayers() extracts the per-layer numbers once
// per model variant, and rooflineFromPrepared() evaluates a device + precision
// over those arrays without touching the Model again (the optimization search
// evaluates thousands of device × precision combinations).
//
// Pure module: no React, no side effects.

import type { Model } from "@modelvisio/parsers";
import { BYTES, type DeviceSpec, type Precision } from "../data/hardware";
import { calibratedUtil, classifyWorkload, peakOpsPerSec, pickPrecision, type WorkloadClass } from "../scoring/roofline";

/** Element count of a tensor shape (dynamic/unknown dims count as 1; empty → 0).
 *  Same convention as scoring/roofline.ts so totals line up exactly. */
export function elems(s?: number[]): number {
  if (!s || s.length === 0) return 0;
  return s.reduce((a, b) => a * (b > 0 ? b : 1), 1);
}

export type PreparedLayers = {
  ids: number[];
  names: string[];
  ops: string[];
  /** FLOPs (2 × MACs when flops is missing) per layer. */
  flops: Float64Array;
  /** Activation elements read + written per layer. */
  actElems: Float64Array;
  params: Float64Array;
  workload: WorkloadClass;
};

export function prepareLayers(model: Model): PreparedLayers {
  const n = model.layers.length;
  const flops = new Float64Array(n);
  const actElems = new Float64Array(n);
  const params = new Float64Array(n);
  const ids: number[] = [];
  const names: string[] = [];
  const ops: string[] = [];
  model.layers.forEach((l, i) => {
    ids.push(l.id);
    names.push(l.name);
    ops.push(l.op || l.type);
    flops[i] = l.flops || l.macs * 2;
    let a = 0;
    for (const x of l.ins) a += elems(x.s);
    for (const x of l.outs) a += elems(x.s);
    if (a === 0 && l.mem > 0) a = l.mem / 4; // mem is float32 bytes
    actElems[i] = a;
    params[i] = l.params;
  });
  return { ids, names, ops, flops, actElems, params, workload: classifyWorkload(model) };
}

export type LayerTime = {
  id: number;
  name: string;
  op: string;
  flops: number;
  bytes: number;
  /** Arithmetic intensity (FLOP/byte); Infinity for a zero-byte layer. */
  ai: number;
  /** De-rated time attributed to this layer (ms). */
  timeMs: number;
  bound: "compute" | "memory";
};

export type LayerRooflineResult = {
  prec: Precision;
  layers: LayerTime[];
  /** Sum of layer times before utilization de-rating (ms). */
  idealMs: number;
  /** Sum of de-rated layer times (ms), excluding dispatch. */
  computeMs: number;
  /** computeMs + dispatch overhead. */
  latencyMs: number;
  dispatchMs: number;
  util: number;
  calibrated: boolean;
  peakOps: number;
  bandwidth: number;
  ridgeAI: number;
  workload: WorkloadClass;
};

export type RooflineOpts = {
  /** Override peak ops/s (e.g. a derived FP16 peak on a part that only lists INT8). */
  peakOps?: number;
  /** Skip building the per-layer LayerTime array (totals only; faster). */
  totalsOnly?: boolean;
};

/** Evaluate one device + precision over prepared layers. */
export function rooflineFromPrepared(
  p: PreparedLayers,
  dev: DeviceSpec,
  prec: Precision = pickPrecision(dev),
  opts: RooflineOpts = {},
): LayerRooflineResult {
  const bpe = BYTES[prec];
  const peak = opts.peakOps ?? peakOpsPerSec(dev, prec);
  const bw = dev.bandwidthGBs * 1e9;
  const onchip = dev.memType === "onchip";
  const cal = calibratedUtil(dev, p.workload);
  const u = cal.util;
  const layers: LayerTime[] = [];
  let idealSec = 0;
  const n = p.flops.length;
  for (let i = 0; i < n; i++) {
    const ops = p.flops[i];
    const bytes = p.actElems[i] * bpe + (onchip ? 0 : p.params[i] * bpe);
    const tc = ops / peak;
    const tm = bytes / bw;
    const t = Math.max(tc, tm);
    idealSec += t;
    if (!opts.totalsOnly) {
      layers.push({
        id: p.ids[i], name: p.names[i], op: p.ops[i], flops: ops, bytes,
        ai: bytes > 0 ? ops / bytes : Infinity,
        timeMs: (t / u) * 1000,
        bound: tc >= tm ? "compute" : "memory",
      });
    }
  }
  const computeMs = (idealSec / u) * 1000;
  return {
    prec, layers, idealMs: idealSec * 1000, computeMs,
    latencyMs: computeMs + dev.dispatchMs, dispatchMs: dev.dispatchMs,
    util: u, calibrated: cal.calibrated, peakOps: peak, bandwidth: bw,
    ridgeAI: bw > 0 ? peak / bw : Infinity, workload: p.workload,
  };
}

/** Convenience: per-layer roofline for a model on a device. */
export function layerRoofline(model: Model, dev: DeviceSpec, prec?: Precision, opts?: RooflineOpts): LayerRooflineResult {
  return rooflineFromPrepared(prepareLayers(model), dev, prec, opts);
}
