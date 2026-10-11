// Measured-vs-predicted comparison, run-history chart geometry and the saved
// JSON report. Pure: no React, no DOM.

import type { Model } from "@modelvisio/parsers";
import { HW, type DeviceSpec } from "../data/hardware";
import { estimateRoofline, type Roofline } from "../scoring";
import type { BenchResult } from "./types";

export type Prediction = {
  deviceId: string;
  deviceName: string;
  precision: Roofline["prec"];
  latencyMs: number;
  fps: number;
  bound: Roofline["bound"];
  calibrated: boolean;
};

/** Roofline-predicted latency of `model` on a hardware-DB device (null if unknown id). */
export function predictFor(model: Model, deviceId: string, devices: readonly DeviceSpec[] = HW): Prediction | null {
  const dev = devices.find((d) => d.id === deviceId);
  if (!dev) return null;
  const r = estimateRoofline(model, dev);
  return {
    deviceId: dev.id, deviceName: dev.n, precision: r.prec, latencyMs: r.latencyMs, fps: r.fps, bound: r.bound, calibrated: r.calibrated,
  };
}

/** Short label for a run: EP, threads/device, and a sequence number. */
export function runLabel(r: BenchResult, index: number): string {
  const c = r.config;
  const ep = c.ep === "wasm" ? `wasm·${r.env.effectiveThreads}t` : c.ep === "webnn" ? `webnn·${c.webnnDevice}` : "webgpu";
  return `#${index + 1} ${ep}`;
}

export type ChartBar = {
  id: string;
  label: string;
  /** Bar length in px (median latency). */
  w: number;
  /** p90 whisker end in px. */
  p90x: number;
  value: number;
  p90: number;
  y: number;
  kind: "measured" | "predicted";
};

export type ChartGeometry = { bars: ChartBar[]; height: number; maxMs: number; ticks: { x: number; ms: number }[] };

/** Nice axis maximum (1/2/2.5/5 × 10^k) at or above `v`. */
export function niceMax(v: number): number {
  if (!(v > 0) || !Number.isFinite(v)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

/**
 * Horizontal bar chart of median latency per run (p90 as a whisker), plus an
 * optional predicted bar. Lower is better. Runs with no samples are skipped.
 */
export function chartGeometry(
  runs: readonly BenchResult[],
  prediction: Prediction | null,
  opts: { width: number; rowH?: number; gap?: number } = { width: 240 },
): ChartGeometry {
  const rowH = opts.rowH ?? 14;
  const gap = opts.gap ?? 6;
  const rows: Omit<ChartBar, "w" | "p90x" | "y">[] = runs
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => r.latency.count > 0)
    .map(({ r, i }) => ({ id: r.id, label: runLabel(r, i), value: r.latency.median, p90: r.latency.p90, kind: "measured" as const }));
  if (prediction && Number.isFinite(prediction.latencyMs)) {
    rows.push({ id: `pred-${prediction.deviceId}`, label: `pred. ${prediction.deviceName}`, value: prediction.latencyMs, p90: prediction.latencyMs, kind: "predicted" });
  }
  const maxMs = niceMax(Math.max(0, ...rows.map((r) => Math.max(r.value, r.p90))));
  const scale = (ms: number) => Math.max(0, Math.min(opts.width, (ms / maxMs) * opts.width));
  const bars = rows.map((r, k) => ({ ...r, w: scale(r.value), p90x: scale(r.p90), y: k * (rowH + gap) }));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => ({ x: f * opts.width, ms: f * maxMs }));
  return { bars, height: Math.max(0, bars.length * (rowH + gap) - gap), maxMs, ticks };
}

/** The JSON document written by "Save result". */
export function benchReport(results: readonly BenchResult[], prediction: Prediction | null) {
  return {
    tool: "ModelVisio live benchmark",
    schema: 1,
    savedAt: new Date().toISOString(),
    prediction: prediction
      ? { ...prediction, note: "Roofline estimate at the device's deployment precision; measured runs are fp32/fp16 in this browser on this machine." }
      : null,
    runs: results,
  };
}

/** Compact latency label: 0.42 ms, 12.3 ms, 1.25 s. */
export function fmtMs(ms: number): string {
  if (!Number.isFinite(ms)) return "–";
  if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
  if (ms >= 100) return ms.toFixed(0) + " ms";
  if (ms >= 10) return ms.toFixed(1) + " ms";
  return ms.toFixed(2) + " ms";
}
