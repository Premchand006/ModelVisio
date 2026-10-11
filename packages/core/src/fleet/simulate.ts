// Fleet simulation: "if I ship this model to N devices of these kinds, what
// happens?" Pure module (no React), deterministic for a given seed.
//
// Per device class it combines:
//   • service time S: the roofline latency at the class's precision + runtime
//     (fleet/estimate.ts), slowed by its thermal and power profile;
//   • a single-server queue at the class's request rate λ (per device):
//       ρ = λ·S; waiting-time tail P(Wq > t) ≈ ρ·exp(−(1−ρ)·t / (k·S)),
//     with k = 1 for M/M/1 (exact for exponential service) and k = ½ for M/D/1
//     (Kingman's G/G/1 factor (ca²+cs²)/2 with cs² = 0). The mean wait this
//     implies is ρ·k·S/(1−ρ): exactly M/M/1's ρS/(1−ρ) and M/D/1's ρS/(2(1−ρ)).
//     Response = S + Wq, so p95 = S + k·S·ln(ρ/0.05)/(1−ρ) (0 wait when ρ ≤ 0.05).
//     Inference latency is near-deterministic, so M/D/1 is the default; camera
//     feeds with fixed frame intervals queue even less (D/D/1), so this is
//     conservative.
//   • memory fit, compile failures and op-compat issues for that runtime;
//   • energy per inference (typical power × latency).
// Fleet aggregates mix the classes' response-time distributions weighted by
// request volume (count × λ) and invert the mixture CDF for the fleet p95.
//
// Monte-Carlo mode draws a per-device slowdown (thermal throttling, background
// load, unit-to-unit variance) from a seeded RNG and repeats the fleet
// computation per trial, giving a distribution of the fleet p95.

import type { Model } from "@modelvisio/parsers";
import { HW, type DeviceSpec } from "../data/hardware";
import { canRun, estimateDevice, type DeviceEstimate } from "./estimate";
import type { DeployPrecision, RuntimeId } from "../deploy/runtimes";

export type ThermalProfile = "cooled" | "passive" | "hot";
export type PowerMode = "max" | "balanced" | "low";
export type QueueModel = "md1" | "mm1";

export type FleetEntry = {
  id: string;
  deviceId: string;
  count: number;
  runtime: RuntimeId;
  precision: DeployPrecision;
  thermal?: ThermalProfile;
  powerMode?: PowerMode;
  /** Throughput the application needs per device (e.g. camera FPS). */
  targetFps?: number;
  /** Latency SLO (ms, p95 response). Default: 1000/targetFps, else 100 ms. */
  sloMs?: number;
  /** Requests per second per device. Default: targetFps, else 10. */
  requestRate?: number;
  label?: string;
};

export type MonteCarloOptions = { trials?: number; seed?: number; samplesPerClass?: number };
export type FleetOptions = { queue?: QueueModel; monteCarlo?: MonteCarloOptions | null };

/** Mean slowdown (latency multiplier) for each thermal profile. */
export const THERMAL_SLOWDOWN: Record<ThermalProfile, number> = { cooled: 1.0, passive: 1.1, hot: 1.3 };
/** Compute speed and power multipliers per power mode. */
export const POWER_MODES: Record<PowerMode, { speed: number; power: number }> = {
  max: { speed: 1, power: 1 },
  balanced: { speed: 0.85, power: 0.75 },
  low: { speed: 0.6, power: 0.5 },
};

const DEFAULT_RATE = 10;
const DEFAULT_SLO_MS = 100;

// ─── Queueing math ───────────────────────────────────────────────────────────

const kFor = (q: QueueModel) => (q === "mm1" ? 1 : 0.5);

export type QueueStats = {
  rho: number;
  stable: boolean;
  meanWaitMs: number;
  meanMs: number;
  p95WaitMs: number;
  p95Ms: number;
};

/** Single-server queue stats for service time `sMs` and arrival rate `lambda` (req/s). */
export function queueStats(sMs: number, lambda: number, q: QueueModel = "md1"): QueueStats {
  const rho = (Math.max(0, lambda) * sMs) / 1000;
  if (!isFinite(sMs) || rho >= 1) {
    return { rho, stable: false, meanWaitMs: Infinity, meanMs: Infinity, p95WaitMs: Infinity, p95Ms: Infinity };
  }
  const k = kFor(q);
  const meanWaitMs = (rho * k * sMs) / (1 - rho);
  const p95WaitMs = rho <= 0.05 ? 0 : (k * sMs * Math.log(rho / 0.05)) / (1 - rho);
  return { rho, stable: true, meanWaitMs, meanMs: sMs + meanWaitMs, p95WaitMs, p95Ms: sMs + p95WaitMs };
}

/** P(response ≤ t) for one device: 0 before the service time, then the wait tail. */
export function responseCdf(tMs: number, sMs: number, lambda: number, q: QueueModel = "md1"): number {
  if (!isFinite(sMs) || tMs < sMs) return 0;
  const rho = (Math.max(0, lambda) * sMs) / 1000;
  if (rho >= 1) return 0;
  return Math.max(0, 1 - rho * Math.exp((-(1 - rho) * (tMs - sMs)) / (kFor(q) * sMs)));
}

/** Fraction of requests meeting a latency SLO. */
export function sloAttainment(sloMs: number, sMs: number, lambda: number, q: QueueModel = "md1"): number {
  return responseCdf(sloMs, sMs, lambda, q);
}

export type MixComponent = { sMs: number; lambda: number; weight: number };

/** Quantile of a weighted mixture of per-device response distributions. Infinity if never reached. */
export function mixtureQuantile(comps: MixComponent[], q: number, queue: QueueModel = "md1"): number {
  const live = comps.filter((c) => c.weight > 0);
  if (live.length === 0) return NaN;
  const W = live.reduce((s, c) => s + c.weight, 0);
  const cdf = (t: number) => live.reduce((s, c) => s + c.weight * responseCdf(t, c.sMs, c.lambda, queue), 0) / W;
  let lo = 0;
  let hi = Math.max(...live.map((c) => (isFinite(c.sMs) ? c.sMs : 0)), 1e-3);
  while (cdf(hi) < q) {
    lo = hi;
    hi *= 2;
    if (hi > 1e8) return Infinity;
  }
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) >= q) hi = mid; else lo = mid;
  }
  return hi;
}

// ─── Seeded RNG ──────────────────────────────────────────────────────────────

/** mulberry32: tiny, fast, deterministic PRNG in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Per-device latency multiplier for one Monte-Carlo draw. */
export function sampleSlowdown(rand: () => number, thermal: ThermalProfile): number {
  const unit = 1 + 0.03 * gaussian(rand);                    // unit-to-unit variance
  const th = thermal === "hot" ? 1 + Math.max(0, 0.3 + 0.12 * gaussian(rand))
    : thermal === "passive" ? 1 + Math.max(0, 0.1 + 0.06 * gaussian(rand))
    : 1 + Math.abs(0.02 * gaussian(rand));
  const bg = rand() < 0.1 ? 1 + 0.1 + 0.5 * rand() : 1;     // background load spike
  return Math.max(0.8, unit) * th * bg;
}

// ─── Histogram ───────────────────────────────────────────────────────────────

export type HistBin = { lo: number; hi: number; count: number };

/** Equal-width histogram of the finite values. */
export function histogram(values: number[], bins = 20): HistBin[] {
  const v = values.filter((x) => isFinite(x));
  if (v.length === 0) return [];
  const min = Math.min(...v);
  const max = Math.max(...v);
  if (max === min) return [{ lo: min, hi: max, count: v.length }];
  const w = (max - min) / bins;
  const out: HistBin[] = Array.from({ length: bins }, (_, i) => ({ lo: min + i * w, hi: min + (i + 1) * w, count: 0 }));
  for (const x of v) out[Math.min(bins - 1, Math.floor((x - min) / w))].count++;
  return out;
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
  return sorted[i];
}

// ─── Simulation ──────────────────────────────────────────────────────────────

export type ClassResult = {
  entry: FleetEntry;
  device: DeviceSpec | null;
  name: string;
  estimate: DeviceEstimate | null;
  canRun: boolean;
  /** Service time after thermal + power-mode effects. */
  latencyMs: number;
  capacityFps: number;
  requestRate: number;
  sloMs: number;
  utilization: number;
  stable: boolean;
  meanMs: number;
  p95Ms: number;
  sloAttainment: number;
  meetsSlo: boolean;
  powerW: number;
  energyMj: number;
  compatErrors: number;
  compatWarns: number;
  reasons: string[];
};

export type FleetAggregate = {
  totalDevices: number;
  runnableDevices: number;
  cantRunDevices: number;
  devicesMeetingSlo: number;
  pctMeetingSlo: number;
  /** p95 response over all requests from runnable devices (mixture). */
  fleetP95Ms: number;
  /** Device-count-weighted mean of the per-class p95. */
  weightedP95Ms: number;
  capacityFps: number;
  demandFps: number;
  /** Request-weighted mean energy per inference (mJ). */
  meanEnergyMj: number;
  /** Sum of typical power across runnable devices (W). */
  fleetPowerW: number;
  bottleneck: { id: string; name: string; reason: string } | null;
};

export type MonteCarloResult = {
  trials: number;
  seed: number;
  fleetP95: number[];
  unstableTrials: number;
  p95Quantiles: { p5: number; p50: number; p95: number };
  pctMeetingSlo: { p5: number; p50: number; p95: number };
  histogram: HistBin[];
};

export type FleetResult = {
  classes: ClassResult[];
  aggregate: FleetAggregate;
  queue: QueueModel;
  monteCarlo: MonteCarloResult | null;
};

function resolveEntry(e: FleetEntry): { sloMs: number; lambda: number } {
  const fps = e.targetFps && e.targetFps > 0 ? e.targetFps : undefined;
  const sloMs = e.sloMs && e.sloMs > 0 ? e.sloMs : fps ? 1000 / fps : DEFAULT_SLO_MS;
  const lambda = e.requestRate != null && e.requestRate >= 0 ? e.requestRate : fps ?? DEFAULT_RATE;
  return { sloMs, lambda };
}

function simulateClass(model: Model, e: FleetEntry, devices: DeviceSpec[], q: QueueModel): ClassResult {
  const dev = devices.find((d) => d.id === e.deviceId) ?? null;
  const { sloMs, lambda } = resolveEntry(e);
  const count = Math.max(0, Math.floor(e.count || 0));
  if (!dev) {
    return {
      entry: { ...e, count }, device: null, name: e.label || e.deviceId, estimate: null, canRun: false,
      latencyMs: Infinity, capacityFps: 0, requestRate: lambda, sloMs, utilization: Infinity, stable: false,
      meanMs: Infinity, p95Ms: Infinity, sloAttainment: 0, meetsSlo: false, powerW: 0, energyMj: Infinity,
      compatErrors: 0, compatWarns: 0, reasons: [`Unknown device "${e.deviceId}"`],
    };
  }
  const est = estimateDevice(model, dev, e.precision, e.runtime);
  const pm = POWER_MODES[e.powerMode ?? "max"];
  const latencyMs = (est.latencyMs / pm.speed) * THERMAL_SLOWDOWN[e.thermal ?? "cooled"];
  const runnable = canRun(est);
  const qs = queueStats(latencyMs, lambda, q);
  const capacityFps = isFinite(latencyMs) && latencyMs > 0 ? 1000 / latencyMs : 0;
  const attain = runnable ? sloAttainment(sloMs, latencyMs, lambda, q) : 0;
  const reasons: string[] = [];
  if (!est.supported) reasons.push(est.reason ?? "unsupported");
  if (!est.fits) reasons.push("model does not fit in device memory");
  if (est.compileFail) reasons.push("unsupported ops with no CPU fallback (won't compile)");
  if (runnable && !qs.stable) reasons.push(`overloaded: ${lambda} req/s > capacity ${capacityFps.toFixed(1)} FPS`);
  if (runnable && qs.stable && qs.p95Ms > sloMs) reasons.push(`p95 ${qs.p95Ms.toFixed(1)} ms > SLO ${sloMs.toFixed(1)} ms`);
  if (runnable && e.targetFps && capacityFps < e.targetFps) reasons.push(`capacity ${capacityFps.toFixed(1)} FPS < target ${e.targetFps} FPS`);
  if (est.errors > 0 && runnable) reasons.push(`${est.errors} op(s) fall back to CPU on ${e.runtime}`);
  const meetsSlo = runnable && qs.stable && qs.p95Ms <= sloMs && (!e.targetFps || capacityFps >= e.targetFps);
  const powerW = est.powerW * pm.power;
  return {
    entry: { ...e, count }, device: dev, name: e.label || dev.n, estimate: est, canRun: runnable,
    latencyMs, capacityFps, requestRate: lambda, sloMs, utilization: qs.rho, stable: qs.stable,
    meanMs: qs.meanMs, p95Ms: qs.p95Ms, sloAttainment: attain, meetsSlo,
    powerW, energyMj: isFinite(latencyMs) ? powerW * latencyMs : Infinity,
    compatErrors: est.errors, compatWarns: est.warns, reasons,
  };
}

function aggregate(classes: ClassResult[], q: QueueModel): FleetAggregate {
  const total = classes.reduce((s, c) => s + c.entry.count, 0);
  const runnable = classes.filter((c) => c.canRun && c.entry.count > 0);
  const runnableDevices = runnable.reduce((s, c) => s + c.entry.count, 0);
  const meeting = classes.filter((c) => c.meetsSlo).reduce((s, c) => s + c.entry.count, 0);
  const comps: MixComponent[] = runnable.map((c) => ({ sMs: c.latencyMs, lambda: c.requestRate, weight: c.entry.count * Math.max(c.requestRate, 1e-9) }));
  const fleetP95Ms = comps.length ? mixtureQuantile(comps, 0.95, q) : NaN;
  const weightedP95Ms = runnableDevices > 0 ? runnable.reduce((s, c) => s + c.p95Ms * c.entry.count, 0) / runnableDevices : NaN;
  const reqW = runnable.reduce((s, c) => s + c.entry.count * c.requestRate, 0);
  const meanEnergyMj = reqW > 0 ? runnable.reduce((s, c) => s + c.energyMj * c.entry.count * c.requestRate, 0) / reqW : NaN;

  let bottleneck: FleetAggregate["bottleneck"] = null;
  const worst = [...runnable].sort((a, b) => b.p95Ms / b.sloMs - a.p95Ms / a.sloMs)[0];
  if (worst) {
    bottleneck = {
      id: worst.entry.id, name: worst.name,
      reason: worst.stable
        ? `highest p95/SLO ratio: ${worst.p95Ms.toFixed(1)} ms vs ${worst.sloMs.toFixed(1)} ms (utilization ${(worst.utilization * 100).toFixed(0)}%)`
        : `overloaded at ${worst.requestRate} req/s (capacity ${worst.capacityFps.toFixed(1)} FPS)`,
    };
  } else {
    const big = [...classes].sort((a, b) => b.entry.count - a.entry.count)[0];
    if (big) bottleneck = { id: big.entry.id, name: big.name, reason: big.reasons[0] ?? "cannot run" };
  }

  return {
    totalDevices: total,
    runnableDevices,
    cantRunDevices: total - runnableDevices,
    devicesMeetingSlo: meeting,
    pctMeetingSlo: total > 0 ? (100 * meeting) / total : 0,
    fleetP95Ms,
    weightedP95Ms,
    capacityFps: runnable.reduce((s, c) => s + c.capacityFps * c.entry.count, 0),
    demandFps: classes.reduce((s, c) => s + c.requestRate * c.entry.count, 0),
    meanEnergyMj,
    fleetPowerW: runnable.reduce((s, c) => s + c.powerW * c.entry.count, 0),
    bottleneck,
  };
}

function monteCarlo(classes: ClassResult[], q: QueueModel, opts: MonteCarloOptions): MonteCarloResult {
  const trials = Math.max(1, Math.min(5000, Math.floor(opts.trials ?? 200)));
  const seed = Math.floor(opts.seed ?? 42);
  const perClass = Math.max(1, Math.floor(opts.samplesPerClass ?? 48));
  const rand = mulberry32(seed);
  const total = classes.reduce((s, c) => s + c.entry.count, 0);
  const runnable = classes.filter((c) => c.canRun && c.entry.count > 0);
  const p95s: number[] = [];
  const pcts: number[] = [];
  let unstable = 0;

  for (let t = 0; t < trials; t++) {
    const comps: MixComponent[] = [];
    let meeting = 0;
    for (const c of runnable) {
      // Base latency without the class's mean thermal factor: the draw re-applies it.
      const base = c.latencyMs / THERMAL_SLOWDOWN[c.entry.thermal ?? "cooled"];
      const n = Math.min(c.entry.count, perClass);
      const devWeight = c.entry.count / n;
      for (let i = 0; i < n; i++) {
        const s = base * sampleSlowdown(rand, c.entry.thermal ?? "cooled");
        comps.push({ sMs: s, lambda: c.requestRate, weight: devWeight * Math.max(c.requestRate, 1e-9) });
        const qs = queueStats(s, c.requestRate, q);
        const cap = 1000 / s;
        if (qs.stable && qs.p95Ms <= c.sloMs && (!c.entry.targetFps || cap >= c.entry.targetFps)) meeting += devWeight;
      }
    }
    const p = comps.length ? mixtureQuantile(comps, 0.95, q) : NaN;
    if (!isFinite(p)) unstable++;
    p95s.push(p);
    pcts.push(total > 0 ? (100 * meeting) / total : 0);
  }

  const finite = p95s.filter((x) => isFinite(x)).sort((a, b) => a - b);
  const sortedPct = [...pcts].sort((a, b) => a - b);
  return {
    trials, seed, fleetP95: p95s, unstableTrials: unstable,
    p95Quantiles: { p5: quantile(finite, 0.05), p50: quantile(finite, 0.5), p95: quantile(finite, 0.95) },
    pctMeetingSlo: { p5: quantile(sortedPct, 0.05), p50: quantile(sortedPct, 0.5), p95: quantile(sortedPct, 0.95) },
    histogram: histogram(finite, 24),
  };
}

/** Simulate a fleet. `devices` defaults to the hardware DB. */
export function simulateFleet(model: Model, entries: FleetEntry[], opts: FleetOptions = {}, devices: DeviceSpec[] = HW): FleetResult {
  const q = opts.queue ?? "md1";
  const classes = entries.map((e) => simulateClass(model, e, devices, q));
  return {
    classes,
    aggregate: aggregate(classes, q),
    queue: q,
    monteCarlo: opts.monteCarlo ? monteCarlo(classes, q, opts.monteCarlo) : null,
  };
}

// ─── Export ──────────────────────────────────────────────────────────────────

const num = (n: number, d = 2) => (isFinite(n) ? Number(n.toFixed(d)) : null);

/** JSON-safe summary (Infinity → null) for export. */
export function fleetResultToJson(r: FleetResult, modelName = ""): string {
  return JSON.stringify({
    model: modelName,
    queue: r.queue,
    aggregate: {
      ...r.aggregate,
      fleetP95Ms: num(r.aggregate.fleetP95Ms), weightedP95Ms: num(r.aggregate.weightedP95Ms),
      meanEnergyMj: num(r.aggregate.meanEnergyMj), pctMeetingSlo: num(r.aggregate.pctMeetingSlo, 1),
      capacityFps: num(r.aggregate.capacityFps, 1), demandFps: num(r.aggregate.demandFps, 1), fleetPowerW: num(r.aggregate.fleetPowerW, 1),
    },
    classes: r.classes.map((c) => ({
      ...c.entry, name: c.name, canRun: c.canRun, latencyMs: num(c.latencyMs), capacityFps: num(c.capacityFps, 1),
      requestRate: c.requestRate, sloMs: num(c.sloMs), utilization: num(c.utilization, 3), meanMs: num(c.meanMs), p95Ms: num(c.p95Ms),
      sloAttainment: num(c.sloAttainment, 4), meetsSlo: c.meetsSlo, powerW: num(c.powerW, 1), energyMj: num(c.energyMj),
      compatErrors: c.compatErrors, compatWarns: c.compatWarns, approx: c.estimate?.approx ?? null, reasons: c.reasons,
    })),
    monteCarlo: r.monteCarlo && {
      trials: r.monteCarlo.trials, seed: r.monteCarlo.seed, unstableTrials: r.monteCarlo.unstableTrials,
      p95Quantiles: { p5: num(r.monteCarlo.p95Quantiles.p5), p50: num(r.monteCarlo.p95Quantiles.p50), p95: num(r.monteCarlo.p95Quantiles.p95) },
      pctMeetingSlo: r.monteCarlo.pctMeetingSlo,
      histogram: r.monteCarlo.histogram.map((b) => ({ lo: num(b.lo), hi: num(b.hi), count: b.count })),
    },
  }, null, 2);
}

const csvCell = (v: unknown) => {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** Per-class CSV for export. */
export function fleetResultToCsv(r: FleetResult): string {
  const head = ["id", "device", "deviceId", "count", "runtime", "precision", "thermal", "powerMode", "targetFps", "requestRate", "sloMs",
    "canRun", "latencyMs", "capacityFps", "utilization", "meanMs", "p95Ms", "sloAttainment", "meetsSlo", "powerW", "energyMj", "compatErrors", "compatWarns", "reasons"];
  const rows = r.classes.map((c) => [
    c.entry.id, c.name, c.entry.deviceId, c.entry.count, c.entry.runtime, c.entry.precision, c.entry.thermal ?? "cooled", c.entry.powerMode ?? "max",
    c.entry.targetFps ?? "", c.requestRate, num(c.sloMs), c.canRun, num(c.latencyMs), num(c.capacityFps, 1), num(c.utilization, 3),
    num(c.meanMs), num(c.p95Ms), num(c.sloAttainment, 4), c.meetsSlo, num(c.powerW, 1), num(c.energyMj), c.compatErrors, c.compatWarns, c.reasons.join("; "),
  ].map(csvCell).join(","));
  return [head.join(","), ...rows].join("\n") + "\n";
}
