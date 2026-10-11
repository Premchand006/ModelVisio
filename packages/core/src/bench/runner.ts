// The benchmark loop. Runtime-agnostic: it takes an onnxruntime-like object
// (`OrtLike`) so it runs unchanged on onnxruntime-web inside the benchmark
// worker and on a mock in unit tests. No React, no DOM, no direct ORT import.

import { resolveInputs, synthesizeData, mulberry32, type InputSpec, type ResolvedInput } from "./inputs";
import { aggregateProfile, parseTraceLines } from "./profile";
import { summarize } from "./stats";
import { tinyIdentityModel } from "./tinyModel";
import type { BenchConfig, BenchEnvInfo, BenchProgress, BenchResult, MemoryInfo } from "./types";

// ─── Minimal structural view of onnxruntime-web ─────────────────────────────

export type OrtValueMeta = { name: string; isTensor: boolean; type?: string; shape?: ReadonlyArray<number | string> };

export interface OrtSessionLike {
  readonly inputNames: readonly string[];
  readonly inputMetadata?: readonly OrtValueMeta[];
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
  release(): Promise<void>;
  endProfiling?(): unknown;
}

export interface OrtLike {
  env: {
    wasm: { numThreads?: number; wasmPaths?: unknown; wasmBinary?: ArrayBufferLike | Uint8Array; proxy?: boolean };
    versions?: { web?: string; common?: string };
    logLevel?: string;
  };
  InferenceSession: { create(model: Uint8Array, options?: Record<string, unknown>): Promise<OrtSessionLike> };
  Tensor: new (type: string, data: unknown, dims: readonly number[]) => unknown;
}

export type RunnerContext = {
  /** Monotonic clock in ms (performance.now). */
  now?: () => number;
  onProgress?: (p: BenchProgress) => void;
  isCancelled?: () => boolean;
  /** Let queued events (e.g. a cancel message) run. Called outside timed regions. */
  yieldToEvents?: () => Promise<void>;
  /** Run `fn` while collecting console output (the ORT profiler prints there). */
  captureConsole?: <T>(fn: () => Promise<T>) => Promise<{ value: T; lines: string[] }>;
  /** Current JS heap in bytes, if the browser exposes it. */
  sampleHeap?: () => number | undefined;
  /** Whole-agent memory in bytes (measureUserAgentSpecificMemory), if available. */
  measureMemory?: () => Promise<number | undefined>;
  env?: Partial<BenchEnvInfo>;
  wasmUrl?: string;
  wasmBinary?: ArrayBuffer;
  fallbackSpecs?: InputSpec[];
  /** Factory for result ids (tests pin it). */
  makeId?: () => string;
};

export class BenchCancelled extends Error {
  constructor() {
    super("Benchmark cancelled");
    this.name = "BenchCancelled";
  }
}

/** Apply the global runtime flags. Must run before the first session is created. */
export function configureOrtEnv(
  env: OrtLike["env"],
  cfg: Pick<BenchConfig, "threads">,
  ctx: { crossOriginIsolated: boolean; hardwareConcurrency: number; wasmUrl?: string; wasmBinary?: ArrayBuffer },
): { effectiveThreads: number; warnings: string[] } {
  const warnings: string[] = [];
  // We already run inside our own worker; ORT's proxy worker would only add a hop.
  env.wasm.proxy = false;
  // Bytes win over a URL: with either set (and no .mjs path) ORT uses the
  // Emscripten glue embedded in its bundle, so nothing is dynamically imported.
  if (ctx.wasmBinary) env.wasm.wasmBinary = new Uint8Array(ctx.wasmBinary);
  else if (ctx.wasmUrl) env.wasm.wasmPaths = { wasm: ctx.wasmUrl };
  let threads = cfg.threads > 0 ? Math.floor(cfg.threads) : 0;
  if (!ctx.crossOriginIsolated) {
    if (threads > 1) warnings.push(`Requested ${threads} threads, but this page isn't cross-origin isolated (no SharedArrayBuffer), so WebAssembly runs single-threaded.`);
    threads = 1;
  }
  env.wasm.numThreads = threads;
  const effectiveThreads = threads > 0 ? threads : Math.max(1, Math.min(4, Math.ceil(ctx.hardwareConcurrency / 2)));
  return { effectiveThreads, warnings };
}

/** Session options for the chosen execution provider. */
export function sessionOptions(cfg: BenchConfig, enableProfiling = false): Record<string, unknown> {
  const ep = cfg.ep === "webnn" ? { name: "webnn", deviceType: cfg.webnnDevice } : cfg.ep;
  return {
    executionProviders: [ep],
    graphOptimizationLevel: cfg.graphOptimization,
    enableProfiling,
    // 3 = error: keep the console readable; the profiler output is separate.
    logSeverityLevel: 3,
  };
}

/** Input specs from the runtime's metadata (authoritative: has symbolic dim names). */
export function specsFromSession(session: OrtSessionLike, fallback: InputSpec[] = []): InputSpec[] {
  const meta = session.inputMetadata;
  if (meta && meta.length) {
    return meta.map((m) => {
      if (!m.isTensor) throw new Error(`Input "${m.name}" is not a tensor (sequence/map inputs aren't supported).`);
      return {
        name: m.name,
        dtype: m.type ?? "float32",
        shape: (m.shape ?? []).map((d) => (typeof d === "number" ? (d > 0 ? d : null) : d || null)),
      };
    });
  }
  return session.inputNames.map((n) => fallback.find((s) => s.name === n) ?? { name: n, dtype: "float32", shape: [] });
}

function disposeOutputs(out: Record<string, unknown>) {
  for (const v of Object.values(out)) {
    const d = (v as { dispose?: () => void } | null)?.dispose;
    if (typeof d === "function") {
      try { d.call(v); } catch { /* already released */ }
    }
  }
}

const defaultYield = () => new Promise<void>((r) => setTimeout(r, 0));

function epLabel(cfg: BenchConfig) {
  return cfg.ep === "webnn" ? `WebNN (${cfg.webnnDevice})` : cfg.ep === "webgpu" ? "WebGPU" : "WebAssembly";
}

/**
 * Benchmark a model: runtime init (tiny model) → session create → first
 * inference → W warm-ups → up to N timed runs within the time budget →
 * optional profiled pass. Cancellation is honored between runs; a cancelled
 * run with at least one timed sample returns partial results.
 */
export async function runBenchmark(
  ort: OrtLike,
  modelBytes: Uint8Array,
  cfg: BenchConfig,
  model: BenchResult["model"],
  ctx: RunnerContext = {},
): Promise<BenchResult> {
  const now = ctx.now ?? (() => performance.now());
  const progress = ctx.onProgress ?? (() => {});
  const cancelled = ctx.isCancelled ?? (() => false);
  const yieldEv = ctx.yieldToEvents ?? defaultYield;
  const startedAt = new Date().toISOString();

  const isolated = ctx.env?.crossOriginIsolated ?? false;
  const hwc = ctx.env?.hardwareConcurrency ?? 1;
  const { effectiveThreads, warnings } = configureOrtEnv(ort.env, cfg, { crossOriginIsolated: isolated, hardwareConcurrency: hwc, wasmUrl: ctx.wasmUrl, wasmBinary: ctx.wasmBinary });

  let heapPeak: number | undefined;
  const sample = () => {
    const h = ctx.sampleHeap?.();
    if (typeof h === "number" && (heapPeak === undefined || h > heapPeak)) heapPeak = h;
  };
  let lastYield = now();
  const maybeYield = async () => {
    if (now() - lastYield > 40) {
      await yieldEv();
      lastYield = now();
    }
    if (cancelled()) throw new BenchCancelled();
  };

  // 1) Runtime init on a trivial model, same EP, so start-up isn't billed to the model.
  progress({ phase: "init", done: 0, total: 1 });
  const t0 = now();
  try {
    const tiny = await ort.InferenceSession.create(tinyIdentityModel(), sessionOptions(cfg));
    const out = await tiny.run({ X: new ort.Tensor("float32", new Float32Array([1]), [1]) });
    disposeOutputs(out);
    await tiny.release();
  } catch (e) {
    throw new Error(`${epLabel(cfg)} runtime failed to initialize: ${e instanceof Error ? e.message : String(e)}`);
  }
  const runtimeInitMs = now() - t0;
  sample();
  await maybeYield();

  // 2) Session for the real model.
  progress({ phase: "session", done: 0, total: 1 });
  const t1 = now();
  let session: OrtSessionLike;
  try {
    session = await ort.InferenceSession.create(modelBytes, sessionOptions(cfg));
  } catch (e) {
    throw new Error(`Couldn't create an inference session on ${epLabel(cfg)}: ${e instanceof Error ? e.message : String(e)}`);
  }
  const sessionCreateMs = now() - t1;
  sample();

  let inputs: ResolvedInput[] = [];
  const warmupMs: number[] = [];
  const samplesMs: number[] = [];
  let firstInferenceMs = NaN;
  let wasCancelled = false;
  try {
    const specs = specsFromSession(session, ctx.fallbackSpecs);
    inputs = resolveInputs(specs, cfg.dimOverrides);
    const rand = mulberry32(cfg.seed);
    const feeds: Record<string, unknown> = {};
    for (const inp of inputs) {
      const n = inp.dims.reduce((a, b) => a * b, 1);
      feeds[inp.name] = new ort.Tensor(inp.dtype, synthesizeData(inp.dtype, n, cfg.fill, rand), inp.dims);
    }
    const once = async () => {
      const s = now();
      const out = await session.run(feeds);
      const ms = now() - s;
      disposeOutputs(out);
      sample();
      return ms;
    };

    // 3) First inference (cold).
    await maybeYield();
    progress({ phase: "first", done: 0, total: 1 });
    try {
      firstInferenceMs = await once();
    } catch (e) {
      const shapes = inputs.map((i) => `${i.name}[${i.dims.join("×")}] ${i.dtype}`).join(", ");
      throw new Error(`Inference failed with inputs ${shapes}: ${e instanceof Error ? e.message : String(e)}`);
    }

    // 4) Warm-up and timed runs.
    try {
      for (let i = 0; i < cfg.warmup; i++) {
        await maybeYield();
        warmupMs.push(await once());
        progress({ phase: "warmup", done: i + 1, total: cfg.warmup, lastMs: warmupMs[i] });
      }
      const tStart = now();
      const total = Math.max(1, cfg.runs);
      for (let i = 0; i < total; i++) {
        await maybeYield();
        samplesMs.push(await once());
        progress({ phase: "timed", done: i + 1, total, lastMs: samplesMs[i] });
        if (cfg.timeBudgetMs > 0 && now() - tStart >= cfg.timeBudgetMs && i + 1 < total) {
          warnings.push(`Time budget of ${(cfg.timeBudgetMs / 1000).toFixed(0)} s reached after ${i + 1} of ${total} timed runs.`);
          break;
        }
      }
    } catch (e) {
      if (!(e instanceof BenchCancelled)) throw e;
      if (samplesMs.length === 0) throw e;
      wasCancelled = true;
      warnings.push(`Cancelled after ${samplesMs.length} timed runs; statistics cover those runs only.`);
    }
  } finally {
    try { await session.release(); } catch { /* ignore */ }
  }

  // 5) Memory (best-effort).
  const memory: MemoryInfo = {};
  if (heapPeak !== undefined) memory.jsHeapPeakBytes = heapPeak;
  if (ctx.measureMemory && !wasCancelled) {
    progress({ phase: "memory", done: 0, total: 1 });
    try {
      const m = await ctx.measureMemory();
      if (typeof m === "number") memory.uaSpecificBytes = m;
    } catch { /* unavailable */ }
  }
  if (memory.jsHeapPeakBytes === undefined && memory.uaSpecificBytes === undefined) {
    memory.note = isolated
      ? "This browser exposes no memory API to workers."
      : "Memory measurement needs Chromium's performance.memory or a cross-origin-isolated page.";
  }

  // 6) Optional per-op profile in a separate session.
  let profile: BenchResult["profile"];
  if (cfg.profile && !wasCancelled) {
    try {
      profile = await profilePass(ort, modelBytes, cfg, inputs, ctx, progress, cancelled);
    } catch (e) {
      if (e instanceof BenchCancelled) {
        wasCancelled = true;
      } else {
        warnings.push(`Per-op profiling failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }

  return {
    id: ctx.makeId?.() ?? `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`,
    startedAt,
    model,
    config: cfg,
    env: {
      userAgent: ctx.env?.userAgent ?? "",
      hardwareConcurrency: hwc,
      crossOriginIsolated: isolated,
      effectiveThreads,
      ortVersion: ort.env.versions?.web ?? ort.env.versions?.common,
      webgpuAdapter: ctx.env?.webgpuAdapter,
    },
    inputs,
    runtimeInitMs,
    sessionCreateMs,
    firstInferenceMs,
    warmupMs,
    samplesMs,
    latency: summarize(samplesMs),
    memory,
    profile,
    cancelled: wasCancelled,
    warnings,
  };
}

async function profilePass(
  ort: OrtLike,
  modelBytes: Uint8Array,
  cfg: BenchConfig,
  inputs: ResolvedInput[],
  ctx: RunnerContext,
  progress: (p: BenchProgress) => void,
  cancelled: () => boolean,
): Promise<NonNullable<BenchResult["profile"]>> {
  const runs = Math.max(1, Math.min(10, cfg.runs));
  const session = await ort.InferenceSession.create(modelBytes, sessionOptions(cfg, true));
  try {
    const rand = mulberry32(cfg.seed);
    const feeds: Record<string, unknown> = {};
    for (const inp of inputs) {
      const n = inp.dims.reduce((a, b) => a * b, 1);
      feeds[inp.name] = new ort.Tensor(inp.dtype, synthesizeData(inp.dtype, n, cfg.fill, rand), inp.dims);
    }
    // One extra (cold) run first; aggregateProfile drops each node's first event.
    for (let i = 0; i <= runs; i++) {
      if (cancelled()) throw new BenchCancelled();
      disposeOutputs(await session.run(feeds));
      progress({ phase: "profile", done: i + 1, total: runs + 1 });
    }
    if (!ctx.captureConsole || typeof session.endProfiling !== "function") {
      return { runs, rows: [], note: "Per-op profiling isn't available in this environment." };
    }
    const { lines } = await ctx.captureConsole(async () => session.endProfiling!());
    const rows = aggregateProfile(parseTraceLines(lines), runs, true);
    return rows.length
      ? { runs, rows }
      : { runs, rows: [], note: `The runtime produced no per-op events for ${epLabel(cfg)}.` };
  } finally {
    try { await session.release(); } catch { /* ignore */ }
  }
}
