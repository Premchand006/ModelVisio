// Shared types for the live benchmark: run configuration, results and the
// main-thread ↔ worker message protocol. Pure types only.

import type { DimOverrides, InputFill, InputSpec, ResolvedInput } from "./inputs";
import type { OpProfileRow } from "./profile";
import type { LatencyStats } from "./stats";

/** onnxruntime-web execution providers the benchmark can target. */
export type BenchEp = "wasm" | "webgpu" | "webnn";

export type WebnnDevice = "cpu" | "gpu" | "npu";

export type GraphOptLevel = "disabled" | "basic" | "extended" | "all";

export type BenchConfig = {
  ep: BenchEp;
  /** WebAssembly threads (wasm EP and CPU fallback). 0 = runtime default. Needs cross-origin isolation for > 1. */
  threads: number;
  webnnDevice: WebnnDevice;
  warmup: number;
  /** Maximum timed runs. */
  runs: number;
  /** Stop timed runs once this much time has elapsed (ms, after at least one run). 0 = no budget. */
  timeBudgetMs: number;
  fill: InputFill;
  dimOverrides: DimOverrides;
  graphOptimization: GraphOptLevel;
  /** Extra profiled pass for a per-op breakdown (best-effort; separate session so it doesn't skew latencies). */
  profile: boolean;
  /** Seed for random inputs (reproducible runs). */
  seed: number;
};

export const DEFAULT_CONFIG: BenchConfig = {
  ep: "wasm", threads: 0, webnnDevice: "gpu", warmup: 3, runs: 30, timeBudgetMs: 20000,
  fill: "random", dimOverrides: {}, graphOptimization: "all", profile: false, seed: 1,
};

export type MemoryInfo = {
  /** Peak `performance.memory.usedJSHeapSize` sampled in the worker (Chromium only). */
  jsHeapPeakBytes?: number;
  /** `performance.measureUserAgentSpecificMemory()` total (needs cross-origin isolation). */
  uaSpecificBytes?: number;
  /** Why a figure is missing (e.g. API unavailable in this browser). */
  note?: string;
};

export type BenchEnvInfo = {
  userAgent: string;
  hardwareConcurrency: number;
  crossOriginIsolated: boolean;
  /** Threads actually used (1 when SharedArrayBuffer is unavailable). */
  effectiveThreads: number;
  ortVersion?: string;
  webgpuAdapter?: string;
};

export type BenchResult = {
  id: string;
  /** ISO timestamp. */
  startedAt: string;
  model: { name: string; format: string; sizeBytes: number };
  config: BenchConfig;
  env: BenchEnvInfo;
  inputs: ResolvedInput[];
  /** Runtime start-up: wasm fetch + compile and EP device/context init (tiny warm-up model). */
  runtimeInitMs: number;
  /** InferenceSession.create for the user's model (graph load + optimization + EP compile). */
  sessionCreateMs: number;
  /** The very first inference (shader compilation, allocations, caches cold). */
  firstInferenceMs: number;
  warmupMs: number[];
  samplesMs: number[];
  latency: LatencyStats;
  memory: MemoryInfo;
  profile?: { runs: number; rows: OpProfileRow[]; note?: string };
  /** True when the user cancelled; stats cover the runs completed so far. */
  cancelled: boolean;
  warnings: string[];
};

export type BenchPhase = "init" | "session" | "first" | "warmup" | "timed" | "profile" | "memory";

export type BenchProgress = { phase: BenchPhase; done: number; total: number; lastMs?: number };

/** Main thread → worker. */
export type ToWorker =
  | {
      type: "run";
      bytes: ArrayBuffer;
      config: BenchConfig;
      model: BenchResult["model"];
      /** Absolute URL of the onnxruntime .wasm binary when the shell serves it from a custom place. */
      wasmUrl?: string;
      /** The .wasm bytes, prefetched on the main thread (shells whose workers can't fetch local resources). */
      wasmBinary?: ArrayBuffer;
      /** Input specs from the parsed Model, used only if the runtime reports no input metadata. */
      fallbackSpecs?: InputSpec[];
    }
  | { type: "cancel" };

/** Worker → main thread. */
export type FromWorker =
  | { type: "progress"; progress: BenchProgress }
  | { type: "result"; result: BenchResult }
  | { type: "error"; message: string };
