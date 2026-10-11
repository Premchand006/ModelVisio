// Main-thread client for the benchmark worker. The worker (and onnxruntime-web
// inside it) is only fetched when a benchmark starts, so ORT never weighs on
// the main bundle. One worker per run; it is terminated when the run ends.

import type { InputSpec } from "./inputs";
import { GPU_WASM_MB, GPU_WASM_URL, verifyGpuWasm } from "./gpuWasm";
import type { BenchConfig, BenchProgress, BenchResult, FromWorker, ToWorker } from "./types";

// Shells whose build can't let the bundler place ORT's .wasm next to the
// worker (the VS Code WebView's single-file IIFE build) define this, relative
// to the main script, and copy the binary there. Unset in the web/desktop build.
declare const __MODELVISIO_ORT_WASM__: string | undefined;

let wasmOverride: string | undefined;

/** Point onnxruntime at a .wasm served from a custom URL (absolute). Pass undefined to reset. */
export function setBenchWasmUrl(url: string | undefined): void {
  wasmOverride = url;
  wasmBytes.clear();
}

/** The .wasm URL to hand the worker, or undefined to let ORT resolve its bundled asset. */
export function benchWasmUrl(): string | undefined {
  if (wasmOverride) return wasmOverride;
  const rel = typeof __MODELVISIO_ORT_WASM__ === "string" ? __MODELVISIO_ORT_WASM__ : "";
  if (!rel) return undefined;
  // A variable base (not a literal import.meta.url) so bundlers don't try to
  // resolve the path at build time.
  const base = import.meta.url;
  try {
    return new URL(rel, base).href;
  } catch {
    return undefined;
  }
}

// Main-thread copies of fetched .wasm binaries, once per URL per page session.
const wasmBytes = new Map<string, Promise<ArrayBuffer | undefined>>();

/**
 * When a custom .wasm URL is configured (the VS Code WebView), fetch it here on
 * the main thread, where the shell's resource loading is known to work, and
 * hand the bytes to the worker. Undefined → the worker resolves the URL itself.
 */
function prefetchWasm(
  url: string | undefined,
  fetchImpl: typeof fetch | undefined,
  verify?: (b: ArrayBuffer) => Promise<boolean>,
): Promise<ArrayBuffer | undefined> {
  if (!url || typeof fetchImpl !== "function") return Promise.resolve(undefined);
  let p = wasmBytes.get(url);
  if (!p) {
    p = fetchImpl(url)
      .then((r) => (r.ok ? r.arrayBuffer() : undefined))
      .catch(() => undefined)
      .then(async (b) => (b && (!verify || (await verify(b).catch(() => false))) ? b : undefined))
      .then((b) => {
        if (!b) wasmBytes.delete(url); // retry next run
        return b;
      });
    wasmBytes.set(url, p);
  }
  return p;
}

const isGpu = (c: BenchConfig) => c.ep === "webgpu" || c.ep === "webnn";

export type BenchHandle = {
  promise: Promise<BenchResult>;
  /** Ask the run to stop; resolves with partial stats when some runs finished, else rejects "Cancelled". */
  cancel: () => void;
};

export type StartArgs = {
  bytes: ArrayBuffer;
  config: BenchConfig;
  model: BenchResult["model"];
  fallbackSpecs?: InputSpec[];
  onProgress?: (p: BenchProgress) => void;
  /** Override for tests (`gpu`: the WebGPU/WebNN build is wanted). */
  createWorker?: (gpu: boolean) => Worker;
  /** Override for tests (defaults to the SHA-256 check in gpuWasm.ts). */
  verifyGpuWasm?: (bytes: ArrayBuffer) => Promise<boolean>;
  /** Override for tests (defaults to globalThis.fetch). */
  fetchImpl?: typeof fetch;
  /** Grace period before a cancelled run's worker is killed outright (ms). */
  cancelGraceMs?: number;
};

// Two literal `new Worker(new URL(…))` forms so bundlers emit both workers.
const defaultWorker = (gpu: boolean) =>
  gpu
    ? new Worker(new URL("./bench-gpu.worker.ts", import.meta.url), { type: "module" })
    : new Worker(new URL("./bench.worker.ts", import.meta.url), { type: "module" });

/** Start a benchmark in a fresh worker. The caller's `bytes` are copied, never transferred away. */
export function startBenchmark(args: StartArgs): BenchHandle {
  let worker: Worker | null = null;
  let settle: { resolve: (r: BenchResult) => void; reject: (e: Error) => void } | null = null;
  let killTimer: ReturnType<typeof setTimeout> | undefined;
  let posted = false;

  const finish = () => {
    if (killTimer) clearTimeout(killTimer);
    worker?.terminate();
    worker = null;
    settle = null;
  };

  const promise = new Promise<BenchResult>((resolve, reject) => {
    settle = { resolve, reject };
    try {
      worker = (args.createWorker ?? defaultWorker)(isGpu(args.config));
    } catch (e) {
      settle = null;
      reject(new Error(`Benchmarking needs a Web Worker, which this environment blocked (${e instanceof Error ? e.message : String(e)}).`));
      return;
    }
    worker.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === "progress") {
        args.onProgress?.(m.progress);
        return;
      }
      const s = settle;
      finish();
      if (!s) return;
      if (m.type === "result") s.resolve(m.result);
      else s.reject(new Error(m.message));
    };
    worker.onerror = (e: ErrorEvent) => {
      e.preventDefault?.();
      const s = settle;
      finish();
      s?.reject(new Error(e.message || "The benchmark worker crashed (the model may be too large for this device's memory)."));
    };
    const gpu = isGpu(args.config);
    // The GPU runtime is never bundled: it always comes from the pinned CDN URL, verified.
    const wasmUrl = gpu ? GPU_WASM_URL : benchWasmUrl();
    const bytes = args.bytes.slice(0);
    const verify = gpu ? args.verifyGpuWasm ?? ((b: ArrayBuffer) => verifyGpuWasm(b)) : undefined;
    void prefetchWasm(wasmUrl, args.fetchImpl ?? globalThis.fetch?.bind(globalThis), verify).then((bin) => {
      if (!worker) return; // cancelled while fetching
      if (gpu && !bin) {
        const s = settle;
        finish();
        s?.reject(new Error(`WebGPU/WebNN benchmarking downloads ONNX Runtime's GPU build (${GPU_WASM_MB} MB, once per session) from cdn.jsdelivr.net, and that download failed or didn't match the expected checksum. Check your connection, or use the WebAssembly (CPU) backend, which works offline.`));
        return;
      }
      const wasmBinary = bin ? bin.slice(0) : undefined;
      const msg: ToWorker = {
        type: "run",
        bytes,
        config: args.config,
        model: args.model,
        wasmUrl,
        wasmBinary,
        fallbackSpecs: args.fallbackSpecs,
      };
      posted = true;
      worker.postMessage(msg, wasmBinary ? [bytes, wasmBinary] : [bytes]);
    });
  });

  const cancel = () => {
    if (!worker || !settle) return;
    if (!posted) {
      const s = settle;
      finish();
      s.reject(new Error("Cancelled"));
      return;
    }
    worker.postMessage({ type: "cancel" } satisfies ToWorker);
    // A single session.run can't be interrupted; if it doesn't yield soon, kill the worker.
    killTimer = setTimeout(() => {
      const s = settle;
      finish();
      s?.reject(new Error("Cancelled"));
    }, args.cancelGraceMs ?? 3000);
  };

  return { promise, cancel };
}
