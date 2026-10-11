/// <reference lib="webworker" />
// Shared body of the benchmark workers (bench.worker.ts: CPU build,
// bench-gpu.worker.ts: WebGPU/WebNN build). One worker per run: ORT's
// WebAssembly flags (threads) are fixed at first init, and terminating the
// worker is the only way to reclaim its WebAssembly heap.
import { BenchCancelled, runBenchmark, type OrtLike } from "./runner";
import type { FromWorker, ToWorker } from "./types";

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// Emscripten pthread workers (multi-threaded wasm) re-load this script by URL
// with a name like "em-pthread-N"; they must be left to the runtime.
const isPthread = typeof ctx.name === "string" && ctx.name.startsWith("em-pthread");

// The ORT profiler prints its trace through Emscripten's `out`, which binds
// console.log at wasm init, so the capture hook must wrap console first.
let capture: string[] | null = null;
function installConsoleCapture() {
  const c = console as unknown as Record<string, (...a: unknown[]) => void>;
  for (const k of ["log", "info", "debug"]) {
    const orig = c[k].bind(console);
    c[k] = (...args: unknown[]) => {
      if (capture) capture.push(args.map(String).join(" "));
      else orig(...args);
    };
  }
}

const post = (m: FromWorker) => ctx.postMessage(m);

async function webgpuAdapterName(): Promise<string | undefined> {
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<{ info?: { vendor?: string; architecture?: string; description?: string } } | null> } }).gpu;
  if (!gpu) throw new Error("WebGPU isn't available in this browser (navigator.gpu is missing).");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter is available (GPU blocklisted or disabled).");
  const i = adapter.info;
  return [i?.vendor, i?.architecture, i?.description].filter(Boolean).join(" ") || undefined;
}

async function measureMemory(): Promise<number | undefined> {
  const p = performance as unknown as { measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }> };
  if (!ctx.crossOriginIsolated || typeof p.measureUserAgentSpecificMemory !== "function") return undefined;
  const timeout = new Promise<undefined>((r) => setTimeout(() => r(undefined), 5000));
  const m = await Promise.race([p.measureUserAgentSpecificMemory(), timeout]);
  return m?.bytes;
}

function sampleHeap(): number | undefined {
  const m = (performance as unknown as { memory?: { usedJSHeapSize?: number } }).memory;
  return typeof m?.usedJSHeapSize === "number" ? m.usedJSHeapSize : undefined;
}

/** Wire the worker's message loop to an onnxruntime-web build. */
export function startBenchWorker(ort: unknown): void {
  if (isPthread) return;
  installConsoleCapture();
  let cancelled = false;
  let started = false;

  ctx.onmessage = async (e: MessageEvent<ToWorker>) => {
    const msg = e.data;
    if (msg.type === "cancel") {
      cancelled = true;
      return;
    }
    if (msg.type !== "run" || started) return;
    started = true;
    try {
      const webgpuAdapter = msg.config.ep === "webgpu" ? await webgpuAdapterName() : undefined;
      if (msg.config.ep === "webnn" && !(navigator as unknown as { ml?: unknown }).ml) {
        throw new Error("WebNN isn't available in this browser (navigator.ml is missing).");
      }
      const result = await runBenchmark(ort as OrtLike, new Uint8Array(msg.bytes), msg.config, msg.model, {
        onProgress: (progress) => post({ type: "progress", progress }),
        isCancelled: () => cancelled,
        captureConsole: async (fn) => {
          capture = [];
          try {
            const value = await fn();
            return { value, lines: capture };
          } finally {
            capture = null;
          }
        },
        sampleHeap,
        measureMemory,
        wasmUrl: msg.wasmUrl,
        wasmBinary: msg.wasmBinary,
        fallbackSpecs: msg.fallbackSpecs,
        env: {
          userAgent: navigator.userAgent,
          hardwareConcurrency: navigator.hardwareConcurrency || 1,
          crossOriginIsolated: !!ctx.crossOriginIsolated,
          webgpuAdapter,
        },
      });
      post({ type: "result", result });
    } catch (err) {
      post({ type: "error", message: err instanceof BenchCancelled ? "Cancelled" : err instanceof Error ? err.message : String(err) });
    }
  };
}
