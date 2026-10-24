// Feature detection for live benchmarking. Never throws: a missing API just
// reports false. Pure functions over an injectable global (testable in Node).

import type { Model } from "@modelvisio/parsers";
import type { BenchEp } from "./types";

export type BenchCapabilities = {
  worker: boolean;
  wasm: boolean;
  /** Fixed-width WebAssembly SIMD (onnxruntime-web ≥ 1.19 ships SIMD-only binaries). */
  simd: boolean;
  /** Multi-threaded wasm: needs SharedArrayBuffer, i.e. a cross-origin-isolated page. */
  threads: boolean;
  crossOriginIsolated: boolean;
  webgpu: boolean;
  webnn: boolean;
  hardwareConcurrency: number;
};

// (module (func (result v128) i32.const 0 i8x16.splat i8x16.popcnt)), from wasm-feature-detect.
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

type G = {
  Worker?: unknown;
  WebAssembly?: { validate?: (b: Uint8Array) => boolean };
  SharedArrayBuffer?: unknown;
  crossOriginIsolated?: boolean;
  navigator?: { gpu?: unknown; ml?: unknown; hardwareConcurrency?: number };
};

export function detectCapabilities(g: G = globalThis as unknown as G): BenchCapabilities {
  const safe = <T>(f: () => T, d: T): T => {
    try {
      return f();
    } catch {
      return d;
    }
  };
  const wasm = safe(() => typeof g.WebAssembly?.validate === "function", false);
  const isolated = safe(() => g.crossOriginIsolated === true, false);
  return {
    worker: safe(() => typeof g.Worker === "function", false),
    wasm,
    simd: wasm && safe(() => g.WebAssembly!.validate!(SIMD_PROBE), false),
    threads: isolated && safe(() => typeof g.SharedArrayBuffer === "function", false),
    crossOriginIsolated: isolated,
    webgpu: safe(() => !!g.navigator?.gpu, false),
    webnn: safe(() => !!g.navigator?.ml, false),
    hardwareConcurrency: safe(() => g.navigator?.hardwareConcurrency ?? 1, 1) || 1,
  };
}

/** Execution providers worth offering given the detected features. */
export function availableEps(c: BenchCapabilities): BenchEp[] {
  const eps: BenchEp[] = [];
  if (c.wasm && c.simd) eps.push("wasm");
  if (c.webgpu) eps.push("webgpu");
  if (c.webnn) eps.push("webnn");
  return eps;
}

/** Live benchmarking runs the original file through onnxruntime, so it needs ONNX bytes. */
export function isBenchmarkable(model: Pick<Model, "format">): boolean {
  return /onnx/i.test(model.format);
}
