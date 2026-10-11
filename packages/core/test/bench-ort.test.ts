import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { runBenchmark, type OrtLike } from "../src/bench/runner";
import { DEFAULT_CONFIG } from "../src/bench/types";
import { modelProtoType } from "../../parsers/src/onnx/proto";

// Integration: the real onnxruntime-web builds the benchmark workers use
// (bench.worker.ts: `onnxruntime-web/wasm`; bench-gpu.worker.ts:
// `onnxruntime-web/webgpu`, here on its WebAssembly EP) running a tiny model
// built in the test. Node resolves the package's "node" export to an
// onnxruntime-node shim, so each browser bundle is loaded by path and its
// .wasm handed over as bytes (the `wasmBinary` path the app uses) instead of fetched.
const req = createRequire(import.meta.url);
const dist = dirname(req.resolve("onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm"));

const BUILDS = [
  { label: "CPU build (bench.worker)", js: "ort.wasm.bundle.min.mjs", wasm: "ort-wasm-simd-threaded.wasm" },
  { label: "GPU build (bench-gpu.worker)", js: "ort.webgpu.bundle.min.mjs", wasm: "ort-wasm-simd-threaded.asyncify.wasm" },
];

async function loadOrt(js: string): Promise<OrtLike> {
  return import(/* @vite-ignore */ pathToFileURL(join(dist, js)).href);
}

function wasmBytes(file: string): ArrayBuffer {
  const b = readFileSync(join(dist, file));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}

/** X[N, 4] → Relu → MatMul(W[4,3]) → Y[N, 3], with a symbolic batch dim. */
function tinyMlp(): Uint8Array {
  const M = modelProtoType();
  return M.encode(M.fromObject({
    ir_version: 8, producer_name: "bench-ort-test", opset_import: [{ domain: "", version: 17 }],
    graph: {
      name: "mlp",
      input: [{ name: "X", type: { tensor_type: { elem_type: 1, shape: { dim: [{ dim_param: "N" }, { dim_value: 4 }] } } } }],
      output: [{ name: "Y", type: { tensor_type: { elem_type: 1, shape: { dim: [{ dim_param: "N" }, { dim_value: 3 }] } } } }],
      initializer: [{ name: "W", data_type: 1, dims: [4, 3], float_data: new Array(12).fill(0.5) }],
      node: [
        { op_type: "Relu", name: "relu", input: ["X"], output: ["r"] },
        { op_type: "MatMul", name: "mm", input: ["r", "W"], output: ["Y"] },
      ],
    },
  })).finish();
}

describe.each(BUILDS)("live benchmark on real onnxruntime-web, $label", ({ js, wasm }) => {
  it("runs, resolves the symbolic batch dim, and profiles per op", async () => {
    const ort = await loadOrt(js);
    // Mirror the worker: the profiler prints through console.log, which the
    // runtime binds at init, so wrap it before the first session is created.
    let capture: string[] | null = null;
    const orig = console.log;
    console.log = (...a: unknown[]) => { if (capture) capture.push(a.map(String).join(" ")); else orig(...a); };
    try {
      const r = await runBenchmark(ort, tinyMlp(), { ...DEFAULT_CONFIG, warmup: 2, runs: 8, profile: true, dimOverrides: { N: 2 } },
        { name: "mlp.onnx", format: "ONNX", sizeBytes: 1 }, {
          env: { crossOriginIsolated: false, hardwareConcurrency: 2, userAgent: "node" },
          wasmBinary: wasmBytes(wasm),
          captureConsole: async (fn) => {
            capture = [];
            try { return { value: await fn(), lines: capture }; } finally { capture = null; }
          },
        });
      expect(r.inputs).toEqual([{ name: "X", dtype: "float32", dims: [2, 4], bytes: 32, dynamicAxes: [0] }]);
      expect(r.samplesMs).toHaveLength(8);
      expect(r.latency.count).toBe(8);
      expect(r.latency.min).toBeGreaterThanOrEqual(0);
      expect(r.latency.p99).toBeGreaterThanOrEqual(r.latency.median);
      expect(r.sessionCreateMs).toBeGreaterThan(0);
      expect(r.runtimeInitMs).toBeGreaterThan(0);
      expect(r.env.ortVersion).toMatch(/^\d+\.\d+/);
      expect(r.env.effectiveThreads).toBe(1);
      expect(r.profile?.rows.map((x) => x.op).sort()).toEqual(["MatMul", "Relu"]);
      expect(r.warnings).toEqual([]);
    } finally {
      console.log = orig;
    }
  }, 60000);
});
