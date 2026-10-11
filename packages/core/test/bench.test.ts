import { beforeEach, describe, expect, it, vi } from "vitest";
import { demoModel } from "../src/demo/demoModel";
import { summarize, percentile, percentileSorted } from "../src/bench/stats";
import {
  defaultDynamicDim, dimKey, inputSpecsFromModel, mulberry32, resolveDims, resolveInputs,
  synthesizeData, toHalfBits, MAX_INPUT_ELEMENTS, type InputSpec,
} from "../src/bench/inputs";
import { aggregateProfile, byOpType, parseTraceLines } from "../src/bench/profile";
import { tinyIdentityModel } from "../src/bench/tinyModel";
import { chartGeometry, niceMax, predictFor, benchReport, fmtMs } from "../src/bench/compare";
import { availableEps, detectCapabilities, isBenchmarkable } from "../src/bench/capabilities";
import { configureOrtEnv, runBenchmark, sessionOptions, specsFromSession, type OrtLike, type OrtSessionLike } from "../src/bench/runner";
import { createBenchStore } from "../src/bench/store";
import { setBenchWasmUrl, startBenchmark } from "../src/bench/benchClient";
import { GPU_WASM_FILE, GPU_WASM_URL, ORT_WEB_VERSION, verifyGpuWasm } from "../src/bench/gpuWasm";
import { readFileSync } from "node:fs";
import { DEFAULT_CONFIG, type BenchConfig, type BenchResult, type FromWorker, type ToWorker } from "../src/bench/types";
import { modelProtoType } from "../../parsers/src/onnx/proto";
import { parseOnnx } from "@modelvisio/parsers";
import type { Model } from "@modelvisio/parsers";

describe("bench stats", () => {
  it("percentiles interpolate linearly between closest ranks (NumPy default)", () => {
    const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentileSorted(xs, 50)).toBeCloseTo(5.5);
    expect(percentileSorted(xs, 90)).toBeCloseTo(9.1);
    expect(percentileSorted(xs, 99)).toBeCloseTo(9.91);
    expect(percentileSorted(xs, 0)).toBe(1);
    expect(percentileSorted(xs, 100)).toBe(10);
    expect(percentile([10, 1, 5], 50)).toBe(5);
    expect(percentileSorted([], 50)).toBeNaN();
    expect(percentileSorted([7], 99)).toBe(7);
  });

  it("summarize computes mean/median/std/min/max/throughput", () => {
    const s = summarize([4, 2, 6, 8]);
    expect(s.count).toBe(4);
    expect(s.mean).toBe(5);
    expect(s.median).toBe(5);
    expect(s.min).toBe(2);
    expect(s.max).toBe(8);
    expect(s.std).toBeCloseTo(Math.sqrt(20 / 3)); // sample std
    expect(s.throughput).toBe(200);
  });

  it("summarize handles one sample, empty input and non-finite values", () => {
    expect(summarize([3]).std).toBe(0);
    expect(summarize([3]).p99).toBe(3);
    const e = summarize([]);
    expect(e.count).toBe(0);
    expect(e.throughput).toBe(0);
    expect(summarize([1, NaN, Infinity, 3]).count).toBe(2);
  });
});

describe("bench inputs: dynamic dims", () => {
  it("defaults: batch → 1, image NCHW → 3×224×224, sequence → 128", () => {
    expect(defaultDynamicDim([null, null, null, null], 0)).toBe(1);
    expect(defaultDynamicDim([null, null, null, null], 1)).toBe(3);
    expect(defaultDynamicDim([null, 3, null, null], 2)).toBe(224);
    expect(defaultDynamicDim([null, null], 1)).toBe(128);
    expect(defaultDynamicDim([null, null, 768], 1)).toBe(128);
    expect(defaultDynamicDim([null, 5, null, null, null], 4)).toBe(1);
  });

  it("resolves fixed dims as declared and dynamic ones via overrides (axis key beats symbolic name)", () => {
    const spec: InputSpec = { name: "ids", dtype: "int64", shape: ["batch", "seq", 16] };
    expect(resolveDims(spec)).toEqual({ dims: [1, 128, 16], dynamicAxes: [0, 1] });
    expect(resolveDims(spec, { seq: 64 }).dims).toEqual([1, 64, 16]);
    expect(resolveDims(spec, { seq: 64, [dimKey("ids", 1)]: 32 }).dims).toEqual([1, 32, 16]);
    expect(resolveDims(spec, { batch: 4 }).dims).toEqual([4, 128, 16]);
    // Invalid overrides are ignored.
    expect(resolveDims(spec, { [dimKey("ids", 1)]: 0, seq: -3 }).dims).toEqual([1, 128, 16]);
  });

  it("resolveInputs computes byte sizes and rejects huge or unsupported inputs", () => {
    const r = resolveInputs([{ name: "x", dtype: "float16", shape: [null, 3, 8, 8] }]);
    expect(r[0]).toMatchObject({ name: "x", dims: [1, 3, 8, 8], bytes: 1 * 3 * 8 * 8 * 2, dynamicAxes: [0] });
    expect(() => resolveInputs([{ name: "x", dtype: "float32", shape: [MAX_INPUT_ELEMENTS + 1] }])).toThrow(/too large/);
    expect(() => resolveInputs([{ name: "x", dtype: "bfloat16", shape: [1] }])).toThrow(/bfloat16/);
  });

  it("reads graph inputs from the parsed Model (dynamic dims → null)", () => {
    const model = parseOnnx(buildModel({ dynBatch: true }), { name: "m.onnx" });
    expect(inputSpecsFromModel(model)).toEqual([{ name: "X", dtype: "float32", shape: [null, 4] }]);
    const noInputs = { ...demoModel, layers: demoModel.layers.filter((l) => l.op !== "Input"), inputShape: [1, 3, 640, 640] } as Model;
    expect(inputSpecsFromModel(noInputs)[0].shape).toEqual([1, 3, 640, 640]);
  });
});

describe("bench inputs: tensor synthesis", () => {
  it("returns the typed array onnxruntime expects for each dtype", () => {
    const r = mulberry32(7);
    expect(synthesizeData("float32", 4, "random", r)).toBeInstanceOf(Float32Array);
    expect(synthesizeData("float64", 4, "random", r)).toBeInstanceOf(Float64Array);
    expect(synthesizeData("float16", 4, "random", r)).toBeInstanceOf(Uint16Array);
    expect(synthesizeData("int64", 4, "random", r)).toBeInstanceOf(BigInt64Array);
    expect(synthesizeData("uint64", 4, "random", r)).toBeInstanceOf(BigUint64Array);
    expect(synthesizeData("int32", 4, "random", r)).toBeInstanceOf(Int32Array);
    expect(synthesizeData("int8", 4, "random", r)).toBeInstanceOf(Int8Array);
    expect(synthesizeData("uint8", 4, "random", r)).toBeInstanceOf(Uint8Array);
    expect(synthesizeData("bool", 4, "random", r)).toBeInstanceOf(Uint8Array);
    expect(synthesizeData("string", 2, "zeros", r)).toEqual(["", ""]);
    expect(() => synthesizeData("complex64", 1, "random", r)).toThrow();
  });

  it("random floats are in [0,1); integers/bools are 0/1 (valid indices & masks); zeros are zero", () => {
    const f = synthesizeData("float32", 1000, "random", mulberry32(1)) as Float32Array;
    expect(Math.min(...f)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...f)).toBeLessThan(1);
    const ids = synthesizeData("int64", 200, "random", mulberry32(1)) as BigInt64Array;
    expect(new Set(Array.from(ids, String))).toEqual(new Set(["0", "1"]));
    const z = synthesizeData("float32", 16, "zeros") as Float32Array;
    expect(z.every((v) => v === 0)).toBe(true);
  });

  it("is reproducible from a seed", () => {
    const a = synthesizeData("float32", 8, "random", mulberry32(42));
    const b = synthesizeData("float32", 8, "random", mulberry32(42));
    expect(Array.from(a as Float32Array)).toEqual(Array.from(b as Float32Array));
  });

  it("encodes float16 bits correctly", () => {
    expect(toHalfBits(0)).toBe(0);
    expect(toHalfBits(1)).toBe(0x3c00);
    expect(toHalfBits(-2)).toBe(0xc000);
    expect(toHalfBits(0.5)).toBe(0x3800);
    expect(toHalfBits(65504)).toBe(0x7bff);
    expect(toHalfBits(1e6)).toBe(0x7c00);
    expect(toHalfBits(Infinity)).toBe(0x7c00);
    expect(toHalfBits(NaN) & 0x7c00).toBe(0x7c00);
    expect(toHalfBits(2 ** -24)).toBe(1); // smallest subnormal
    expect(toHalfBits(0.1)).toBe(0x2e66);
  });
});

describe("bench tiny warm-up model", () => {
  it("is a valid ONNX Identity model", () => {
    const bytes = tinyIdentityModel();
    const m = modelProtoType().decode(bytes) as unknown as {
      ir_version: unknown; graph: { node: { op_type: string; input: string[]; output: string[] }[]; input: { name: string }[]; output: { name: string }[] };
      opset_import: { version: unknown }[];
    };
    expect(Number(m.ir_version)).toBe(8);
    expect(Number(m.opset_import[0].version)).toBe(13);
    expect(m.graph.node[0]).toMatchObject({ op_type: "Identity", input: ["X"], output: ["Y"] });
    const parsed = parseOnnx(bytes, { name: "tiny.onnx" });
    expect(parsed.inputShape).toEqual([1]);
    expect(parsed.layers.some((l) => l.op === "Identity")).toBe(true);
  });
});

describe("bench per-op profile parsing", () => {
  const line = (name: string, op: string, dur: number) =>
    `{"cat" : "Node","pid" :42,"tid" :1,"dur" :${dur},"ts" :1,"ph" : "X","name" :"${name}_kernel_time","args" : {"op_name" : "${op}","provider" : "CPUExecutionProvider"}},`;
  const lines = ["[", line("conv1", "Conv", 9000), line("relu1", "Relu", 1000), `{"cat" : "Session","dur" :5,"name" :"SequentialExecutor::Execute"},`,
    line("conv1", "Conv", 3000), line("relu1", "Relu", 1000), "]", "not json"];

  it("parses Chrome-trace lines and aggregates per node", () => {
    const ev = parseTraceLines(lines);
    expect(ev.length).toBe(5);
    const rows = aggregateProfile(ev, 2);
    expect(rows[0]).toMatchObject({ node: "conv1", op: "Conv", provider: "CPU", calls: 2 });
    expect(rows[0].meanMs).toBeCloseTo(6);
    expect(rows[0].share).toBeCloseTo(12 / 14);
  });

  it("skipFirst drops each node's cold first event", () => {
    const rows = aggregateProfile(parseTraceLines(lines), 1, true);
    expect(rows.find((r) => r.node === "conv1")!.meanMs).toBeCloseTo(3);
    expect(rows.find((r) => r.node === "conv1")!.calls).toBe(1);
  });

  it("rolls up to op types", () => {
    const rows = aggregateProfile(parseTraceLines([line("a", "Conv", 2000), line("b", "Conv", 2000), line("c", "Add", 1000)]), 1);
    expect(byOpType(rows)[0]).toMatchObject({ op: "Conv", nodes: 2 });
    expect(byOpType(rows)[0].meanMs).toBeCloseTo(4);
  });
});

describe("bench compare + chart", () => {
  it("predicts latency for a hardware-DB device", () => {
    const p = predictFor(demoModel, "orin_nano")!;
    expect(p.deviceName).toMatch(/Orin Nano/);
    expect(p.latencyMs).toBeGreaterThan(0);
    expect(predictFor(demoModel, "nope")).toBeNull();
  });

  it("niceMax rounds up to 1/2/2.5/5×10^k", () => {
    expect(niceMax(7)).toBe(10);
    expect(niceMax(1.7)).toBe(2);
    expect(niceMax(22)).toBe(25);
    expect(niceMax(0)).toBe(1);
  });

  it("chart geometry scales bars to the axis max and adds the predicted bar", () => {
    const runs = [fakeResult("a", [10, 10, 10]), fakeResult("b", [20, 20, 30]), fakeResult("c", [])];
    const g = chartGeometry(runs, { deviceId: "x", deviceName: "X", precision: "int8", latencyMs: 5, fps: 200, bound: "compute", calibrated: false }, { width: 100 });
    expect(g.bars.map((b) => b.kind)).toEqual(["measured", "measured", "predicted"]); // empty run skipped
    expect(g.maxMs).toBe(50);
    expect(g.bars[0].w).toBeCloseTo(20);
    expect(g.bars[1].p90x).toBeGreaterThan(g.bars[1].w);
    expect(g.bars[2].w).toBeCloseTo(10);
    expect(g.ticks[4]).toEqual({ x: 100, ms: 50 });
  });

  it("report JSON carries runs and the prediction caveat", () => {
    const doc = benchReport([fakeResult("a", [1])], predictFor(demoModel, "orin_nano"));
    expect(doc.runs).toHaveLength(1);
    expect(doc.prediction?.note).toMatch(/deployment precision/);
    expect(JSON.parse(JSON.stringify(doc)).schema).toBe(1);
  });

  it("formats milliseconds", () => {
    expect(fmtMs(0.4234)).toBe("0.42 ms");
    expect(fmtMs(12.34)).toBe("12.3 ms");
    expect(fmtMs(1250)).toBe("1.25 s");
    expect(fmtMs(NaN)).toBe("–");
  });
});

describe("bench capabilities", () => {
  it("never throws and detects features from an injected global", () => {
    const c = detectCapabilities({ Worker: function () {}, WebAssembly: { validate: () => true }, SharedArrayBuffer: function () {}, crossOriginIsolated: true, navigator: { gpu: {}, hardwareConcurrency: 8 } });
    expect(c).toMatchObject({ worker: true, wasm: true, simd: true, threads: true, webgpu: true, webnn: false, hardwareConcurrency: 8 });
    expect(availableEps(c)).toEqual(["wasm", "webgpu"]);
    const bare = detectCapabilities({});
    expect(bare).toMatchObject({ worker: false, wasm: false, threads: false, webgpu: false, hardwareConcurrency: 1 });
    expect(availableEps(bare)).toEqual([]);
    const throwing = detectCapabilities({ WebAssembly: { validate: () => { throw new Error("x"); } } });
    expect(throwing.simd).toBe(false);
  });

  it("real Node has wasm SIMD", () => {
    expect(detectCapabilities().simd).toBe(true);
  });

  it("only ONNX is benchmarkable", () => {
    expect(isBenchmarkable({ format: "ONNX" })).toBe(true);
    expect(isBenchmarkable({ format: "TFLite" })).toBe(false);
  });
});

// ─── Runner on a mocked onnxruntime ──────────────────────────────────────────

class FakeTensor {
  constructor(public type: string, public data: unknown, public dims: readonly number[]) {}
  disposed = false;
  dispose() { this.disposed = true; }
}

function mockOrt(opts: { runMs?: number; meta?: OrtSessionLike["inputMetadata"]; failCreate?: (n: number) => boolean; failRun?: boolean; profileLines?: string[] } = {}) {
  let clock = 0;
  let creates = 0;
  const feedsSeen: Record<string, FakeTensor>[] = [];
  const opts2: Record<string, unknown>[] = [];
  const env: OrtLike["env"] = { wasm: {}, versions: { web: "9.9.9" } };
  const ort: OrtLike = {
    env,
    Tensor: FakeTensor as unknown as OrtLike["Tensor"],
    InferenceSession: {
      async create(_bytes, o) {
        creates++;
        opts2.push(o ?? {});
        clock += 50;
        if (opts.failCreate?.(creates)) throw new Error("no backend found");
        const isTiny = creates === 1;
        return {
          inputNames: isTiny ? ["X"] : (opts.meta ?? []).map((m) => m.name),
          inputMetadata: isTiny ? [{ name: "X", isTensor: true, type: "float32", shape: [1] }] : opts.meta,
          async run(feeds) {
            if (opts.failRun && !isTiny) throw new Error("bad shape");
            feedsSeen.push(feeds as Record<string, FakeTensor>);
            clock += opts.runMs ?? 2;
            return { Y: new FakeTensor("float32", new Float32Array(1), [1]) };
          },
          async release() {},
          endProfiling() { if (opts.profileLines) opts.profileLines.forEach((l) => console.log(l)); },
        };
      },
    },
  };
  return { ort, now: () => clock, feedsSeen, sessionOpts: opts2, get creates() { return creates; } };
}

const META = [
  { name: "input_ids", isTensor: true, type: "int64", shape: ["batch", "sequence"] },
  { name: "pixel", isTensor: true, type: "float32", shape: [-1, 3, 4, 4] },
] as const;
const MODEL_INFO = { name: "m.onnx", format: "ONNX", sizeBytes: 10 };
const cfg = (p: Partial<BenchConfig> = {}): BenchConfig => ({ ...DEFAULT_CONFIG, warmup: 2, runs: 5, timeBudgetMs: 0, ...p });

describe("bench runner (mock runtime)", () => {
  it("configureOrtEnv clamps threads without cross-origin isolation and sets wasm paths", () => {
    const env: OrtLike["env"] = { wasm: {} };
    const r = configureOrtEnv(env, { threads: 8 }, { crossOriginIsolated: false, hardwareConcurrency: 8, wasmUrl: "https://x/ort.wasm" });
    expect(env.wasm).toMatchObject({ numThreads: 1, proxy: false, wasmPaths: { wasm: "https://x/ort.wasm" } });
    const env3: OrtLike["env"] = { wasm: {} };
    configureOrtEnv(env3, { threads: 1 }, { crossOriginIsolated: false, hardwareConcurrency: 2, wasmUrl: "https://x/ort.wasm", wasmBinary: new Uint8Array([1, 2]).buffer });
    expect(env3.wasm.wasmBinary).toEqual(new Uint8Array([1, 2]));
    expect(env3.wasm.wasmPaths).toBeUndefined();
    expect(r.effectiveThreads).toBe(1);
    expect(r.warnings[0]).toMatch(/cross-origin isolated/);
    const env2: OrtLike["env"] = { wasm: {} };
    expect(configureOrtEnv(env2, { threads: 0 }, { crossOriginIsolated: true, hardwareConcurrency: 12 }).effectiveThreads).toBe(4);
    expect(env2.wasm.numThreads).toBe(0);
    expect(env2.wasm.wasmPaths).toBeUndefined();
  });

  it("sessionOptions maps EPs", () => {
    expect(sessionOptions(cfg({ ep: "wasm" })).executionProviders).toEqual(["wasm"]);
    expect(sessionOptions(cfg({ ep: "webgpu" })).executionProviders).toEqual(["webgpu"]);
    expect(sessionOptions(cfg({ ep: "webnn", webnnDevice: "npu" })).executionProviders).toEqual([{ name: "webnn", deviceType: "npu" }]);
    expect(sessionOptions(cfg(), true).enableProfiling).toBe(true);
  });

  it("specsFromSession prefers runtime metadata, else falls back to Model specs", () => {
    const s = { inputNames: ["a"], inputMetadata: [{ name: "a", isTensor: true, type: "int32", shape: ["n", 0, 5] }] } as unknown as OrtSessionLike;
    expect(specsFromSession(s)).toEqual([{ name: "a", dtype: "int32", shape: ["n", null, 5] }]);
    const noMeta = { inputNames: ["a"] } as unknown as OrtSessionLike;
    expect(specsFromSession(noMeta, [{ name: "a", dtype: "float16", shape: [2] }])).toEqual([{ name: "a", dtype: "float16", shape: [2] }]);
    const nonTensor = { inputNames: ["a"], inputMetadata: [{ name: "a", isTensor: false }] } as unknown as OrtSessionLike;
    expect(() => specsFromSession(nonTensor)).toThrow(/not a tensor/);
  });

  it("times init, session, first, warm-up and N timed runs with synthesized feeds", async () => {
    const m = mockOrt({ runMs: 3, meta: [...META] });
    const progress: string[] = [];
    const r = await runBenchmark(m.ort, new Uint8Array(4), cfg({ dimOverrides: { sequence: 16 } }), MODEL_INFO, {
      now: m.now, onProgress: (p) => progress.push(p.phase), yieldToEvents: async () => {}, makeId: () => "id1",
      env: { crossOriginIsolated: false, hardwareConcurrency: 4, userAgent: "test" },
    });
    expect(r.id).toBe("id1");
    expect(r.runtimeInitMs).toBe(53); // tiny create (50) + tiny run (3)
    expect(r.sessionCreateMs).toBe(50);
    expect(r.firstInferenceMs).toBe(3);
    expect(r.warmupMs).toEqual([3, 3]);
    expect(r.samplesMs).toHaveLength(5);
    expect(r.latency.median).toBe(3);
    expect(r.env).toMatchObject({ ortVersion: "9.9.9", effectiveThreads: 1, userAgent: "test" });
    expect(r.inputs).toEqual([
      { name: "input_ids", dtype: "int64", dims: [1, 16], bytes: 128, dynamicAxes: [0, 1] },
      { name: "pixel", dtype: "float32", dims: [1, 3, 4, 4], bytes: 192, dynamicAxes: [0] },
    ]);
    const feeds = m.feedsSeen[1];
    expect(feeds.input_ids.data).toBeInstanceOf(BigInt64Array);
    expect(feeds.pixel.dims).toEqual([1, 3, 4, 4]);
    expect(progress).toEqual(expect.arrayContaining(["init", "session", "first", "warmup", "timed"]));
    expect(r.cancelled).toBe(false);
    expect(r.memory.note).toBeTruthy();
  });

  it("stops at the time budget", async () => {
    const m = mockOrt({ runMs: 100, meta: [META[1]] });
    const r = await runBenchmark(m.ort, new Uint8Array(4), cfg({ warmup: 0, runs: 50, timeBudgetMs: 350 }), MODEL_INFO, { now: m.now, yieldToEvents: async () => {} });
    expect(r.samplesMs.length).toBe(4);
    expect(r.warnings.join(" ")).toMatch(/Time budget/);
  });

  it("cancel mid-run returns partial results; cancel before any timed run rejects", async () => {
    const m = mockOrt({ runMs: 1, meta: [META[1]] });
    let n = 0;
    let cancelled = false;
    const r = await runBenchmark(m.ort, new Uint8Array(4), cfg({ warmup: 0, runs: 100 }), MODEL_INFO, {
      now: () => (n++, m.now() + n * 100), isCancelled: () => cancelled,
      onProgress: (p) => { if (p.phase === "timed" && p.done === 3) cancelled = true; },
      yieldToEvents: async () => {},
    });
    expect(r.cancelled).toBe(true);
    expect(r.samplesMs.length).toBe(3);
    const m2 = mockOrt({ meta: [META[1]] });
    await expect(runBenchmark(m2.ort, new Uint8Array(4), cfg(), MODEL_INFO, { now: m2.now, isCancelled: () => true, yieldToEvents: async () => {} })).rejects.toThrow(/cancelled/i);
  });

  it("reports EP init and inference failures clearly", async () => {
    const m = mockOrt({ meta: [META[1]], failCreate: (n) => n === 1 });
    await expect(runBenchmark(m.ort, new Uint8Array(4), cfg({ ep: "webgpu" }), MODEL_INFO, { now: m.now })).rejects.toThrow(/WebGPU runtime failed to initialize: no backend found/);
    const m2 = mockOrt({ meta: [META[1]], failCreate: (n) => n === 2 });
    await expect(runBenchmark(m2.ort, new Uint8Array(4), cfg(), MODEL_INFO, { now: m2.now })).rejects.toThrow(/Couldn't create an inference session on WebAssembly/);
    const m3 = mockOrt({ meta: [META[1]], failRun: true });
    await expect(runBenchmark(m3.ort, new Uint8Array(4), cfg(), MODEL_INFO, { now: m3.now })).rejects.toThrow(/Inference failed with inputs pixel\[1×3×4×4\] float32: bad shape/);
  });

  it("profile pass uses a separate profiled session and parses captured console output", async () => {
    const lines = [
      `{"cat" : "Node","dur" :5000,"name" :"conv_kernel_time","args" : {"op_name" : "Conv","provider" : "CPUExecutionProvider"}},`,
      `{"cat" : "Node","dur" :2000,"name" :"conv_kernel_time","args" : {"op_name" : "Conv","provider" : "CPUExecutionProvider"}},`,
    ];
    const m = mockOrt({ meta: [META[1]], profileLines: lines });
    const captured: string[] = [];
    const r = await runBenchmark(m.ort, new Uint8Array(4), cfg({ profile: true, runs: 1 }), MODEL_INFO, {
      now: m.now, yieldToEvents: async () => {},
      captureConsole: async (fn) => {
        const orig = console.log;
        console.log = (...a: unknown[]) => { captured.push(a.join(" ")); };
        try { return { value: await fn(), lines: captured }; } finally { console.log = orig; }
      },
    });
    expect(m.creates).toBe(3);
    expect(m.sessionOpts[2].enableProfiling).toBe(true);
    expect(m.sessionOpts[1].enableProfiling).toBe(false);
    expect(r.profile?.runs).toBe(1);
    expect(r.profile?.rows[0]).toMatchObject({ node: "conv", op: "Conv" });
    expect(r.profile?.rows[0].meanMs).toBeCloseTo(2); // cold first event dropped
  });
});

// ─── Worker protocol (client ↔ fake worker) ──────────────────────────────────

class FakeWorker {
  onmessage: ((e: MessageEvent<FromWorker>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  sent: ToWorker[] = [];
  transfers: unknown[][] = [];
  terminated = false;
  postMessage(m: ToWorker, t?: unknown[]) { this.sent.push(m); this.transfers.push(t ?? []); }
  terminate() { this.terminated = true; }
  emit(m: FromWorker) { this.onmessage?.({ data: m } as MessageEvent<FromWorker>); }
}

describe("bench client protocol", () => {
  it("copies the bytes, relays progress, resolves the result and terminates the worker", async () => {
    const w = new FakeWorker();
    const bytes = new Uint8Array([1, 2, 3]).buffer;
    const progress: number[] = [];
    const h = startBenchmark({ bytes, config: cfg(), model: MODEL_INFO, createWorker: () => w as unknown as Worker, onProgress: (p) => progress.push(p.done) });
    await vi.waitFor(() => expect(w.sent).toHaveLength(1));
    const run = w.sent[0] as Extract<ToWorker, { type: "run" }>;
    expect(run.type).toBe("run");
    expect(run.bytes).not.toBe(bytes); // caller's buffer stays usable
    expect(bytes.byteLength).toBe(3);
    expect(w.transfers[0]).toEqual([run.bytes]);
    expect(run.wasmUrl).toBeUndefined();
    expect(run.wasmBinary).toBeUndefined();
    w.emit({ type: "progress", progress: { phase: "timed", done: 1, total: 2 } });
    w.emit({ type: "result", result: fakeResult("r", [1]) });
    await expect(h.promise).resolves.toMatchObject({ id: "r" });
    expect(progress).toEqual([1]);
    expect(w.terminated).toBe(true);
  });

  it("rejects on worker error messages and crashes", async () => {
    const w = new FakeWorker();
    const h = startBenchmark({ bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO, createWorker: () => w as unknown as Worker });
    await vi.waitFor(() => expect(w.sent).toHaveLength(1));
    w.emit({ type: "error", message: "boom" });
    await expect(h.promise).rejects.toThrow("boom");
    const w2 = new FakeWorker();
    const h2 = startBenchmark({ bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO, createWorker: () => w2 as unknown as Worker });
    w2.onerror?.({ message: "" } as ErrorEvent);
    await expect(h2.promise).rejects.toThrow(/crashed/);
    const h3 = startBenchmark({ bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO, createWorker: () => { throw new Error("CSP"); } });
    await expect(h3.promise).rejects.toThrow(/Web Worker.*CSP/);
  });

  it("prefetches a custom-URL wasm on the main thread once and transfers a copy per run", async () => {
    const fetchImpl = vi.fn(async () => new Response(new Uint8Array([0, 97, 115, 109])));
    setBenchWasmUrl("https://cdn.example/media/assets/ort.wasm");
    try {
      for (let i = 0; i < 2; i++) {
        const w = new FakeWorker();
        startBenchmark({ bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO, createWorker: () => w as unknown as Worker, fetchImpl: fetchImpl as unknown as typeof fetch });
        await vi.waitFor(() => expect(w.sent).toHaveLength(1));
        const run = w.sent[0] as Extract<ToWorker, { type: "run" }>;
        expect(run.wasmUrl).toBe("https://cdn.example/media/assets/ort.wasm");
        expect(new Uint8Array(run.wasmBinary!)).toEqual(new Uint8Array([0, 97, 115, 109]));
        expect(w.transfers[0]).toContain(run.wasmBinary);
      }
      expect(fetchImpl).toHaveBeenCalledTimes(1);
    } finally {
      setBenchWasmUrl(undefined);
    }
  });

  it("cancel before the run is posted rejects at once and kills the worker", async () => {
    const w = new FakeWorker();
    const h = startBenchmark({ bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO, createWorker: () => w as unknown as Worker });
    h.cancel();
    await expect(h.promise).rejects.toThrow("Cancelled");
    expect(w.terminated).toBe(true);
    await Promise.resolve();
    expect(w.sent).toHaveLength(0);
  });

  it("cancel posts a cancel message, then kills a stuck worker after the grace period", async () => {
    const w = new FakeWorker();
    const h = startBenchmark({ bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO, createWorker: () => w as unknown as Worker, cancelGraceMs: 100 });
    await vi.waitFor(() => expect(w.sent).toHaveLength(1));
    vi.useFakeTimers();
    try {
      h.cancel();
      expect(w.sent[1]).toEqual({ type: "cancel" });
      vi.advanceTimersByTime(150);
      await expect(h.promise).rejects.toThrow("Cancelled");
      expect(w.terminated).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("bench session store", () => {
  it("keeps history per model, one active run at a time, and swallows plain cancels", async () => {
    let resolveRun!: (r: BenchResult) => void;
    let rejectRun!: (e: Error) => void;
    const cancel = vi.fn();
    const store = createBenchStore(() => ({ promise: new Promise<BenchResult>((res, rej) => { resolveRun = res; rejectRun = rej; }), cancel }));
    const seen: number[] = [];
    store.subscribe(() => seen.push(1));
    expect(store.run("k", { bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO })).toBe(true);
    expect(store.getSnapshot().active?.modelKey).toBe("k");
    expect(store.run("k", { bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO })).toBe(false);
    resolveRun(fakeResult("one", [1]));
    await Promise.resolve();
    await Promise.resolve();
    expect(store.getSnapshot().active).toBeNull();
    expect(store.getSnapshot().runs.k.map((r) => r.id)).toEqual(["one"]);

    store.run("k", { bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO });
    store.cancel();
    expect(cancel).toHaveBeenCalled();
    expect(store.getSnapshot().active?.cancelling).toBe(true);
    rejectRun(new Error("Cancelled"));
    await Promise.resolve();
    await Promise.resolve();
    expect(store.getSnapshot().errors.k).toBeNull();

    store.run("k", { bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO });
    rejectRun(new Error("OOM"));
    await Promise.resolve();
    await Promise.resolve();
    expect(store.getSnapshot().errors.k).toBe("OOM");

    store.setConfig("k", cfg({ runs: 7 }));
    expect(store.configFor("k").runs).toBe(7);
    expect(store.configFor("other")).toEqual(DEFAULT_CONFIG);
    store.remove("k", "one");
    expect(store.getSnapshot().runs.k).toEqual([]);
    expect(seen.length).toBeGreaterThan(5);
  });
});

// ─── helpers ─────────────────────────────────────────────────────────────────

function fakeResult(id: string, samples: number[]): BenchResult {
  return {
    id, startedAt: "2026-01-01T00:00:00.000Z", model: MODEL_INFO, config: cfg(),
    env: { userAgent: "", hardwareConcurrency: 4, crossOriginIsolated: false, effectiveThreads: 1 },
    inputs: [], runtimeInitMs: 1, sessionCreateMs: 2, firstInferenceMs: 3, warmupMs: [], samplesMs: samples,
    latency: summarize(samples), memory: {}, cancelled: false, warnings: [],
  };
}

/** X[N?,4] → Relu → Y, encoded through the parser's own protobuf schema. */
function buildModel({ dynBatch }: { dynBatch: boolean }): Uint8Array {
  const M = modelProtoType();
  const dims = [dynBatch ? { dim_param: "N" } : { dim_value: 2 }, { dim_value: 4 }];
  const vi = (name: string) => ({ name, type: { tensor_type: { elem_type: 1, shape: { dim: dims } } } });
  return M.encode(M.fromObject({
    ir_version: 8, producer_name: "bench-test", opset_import: [{ domain: "", version: 17 }],
    graph: { name: "g", input: [vi("X")], output: [vi("Y")], node: [{ op_type: "Relu", name: "relu", input: ["X"], output: ["Y"] }] },
  })).finish();
}

describe("GPU runtime download", () => {
  // The verified binary is cached per page session; start each test without it.
  beforeEach(() => setBenchWasmUrl(undefined));
  const gpuCfg = () => ({ ...cfg(), ep: "webgpu" as const });

  it("pins the installed onnxruntime-web release and its GPU binary's hash", async () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(pkg.dependencies["onnxruntime-web"]).toBe(ORT_WEB_VERSION);
    const local = readFileSync(new URL(`../node_modules/onnxruntime-web/dist/${GPU_WASM_FILE}`, import.meta.url));
    const ab = local.buffer.slice(local.byteOffset, local.byteOffset + local.byteLength);
    expect(await verifyGpuWasm(ab)).toBe(true);
    expect(await verifyGpuWasm(new ArrayBuffer(8))).toBe(false);
  });

  it("uses the GPU worker with the verified CDN binary, and the CPU worker without any download", async () => {
    const fetchImpl = vi.fn(async () => new Response(new Uint8Array([1, 2, 3])));
    const kinds: boolean[] = [];
    const w = new FakeWorker();
    startBenchmark({
      bytes: new ArrayBuffer(1), config: gpuCfg(), model: MODEL_INFO, fetchImpl: fetchImpl as unknown as typeof fetch,
      verifyGpuWasm: async () => true, createWorker: (gpu) => { kinds.push(gpu); return w as unknown as Worker; },
    });
    await vi.waitFor(() => expect(w.sent).toHaveLength(1));
    const run = w.sent[0] as Extract<ToWorker, { type: "run" }>;
    expect(run.wasmUrl).toBe(GPU_WASM_URL);
    expect(new Uint8Array(run.wasmBinary!)).toEqual(new Uint8Array([1, 2, 3]));
    expect(fetchImpl).toHaveBeenCalledWith(GPU_WASM_URL);

    const c = new FakeWorker();
    startBenchmark({ bytes: new ArrayBuffer(1), config: cfg(), model: MODEL_INFO, createWorker: (gpu) => { kinds.push(gpu); return c as unknown as Worker; } });
    await vi.waitFor(() => expect(c.sent).toHaveLength(1));
    expect(kinds).toEqual([true, false]);
  });

  it("refuses a download that fails the checksum, then retries on the next run", async () => {
    const fetchImpl = vi.fn(async () => new Response(new Uint8Array([9])));
    const w = new FakeWorker();
    const h = startBenchmark({
      bytes: new ArrayBuffer(1), config: { ...cfg(), ep: "webnn" as const }, model: MODEL_INFO, fetchImpl: fetchImpl as unknown as typeof fetch,
      verifyGpuWasm: async () => false, createWorker: () => w as unknown as Worker,
    });
    await expect(h.promise).rejects.toThrow(/checksum.*WebAssembly \(CPU\)/s);
    expect(w.sent).toHaveLength(0);
    expect(w.terminated).toBe(true);
    const w2 = new FakeWorker();
    startBenchmark({
      bytes: new ArrayBuffer(1), config: gpuCfg(), model: MODEL_INFO, fetchImpl: fetchImpl as unknown as typeof fetch,
      verifyGpuWasm: async () => true, createWorker: () => w2 as unknown as Worker,
    });
    await vi.waitFor(() => expect(w2.sent).toHaveLength(1));
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});
