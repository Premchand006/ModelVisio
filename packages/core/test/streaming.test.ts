import { afterEach, describe, expect, it, vi } from "vitest";
import { createElement as h } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { parseModel, type Model, type ParseChunk } from "@modelvisio/parsers";
import {
  ChunkCoalescer,
  PartialModelBuilder,
  parseStreamingOnMainThread,
  throttleFrames,
} from "../src/workers/streaming";
import { ThemeCtx } from "../src/theme/ThemeContext";
import { TH } from "../src/theme/theme";
import { GraphPanel } from "../src/components/Graph/GraphPanel";
import { demoModel } from "../src/demo/demoModel";

// A real GGUF file with `n` tensors (header only; GGUF weights are never read).
function gguf(n: number): ArrayBuffer {
  const b: number[] = [];
  const u32 = (v: number) => { for (let i = 0; i < 4; i++) b.push((v >>> (8 * i)) & 255); };
  const u64 = (v: number) => { u32(v >>> 0); u32(Math.floor(v / 2 ** 32)); };
  const str = (s: string) => { const e = new TextEncoder().encode(s); u64(e.length); b.push(...e); };
  b.push(...new TextEncoder().encode("GGUF")); u32(3); u64(n); u64(1);
  str("general.architecture"); u32(8); str("llama");
  for (let i = 0; i < n; i++) { str(`blk.${i}.attn_q.weight`); u32(2); u64(32); u64(32); u32(1); u64(0); }
  return new Uint8Array(b).buffer;
}

const chunk = (ids: number[], edges: [number, number][] = [], progress = 0.5): ParseChunk => ({
  layers: ids.map((id) => ({ ...demoModel.layers[0], id, name: `l${id}` })),
  edges,
  progress,
});

describe("ChunkCoalescer", () => {
  it("flushes the first chunk at once, then batches by layer count and time", () => {
    let t = 0;
    const out: ParseChunk[] = [];
    const co = new ChunkCoalescer((c) => out.push(c), 10, 50, () => t);
    co.push({ ...chunk([0]), header: { name: "m", format: "ONNX" } });
    expect(out).toHaveLength(1); // header reaches the UI immediately
    expect(out[0].header?.name).toBe("m");
    co.push(chunk([1, 2, 3]));
    co.push(chunk([4, 5, 6], [[1, 2]]));
    expect(out).toHaveLength(1);
    co.push(chunk([7, 8, 9, 10]));
    expect(out).toHaveLength(2); // 10 layers buffered
    expect(out[1].layers.map((l) => l.id)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(out[1].edges).toEqual([[1, 2]]);
    co.push(chunk([11]));
    expect(out).toHaveLength(2);
    t = 60;
    co.push(chunk([12], [], 0.9));
    expect(out).toHaveLength(3); // 50ms elapsed
    expect(out[2].progress).toBe(0.9);
    co.flush();
    expect(out).toHaveLength(3); // nothing pending: no empty message
  });
});

describe("PartialModelBuilder", () => {
  it("holds back edges until both endpoints arrive and ignores repeated ids", () => {
    const b = new PartialModelBuilder();
    b.push({ ...chunk([0, 1], [[0, 1], [1, 2]], 0.3), header: { name: "big", format: "ONNX" } });
    let m = b.snapshot();
    expect(m.name).toBe("big");
    expect(m.layers).toHaveLength(2);
    expect(m.edges).toEqual([[0, 1]]); // 1→2 waits for layer 2
    expect(m.inputShape).toEqual([]); // header defaults keep the Model shape valid
    b.push(chunk([2, 1], [], 0.6));
    m = b.snapshot();
    expect(m.layers.map((l) => l.id)).toEqual([0, 1, 2]);
    expect(m.edges).toEqual([[0, 1], [1, 2]]);
    expect(b.progress).toBe(0.6);
    // Snapshots are fresh arrays (React state must change identity).
    expect(b.snapshot().layers).not.toBe(m.layers);
  });
});

describe("main-thread progressive fallback", () => {
  it("emits several chunks, yields between slices, and rebuilds parseModel exactly", async () => {
    const buf = gguf(1000);
    const reference = parseModel(buf, "big.gguf");
    let t = 0;
    const chunks: ParseChunk[] = [];
    let yields = 0;
    const model = await parseStreamingOnMainThread(buf, "big.gguf", (c) => chunks.push(c), {
      batchSize: 64,
      sliceMs: 10,
      now: () => (t += 4),
      yieldFn: async () => { yields++; },
    });
    expect(model).toEqual(reference);
    expect(chunks.length).toBeGreaterThan(3);
    expect(yields).toBeGreaterThan(3);
    expect(chunks.flatMap((c) => c.layers).map((l) => l.id)).toEqual(reference.layers.map((l) => l.id));
    expect(chunks.flatMap((c) => c.edges)).toEqual(reference.edges);
    expect(chunks[chunks.length - 1].progress).toBe(1);
  });

  it("parseBufferStreaming falls back to the main thread when no Worker exists", async () => {
    expect(typeof (globalThis as { Worker?: unknown }).Worker).toBe("undefined");
    const { parseBufferStreaming } = await import("../src/workers/parseClient");
    const buf = gguf(600);
    const chunks: ParseChunk[] = [];
    const model = await parseBufferStreaming(buf, "big.gguf", (c) => chunks.push(c));
    expect(model.layers).toHaveLength(600);
    expect(chunks.length).toBeGreaterThan(1);
    const b = new PartialModelBuilder();
    chunks.forEach((c) => b.push(c));
    expect(b.snapshot().layers).toEqual(model.layers);
    expect(b.snapshot().edges).toEqual(model.edges);
  });

  it("shouldStream switches at the 100MB threshold", async () => {
    const { shouldStream, STREAM_THRESHOLD_BYTES } = await import("../src/workers/parseClient");
    expect(STREAM_THRESHOLD_BYTES).toBe(100 * 1024 * 1024);
    expect(shouldStream(STREAM_THRESHOLD_BYTES - 1)).toBe(false);
    expect(shouldStream(STREAM_THRESHOLD_BYTES)).toBe(true);
  });
});

describe("worker streaming protocol", () => {
  afterEach(() => {
    delete (globalThis as { Worker?: unknown }).Worker;
    vi.resetModules();
  });

  it("routes {id, chunk} messages to onChunk and resolves on {id, model}", async () => {
    const sent: { kind: string; id: number; name: string }[] = [];
    class FakeWorker {
      onmessage: ((e: { data: unknown }) => void) | null = null;
      onerror: ((e: { message: string }) => void) | null = null;
      postMessage(msg: { kind: string; id: number; name: string }) {
        sent.push(msg);
        const reply = (data: unknown) => setTimeout(() => this.onmessage?.({ data }), 0);
        reply({ id: msg.id, chunk: { layers: demoModel.layers.slice(0, 2), edges: [], header: { name: "w" }, progress: 0.5 } });
        reply({ id: msg.id, chunk: { layers: demoModel.layers.slice(2, 3), edges: [], progress: 1 } });
        reply({ id: msg.id, model: demoModel });
      }
    }
    (globalThis as { Worker?: unknown }).Worker = FakeWorker;
    vi.resetModules();
    const { parseBufferStreaming } = await import("../src/workers/parseClient");
    const got: ParseChunk[] = [];
    const model = await parseBufferStreaming(new ArrayBuffer(8), "x.onnx", (c) => got.push(c));
    expect(sent[0].kind).toBe("parseStream");
    expect(got.map((c) => c.progress)).toEqual([0.5, 1]);
    expect(model).toEqual(demoModel);
  });

  it("rejects the stream on a worker error message", async () => {
    class FakeWorker {
      onmessage: ((e: { data: unknown }) => void) | null = null;
      onerror: ((e: { message: string }) => void) | null = null;
      postMessage(msg: { id: number }) {
        setTimeout(() => this.onmessage?.({ data: { id: msg.id, error: "bad model" } }), 0);
      }
    }
    (globalThis as { Worker?: unknown }).Worker = FakeWorker;
    vi.resetModules();
    const { parseBufferStreaming } = await import("../src/workers/parseClient");
    await expect(parseBufferStreaming(new ArrayBuffer(8), "x.onnx", () => {})).rejects.toThrow("bad model");
  });
});

describe("throttleFrames", () => {
  it("coalesces bursts into one trailing call per window", () => {
    vi.useFakeTimers();
    try {
      const fn = vi.fn();
      const th = throttleFrames(fn, 250);
      th(); th(); th();
      vi.advanceTimersByTime(1);
      expect(fn).toHaveBeenCalledTimes(1);
      th(); th();
      vi.advanceTimersByTime(100);
      expect(fn).toHaveBeenCalledTimes(1);
      vi.advanceTimersByTime(200);
      expect(fn).toHaveBeenCalledTimes(2);
      th();
      th.cancel();
      vi.advanceTimersByTime(1000);
      expect(fn).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("graph renders partial (streaming) and final models", () => {
  const themed = (m: Model, streaming: boolean) =>
    renderToStaticMarkup(h(ThemeCtx.Provider, { value: TH.dark }, h(GraphPanel, { model: m, selected: null, onSelect: () => {}, streaming })));

  it("renders a partial model whose edges point at layers not yet emitted, then the final one", () => {
    const partial: Model = {
      ...demoModel,
      layers: demoModel.layers.slice(0, 5),
      // Includes edges into layers that haven't streamed in yet.
      edges: demoModel.edges.filter(([a]) => a < 5),
    };
    const html = themed(partial, true);
    expect(html).toContain("<svg");
    const finalHtml = themed(demoModel, false);
    expect(finalHtml.length).toBeGreaterThan(html.length);
  });

  it("renders a header-only partial model (no layers yet)", () => {
    expect(() => themed({ ...demoModel, layers: [], edges: [] }, true)).not.toThrow();
  });
});
