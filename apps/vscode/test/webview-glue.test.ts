import { describe, expect, it, vi } from "vitest";
import type { AppApi } from "@modelvisio/core";
import type { HostToWebview, WebviewToHost } from "../src/protocol";
import { createModelSink } from "../webview/modelSink";
import { createRpc } from "../webview/rpc";
import { createSaveHandler } from "../webview/save";
import { createWorkerClass, prefetchWorkerScripts, resolveWorkerUrl, workerAssets } from "../webview/workerShim";
import type { FetchLike } from "../webview/fetchBridge";

describe("save handler", () => {
  function setup(reply: Omit<Extract<HostToWebview, { type: "saveResult" }>, "id">) {
    const sent: WebviewToHost[] = [];
    const rpc = createRpc((m) => {
      sent.push(m);
      if (m.type === "save") queueMicrotask(() => rpc.handle({ ...reply, id: m.id }));
    });
    return { save: createSaveHandler(rpc, (m) => sent.push(m)), sent };
  }

  it("posts the bytes and stays quiet on success or cancel", async () => {
    for (const reply of [{ type: "saveResult", saved: true, path: "/x" }, { type: "saveResult", saved: false }] as const) {
      const { save, sent } = setup(reply);
      save(new Uint8Array([1, 2, 3]), "m.safetensors", "application/octet-stream");
      await vi.waitFor(() => expect(sent).toHaveLength(1));
      await new Promise((r) => setTimeout(r, 0));
      expect(sent).toEqual([{ type: "save", id: expect.any(Number), filename: "m.safetensors", mime: "application/octet-stream", data: new Uint8Array([1, 2, 3]) }]);
    }
  });

  it("sends only the viewed bytes of a larger buffer", () => {
    const { save, sent } = setup({ type: "saveResult", saved: true });
    const big = new Uint8Array([9, 9, 1, 2, 9]);
    save(big.subarray(2, 4), "a.bin", "x");
    const data = (sent[0] as { data: Uint8Array }).data;
    expect(Array.from(data)).toEqual([1, 2]);
    expect(data.buffer.byteLength).toBe(2);
  });

  it("notifies on a save error", async () => {
    const { save, sent } = setup({ type: "saveResult", saved: false, error: "EACCES" });
    save("text", "recipe.sh", "text/x-sh");
    await vi.waitFor(() => expect(sent).toHaveLength(2));
    expect(sent[1]).toEqual({ type: "notify", level: "error", message: "Couldn't save recipe.sh: EACCES" });
  });
});

describe("model sink", () => {
  const api = () => ({ openFile: vi.fn<AppApi["openFile"]>() });

  it("buffers the latest model until onReady, then opens it once", () => {
    const sink = createModelSink();
    sink.push("old.onnx", new Uint8Array([1]));
    sink.push("new.onnx", new Uint8Array([2, 3]));
    const a = api();
    sink.setApi(a);
    sink.setApi(a); // onReady can fire again — must not re-open
    expect(a.openFile).toHaveBeenCalledTimes(1);
    const f = a.openFile.mock.calls[0][0];
    expect(f.name).toBe("new.onnx");
    expect(f.size).toBe(2);
  });

  it("opens every later model (incl. reloads) immediately", () => {
    const sink = createModelSink();
    const a = api();
    sink.setApi(a);
    sink.push("m.onnx", new Uint8Array([1]));
    sink.push("m.onnx", new Uint8Array([1, 2]));
    expect(a.openFile.mock.calls.map(([f]) => f.size)).toEqual([1, 2]);
  });
});

describe("worker shim", () => {
  it("parses the build-injected asset list", () => {
    expect(workerAssets()).toEqual([]); // placeholder not replaced outside a build
    expect(workerAssets('["assets/parse.worker-abc.js"]')).toEqual(["assets/parse.worker-abc.js"]);
    expect(workerAssets('{"a":1}')).toEqual([]);
  });

  it("prefetches scripts into blob URLs and skips failures", async () => {
    const fetchImpl: FetchLike = async (input) => {
      const u = String(input);
      if (u.endsWith("ok.js")) return new Response("self.onmessage=()=>{}");
      if (u.endsWith("404.js")) return new Response("", { status: 404 });
      throw new TypeError("network");
    };
    const scripts = await prefetchWorkerScripts(
      ["https://cdn/m/ok.js", "https://cdn/m/404.js", "https://cdn/m/down.js"],
      fetchImpl,
      (code) => `blob:test/${code.length}`,
    );
    expect([...scripts]).toEqual([["https://cdn/m/ok.js", "blob:test/21"]]);
  });

  it("times out a hung prefetch", async () => {
    const hang: FetchLike = (_i, init) =>
      new Promise((_r, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))));
    const scripts = await prefetchWorkerScripts(["https://cdn/x.js"], hang, () => "blob:x", 10);
    expect(scripts.size).toBe(0);
  });

  it("swaps prefetched URLs for blobs and leaves others alone", () => {
    const scripts = new Map([["https://cdn/media/assets/w.js", "blob:doc/1"]]);
    const base = "https://cdn/media/webview.js";
    expect(resolveWorkerUrl(new URL("./assets/w.js", base), scripts, "vscode-webview://id/index.html")).toBe("blob:doc/1");
    expect(resolveWorkerUrl("assets/w.js", scripts, base)).toBe("blob:doc/1");
    expect(resolveWorkerUrl("https://cdn/other.js", scripts, base)).toBe("https://cdn/other.js");

    const seen: [string | URL, WorkerOptions | undefined][] = [];
    class FakeWorker {
      constructor(url: string | URL, opts?: WorkerOptions) {
        seen.push([url, opts]);
      }
    }
    const W = createWorkerClass(FakeWorker as unknown as typeof Worker, scripts, base);
    const w = new W(new URL("https://cdn/media/assets/w.js"), { type: "module" });
    expect(w).toBeInstanceOf(FakeWorker);
    expect(seen).toEqual([["blob:doc/1", { type: "module" }]]);
  });
});
