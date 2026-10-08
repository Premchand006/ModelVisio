import { describe, expect, it, vi } from "vitest";
import type { WebviewToHost } from "../src/protocol";
import { createRpc } from "../webview/rpc";

function setup() {
  const sent: WebviewToHost[] = [];
  const rpc = createRpc((m) => sent.push(m));
  return { rpc, sent };
}

describe("rpc", () => {
  it("assigns fresh ids and correlates out-of-order replies", async () => {
    const { rpc, sent } = setup();
    const a = rpc.request({ type: "scrape", url: "https://a.example" });
    const b = rpc.request({ type: "scrape", url: "https://b.example" });
    const [ma, mb] = sent as Extract<WebviewToHost, { type: "scrape" }>[];
    expect(ma.id).not.toBe(mb.id);
    expect(ma.url).toBe("https://a.example");

    rpc.handle({ type: "scrapeError", id: mb.id, error: "nope", status: 404 });
    rpc.handle({ type: "scrapeError", id: ma.id, error: "gone", status: 410 });
    await expect(b).resolves.toMatchObject({ error: "nope", status: 404 });
    await expect(a).resolves.toMatchObject({ error: "gone", status: 410 });
    expect(rpc.pending()).toBe(0);
  });

  it("seeds ids randomly so a reloaded page doesn't reuse the previous page's ids", () => {
    const spy = vi.spyOn(Math, "random").mockReturnValueOnce(0.25).mockReturnValueOnce(0.75);
    try {
      const before = setup();
      const after = setup(); // the same WebView after a reload
      void before.rpc.request({ type: "scrape", url: "https://x" });
      void after.rpc.request({ type: "scrape", url: "https://x" });
      expect((before.sent[0] as { id: number }).id).toBe(250_000_001);
      expect((after.sent[0] as { id: number }).id).toBe(750_000_001);
    } finally {
      spy.mockRestore();
    }
  });

  it("only consumes reply messages", () => {
    const { rpc } = setup();
    expect(rpc.handle({ type: "theme", theme: "dark" })).toBe(false);
    expect(rpc.handle({ type: "chatResult" })).toBe(false); // no id
    expect(rpc.handle(null)).toBe(false);
    expect(rpc.handle("chatResult")).toBe(false);
    // Unknown id: consumed (it's a reply) but resolves nothing.
    expect(rpc.handle({ type: "chatResult", id: 999, text: "", sources: [] })).toBe(true);
  });

  it("rejects with AbortError without posting when already aborted", async () => {
    const { rpc, sent } = setup();
    const ctl = new AbortController();
    ctl.abort();
    await expect(rpc.request({ type: "scrape", url: "https://x" }, ctl.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(sent).toHaveLength(0);
    expect(rpc.pending()).toBe(0);
  });

  it("drops the pending entry on abort and ignores the late reply", async () => {
    const { rpc, sent } = setup();
    const ctl = new AbortController();
    const p = rpc.request({ type: "scrape", url: "https://x" }, ctl.signal);
    expect(rpc.pending()).toBe(1);
    ctl.abort();
    await expect(p).rejects.toMatchObject({ name: "AbortError" });
    expect(rpc.pending()).toBe(0);
    const id = (sent[0] as { id: number }).id;
    expect(rpc.handle({ type: "scrapeError", id, error: "late", status: 500 })).toBe(true);
  });

  it("rejects and cleans up when posting throws", async () => {
    const rpc = createRpc(() => {
      throw new Error("could not clone");
    });
    await expect(rpc.request({ type: "save", filename: "a.bin", mime: "x", data: "hi" })).rejects.toThrow("could not clone");
    expect(rpc.pending()).toBe(0);
  });
});
