import { describe, expect, it, vi } from "vitest";
import type { HostToWebview, WebviewToHost } from "../src/protocol";
import { createFetchBridge, routeOf, type FetchLike } from "../webview/fetchBridge";
import { createRpc } from "../webview/rpc";

type Reply = Extract<HostToWebview, { id: number }>;

/** A bridge wired to a fake host. `answer` builds the host's reply (or undefined to stay silent). */
function setup(answer: (m: WebviewToHost & { id: number }) => Omit<Reply, "id"> | undefined = () => undefined) {
  const sent: WebviewToHost[] = [];
  const rpc = createRpc((m) => {
    sent.push(m);
    const r = "id" in m ? answer(m) : undefined;
    if (r) queueMicrotask(() => rpc.handle({ ...r, id: (m as { id: number }).id }));
  });
  const realFetch = vi.fn<FetchLike>(async () => new Response("real"));
  return { fetch: createFetchBridge(rpc, realFetch), sent, realFetch, rpc };
}

const post = (body: unknown, extra: RequestInit = {}): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
  ...extra,
});

describe("routeOf", () => {
  it("bridges only POSTs to …/api/chat and …/api/scrape", () => {
    expect(routeOf("/api/chat", { method: "post" })).toBe("chat");
    expect(routeOf("/api/scrape?x=1", { method: "POST" })).toBe("scrape");
    expect(routeOf(new URL("vscode-webview://abc/api/chat"), { method: "POST" })).toBe("chat");
    expect(routeOf(new Request("http://h/api/scrape", { method: "POST", body: "{}" }))).toBe("scrape");
    expect(routeOf("/api/chat")).toBeNull(); // GET
    expect(routeOf("/api/chatty", { method: "POST" })).toBeNull();
    expect(routeOf("/api/chat/x", { method: "POST" })).toBeNull();
  });
});

describe("fetch bridge", () => {
  it("passes everything else to the real fetch", async () => {
    const { fetch, realFetch, sent } = setup();
    const res = await fetch("https://example.com/data.json");
    expect(await res.text()).toBe("real");
    await fetch("/api/chat"); // GET
    expect(realFetch).toHaveBeenCalledTimes(2);
    expect(sent).toHaveLength(0);
  });

  it("acks usage beacons locally without messaging the host", async () => {
    const { fetch, sent, realFetch } = setup();
    for (const body of [{ event: { kind: "upload" } }, { messages: [] }, "not json"]) {
      const res = await fetch("/api/chat", { method: "POST", body: typeof body === "string" ? body : JSON.stringify(body) });
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: true });
    }
    expect(sent).toHaveLength(0);
    expect(realFetch).not.toHaveBeenCalled();
  });

  it("round-trips a chat turn", async () => {
    const { fetch, sent } = setup(() => ({ type: "chatResult", text: "hi there", sources: [{ title: "T", url: "https://t" }] }));
    const messages = [{ role: "user", content: "hello" }];
    const res = await fetch("/api/chat", post({ system: "SYS", messages }));
    expect(sent).toEqual([{ type: "chat", id: expect.any(Number), system: "SYS", messages }]);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ text: "hi there", sources: [{ title: "T", url: "https://t" }] });
  });

  it("maps chatError to 500 {error}", async () => {
    const { fetch } = setup(() => ({ type: "chatError", error: "No API key" }));
    const res = await fetch("/api/chat", post({ messages: [{ role: "user", content: "x" }] }));
    expect(res.status).toBe(500);
    expect(await res.json()).toEqual({ error: "No API key" });
  });

  it("reads the body from a Request when init has none", async () => {
    const { fetch, sent } = setup(() => ({ type: "chatResult", text: "ok", sources: [] }));
    const req = new Request("http://localhost/api/chat", post({ system: "S", messages: [{ role: "user", content: "q" }] }));
    const res = await fetch(req);
    expect(res.status).toBe(200);
    expect(sent[0]).toMatchObject({ type: "chat", system: "S", messages: [{ role: "user", content: "q" }] });
  });

  it("returns scraped page fields at the top level", async () => {
    const page = { url: "https://a", finalUrl: "https://a/", title: "A", text: "body", truncated: false, bytes: 4, contentType: "text/html" };
    const { fetch, sent } = setup(() => ({ type: "scrapeResult", page }));
    const res = await fetch(new URL("http://localhost/api/scrape"), post({ url: " https://a " }));
    expect(sent[0]).toMatchObject({ type: "scrape", url: "https://a" });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual(page);
  });

  it("maps scrapeError to its status (or 502 when the status is unusable)", async () => {
    for (const [status, expected] of [[403, 403], [504, 504], [0, 502], [200, 502]] as const) {
      const { fetch } = setup(() => ({ type: "scrapeError", error: "blocked", status }));
      const res = await fetch("/api/scrape", post({ url: "https://a" }));
      expect(res.status).toBe(expected);
      expect(await res.json()).toEqual({ error: "blocked" });
    }
  });

  it("rejects a scrape without a url locally", async () => {
    const { fetch, sent } = setup();
    const res = await fetch("/api/scrape", post({}));
    expect(res.status).toBe(400);
    expect(sent).toHaveLength(0);
  });

  it("rejects with AbortError when aborted before the call", async () => {
    const { fetch, sent } = setup(() => ({ type: "chatResult", text: "x", sources: [] }));
    const ctl = new AbortController();
    ctl.abort();
    await expect(fetch("/api/chat", post({ messages: [{ role: "user", content: "x" }] }, { signal: ctl.signal }))).rejects.toMatchObject({ name: "AbortError" });
    expect(sent).toHaveLength(0);
  });

  it("rejects with AbortError and drops the pending entry when aborted in flight", async () => {
    const { fetch, sent, rpc } = setup(); // host never answers
    const ctl = new AbortController();
    const p = fetch("/api/scrape", post({ url: "https://slow" }, { signal: ctl.signal }));
    await vi.waitFor(() => expect(sent).toHaveLength(1));
    expect(rpc.pending()).toBe(1);
    ctl.abort();
    const err = await p.catch((e: unknown) => e);
    expect(err).toBeInstanceOf(DOMException);
    expect((err as DOMException).name).toBe("AbortError");
    expect(rpc.pending()).toBe(0);
  });

  it("honours a Request's own signal", async () => {
    const { fetch, rpc } = setup();
    const ctl = new AbortController();
    const p = fetch(new Request("http://localhost/api/chat", post({ messages: [{ role: "user", content: "x" }] }, { signal: ctl.signal })));
    await vi.waitFor(() => expect(rpc.pending()).toBe(1));
    ctl.abort();
    await expect(p).rejects.toMatchObject({ name: "AbortError" });
    expect(rpc.pending()).toBe(0);
  });
});
