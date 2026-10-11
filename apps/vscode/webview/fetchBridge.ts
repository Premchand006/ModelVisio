import type { Rpc } from "./rpc";
import { abortError } from "./rpc";

// The copilot POSTs to /api/chat and /api/scrape, which don't exist inside a
// webview. We intercept exactly those two and forward them to the extension
// host (which holds the Gemini key and runs the SSRF-guarded scraper), then
// answer with the same JSON the web proxies return — so core's sendChat /
// fetchScraped run unchanged. Everything else goes to the real fetch.

export type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

type Route = "chat" | "scrape";

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function asRequest(input: RequestInfo | URL): Request | null {
  return typeof input === "object" && !(input instanceof URL) ? input : null;
}

/** Which bridged endpoint (if any) a fetch call targets. Only POSTs are bridged. */
export function routeOf(input: RequestInfo | URL, init?: RequestInit): Route | null {
  const req = asRequest(input);
  const method = (init?.method ?? req?.method ?? "GET").toUpperCase();
  if (method !== "POST") return null;
  const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const path = href.split(/[?#]/, 1)[0];
  const m = /\/api\/(chat|scrape)$/.exec(path);
  return m ? (m[1] as Route) : null;
}

async function readJson(input: RequestInfo | URL, init?: RequestInit): Promise<Record<string, unknown>> {
  const req = asRequest(input);
  try {
    // new Response(body).text() reads any BodyInit (string, Blob, buffer, …).
    const text = init?.body != null ? await new Response(init.body).text() : req ? await req.clone().text() : "";
    const v: unknown = JSON.parse(text || "{}");
    return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

export function createFetchBridge(rpc: Rpc, realFetch: FetchLike): FetchLike {
  return (input, init) => {
    const route = routeOf(input, init);
    if (!route) return realFetch(input, init);
    const signal = init?.signal ?? asRequest(input)?.signal ?? undefined;
    return bridged(route, input, init, signal);
  };

  async function bridged(route: Route, input: RequestInfo | URL, init: RequestInit | undefined, signal?: AbortSignal): Promise<Response> {
    if (signal?.aborted) throw abortError();
    const body = await readJson(input, init);

    if (route === "chat") {
      // Usage beacons (logUpload) share this endpoint but carry `{event}` and no
      // messages. Ack locally: there is no server-side log in the extension.
      const messages = Array.isArray(body.messages) ? body.messages : [];
      if (messages.length === 0) return json(200, { ok: true });
      const system = typeof body.system === "string" ? body.system : "";
      const r = await rpc.request({ type: "chat", system, messages: messages as { role: "user" | "assistant"; content: string }[] }, signal);
      // core's sendChat reads `data.error` on failure.
      return r.type === "chatResult" ? json(200, { text: r.text, sources: r.sources }) : json(500, { error: r.error });
    }

    const url = typeof body.url === "string" ? body.url.trim() : "";
    if (!url) return json(400, { error: "Missing url." });
    const r = await rpc.request({ type: "scrape", url }, signal);
    // core's fetchScraped expects the page fields at the top level.
    if (r.type === "scrapeResult") return json(200, r.page);
    const status = r.status >= 400 && r.status <= 599 ? r.status : 502;
    return json(status, { error: r.error });
  }
}

/** Replace window.fetch with the bridge. Returns the original fetch. */
export function installFetchBridge(win: Window, rpc: Rpc): FetchLike {
  const realFetch: FetchLike = win.fetch.bind(win);
  win.fetch = createFetchBridge(rpc, realFetch);
  return realFetch;
}
