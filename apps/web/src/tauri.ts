// Optional desktop (Tauri) integration. The web build runs unchanged in a
// browser; when hosted inside the Tauri WebView, `withGlobalTauri` exposes the
// runtime on `window.__TAURI__`, so we (a) wire the native "File → Open Model"
// menu (and File → Open Recent) to the same openFile handler the browser UI
// uses, (b) route the AI copilot's /api/chat call through a Rust command,
// passing the selected provider (Gemini or Grok) and the user's own key for it
// (bring-your-own-key, entered in the AI settings and stored locally; no key
// ships in the app), and (c) route the copilot's /api/scrape call (reading
// URLs pasted into the chat) to the Rust port of the SSRF-guarded scraper.

import { getAiProvider, getUserApiKey } from "@modelvisio/core";

type InvokeFn = (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
type GlobalTauri = {
  event?: {
    listen: (event: string, cb: (e: { payload: unknown }) => void) => Promise<() => void>;
  };
  core?: { invoke: InvokeFn };
  invoke?: InvokeFn;
};

function tauri(): GlobalTauri | null {
  if (typeof window === "undefined") return null;
  return (window as unknown as { __TAURI__?: GlobalTauri }).__TAURI__ ?? null;
}

export function isTauri(): boolean {
  return tauri() !== null;
}

function invoke(cmd: string, args?: Record<string, unknown>): Promise<unknown> {
  const t = tauri();
  const fn = t?.core?.invoke ?? t?.invoke;
  if (!fn) return Promise.reject(new Error("Tauri invoke is unavailable"));
  return fn(cmd, args);
}

/**
 * Listen for the native menu's "model-opened" event (payload: file name + bytes
 * read on the Rust side) and feed it into the app as a File.
 */
export async function wireTauriMenu(openFile: (f: File, companions?: File[]) => void): Promise<void> {
  const t = tauri();
  if (!t?.event?.listen) return;
  await t.event.listen("model-opened", (e) => {
    const { name, bytes, companions } = (e.payload || {}) as {
      name?: string;
      bytes?: number[];
      // Sibling weight files (OpenVINO .bin, Paddle .pdiparams, ...) read next to the model.
      companions?: { name: string; bytes: number[] }[];
    };
    if (!name || !bytes) return;
    const extra = (companions ?? []).map((c) => new File([new Uint8Array(c.bytes)], c.name));
    openFile(new File([new Uint8Array(bytes)], name), extra.length ? extra : undefined);
  });
}

/** One recently opened model, as listed by File → Open Recent. */
export type RecentModel = { path: string; name: string };

/**
 * The recently opened models (most recent first), recorded by the Rust side
 * when a model is opened via File → Open Model or Open Recent. Files dropped on
 * the in-window drop zone have no path and are never listed. Empty outside Tauri.
 */
export async function getRecentModels(): Promise<RecentModel[]> {
  if (!isTauri()) return [];
  const r = await invoke("recent_models");
  return Array.isArray(r) ? (r as RecentModel[]) : [];
}

/**
 * Open a model from the recent list. The Rust side reads it and emits the same
 * "model-opened" event as the menu, so `wireTauriMenu`'s handler opens it.
 * Rejects with a readable message if the file is gone (it is then dropped from
 * the list) or the path isn't in the list.
 */
export async function openRecentModel(path: string): Promise<void> {
  await invoke("open_recent", { path });
}

/** Empty the recent list (same as File → Open Recent → Clear Recently Opened). */
export async function clearRecentModels(): Promise<void> {
  if (isTauri()) await invoke("clear_recent_models");
}

/**
 * Call `cb` with the new list whenever it changes (a model opened, a missing
 * file dropped, the list cleared). Returns an unsubscribe function.
 */
export async function onRecentModelsChanged(cb: (list: RecentModel[]) => void): Promise<() => void> {
  const t = tauri();
  if (!t?.event?.listen) return () => {};
  return t.event.listen("recent-models-changed", (e) => {
    if (Array.isArray(e.payload)) cb(e.payload as RecentModel[]);
  });
}

type ChatReply = { text?: string; sources?: { title: string; url: string }[]; error?: string };

const json = (body: unknown, status: number) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

/**
 * Answer POST /api/scrape {url} from the Rust `scrape` command. Same contract as
 * the web proxy (packages/ai/src/scrape.ts): 200 + the page, or {error} with
 * 400 (bad/blocked URL) / 502 (origin failure), so fetchScraped() and
 * scrapeMessageUrls() in @modelvisio/ai work unchanged.
 */
function scrapeViaRust(init: RequestInit | undefined): Promise<Response> {
  let body: { url?: unknown } = {};
  try { body = JSON.parse((init?.body as string) || "{}"); } catch {
    return Promise.resolve(json({ error: "Request body must be JSON." }, 400));
  }
  if (typeof body.url !== "string" || !body.url) {
    return Promise.resolve(json({ error: "Missing `url` (string) in request body." }, 400));
  }
  const reply = invoke("scrape", { url: body.url })
    .then((page) => json(page, 200))
    .catch((e: unknown) => {
      const err = e as { error?: unknown; status?: unknown } | null;
      if (err && typeof err.error === "string") {
        return json({ error: err.error }, typeof err.status === "number" ? err.status : 500);
      }
      return json({ error: e instanceof Error ? e.message : String(e) }, 500);
    });
  // The Rust call can't be cancelled, but honour the caller's AbortSignal so a
  // stopped chat turn doesn't wait for it (the Rust side has its own deadline).
  const signal = init?.signal;
  if (!signal) return reply;
  if (signal.aborted) return Promise.reject(new DOMException("Aborted", "AbortError"));
  return new Promise<Response>((resolve, reject) => {
    const onAbort = () => reject(new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
    reply.then(
      (r) => { signal.removeEventListener("abort", onAbort); resolve(r); },
      (e) => { signal.removeEventListener("abort", onAbort); reject(e); },
    );
  });
}

/**
 * Intercept the copilot's POST /api/chat (which has no server in the desktop
 * shell) and forward it to the Rust `chat` command. Installed once, only under
 * Tauri; everything else uses the real fetch. Keeps core's Chat/sendChat as-is.
 */
export function installTauriChatProxy(): void {
  if (!isTauri() || typeof window === "undefined") return;
  const realFetch = window.fetch.bind(window);
  window.fetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const method = (init?.method || (typeof input === "object" && "method" in input ? input.method : "GET") || "GET").toUpperCase();
    if (url && /\/api\/chat$/.test(url) && method === "POST") {
      let body: { system?: string; messages?: unknown[] } = {};
      try { body = JSON.parse((init?.body as string) || "{}"); } catch { /* ignore */ }
      // Usage/upload beacons (logUpload) also POST here but carry no messages and
      // have no server to receive them in the desktop shell, so ack and drop them.
      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        return Promise.resolve(new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "content-type": "application/json" } }));
      }
      // Bring-your-own-key: read the selected provider (default Gemini) and the
      // user's locally-stored key for it, and hand both to the Rust command
      // (which never stores or bundles a key of its own). Neither helper throws.
      const provider = getAiProvider();
      const apiKey = getUserApiKey(provider);
      return invoke("chat", { provider, apiKey, system: body.system ?? "", messages: body.messages })
        .then((r) => new Response(JSON.stringify(r as ChatReply), { status: 200, headers: { "content-type": "application/json" } }))
        .catch((e) => new Response(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }), { status: 500, headers: { "content-type": "application/json" } }));
    }
    // URLs pasted into the chat: read by the Rust port of the scraper.
    if (url && /\/api\/scrape$/.test(url) && method === "POST") {
      return scrapeViaRust(init);
    }
    return realFetch(input, init);
  };
}
