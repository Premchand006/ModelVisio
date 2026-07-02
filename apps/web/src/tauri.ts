// Optional desktop (Tauri) integration. The web build runs unchanged in a
// browser; when hosted inside the Tauri WebView, `withGlobalTauri` exposes the
// runtime on `window.__TAURI__`, so we (a) wire the native "File → Open Model"
// menu to the same openFile handler the browser UI uses, and (b) route the AI
// copilot's /api/chat call through a Rust command — passing the user's own
// Gemini key (bring-your-own-key, entered in the AI settings and stored locally;
// no key ships in the app).

import { GEMINI_KEY_STORAGE } from "@modelvisio/core";

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
export async function wireTauriMenu(openFile: (f: File) => void): Promise<void> {
  const t = tauri();
  if (!t?.event?.listen) return;
  await t.event.listen("model-opened", (e) => {
    const { name, bytes } = (e.payload || {}) as { name?: string; bytes?: number[] };
    if (!name || !bytes) return;
    openFile(new File([new Uint8Array(bytes)], name));
  });
}

type ChatReply = { text?: string; sources?: { title: string; url: string }[]; error?: string };

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
      // have no server to receive them in the desktop shell — ack and drop them.
      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        return Promise.resolve(new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "content-type": "application/json" } }));
      }
      // Bring-your-own-key: read the user's locally-stored Gemini key and hand it
      // to the Rust command (which never stores or bundles a key of its own).
      let apiKey = "";
      try { apiKey = localStorage.getItem(GEMINI_KEY_STORAGE) ?? ""; } catch { /* ignore */ }
      return invoke("chat", { apiKey, system: body.system ?? "", messages: body.messages })
        .then((r) => new Response(JSON.stringify(r as ChatReply), { status: 200, headers: { "content-type": "application/json" } }))
        .catch((e) => new Response(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }), { status: 500, headers: { "content-type": "application/json" } }));
    }
    return realFetch(input, init);
  };
}
