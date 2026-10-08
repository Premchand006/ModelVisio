import type { FetchLike } from "./fetchBridge";

// ─── Why this exists ─────────────────────────────────────────────────────────
// core's parseClient does `new Worker(new URL("./parse.worker.ts", import.meta.url))`.
// In a webview the document lives on vscode-webview://… while our scripts are
// served from https://…vscode-cdn.net, so that constructor throws a
// SecurityError (workers must be same-origin) and core falls back to parsing
// on the main thread, freezing the UI on big models.
//
// ─── Why a prefetched blob, and not `import "<url>"` in a blob ───────────────
// A blob: worker is same-origin with the document and allowed by
// `worker-src blob:`, but it inherits the document's CSP. Our script-src is
// nonce-only and a worker's top-level script carries no nonce, so a blob that
// does `import "<cdn url>"` / `importScripts(url)` is blocked inside the worker.
// Instead we fetch the worker script's TEXT up front (connect-src allows it)
// and build the blob from the code itself: nothing is loaded from inside the
// worker, so script-src never comes into play. This needs the worker chunk to
// be self-contained (no imports) — vite.webview.config.ts checks that at build.
//
// Prefetching (rather than a lazy proxy Worker) keeps `new Worker()`
// synchronous and returns a REAL Worker, so core's onmessage/onerror wiring is
// untouched. If the prefetch failed we pass the original URL through, the
// constructor throws as before, and core keeps its main-thread fallback.
//
// The bundle's worker asset paths are injected at build time (see
// vite.webview.config.ts) in place of this placeholder.
const WORKER_ASSETS = "__MODELVISIO_WORKER_ASSETS__";

/** Worker asset paths (relative to webview.js) baked in at build; [] when unbuilt. */
export function workerAssets(raw: string = WORKER_ASSETS): string[] {
  try {
    const v: unknown = JSON.parse(raw);
    return Array.isArray(v) ? v.filter((p): p is string => typeof p === "string") : [];
  } catch {
    return [];
  }
}

/** Absolute script URL → blob: URL holding the same code. */
export type WorkerScripts = Map<string, string>;

/** Fetch each worker script and wrap it in a same-origin blob URL. Failures are skipped. */
export async function prefetchWorkerScripts(
  hrefs: string[],
  fetchImpl: FetchLike,
  toBlobUrl: (code: string) => string = (code) => URL.createObjectURL(new Blob([code], { type: "text/javascript" })),
  timeoutMs = 5000,
): Promise<WorkerScripts> {
  const out: WorkerScripts = new Map();
  await Promise.all(
    hrefs.map(async (href) => {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), timeoutMs);
      try {
        const res = await fetchImpl(href, { signal: ctl.signal });
        if (res.ok) out.set(href, toBlobUrl(await res.text()));
      } catch {
        // leave it out: the Worker constructor falls through to the original URL
      } finally {
        clearTimeout(timer);
      }
    }),
  );
  return out;
}

/** Map a Worker URL to its prefetched blob, or return it unchanged. */
export function resolveWorkerUrl(url: string | URL, scripts: WorkerScripts, base: string): string | URL {
  try {
    return scripts.get(new URL(String(url), base).href) ?? url;
  } catch {
    return url;
  }
}

/** A Worker subclass that swaps prefetched script URLs for their blob copies. */
export function createWorkerClass(Real: typeof Worker, scripts: WorkerScripts, base: string): typeof Worker {
  return class ShimWorker extends Real {
    constructor(url: string | URL, options?: WorkerOptions) {
      super(resolveWorkerUrl(url, scripts, base), options);
    }
  };
}

/** Install the shim on `win` (no-op when nothing was prefetched). Must run before core creates its worker. */
export function installWorkerShim(win: Window & typeof globalThis, scripts: WorkerScripts): void {
  if (!scripts.size || typeof win.Worker !== "function") return;
  win.Worker = createWorkerClass(win.Worker, scripts, win.document.baseURI);
}
