// Message contract between the extension host (src/) and the WebView
// (webview/). Type-only: both sides import it with `import type`, so nothing
// here may emit runtime code.
//
// Binary payloads travel as Uint8Array — VS Code's postMessage transfers typed
// arrays natively (no base64 round-trip), which matters for 100MB+ models.

import type { ThemeName } from "@modelvisio/core";

export type Source = { title: string; url: string };

/** Mirrors @modelvisio/ai's ScrapedPage (kept local so the webview bundle
 *  never pulls in the server-side scraper). */
export type ScrapedPage = {
  url: string;
  finalUrl: string;
  title: string;
  text: string;
  truncated: boolean;
  bytes: number;
  contentType: string;
};

// ─── WebView → host ──────────────────────────────────────────────────────────

export type WebviewToHost =
  /** App mounted and listening; host replies with `theme` then `model`. */
  | { type: "ready" }
  /** One copilot turn (Anthropic-style messages). Reply: chatResult | chatError. */
  | { type: "chat"; id: number; system: string; messages: { role: "user" | "assistant"; content: string }[] }
  /** Read a URL for the copilot via the SSRF-guarded scraper. Reply: scrapeResult | scrapeError. */
  | { type: "scrape"; id: number; url: string }
  /** Save a file the app produced (export, converted model, script). Reply: saveResult. */
  | { type: "save"; id: number; filename: string; mime: string; data: Uint8Array | string }
  /** Surface a message in VS Code (notification + output channel). */
  | { type: "notify"; level: "info" | "warn" | "error"; message: string };

// ─── host → WebView ──────────────────────────────────────────────────────────

export type HostToWebview =
  | { type: "theme"; theme: ThemeName }
  /** Model bytes to open. Re-sent whenever the file changes on disk. */
  | { type: "model"; name: string; bytes: Uint8Array; reload?: boolean }
  /** The host could not read the file (too large, deleted, permission). */
  | { type: "modelError"; name: string; error: string }
  | { type: "chatResult"; id: number; text: string; sources: Source[] }
  | { type: "chatError"; id: number; error: string }
  | { type: "scrapeResult"; id: number; page: ScrapedPage }
  | { type: "scrapeError"; id: number; error: string; status: number }
  /** saved=false with no error means the user cancelled the dialog. */
  | { type: "saveResult"; id: number; saved: boolean; path?: string; error?: string };
