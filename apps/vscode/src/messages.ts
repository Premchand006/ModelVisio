// Validation of WebView → host messages. Pure (no `vscode` import) so it can be
// unit tested. The WebView runs third-party-influenced content (model files,
// scraped pages, AI output), so the host treats every message as untrusted:
// anything that does not match the protocol exactly is dropped.

import { ScrapeError } from "@modelvisio/ai/scrape";
import type { WebviewToHost } from "./protocol";

type Chat = Extract<WebviewToHost, { type: "chat" }>;
type ChatTurn = Chat["messages"][number];

/** Cap on `notify` text so a runaway message can't flood the notification UI. */
export const MAX_NOTIFY_CHARS = 2000;

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
const isId = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/**
 * Parse an incoming message into the typed protocol, or null if malformed.
 * Normalizes where it is safe to: chat turns with empty/invalid content are
 * dropped (Gemini rejects them anyway), and binary `save` payloads arriving as
 * ArrayBuffer / other typed-array views / number[] become a Uint8Array.
 */
export function parseWebviewMessage(raw: unknown): WebviewToHost | null {
  if (!isObj(raw)) return null;
  switch (raw.type) {
    case "ready":
      return { type: "ready" };
    case "chat": {
      if (!isId(raw.id) || !Array.isArray(raw.messages)) return null;
      const messages: ChatTurn[] = [];
      for (const m of raw.messages) {
        if (!isObj(m) || (m.role !== "user" && m.role !== "assistant")) continue;
        if (typeof m.content !== "string" || m.content.trim() === "") continue;
        messages.push({ role: m.role, content: m.content });
      }
      return { type: "chat", id: raw.id, system: typeof raw.system === "string" ? raw.system : "", messages };
    }
    case "scrape":
      if (!isId(raw.id) || typeof raw.url !== "string") return null;
      return { type: "scrape", id: raw.id, url: raw.url };
    case "save": {
      if (!isId(raw.id) || typeof raw.filename !== "string") return null;
      const data = toSaveData(raw.data);
      if (data === null) return null;
      return { type: "save", id: raw.id, filename: raw.filename, mime: typeof raw.mime === "string" ? raw.mime : "", data };
    }
    case "notify": {
      const level = raw.level;
      if ((level !== "info" && level !== "warn" && level !== "error") || typeof raw.message !== "string") return null;
      const message = raw.message.length > MAX_NOTIFY_CHARS ? `${raw.message.slice(0, MAX_NOTIFY_CHARS)}…` : raw.message;
      return { type: "notify", level, message };
    }
    default:
      return null;
  }
}

/** The request id of a malformed message, if it had one — so the host can
 *  still answer with an error instead of leaving the WebView's promise hanging. */
export function messageId(raw: unknown): { type: string; id: number } | null {
  if (!isObj(raw) || typeof raw.type !== "string" || !isId(raw.id)) return null;
  return { type: raw.type, id: raw.id };
}

function toSaveData(v: unknown): Uint8Array | string | null {
  if (typeof v === "string" || v instanceof Uint8Array) return v;
  if (v instanceof ArrayBuffer) return new Uint8Array(v);
  if (ArrayBuffer.isView(v)) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  if (Array.isArray(v) && v.every((n) => typeof n === "number" && Number.isInteger(n) && n >= 0 && n <= 255)) {
    return Uint8Array.from(v);
  }
  return null;
}

/** Bytes to write for a `save`: strings are UTF-8 encoded. */
export function saveBytes(data: Uint8Array | string): Uint8Array {
  return typeof data === "string" ? new TextEncoder().encode(data) : data;
}

/** Error + status to relay for a failed scrape: ScrapeError carries its own
 *  400 (caller's fault) / 502 (origin's fault); anything else is a 502. The
 *  shared scraper names the server's env var; here the allowlist is a setting. */
export function scrapeFailure(e: unknown): { error: string; status: number } {
  const error = e instanceof Error && e.message ? e.message : "Scrape failed";
  return {
    error: error.replace(/\bMODELVISIO_SCRAPE_ALLOWLIST\b/g, "the modelvisio.scrapeAllowlist setting"),
    status: e instanceof ScrapeError ? e.status : 502,
  };
}
