import type { VercelRequest, VercelResponse } from "@vercel/node";
import { scrapeUrl } from "@modelvisio/ai/scrape";

// Server-side scrape proxy for the AI copilot. Runs the SSRF-guarded, size-
// capped scraper (packages/ai/src/scrape.ts) so the browser never fetches
// arbitrary cross-origin URLs itself. Companion to /api/chat.
//
// Env knobs (all optional):
//   MODELVISIO_SCRAPE_ALLOWLIST  comma-separated host suffixes (e.g.
//                                 "docs.nvidia.com,hailo.ai"); empty = allow all
//   MODELVISIO_SCRAPE_MAX_BYTES  cap on bytes read (default 524288)
//   MODELVISIO_SCRAPE_MAX_CHARS  cap on chars returned (default 12000)
//   MODELVISIO_SCRAPE_TIMEOUT_MS wall-clock cap in ms  (default 8000)

export const config = { maxDuration: 15 };

const ALLOWLIST = (process.env.MODELVISIO_SCRAPE_ALLOWLIST || "")
  .split(",").map((s) => s.trim()).filter(Boolean);
const MAX_BYTES = Number(process.env.MODELVISIO_SCRAPE_MAX_BYTES) || 512 * 1024;
const MAX_CHARS = Number(process.env.MODELVISIO_SCRAPE_MAX_CHARS) || 12_000;
const TIMEOUT_MS = Number(process.env.MODELVISIO_SCRAPE_TIMEOUT_MS) || 8000;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }
  try {
    const body = (req.body ?? {}) as { url?: string };
    if (!body.url || typeof body.url !== "string") {
      res.status(400).json({ error: "Missing `url` (string) in request body." });
      return;
    }
    const out = await scrapeUrl({
      url: body.url,
      allowlist: ALLOWLIST.length ? ALLOWLIST : undefined,
      maxBytes: MAX_BYTES,
      maxChars: MAX_CHARS,
      timeoutMs: TIMEOUT_MS,
    });
    res.status(200).json(out);
  } catch (e) {
    // 400: user-caused (bad URL, blocked host, wrong content-type).
    // 502: origin failure / timeout.
    const msg = e instanceof Error ? e.message : "Scrape error";
    const status = /timed out|fetch failed|HTTP 5\d\d/i.test(msg) ? 502 : 400;
    res.status(status).json({ error: msg });
  }
}
