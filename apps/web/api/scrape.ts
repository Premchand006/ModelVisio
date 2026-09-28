import type { VercelRequest, VercelResponse } from "@vercel/node";
import { handleScrapeRequest } from "@modelvisio/ai/scrape";

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

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }
  const out = await handleScrapeRequest(req.body ?? {}, process.env);
  res.status(out.status).json(out.body);
}
