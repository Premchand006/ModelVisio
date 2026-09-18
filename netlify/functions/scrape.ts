import { scrapeUrl } from "@modelvisio/ai/scrape";

// Netlify proxy for the AI copilot's URL scraper. Reached at /api/scrape via
// the rewrite in netlify.toml (this function lives at /.netlify/functions/scrape).
//
// Legacy v1 `handler` export on purpose — Netlify's build here invokes functions
// in v1 mode (looks for `.handler`); a v2-style `export default` returns an
// opaque 502 on every call. Same reasoning as netlify/functions/chat.ts.
//
// Netlify free tier kills synchronous functions at 10s — the scraper's default
// 8s timeout keeps it inside that budget with margin for cold start.

const ALLOWLIST = (process.env.MODELVISIO_SCRAPE_ALLOWLIST || "")
  .split(",").map((s) => s.trim()).filter(Boolean);
const MAX_BYTES = Number(process.env.MODELVISIO_SCRAPE_MAX_BYTES) || 512 * 1024;
const MAX_CHARS = Number(process.env.MODELVISIO_SCRAPE_MAX_CHARS) || 12_000;
const TIMEOUT_MS = Number(process.env.MODELVISIO_SCRAPE_TIMEOUT_MS) || 8000;

type NetlifyEvent = { httpMethod: string; body: string | null };

const json = (statusCode: number, body: unknown) => ({
  statusCode,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const handler = async (event: NetlifyEvent) => {
  if (event.httpMethod !== "POST") return json(405, { error: "Method not allowed" });
  try {
    const body = JSON.parse(event.body || "{}") as { url?: string };
    if (!body.url || typeof body.url !== "string") {
      return json(400, { error: "Missing `url` (string) in request body." });
    }
    const out = await scrapeUrl({
      url: body.url,
      allowlist: ALLOWLIST.length ? ALLOWLIST : undefined,
      maxBytes: MAX_BYTES,
      maxChars: MAX_CHARS,
      timeoutMs: TIMEOUT_MS,
    });
    return json(200, out);
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Scrape error";
    const status = /timed out|fetch failed|HTTP 5\d\d/i.test(msg) ? 502 : 400;
    return json(status, { error: msg });
  }
};
