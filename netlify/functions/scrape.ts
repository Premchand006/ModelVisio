import { handleScrapeRequest } from "@modelvisio/ai/scrape";

// Netlify proxy for the AI copilot's URL scraper. Reached at /api/scrape via
// the rewrite in netlify.toml (this function lives at /.netlify/functions/scrape).
//
// Legacy v1 `handler` export on purpose — Netlify's build here invokes functions
// in v1 mode (looks for `.handler`); a v2-style `export default` returns an
// opaque 502 on every call. Same reasoning as netlify/functions/chat.ts.
//
// Netlify free tier kills synchronous functions at 10s — the scraper's default
// 8s timeout keeps it inside that budget with margin for cold start. Env knobs
// (MODELVISIO_SCRAPE_*) are documented in apps/web/api/scrape.ts.

type NetlifyEvent = { httpMethod: string; body: string | null };

const json = (statusCode: number, body: unknown) => ({
  statusCode,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const handler = async (event: NetlifyEvent) => {
  if (event.httpMethod !== "POST") return json(405, { error: "Method not allowed" });
  let body: unknown;
  try {
    body = JSON.parse(event.body || "{}");
  } catch {
    return json(400, { error: "Request body must be JSON." });
  }
  const out = await handleScrapeRequest(body, process.env);
  return json(out.status, out.body);
};
