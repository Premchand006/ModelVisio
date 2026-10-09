import { runChatProxy } from "@modelvisio/ai/proxy";
import { runGrokChat } from "@modelvisio/ai/grok";
import { chatProviderFromEnv } from "@modelvisio/ai/providers";
import { chatWithWebResearch, freeSearchApplies, parseFreeSearchMode } from "@modelvisio/ai/search";
import { scrapeOptionsFromEnv, scrapeUrl } from "@modelvisio/ai/scrape";
import { logChat, logClientEvent, type ClientEvent } from "@modelvisio/ai/log";

// Netlify proxy for the AI copilot. Holds the provider key (GEMINI_API_KEY, or
// XAI_API_KEY for Grok — see packages/ai/src/providers.ts) server-side (set it
// in the Netlify site env) so it NEVER reaches the browser. Reached at /api/chat
// via the rewrite in netlify.toml (this function lives at /.netlify/functions/chat).
//
// Uses the LEGACY v1 `handler` export on purpose: Netlify's build here invokes
// functions in v1 mode (it looks for `.handler`), so a v2-style `export default`
// fails at runtime with "handler is not a function" → an opaque 502 on every
// call. The v1 handler is detected reliably on every plan/build image.
//
// Netlify synchronous functions are killed at a HARD 10s on the free tier (not
// raisable) — and that clock includes cold start. So we keep a conservative time
// budget AND default BOTH kinds of web search OFF: Google-Search grounding and
// the free crawler (search + page reads) are the slow paths and, with a cold
// start, the main cause of free-tier 502s. A plain reply returns in ~1s.
// Re-enable with MODELVISIO_WEB_SEARCH=on (Gemini) or MODELVISIO_FREE_SEARCH=auto
// (Grok / ungrounded Gemini) — best on a paid plan + a higher
// MODELVISIO_TIMEOUT_MS. Vercel's 60s shell keeps both on.
const PROVIDER = chatProviderFromEnv(process.env);
const WEB_SEARCH = (process.env.MODELVISIO_WEB_SEARCH || "off") !== "off";
const FREE_SEARCH = parseFreeSearchMode(process.env.MODELVISIO_FREE_SEARCH, "off");
const SEARXNG_URL = process.env.MODELVISIO_SEARXNG_URL || undefined;
const THINKING = (process.env.MODELVISIO_THINKING || "off") === "on";
const TIME_BUDGET_MS = Number(process.env.MODELVISIO_TIMEOUT_MS) || 7000;
const SCRAPE = scrapeOptionsFromEnv(process.env);

type NetlifyEvent = { httpMethod: string; body: string | null; headers?: Record<string, string | undefined> };

/** Best-effort client IP from the platform's forwarding headers. */
function clientIp(event: NetlifyEvent): string | null {
  const h = event.headers || {};
  const raw = h["x-nf-client-connection-ip"] || h["x-forwarded-for"];
  return raw ? raw.split(",")[0].trim() : null;
}

const json = (statusCode: number, body: unknown) => ({
  statusCode,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const handler = async (event: NetlifyEvent) => {
  if (event.httpMethod !== "POST") return json(405, { error: "Method not allowed" });
  const { provider, key, model, missingKeyError } = PROVIDER;
  if (!key) return json(500, { error: missingKeyError });
  try {
    const body = JSON.parse(event.body || "{}") as { system?: string; messages?: unknown[]; event?: ClientEvent };
    // Client telemetry beacon (e.g. a model upload): record it and return.
    if (body.event) {
      await logClientEvent(body.event, { ip: clientIp(event) });
      return json(200, { ok: true });
    }
    const system = body.system ?? "";
    const messages = body.messages ?? [];
    // Log concurrently with the model call so it adds ~no latency; never rejects.
    const logP = logChat(system, messages, { ip: clientIp(event), user_agent: event.headers?.["user-agent"] ?? null });
    const run = (sys: string, timeBudgetMs: number) =>
      provider === "grok"
        ? runGrokChat({ key, system: sys, messages, model, thinking: THINKING, timeBudgetMs })
        : runChatProxy({ key, system: sys, messages, model, webSearch: WEB_SEARCH, thinking: THINKING, timeBudgetMs });
    const out = freeSearchApplies(provider, WEB_SEARCH, FREE_SEARCH)
      ? await chatWithWebResearch({
          system, messages, mode: FREE_SEARCH, searxngUrl: SEARXNG_URL, timeBudgetMs: TIME_BUDGET_MS,
          read: (url, limits) => scrapeUrl({ url, ...SCRAPE, ...limits }),
          onError: (m) => console.warn(m),
        }, run)
      : await run(system, TIME_BUDGET_MS);
    await logP;
    return json(200, out);
  } catch (e) {
    return json(500, { error: e instanceof Error ? e.message : "Proxy error" });
  }
};
