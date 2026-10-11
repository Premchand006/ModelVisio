import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runChatProxy } from "@modelvisio/ai/proxy";
import { runGrokChat } from "@modelvisio/ai/grok";
import { chatProviderFromEnv } from "@modelvisio/ai/providers";
import { chatWithWebResearch, freeSearchApplies, parseFreeSearchMode } from "@modelvisio/ai/search";
import { scrapeOptionsFromEnv, scrapeUrl } from "@modelvisio/ai/scrape";
import { logChat, logClientEvent, type ClientEvent } from "@modelvisio/ai/log";

/** Best-effort client IP from the platform's forwarding headers. */
function clientIp(req: VercelRequest): string | null {
  const xff = req.headers["x-forwarded-for"];
  const raw = Array.isArray(xff) ? xff[0] : xff;
  return raw ? raw.split(",")[0].trim() : null;
}

// Server-side proxy for the AI copilot. Holds the provider key (GEMINI_API_KEY,
// or XAI_API_KEY for Grok — see packages/ai/src/providers.ts) in the Vercel
// project env so it NEVER reaches the browser. Gemini runs with Google Search
// grounding; Grok (or Gemini with grounding off) gets the free web crawler
// instead (MODELVISIO_FREE_SEARCH=auto|always|off, default auto).
//
// Free Gemini key: https://aistudio.google.com/apikey . Disable Google grounding
// via MODELVISIO_WEB_SEARCH=off.
export const config = { maxDuration: 60 };

const PROVIDER = chatProviderFromEnv(process.env);
const WEB_SEARCH = (process.env.MODELVISIO_WEB_SEARCH || "on") !== "off";
const FREE_SEARCH = parseFreeSearchMode(process.env.MODELVISIO_FREE_SEARCH, "auto");
const SEARXNG_URL = process.env.MODELVISIO_SEARXNG_URL || undefined;
const THINKING = (process.env.MODELVISIO_THINKING || "off") === "on";
// Vercel allows maxDuration (60s); keep a budget just under it so the proxy
// returns a clean error before the platform kills the function.
const TIME_BUDGET_MS = Number(process.env.MODELVISIO_TIMEOUT_MS) || 55000;
const SCRAPE = scrapeOptionsFromEnv(process.env);

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }
  const { provider, key, model, missingKeyError } = PROVIDER;
  if (!key) {
    res.status(500).json({ error: missingKeyError });
    return;
  }
  try {
    const body = (req.body ?? {}) as { system?: string; messages?: unknown[]; event?: ClientEvent };
    // Client telemetry beacon (e.g. a model upload): record it and return.
    if (body.event) {
      await logClientEvent(body.event, { ip: clientIp(req) });
      res.status(200).json({ ok: true });
      return;
    }
    const system = body.system ?? "";
    const messages = body.messages ?? [];
    // Log concurrently with the model call so it adds ~no latency; never rejects.
    const logP = logChat(system, messages, { ip: clientIp(req), user_agent: req.headers["user-agent"] ?? null });
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
    res.status(200).json(out);
  } catch (e) {
    res.status(500).json({ error: e instanceof Error ? e.message : "Proxy error" });
  }
}
