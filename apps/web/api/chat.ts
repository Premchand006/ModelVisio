import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runChatProxy } from "@modelvisio/ai/proxy";

// Server-side proxy for the AI copilot. Holds GEMINI_API_KEY (set in the Vercel
// project env) so it NEVER reaches the browser. Runs Gemini with Google Search
// grounding so the assistant can cite real reference links.
//
// Free key: https://aistudio.google.com/apikey . Disable grounding via
// MODELVISIO_WEB_SEARCH=off.
export const config = { maxDuration: 60 };

const MODEL = process.env.MODELVISIO_MODEL || "gemini-2.5-flash";
const WEB_SEARCH = (process.env.MODELVISIO_WEB_SEARCH || "on") !== "off";
const THINKING = (process.env.MODELVISIO_THINKING || "off") === "on";
// Vercel allows maxDuration (60s); keep a budget just under it so the proxy
// returns a clean error before the platform kills the function.
const TIME_BUDGET_MS = Number(process.env.MODELVISIO_TIMEOUT_MS) || 55000;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) {
    res.status(500).json({ error: "GEMINI_API_KEY is not configured on the server." });
    return;
  }
  try {
    const { system, messages } = (req.body ?? {}) as { system?: string; messages?: unknown[] };
    const out = await runChatProxy({
      key, system: system ?? "", messages: messages ?? [],
      model: MODEL, webSearch: WEB_SEARCH, thinking: THINKING, timeBudgetMs: TIME_BUDGET_MS,
    });
    res.status(200).json(out);
  } catch (e) {
    res.status(500).json({ error: e instanceof Error ? e.message : "Proxy error" });
  }
}
