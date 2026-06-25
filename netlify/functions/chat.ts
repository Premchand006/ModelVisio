import { runChatProxy } from "@modelvisio/ai/proxy";
import { logChat, logClientEvent, type ClientEvent } from "@modelvisio/ai/log";

// Netlify proxy for the AI copilot. Holds GEMINI_API_KEY server-side (set it in
// the Netlify site env) so it NEVER reaches the browser. Reached at /api/chat via
// the rewrite in netlify.toml (this function lives at /.netlify/functions/chat).
//
// Uses the LEGACY v1 `handler` export on purpose: Netlify's build here invokes
// functions in v1 mode (it looks for `.handler`), so a v2-style `export default`
// fails at runtime with "handler is not a function" → an opaque 502 on every
// call. The v1 handler is detected reliably on every plan/build image.
//
// Netlify synchronous functions are killed at a HARD 10s on the free tier (not
// raisable) — and that clock includes cold start. So we keep a conservative time
// budget AND default Google-Search grounding OFF: grounding is the slow path and,
// with a cold start, is the main cause of free-tier 502s. A plain reply returns
// in ~1s. Re-enable citations with MODELVISIO_WEB_SEARCH=on (best on a paid plan
// + a higher MODELVISIO_TIMEOUT_MS). Vercel's 60s shell keeps grounding on.
const MODEL = process.env.MODELVISIO_MODEL || "gemini-2.5-flash";
const WEB_SEARCH = (process.env.MODELVISIO_WEB_SEARCH || "off") !== "off";
const THINKING = (process.env.MODELVISIO_THINKING || "off") === "on";
const TIME_BUDGET_MS = Number(process.env.MODELVISIO_TIMEOUT_MS) || 7000;

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
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) return json(500, { error: "GEMINI_API_KEY is not configured on the server." });
  try {
    const body = JSON.parse(event.body || "{}") as { system?: string; messages?: unknown[]; event?: ClientEvent };
    // Client telemetry beacon (e.g. a model upload): record it and return.
    if (body.event) {
      await logClientEvent(body.event, { ip: clientIp(event) });
      return json(200, { ok: true });
    }
    const { system, messages } = body;
    // Log concurrently with the Gemini call so it adds ~no latency; never rejects.
    const logP = logChat(system ?? "", messages ?? [], { ip: clientIp(event), user_agent: event.headers?.["user-agent"] ?? null });
    const out = await runChatProxy({
      key, system: system ?? "", messages: messages ?? [],
      model: MODEL, webSearch: WEB_SEARCH, thinking: THINKING, timeBudgetMs: TIME_BUDGET_MS,
    });
    await logP;
    return json(200, out);
  } catch (e) {
    return json(500, { error: e instanceof Error ? e.message : "Proxy error" });
  }
};
