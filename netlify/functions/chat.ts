import { runChatProxy } from "@modelvisio/ai/proxy";

// Netlify equivalent of the Vercel /api/chat proxy. Holds GEMINI_API_KEY
// server-side and runs Gemini with Google Search grounding. `config.path`
// routes /api/chat here.
//
// Netlify synchronous functions are killed at a HARD 10s on the free tier (not
// raisable), and the platform then returns an opaque 502 — not our JSON. The
// 10s clock also includes cold-start/init time, so we keep a conservative budget
// AND default Google-Search grounding OFF here: grounding is the slow path and,
// combined with a cold start, is the main cause of the free-tier 502. A plain
// (non-grounded) reply returns in ~1s and stays comfortably inside the limit.
//
// To re-enable citations, set MODELVISIO_WEB_SEARCH=on (best paired with a paid
// plan + a higher MODELVISIO_TIMEOUT_MS). Vercel's 60s shell keeps grounding on.
// MODELVISIO_THINKING=on only if you raise the timeout on a longer-limit plan.
const MODEL = process.env.MODELVISIO_MODEL || "gemini-2.5-flash";
const WEB_SEARCH = (process.env.MODELVISIO_WEB_SEARCH || "off") !== "off";
const THINKING = (process.env.MODELVISIO_THINKING || "off") === "on";
const TIME_BUDGET_MS = Number(process.env.MODELVISIO_TIMEOUT_MS) || 7000;

export default async (req: Request): Promise<Response> => {
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) return Response.json({ error: "GEMINI_API_KEY is not configured on the server." }, { status: 500 });
  try {
    const { system, messages } = (await req.json()) as { system?: string; messages?: unknown[] };
    const out = await runChatProxy({
      key, system: system ?? "", messages: messages ?? [],
      model: MODEL, webSearch: WEB_SEARCH, thinking: THINKING, timeBudgetMs: TIME_BUDGET_MS,
    });
    return Response.json(out);
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "Proxy error" }, { status: 500 });
  }
};

export const config = { path: "/api/chat" };
