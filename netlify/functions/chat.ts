import { runChatProxy } from "@modelvisio/ai/proxy";

// Netlify equivalent of the Vercel /api/chat proxy. Holds GEMINI_API_KEY
// server-side and runs Gemini with Google Search grounding. `config.path`
// routes /api/chat here.
//
// Netlify synchronous functions are killed at a HARD 10s on the free tier (not
// raisable), and the platform then returns an opaque 502 — not our JSON. So we
// give the proxy a sub-10s time budget: it disables slow "thinking", never
// sleeps past the deadline on a 429, and falls back to a fast non-grounded
// answer if grounded search runs long — so the chat answers instead of 502-ing.
// Tune with MODELVISIO_TIMEOUT_MS; set MODELVISIO_WEB_SEARCH=off to skip search
// entirely, or MODELVISIO_THINKING=on if you raise the timeout on a paid plan.
const MODEL = process.env.MODELVISIO_MODEL || "gemini-2.5-flash";
const WEB_SEARCH = (process.env.MODELVISIO_WEB_SEARCH || "on") !== "off";
const THINKING = (process.env.MODELVISIO_THINKING || "off") === "on";
const TIME_BUDGET_MS = Number(process.env.MODELVISIO_TIMEOUT_MS) || 8500;

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
