import { runChatProxy } from "@modelvisio/ai/proxy";

// Netlify equivalent of the Vercel /api/chat proxy. Holds GEMINI_API_KEY
// server-side and runs Gemini with Google Search grounding. `config.path`
// routes /api/chat here.
//
// NOTE: Netlify sync functions cap at ~10s on free tiers; grounded search can
// exceed that. Use MODELVISIO_WEB_SEARCH=off or a lighter model
// (MODELVISIO_MODEL=gemini-2.0-flash-lite) there, or upgrade the function timeout.
const MODEL = process.env.MODELVISIO_MODEL || "gemini-2.5-flash";
const WEB_SEARCH = (process.env.MODELVISIO_WEB_SEARCH || "on") !== "off";

export default async (req: Request): Promise<Response> => {
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) return Response.json({ error: "GEMINI_API_KEY is not configured on the server." }, { status: 500 });
  try {
    const { system, messages } = (await req.json()) as { system?: string; messages?: unknown[] };
    const out = await runChatProxy({ key, system: system ?? "", messages: messages ?? [], model: MODEL, webSearch: WEB_SEARCH });
    return Response.json(out);
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "Proxy error" }, { status: 500 });
  }
};

export const config = { path: "/api/chat" };
