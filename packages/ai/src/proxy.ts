// Server-side Google Gemini call shared by every proxy entrypoint: the Vercel
// function (apps/web/api/chat.ts), the Netlify function (netlify/functions/
// chat.ts), and the Vite dev middleware (apps/web/vite.config.ts).
//
// This runs ONLY on a server/Node/edge runtime — it is intentionally NOT
// re-exported from index.ts, so the API key path never enters the browser
// bundle. Uses global `fetch` (Node 18+, Vercel, Netlify, Vite dev server).
//
// Get a FREE key at https://aistudio.google.com/apikey (no billing required for
// the free tier). The client→proxy contract stays Anthropic-style ({ system,
// messages:[{role:"user"|"assistant", content}] }); we convert to Gemini's
// `contents`/`systemInstruction` shape here so the UI/client need not change.

export type ProxyResult = { text: string; sources: { title: string; url: string }[] };
export type ChatTurn = { role: "user" | "assistant"; content: string };

type GeminiPart = { text?: string };
type GeminiContent = { role?: string; parts?: GeminiPart[] };
type GroundingChunk = { web?: { uri?: string; title?: string } };
type Candidate = { content?: GeminiContent; groundingMetadata?: { groundingChunks?: GroundingChunk[] } };
type GeminiResponse = { candidates?: Candidate[]; error?: { message?: string; status?: string } };

/** Anthropic-style messages → Gemini `contents` (role "assistant" → "model"). */
function toContents(messages: unknown[]): GeminiContent[] {
  return (messages as ChatTurn[])
    .filter((m) => m && typeof m.content === "string" && m.content.length > 0)
    .map((m) => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] }));
}

/** Pull unique {title,url} from Gemini's Google-Search grounding metadata. */
function extractSources(cand: Candidate | undefined): { title: string; url: string }[] {
  const chunks = cand?.groundingMetadata?.groundingChunks ?? [];
  const out: { title: string; url: string }[] = [];
  const seen = new Set<string>();
  for (const c of chunks) {
    const url = c?.web?.uri;
    if (url && !seen.has(url)) {
      seen.add(url);
      out.push({ title: c.web?.title || url, url });
    }
  }
  return out;
}

export type RunChatArgs = {
  key: string;
  system: string;
  messages: unknown[];
  /** Default: gemini-2.0-flash (free tier). Try gemini-2.5-flash for stronger reasoning. */
  model?: string;
  /** Enable Google Search grounding so answers can cite real, current links. */
  webSearch?: boolean;
  maxTokens?: number;
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Seconds to wait from a 429 body ("Please retry in 13.8s" or "retryDelay": "13s"). */
function retryDelaySec(msg: string): number | null {
  const m = msg.match(/retry in ([\d.]+)s/i) || msg.match(/"?retryDelay"?:\s*"?([\d.]+)s/i);
  return m ? parseFloat(m[1]) : null;
}

/** True when the 429 means the quota ceiling itself is 0 — not a transient rate
 *  limit. Retrying/waiting can NOT help; the key/project/region must change. */
function isZeroQuota(msg: string): boolean {
  return /limit:?\s*"?0\b/.test(msg);
}

function formatError(status: number, data: GeminiResponse, model: string): string {
  const msg = data?.error?.message || JSON.stringify(data);
  if (status === 429) {
    if (isZeroQuota(msg)) {
      return `Gemini free-tier quota is 0 for "${model}" on this key — Google grants it no requests, so retrying won't help. `
        + `Fix on the Google side: (1) create a NEW key in a NEW project at https://aistudio.google.com/apikey ; `
        + `(2) try MODELVISIO_MODEL=gemini-2.5-flash-lite or gemini-2.0-flash; `
        + `(3) the free tier may not be offered in your region — enabling billing keeps a free monthly allowance. Original: ${msg}`;
    }
    return `Gemini rate limit (429) on "${model}": ${msg}`;
  }
  return `Gemini API error ${status}: ${msg}`;
}

/**
 * Run one copilot turn against the Gemini API and return the assistant text +
 * any grounded sources.
 *  - Retries once on a transient 429 (real rate limit with a retry delay), but
 *    NOT on a zero-quota 429 (waiting can't help — surfaces an actionable error).
 *  - If grounding is unavailable for this key/region, retries without the search
 *    tool so the chat still works rather than hard-failing.
 */
export async function runChatProxy({
  key,
  system,
  messages,
  model = "gemini-2.5-flash",
  webSearch = true,
  maxTokens = 4096,
}: RunChatArgs): Promise<ProxyResult> {
  const base: Record<string, unknown> = {
    contents: toContents(messages),
    generationConfig: { maxOutputTokens: maxTokens, temperature: 0.4 },
  };
  if (system) base.systemInstruction = { parts: [{ text: system }] };
  const grounded = webSearch ? { ...base, tools: [{ google_search: {} }] } : base;

  const call = async (payload: Record<string, unknown>): Promise<GeminiResponse> => {
    for (let attempt = 0; ; attempt++) {
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
        method: "POST",
        headers: { "content-type": "application/json", "x-goog-api-key": key },
        body: JSON.stringify(payload),
      });
      const data = (await r.json().catch(() => ({}))) as GeminiResponse;
      if (r.ok) return data;
      const msg = data?.error?.message || "";
      const delay = retryDelaySec(msg);
      if (r.status === 429 && attempt === 0 && !isZeroQuota(msg) && delay != null && delay <= 25) {
        await sleep(delay * 1000 + 300);
        continue; // transient rate limit — wait the suggested time, retry once
      }
      throw new Error(formatError(r.status, data, model));
    }
  };

  let data: GeminiResponse;
  try {
    data = await call(grounded);
  } catch (e) {
    // Grounding may be unavailable on this key/region — retry without it, unless
    // the failure is a zero-quota one (which a no-tools retry can't fix either).
    if (!webSearch || (e instanceof Error && /quota is 0/.test(e.message))) throw e;
    data = await call(base);
  }

  const cand = data.candidates?.[0];
  const text = (cand?.content?.parts ?? []).map((p) => p.text).filter(Boolean).join("");
  return { text: text || "(empty response)", sources: extractSources(cand) };
}
