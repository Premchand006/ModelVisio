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
//
// TIME BUDGET: serverless platforms kill a synchronous function at a hard limit
// (Netlify free = 10s, Vercel = up to maxDuration). A grounded Gemini call plus
// gemini-2.5 "thinking" can exceed that, and the platform then returns an opaque
// 502 instead of our JSON. To stay inside the budget we (1) disable thinking by
// default, (2) never sleep past the deadline on a 429 retry, (3) abort the fetch
// at the deadline, and (4) fall back to a fast non-grounded answer if grounding
// runs long — so the chat always responds within budget rather than 502-ing.

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
  /** Default: gemini-2.5-flash. Try gemini-2.5-flash-lite for the lightest free tier. */
  model?: string;
  /** Enable Google Search grounding so answers can cite real, current links. */
  webSearch?: boolean;
  maxTokens?: number;
  /**
   * Let gemini-2.5 models "think" before answering. OFF by default: thinking
   * adds seconds of latency that blow short serverless timeouts (Netlify free =
   * 10s). Ignored by non-2.5 models, which don't support it.
   */
  thinking?: boolean;
  /**
   * Total wall-clock budget (ms) for this call, including any retry wait. The
   * fetch is aborted at the deadline so the function returns a clean error
   * BEFORE the platform kills it with an opaque 502. Default: no limit.
   */
  timeBudgetMs?: number;
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

/** Marker thrown when a fetch is aborted because the time budget ran out. */
const TIMEOUT = "TIMEOUT";

/**
 * Run one copilot turn against the Gemini API and return the assistant text +
 * any grounded sources.
 *  - Disables gemini-2.5 "thinking" by default so the call fits short serverless
 *    timeouts; pass thinking:true to re-enable it where the budget allows.
 *  - Bounds the whole call by timeBudgetMs: aborts the fetch at the deadline and
 *    only waits out a transient 429 if the suggested delay fits the budget — so
 *    it surfaces a clean error instead of being killed mid-flight (opaque 502).
 *  - If grounding is unavailable OR runs past its slice of the budget, retries
 *    once WITHOUT the search tool so the chat still answers rather than failing.
 */
export async function runChatProxy({
  key,
  system,
  messages,
  model = "gemini-2.5-flash",
  webSearch = true,
  maxTokens = 4096,
  thinking = false,
  timeBudgetMs = Infinity,
}: RunChatArgs): Promise<ProxyResult> {
  const deadline = timeBudgetMs === Infinity ? Infinity : Date.now() + timeBudgetMs;
  const remaining = () => deadline - Date.now();

  const generationConfig: Record<string, unknown> = { maxOutputTokens: maxTokens, temperature: 0.4 };
  // thinkingConfig is only valid on gemini-2.5 models; sending it to others errors.
  if (!thinking && /2\.5/.test(model)) generationConfig.thinkingConfig = { thinkingBudget: 0 };

  const base: Record<string, unknown> = { contents: toContents(messages), generationConfig };
  if (system) base.systemInstruction = { parts: [{ text: system }] };
  const grounded = webSearch ? { ...base, tools: [{ google_search: {} }] } : base;

  // One Gemini request, aborted if `budgetMs` elapses. Retries transient errors
  // (429 rate limit, 503/500 overload) but only while the wait still fits inside
  // `budgetMs`, so it never sleeps the function into a platform 502.
  const call = async (payload: Record<string, unknown>, budgetMs: number): Promise<GeminiResponse> => {
    const start = Date.now();
    const left = () => budgetMs - (Date.now() - start);
    for (let attempt = 0; ; attempt++) {
      const ctrl = new AbortController();
      const timer = budgetMs === Infinity ? null : setTimeout(() => ctrl.abort(), Math.max(0, left()));
      let r: Response;
      let data: GeminiResponse;
      try {
        r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
          method: "POST",
          headers: { "content-type": "application/json", "x-goog-api-key": key },
          body: JSON.stringify(payload),
          signal: ctrl.signal,
        });
        data = (await r.json().catch(() => ({}))) as GeminiResponse;
      } catch (e) {
        if (e instanceof Error && e.name === "AbortError") throw new Error(TIMEOUT);
        throw e;
      } finally {
        if (timer) clearTimeout(timer);
      }
      if (r.ok) return data;
      const msg = data?.error?.message || "";
      // Decide how long to wait before retrying this transient failure, or null
      // to give up. 429 = honor the server's suggested delay; 503/500 = short
      // backoff (the free tier throws "high demand" 503s that clear in ~1s).
      let waitMs: number | null = null;
      if (r.status === 429 && !isZeroQuota(msg)) {
        const delay = retryDelaySec(msg);
        if (delay != null && delay <= 25) waitMs = delay * 1000 + 300;
      } else if ((r.status === 503 || r.status === 500) && attempt < 2) {
        waitMs = 700 * (attempt + 1);
      }
      // Retry only if the wait plus a margin for the next call still fits budget.
      if (waitMs != null && (budgetMs === Infinity || waitMs + 1500 < left())) {
        await sleep(waitMs);
        continue;
      }
      throw new Error(formatError(r.status, data, model));
    }
  };

  let data: GeminiResponse;
  if (!webSearch) {
    data = await call(base, remaining());
  } else {
    // Reserve part of the budget so a slow grounded call can still fall back to
    // a fast non-grounded one within the deadline (rather than 502-ing).
    const groundedBudget = deadline === Infinity ? Infinity : Math.max(0, Math.floor(timeBudgetMs * 0.6));
    try {
      data = await call(grounded, groundedBudget);
    } catch (e) {
      const m = e instanceof Error ? e.message : "";
      // A rate-limit/zero-quota failure repeats without grounding too — don't
      // burn the rest of the budget retrying; surface it as-is.
      if (/quota is 0/.test(m) || /rate limit/.test(m)) throw e;
      // Grounding unavailable or too slow → fast non-grounded retry if any
      // budget is left; otherwise the timeout error stands.
      if (deadline !== Infinity && remaining() < 1500) throw e;
      data = await call(base, remaining());
    }
  }

  const cand = data.candidates?.[0];
  const text = (cand?.content?.parts ?? []).map((p) => p.text).filter(Boolean).join("");
  return { text: text || "(empty response)", sources: extractSources(cand) };
}
