// Server-side xAI Grok call — the second copilot provider next to Gemini
// (proxy.ts). Same contract as runChatProxy: Anthropic-style { system,
// messages } in, { text, sources } out, so every proxy entrypoint can switch
// providers without touching the UI or client.
//
// Runs ONLY on a server/Node/edge runtime and is NOT re-exported from index.ts,
// so the xAI key never enters the browser bundle. Uses global `fetch`.
//
// xAI's REST API is OpenAI-compatible: POST https://api.x.ai/v1/chat/completions
// with `Authorization: Bearer <XAI_API_KEY>`. Keys: https://console.x.ai .
// Grok has no free built-in web search on this endpoint, so live grounding
// comes from the free crawler in search.ts (chatWithWebResearch), which the
// entrypoints wrap around this call.
//
// TIME BUDGET: same rules as proxy.ts — abort the fetch at the deadline and
// only wait out a transient 429/5xx if the wait still fits, so a serverless
// function returns a clean error instead of an opaque platform 502.

// NOTE: vite.config.ts imports this file, and Node loads it natively with
// type-stripping. Erasable TS syntax only, and `import type` for anything from
// sibling files (no runtime relative imports).
import type { ChatTurn, ProxyResult } from "./proxy";

export const XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions";
export const DEFAULT_GROK_MODEL = "grok-4.7";
export const GROK_KEY_URL = "https://console.x.ai";

export type RunGrokArgs = {
  key: string;
  system: string;
  messages: unknown[];
  /** Default: grok-4.7. */
  model?: string;
  /**
   * Let reasoning models think at their default (high) effort. OFF by default:
   * the copilot asks for `reasoning_effort: "low"`, which keeps answers fast
   * enough for short serverless budgets.
   */
  thinking?: boolean;
  /** Output cap when not thinking. With thinking on no cap is sent, because
   *  reasoning tokens would otherwise eat the whole allowance. */
  maxTokens?: number;
  /** Total wall-clock budget (ms), including any retry wait. Default: no limit. */
  timeBudgetMs?: number;
};

type GrokMessage = { role: "system" | "user" | "assistant"; content: string };
type GrokResponse = {
  choices?: { message?: { content?: string | null } }[];
  // xAI answers errors as { code, error: "msg" }; OpenAI-style { error: { message } } also seen.
  error?: string | { message?: string };
  code?: string;
};

/** Models that accept `reasoning_effort`: the grok-4.x line and *-reasoning ids. */
export function supportsReasoningEffort(model: string): boolean {
  return /^grok-4\.\d/i.test(model) || (/reasoning/i.test(model) && !/non-reasoning/i.test(model));
}

/** Anthropic-style turns → OpenAI-style messages with the system prompt first. */
export function toGrokMessages(system: string, messages: unknown[]): GrokMessage[] {
  const turns = (messages as ChatTurn[])
    .filter((m) => m && typeof m.content === "string" && m.content.length > 0)
    .map((m): GrokMessage => ({ role: m.role === "assistant" ? "assistant" : "user", content: m.content }));
  return system ? [{ role: "system", content: system }, ...turns] : turns;
}

function errorText(data: GrokResponse): string {
  const e = data?.error;
  if (typeof e === "string") return e;
  if (e && typeof e.message === "string") return e.message;
  return JSON.stringify(data);
}

function formatError(status: number, data: GrokResponse, model: string): string {
  const msg = errorText(data);
  if (status === 401 || status === 403) {
    return `Grok API key rejected (${status}). Check XAI_API_KEY / your key at ${GROK_KEY_URL}. Original: ${msg}`;
  }
  if (status === 404) return `Grok model "${model}" is not available to this key (404): ${msg}`;
  if (status === 429) return `Grok rate limit (429) on "${model}": ${msg}`;
  return `Grok API error ${status}: ${msg}`;
}

/** Seconds from a Retry-After header (delta-seconds form only). */
function retryAfterSec(res: Response): number | null {
  const v = res.headers.get("retry-after");
  const n = v != null ? Number(v) : NaN;
  return Number.isFinite(n) && n >= 0 ? n : null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Marker thrown when the fetch is aborted because the time budget ran out
 *  (same marker as proxy.ts, so callers handle both providers alike). */
const TIMEOUT = "TIMEOUT";

/**
 * Run one copilot turn against xAI Grok and return the assistant text. Grok's
 * chat-completions endpoint returns no citations for plain calls, so `sources`
 * is empty here — the free web research wrapper adds its own.
 */
export async function runGrokChat({
  key,
  system,
  messages,
  model = DEFAULT_GROK_MODEL,
  thinking = false,
  maxTokens = 4096,
  timeBudgetMs = Infinity,
}: RunGrokArgs): Promise<ProxyResult> {
  const deadline = timeBudgetMs === Infinity ? Infinity : Date.now() + timeBudgetMs;
  const left = () => deadline - Date.now();

  const payload: Record<string, unknown> = {
    model,
    messages: toGrokMessages(system, messages),
    temperature: 0.4,
    stream: false,
  };
  if (!thinking) {
    payload.max_tokens = maxTokens;
    if (supportsReasoningEffort(model)) payload.reasoning_effort = "low";
  }

  for (let attempt = 0; ; attempt++) {
    const ctrl = new AbortController();
    const timer = deadline === Infinity ? null : setTimeout(() => ctrl.abort(), Math.max(0, left()));
    let res: Response;
    let data: GrokResponse;
    try {
      res = await fetch(XAI_CHAT_URL, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
        body: JSON.stringify(payload),
        signal: ctrl.signal,
      });
      data = (await res.json().catch(() => ({}))) as GrokResponse;
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") throw new Error(TIMEOUT);
      throw e;
    } finally {
      if (timer) clearTimeout(timer);
    }

    if (res.ok) {
      const text = data.choices?.[0]?.message?.content ?? "";
      return { text: text || "(empty response)", sources: [] };
    }

    // A model that rejects reasoning_effort: drop it once and resend at once.
    if (res.status === 400 && "reasoning_effort" in payload && /reasoning/i.test(errorText(data))) {
      delete payload.reasoning_effort;
      continue;
    }
    // Transient: 429 honors Retry-After (else a short backoff), 5xx backs off.
    let waitMs: number | null = null;
    if (res.status === 429 && attempt < 2) {
      const after = retryAfterSec(res);
      waitMs = after != null ? (after <= 25 ? after * 1000 + 300 : null) : 1000 * (attempt + 1);
    } else if (res.status >= 500 && attempt < 2) {
      waitMs = 700 * (attempt + 1);
    }
    if (waitMs != null && (deadline === Infinity || waitMs + 1500 < left())) {
      await sleep(waitMs);
      continue;
    }
    throw new Error(formatError(res.status, data, model));
  }
}
