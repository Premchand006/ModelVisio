import { describe, it, expect, afterEach, vi } from "vitest";
import {
  runGrokChat, toGrokMessages, supportsReasoningEffort, XAI_CHAT_URL, DEFAULT_GROK_MODEL,
} from "../src/grok";
import { DEFAULT_GROK_MODEL as PROVIDERS_DEFAULT_GROK } from "../src/providers";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; vi.restoreAllMocks(); vi.useRealTimers(); });

const ok = (content: string) => new Response(
  JSON.stringify({ choices: [{ message: { role: "assistant", content } }] }),
  { status: 200, headers: { "content-type": "application/json" } },
);
const fail = (status: number, body: unknown, headers: Record<string, string> = {}) => new Response(
  JSON.stringify(body),
  { status, headers: { "content-type": "application/json", ...headers } },
);

/** Mock fetch that returns the given responses in order and records bodies. */
function mockFetch(...responses: Response[]) {
  const fn = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
    const next = responses.shift();
    if (!next) throw new Error("unexpected extra fetch");
    return next;
  });
  globalThis.fetch = fn as unknown as typeof fetch;
  return fn;
}
const sentBody = (fn: ReturnType<typeof mockFetch>, i = 0) => JSON.parse(String(fn.mock.calls[i][1]?.body));

describe("toGrokMessages", () => {
  it("puts the system prompt first and keeps user/assistant roles", () => {
    expect(toGrokMessages("SYS", [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "" },
      null,
    ])).toEqual([
      { role: "system", content: "SYS" },
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });
  it("omits an empty system prompt", () => {
    expect(toGrokMessages("", [{ role: "user", content: "q" }])).toEqual([{ role: "user", content: "q" }]);
  });
});

describe("supportsReasoningEffort", () => {
  it("is true for the grok-4.x line and *-reasoning ids only", () => {
    expect(supportsReasoningEffort("grok-4.7")).toBe(true);
    expect(supportsReasoningEffort("grok-420-reasoning")).toBe(true);
    expect(supportsReasoningEffort("grok-4-1-fast-non-reasoning")).toBe(false);
    expect(supportsReasoningEffort("grok-3")).toBe(false);
  });
});

describe("runGrokChat", () => {
  it("posts an OpenAI-style request with a Bearer key and returns the text", async () => {
    const fn = mockFetch(ok("Use INT8 on the Orin."));
    const out = await runGrokChat({ key: "xai-test", system: "SYS", messages: [{ role: "user", content: "q" }] });
    expect(out).toEqual({ text: "Use INT8 on the Orin.", sources: [] });
    expect(String(fn.mock.calls[0][0])).toBe(XAI_CHAT_URL);
    const headers = fn.mock.calls[0][1]?.headers as Record<string, string>;
    expect(headers.authorization).toBe("Bearer xai-test");
    const body = sentBody(fn);
    expect(body.model).toBe(DEFAULT_GROK_MODEL);
    expect(body.messages[0]).toEqual({ role: "system", content: "SYS" });
    expect(body.stream).toBe(false);
  });

  it("asks for low reasoning effort and caps output unless thinking is on", async () => {
    const fn = mockFetch(ok("a"), ok("b"));
    await runGrokChat({ key: "k", system: "", messages: [{ role: "user", content: "q" }] });
    expect(sentBody(fn, 0)).toMatchObject({ reasoning_effort: "low", max_tokens: 4096 });
    await runGrokChat({ key: "k", system: "", messages: [{ role: "user", content: "q" }], thinking: true });
    expect(sentBody(fn, 1).reasoning_effort).toBeUndefined();
    expect(sentBody(fn, 1).max_tokens).toBeUndefined();
  });

  it("drops reasoning_effort and retries when the model rejects it", async () => {
    const fn = mockFetch(fail(400, { code: "invalid-argument", error: "reasoning_effort is not supported" }), ok("fine"));
    const out = await runGrokChat({ key: "k", system: "", messages: [{ role: "user", content: "q" }] });
    expect(out.text).toBe("fine");
    expect(sentBody(fn, 0).reasoning_effort).toBe("low");
    expect(sentBody(fn, 1).reasoning_effort).toBeUndefined();
  });

  it("explains a rejected key", async () => {
    mockFetch(fail(401, { code: "unauthenticated", error: "Incorrect API key provided" }));
    await expect(runGrokChat({ key: "bad", system: "", messages: [{ role: "user", content: "q" }] }))
      .rejects.toThrow(/key rejected \(401\).*console\.x\.ai.*Incorrect API key/);
  });

  it("names the model on a 404 and reads OpenAI-style error bodies", async () => {
    mockFetch(fail(404, { error: { message: "model not found" } }));
    await expect(runGrokChat({ key: "k", system: "", messages: [], model: "grok-nope" }))
      .rejects.toThrow(/"grok-nope" is not available.*model not found/);
  });

  it("retries a 503 and then succeeds", async () => {
    const fn = mockFetch(fail(503, { error: "overloaded" }), ok("back"));
    const out = await runGrokChat({ key: "k", system: "", messages: [{ role: "user", content: "q" }] });
    expect(out.text).toBe("back");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("does not wait out a 429 whose Retry-After exceeds the budget", async () => {
    const fn = mockFetch(fail(429, { error: "slow down" }, { "retry-after": "20" }));
    await expect(runGrokChat({ key: "k", system: "", messages: [], timeBudgetMs: 5000 }))
      .rejects.toThrow(/rate limit \(429\)/);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("throws TIMEOUT when the budget runs out mid-request", async () => {
    globalThis.fetch = vi.fn((_i: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((_r, reject) => {
      init?.signal?.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
    })) as unknown as typeof fetch;
    await expect(runGrokChat({ key: "k", system: "", messages: [], timeBudgetMs: 20 })).rejects.toThrow("TIMEOUT");
  });

  it("keeps its default model in sync with providers.ts", () => {
    expect(DEFAULT_GROK_MODEL).toBe(PROVIDERS_DEFAULT_GROK);
  });
});
