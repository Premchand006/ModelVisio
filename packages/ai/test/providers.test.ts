import { describe, it, expect } from "vitest";
import { chatProviderFromEnv, parseProvider, DEFAULT_GEMINI_MODEL, DEFAULT_GROK_MODEL } from "../src/providers";

describe("parseProvider", () => {
  it("accepts provider names and vendor aliases, case-insensitively", () => {
    expect(parseProvider(" Grok ")).toBe("grok");
    expect(parseProvider("xai")).toBe("grok");
    expect(parseProvider("GEMINI")).toBe("gemini");
    expect(parseProvider("google")).toBe("gemini");
    expect(parseProvider("openai")).toBeNull();
    expect(parseProvider(undefined)).toBeNull();
  });
});

describe("chatProviderFromEnv", () => {
  it("defaults to Gemini when its key is set (existing deployments are unchanged)", () => {
    expect(chatProviderFromEnv({ GEMINI_API_KEY: "g", XAI_API_KEY: "x" })).toMatchObject({
      provider: "gemini", key: "g", model: DEFAULT_GEMINI_MODEL,
    });
    expect(chatProviderFromEnv({ GOOGLE_API_KEY: "g2" }).key).toBe("g2");
  });
  it("falls back to Grok when only an xAI key is set", () => {
    expect(chatProviderFromEnv({ XAI_API_KEY: "x" })).toMatchObject({ provider: "grok", key: "x", model: DEFAULT_GROK_MODEL });
  });
  it("honors an explicit provider and per-provider model", () => {
    expect(chatProviderFromEnv({
      MODELVISIO_PROVIDER: "grok", GEMINI_API_KEY: "g", XAI_API_KEY: "x",
      MODELVISIO_MODEL: "gemini-2.5-flash-lite", MODELVISIO_GROK_MODEL: "grok-420-reasoning",
    })).toMatchObject({ provider: "grok", key: "x", model: "grok-420-reasoning" });
  });
  it("reports which key is missing", () => {
    const grok = chatProviderFromEnv({ MODELVISIO_PROVIDER: "grok", GEMINI_API_KEY: "g" });
    expect(grok.key).toBeUndefined();
    expect(grok.missingKeyError).toMatch(/XAI_API_KEY/);
    const none = chatProviderFromEnv({});
    expect(none).toMatchObject({ provider: "gemini", key: undefined });
    expect(none.missingKeyError).toMatch(/GEMINI_API_KEY/);
  });
});
