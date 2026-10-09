import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AI_PROVIDER_STORAGE, GEMINI_KEY_STORAGE, GROK_KEY_STORAGE,
  getAiProvider, setAiProvider, getUserApiKey, setUserApiKey,
} from "../src/utils/apiKey";

// Node env has no Web Storage — stub an in-memory localStorage so the
// bring-your-own-key helpers can be exercised as they run in the WebView.
function memoryStorage() {
  const m = new Map<string, string>();
  return {
    getItem: (k: string) => (m.has(k) ? m.get(k)! : null),
    setItem: (k: string, v: string) => { m.set(k, String(v)); },
    removeItem: (k: string) => { m.delete(k); },
    clear: () => m.clear(),
    key: (i: number) => [...m.keys()][i] ?? null,
    get length() { return m.size; },
  };
}

let store: ReturnType<typeof memoryStorage>;
beforeEach(() => {
  store = memoryStorage();
  vi.stubGlobal("localStorage", store);
});
afterEach(() => vi.unstubAllGlobals());

describe("AI provider selection", () => {
  it("defaults to gemini", () => {
    expect(getAiProvider()).toBe("gemini");
  });

  it("persists the chosen provider", () => {
    setAiProvider("grok");
    expect(store.getItem(AI_PROVIDER_STORAGE)).toBe("grok");
    expect(getAiProvider()).toBe("grok");
    setAiProvider("gemini");
    expect(getAiProvider()).toBe("gemini");
  });

  it("falls back to gemini for unknown stored values", () => {
    store.setItem(AI_PROVIDER_STORAGE, "openai");
    expect(getAiProvider()).toBe("gemini");
  });

  it("never throws when storage is unavailable", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => { throw new Error("denied"); },
      setItem: () => { throw new Error("denied"); },
      removeItem: () => { throw new Error("denied"); },
    });
    expect(getAiProvider()).toBe("gemini");
    expect(() => setAiProvider("grok")).not.toThrow();
    expect(getUserApiKey()).toBe("");
    expect(() => setUserApiKey("k")).not.toThrow();
  });

  it("works when localStorage is not defined at all", () => {
    vi.stubGlobal("localStorage", undefined);
    expect(getAiProvider()).toBe("gemini");
    expect(getUserApiKey("grok")).toBe("");
  });
});

describe("per-provider API keys", () => {
  it("keeps Gemini and Grok keys separate", () => {
    setUserApiKey("AIza-gem", "gemini");
    setUserApiKey("xai-grok", "grok");
    expect(store.getItem(GEMINI_KEY_STORAGE)).toBe("AIza-gem");
    expect(store.getItem(GROK_KEY_STORAGE)).toBe("xai-grok");
    expect(getUserApiKey("gemini")).toBe("AIza-gem");
    expect(getUserApiKey("grok")).toBe("xai-grok");
  });

  it("defaults to the selected provider's key", () => {
    setUserApiKey("AIza-gem", "gemini");
    setUserApiKey("xai-grok", "grok");
    expect(getUserApiKey()).toBe("AIza-gem");
    setAiProvider("grok");
    expect(getUserApiKey()).toBe("xai-grok");
    setUserApiKey("xai-new");
    expect(store.getItem(GROK_KEY_STORAGE)).toBe("xai-new");
    expect(store.getItem(GEMINI_KEY_STORAGE)).toBe("AIza-gem");
  });

  it("trims keys and clears on blank", () => {
    setUserApiKey("  xai-abc  ", "grok");
    expect(getUserApiKey("grok")).toBe("xai-abc");
    setUserApiKey("   ", "grok");
    expect(store.getItem(GROK_KEY_STORAGE)).toBeNull();
    expect(getUserApiKey("grok")).toBe("");
  });

  it("keeps the legacy Gemini storage key", () => {
    // Existing desktop installs stored their key here before Grok was added.
    expect(GEMINI_KEY_STORAGE).toBe("mv_gemini_key");
    store.setItem("mv_gemini_key", "AIza-legacy");
    expect(getUserApiKey()).toBe("AIza-legacy");
  });
});
