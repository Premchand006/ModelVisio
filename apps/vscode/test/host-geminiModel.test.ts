import { describe, expect, it } from "vitest";
import { DEFAULT_GEMINI_MODEL, sanitizeGeminiModel } from "../src/geminiModel";

describe("sanitizeGeminiModel", () => {
  it("keeps plain model ids (trimmed)", () => {
    expect(sanitizeGeminiModel("gemini-2.5-flash-lite")).toBe("gemini-2.5-flash-lite");
    expect(sanitizeGeminiModel(" gemini-2.0-flash ")).toBe("gemini-2.0-flash");
    expect(sanitizeGeminiModel("Gemini-3.0-Pro")).toBe("Gemini-3.0-Pro");
  });

  it("falls back to the default for anything that could alter the request URL", () => {
    for (const bad of [
      "../../v1/files",
      "gemini-2.5-flash:streamGenerateContent",
      "gemini?key=x",
      "gemini/2.5",
      "gemini%2F",
      "gemini 2.5",
      "gemini_2",
      "",
      "   ",
    ]) {
      expect(sanitizeGeminiModel(bad)).toBe(DEFAULT_GEMINI_MODEL);
    }
  });

  it("falls back to the default for non-strings", () => {
    expect(sanitizeGeminiModel(undefined)).toBe(DEFAULT_GEMINI_MODEL);
    expect(sanitizeGeminiModel(null)).toBe(DEFAULT_GEMINI_MODEL);
    expect(sanitizeGeminiModel(42)).toBe(DEFAULT_GEMINI_MODEL);
    expect(sanitizeGeminiModel(["gemini-2.5-flash"])).toBe(DEFAULT_GEMINI_MODEL);
  });
});
