import { describe, expect, it } from "vitest";
import {
  MISSING_KEY_MESSAGE, PROVIDERS, missingKeyMessage, resolveApiKey, shouldMigrate, userSettingKey, validateApiKeyInput,
} from "../src/apiKey";

describe("resolveApiKey", () => {
  it("prefers SecretStorage, then the legacy setting, then the environment", () => {
    expect(resolveApiKey({ secret: "s", setting: "c", env: "e" })).toEqual({ key: "s", source: "secret" });
    expect(resolveApiKey({ secret: undefined, setting: "c", env: "e" })).toEqual({ key: "c", source: "setting" });
    expect(resolveApiKey({ secret: null, setting: "", env: "e" })).toEqual({ key: "e", source: "env" });
  });

  it("skips blank values and trims", () => {
    expect(resolveApiKey({ secret: "   ", setting: " k ", env: "e" })).toEqual({ key: "k", source: "setting" });
  });

  it("returns null when nothing is configured", () => {
    expect(resolveApiKey({})).toBeNull();
    expect(resolveApiKey({ secret: "", setting: " ", env: undefined })).toBeNull();
  });
});

describe("userSettingKey", () => {
  it("reads only the user-level value, never workspace or folder values", () => {
    const workspaceOnly = { globalValue: undefined, workspaceValue: "ws", workspaceFolderValue: "folder" };
    const both = { globalValue: "user", workspaceValue: "ws" };
    expect(userSettingKey({ globalValue: "user" })).toBe("user");
    expect(userSettingKey(workspaceOnly)).toBeUndefined();
    expect(userSettingKey(both)).toBe("user");
  });

  it("ignores missing and non-string values", () => {
    expect(userSettingKey(undefined)).toBeUndefined();
    expect(userSettingKey({})).toBeUndefined();
    expect(userSettingKey({ globalValue: 123 })).toBeUndefined();
  });
});

describe("shouldMigrate", () => {
  it("migrates only when the setting is set and no secret exists", () => {
    expect(shouldMigrate(undefined, "key")).toBe(true);
    expect(shouldMigrate("", "key")).toBe(true);
    expect(shouldMigrate("existing", "key")).toBe(false);
    expect(shouldMigrate(undefined, "  ")).toBe(false);
    expect(shouldMigrate(undefined, undefined)).toBe(false);
  });
});

describe("validateApiKeyInput", () => {
  it("rejects empty and whitespace-containing input", () => {
    expect(validateApiKeyInput("")).not.toBeNull();
    expect(validateApiKeyInput("   ")).not.toBeNull();
    expect(validateApiKeyInput("AIza abc")).not.toBeNull();
    expect(validateApiKeyInput(" AIzaSyabc ")).toBeNull();
  });
});

describe("providers", () => {
  it("keeps Gemini and Grok keys in separate secret slots and env vars", () => {
    expect(PROVIDERS.gemini.secretId).not.toBe(PROVIDERS.grok.secretId);
    expect(PROVIDERS.gemini.envVar).toBe("GEMINI_API_KEY");
    expect(PROVIDERS.grok.envVar).toBe("XAI_API_KEY");
  });

  it("names the provider, its command and env var in the missing-key message", () => {
    expect(MISSING_KEY_MESSAGE).toBe(missingKeyMessage("gemini"));
    expect(missingKeyMessage("gemini")).toMatch(/No Gemini API key.*Set Gemini API Key.*GEMINI_API_KEY/);
    const grok = missingKeyMessage("grok");
    expect(grok).toMatch(/No Grok \(xAI\) API key.*Set Grok \(xAI\) API Key.*console\.x\.ai.*XAI_API_KEY/);
  });

  it("points each provider at a command the manifest contributes", async () => {
    const { readFileSync } = await import("node:fs");
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
      contributes: { commands: { command: string; title: string }[] };
    };
    for (const p of Object.values(PROVIDERS)) {
      const cmd = pkg.contributes.commands.find((c) => c.command === p.setCommand);
      expect(cmd, p.setCommand).toBeDefined();
      expect(p.setTitle).toBe(`ModelVisio: ${cmd!.title}`);
    }
  });
});
