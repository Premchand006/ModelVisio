import { describe, expect, it } from "vitest";
import { resolveApiKey, shouldMigrate, userSettingKey, validateApiKeyInput } from "../src/apiKey";

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
