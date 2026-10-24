// Copilot API keys (Gemini or Grok): per-provider details and the resolution
// order. Pure (no `vscode` import) so the precedence rules are unit tested;
// secrets.ts feeds it the actual values.

import type { ChatProvider } from "@modelvisio/ai/providers";

export type { ChatProvider };

export type KeySource = "secret" | "setting" | "env";

export type KeyCandidates = {
  /** VS Code SecretStorage (encrypted, per machine). */
  secret?: string | null;
  /** Legacy plain-text `modelvisio.geminiApiKey` setting — user level only
   *  (see userSettingKey). Gemini only. */
  setting?: string | null;
  /** GEMINI_API_KEY / XAI_API_KEY in the environment that launched VS Code. */
  env?: string | null;
};

export const API_KEY_URL = "https://aistudio.google.com/apikey";
export const GROK_KEY_URL = "https://console.x.ai";

export type ProviderInfo = {
  label: string;
  keyUrl: string;
  /** Environment variable read as the last fallback. */
  envVar: string;
  /** SecretStorage slot. */
  secretId: string;
  setCommand: string;
  setTitle: string;
  placeholder: string;
};

export const PROVIDERS: Record<ChatProvider, ProviderInfo> = {
  gemini: {
    label: "Gemini",
    keyUrl: API_KEY_URL,
    envVar: "GEMINI_API_KEY",
    // Same id as the legacy setting, for discoverability.
    secretId: "modelvisio.geminiApiKey",
    setCommand: "modelvisio.setApiKey",
    setTitle: "ModelVisio: Set Gemini API Key",
    placeholder: "AIza…",
  },
  grok: {
    label: "Grok (xAI)",
    keyUrl: GROK_KEY_URL,
    envVar: "XAI_API_KEY",
    secretId: "modelvisio.grokApiKey",
    setCommand: "modelvisio.setGrokApiKey",
    setTitle: "ModelVisio: Set Grok (xAI) API Key",
    placeholder: "xai-…",
  },
};

export function missingKeyMessage(provider: ChatProvider): string {
  const p = PROVIDERS[provider];
  const where = provider === "gemini" ? `free key at ${p.keyUrl}` : `get a key at ${p.keyUrl}`;
  return `No ${p.label} API key configured. Run "${p.setTitle}" from the Command Palette ` +
    `(${where}), or set ${p.envVar} in the environment that launches VS Code.`;
}

export const MISSING_KEY_MESSAGE = missingKeyMessage("gemini");

/** First non-blank key in precedence order: SecretStorage → legacy setting → env. */
export function resolveApiKey(c: KeyCandidates): { key: string; source: KeySource } | null {
  const order: [KeySource, string | null | undefined][] = [
    ["secret", c.secret],
    ["setting", c.setting],
    ["env", c.env],
  ];
  for (const [source, value] of order) {
    const key = value?.trim();
    if (key) return { key, source };
  }
  return null;
}

/** The legacy setting's *user* (global) value from `inspect()`. Workspace and
 *  folder values come from a checked-in/shared .vscode/settings.json; adopting
 *  one would bill someone else's key (or route chats through a key the repo
 *  author controls), so they are ignored for both lookup and migration. */
export function userSettingKey(inspected: { globalValue?: unknown } | undefined): string | undefined {
  const v = inspected?.globalValue;
  return typeof v === "string" ? v : undefined;
}

/** Migrate the legacy user setting into SecretStorage only when no secret
 *  exists yet — never overwrite a key the user stored deliberately. */
export function shouldMigrate(secret: string | null | undefined, legacySetting: string | null | undefined): boolean {
  return !secret?.trim() && !!legacySetting?.trim();
}

/** InputBox validator: null = valid, otherwise the message to show. */
export function validateApiKeyInput(value: string): string | null {
  const v = value.trim();
  if (!v) return "The API key cannot be empty.";
  if (/\s/.test(v)) return "API keys don't contain spaces — check what was pasted.";
  return null;
}
