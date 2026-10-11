// Gemini API key resolution order. Pure (no `vscode` import) so the precedence
// rules are unit tested; secrets.ts feeds it the actual values.

export type KeySource = "secret" | "setting" | "env";

export type KeyCandidates = {
  /** VS Code SecretStorage (encrypted, per machine). */
  secret?: string | null;
  /** Legacy plain-text `modelvisio.geminiApiKey` setting — user level only
   *  (see userSettingKey). */
  setting?: string | null;
  /** GEMINI_API_KEY in the environment that launched VS Code. */
  env?: string | null;
};

export const API_KEY_URL = "https://aistudio.google.com/apikey";

export const MISSING_KEY_MESSAGE =
  `No Gemini API key configured. Run "ModelVisio: Set Gemini API Key" from the Command Palette ` +
  `(free key at ${API_KEY_URL}), or set GEMINI_API_KEY in the environment that launches VS Code.`;

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
