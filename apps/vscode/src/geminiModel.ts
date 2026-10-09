// Copilot model id sanitizing (Gemini and Grok). Pure (no `vscode` import) so
// it is unit tested; config.ts applies it to the `modelvisio.geminiModel` and
// `modelvisio.grokModel` settings.

export const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";
export const DEFAULT_GROK_MODEL = "grok-4.7";

/** A plain model id, or `fallback`. The setting's enum is only an editor hint
 *  — a hand-edited value can be anything — and the Gemini id is interpolated
 *  into the request URL path, so only plain ids are accepted. */
function sanitizeModelId(value: unknown, fallback: string): string {
  const id = typeof value === "string" ? value.trim() : "";
  return /^[A-Za-z0-9.-]+$/.test(id) ? id : fallback;
}

export function sanitizeGeminiModel(value: unknown): string {
  return sanitizeModelId(value, DEFAULT_GEMINI_MODEL);
}

/** Grok ids travel in the JSON body, not the URL, but the same plain-id rule
 *  keeps an untrusted value from smuggling anything odd into the request. */
export function sanitizeGrokModel(value: unknown): string {
  return sanitizeModelId(value, DEFAULT_GROK_MODEL);
}
