// Gemini model id sanitizing. Pure (no `vscode` import) so it is unit tested;
// config.ts applies it to the `modelvisio.geminiModel` setting.

export const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";

/** The id is interpolated into the Gemini request URL path, and the setting's
 *  enum is only an editor hint — a hand-edited value can be anything. Accept
 *  plain ids only; anything else falls back to the default model. */
export function sanitizeGeminiModel(value: unknown): string {
  const id = typeof value === "string" ? value.trim() : "";
  return /^[A-Za-z0-9.-]+$/.test(id) ? id : DEFAULT_GEMINI_MODEL;
}
