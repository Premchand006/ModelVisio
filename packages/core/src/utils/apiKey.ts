// Bring-your-own-key storage for the desktop AI copilot. On the web build the
// provider keys live on the server-side /api/chat proxy, so this is used only
// by the desktop (Tauri) shell: the user picks a provider (Gemini or Grok),
// pastes their own key for it, it is kept in localStorage on their machine,
// and the desktop proxy (apps/web/src/tauri.ts) reads it and passes it to the
// Rust `chat` command. No key ships in the app, and a key never leaves the
// user's device except to call its own provider.

/** Copilot providers the desktop shell can call with the user's own key. */
export type AiProvider = "gemini" | "grok";

/** localStorage key holding the user's own Gemini API key (desktop only). */
export const GEMINI_KEY_STORAGE = "mv_gemini_key";
/** localStorage key holding the user's own xAI Grok API key (desktop only). */
export const GROK_KEY_STORAGE = "mv_grok_key";
/** localStorage key holding the selected copilot provider (desktop only). */
export const AI_PROVIDER_STORAGE = "mv_ai_provider";

const KEY_STORAGE: Record<AiProvider, string> = {
  gemini: GEMINI_KEY_STORAGE,
  grok: GROK_KEY_STORAGE,
};

/** True when running inside the Tauri (desktop) WebView, where the user supplies
 *  their own key. The browser build uses the hosted proxy instead, so it never
 *  shows the key field. */
export function isDesktop(): boolean {
  return typeof window !== "undefined" && "__TAURI__" in window;
}

/** True when running inside the VS Code extension's WebView. VS Code defines
 *  the global `acquireVsCodeApi` before any page script runs (the extension
 *  calls it lazily), so this works from first render. */
export function isVsCodeWebview(): boolean {
  return typeof (globalThis as { acquireVsCodeApi?: unknown }).acquireVsCodeApi === "function";
}

/** The selected copilot provider; "gemini" unless the user picked Grok. Never throws. */
export function getAiProvider(): AiProvider {
  try {
    const v = typeof localStorage !== "undefined" ? localStorage.getItem(AI_PROVIDER_STORAGE) : null;
    return v === "grok" ? "grok" : "gemini";
  } catch {
    return "gemini";
  }
}

/** Persist the selected copilot provider. Never throws. */
export function setAiProvider(provider: AiProvider): void {
  try {
    localStorage.setItem(AI_PROVIDER_STORAGE, provider === "grok" ? "grok" : "gemini");
  } catch {
    // ignore — storage unavailable (private mode, etc.)
  }
}

/** The user's stored key for `provider` (default: the selected one), or "" if
 *  none set. Never throws. */
export function getUserApiKey(provider: AiProvider = getAiProvider()): string {
  try {
    return (typeof localStorage !== "undefined" && localStorage.getItem(KEY_STORAGE[provider])) || "";
  } catch {
    return "";
  }
}

/** Persist the user's key for `provider` (default: the selected one), or clear
 *  it when blank. Never throws. */
export function setUserApiKey(key: string, provider: AiProvider = getAiProvider()): void {
  try {
    const k = key.trim();
    if (k) localStorage.setItem(KEY_STORAGE[provider], k);
    else localStorage.removeItem(KEY_STORAGE[provider]);
  } catch {
    // ignore — storage unavailable (private mode, etc.)
  }
}
