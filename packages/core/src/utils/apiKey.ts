// Bring-your-own-key storage for the desktop AI copilot. On the web build the
// Gemini key lives on the server-side /api/chat proxy, so this is used only by
// the desktop (Tauri) shell: the user pastes their own free Gemini key, it is
// kept in localStorage on their machine, and the desktop proxy (apps/web/src/
// tauri.ts) reads it and passes it to the Rust `chat` command. No key ships in
// the app, and the key never leaves the user's device except to call Gemini.

/** localStorage key holding the user's own Gemini API key (desktop only). */
export const GEMINI_KEY_STORAGE = "mv_gemini_key";

/** True when running inside the Tauri (desktop) WebView, where the user supplies
 *  their own key. The browser build uses the hosted proxy instead, so it never
 *  shows the key field. */
export function isDesktop(): boolean {
  return typeof window !== "undefined" && "__TAURI__" in window;
}

/** The user's stored Gemini key, or "" if none set. Never throws. */
export function getUserApiKey(): string {
  try {
    return (typeof localStorage !== "undefined" && localStorage.getItem(GEMINI_KEY_STORAGE)) || "";
  } catch {
    return "";
  }
}

/** Persist the user's Gemini key, or clear it when blank. Never throws. */
export function setUserApiKey(key: string): void {
  try {
    const k = key.trim();
    if (k) localStorage.setItem(GEMINI_KEY_STORAGE, k);
    else localStorage.removeItem(GEMINI_KEY_STORAGE);
  } catch {
    // ignore — storage unavailable (private mode, etc.)
  }
}
