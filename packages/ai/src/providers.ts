// Which copilot provider a server-side proxy uses, read from its environment.
// Shared by the Vercel function, the Netlify function and the Vite dev
// middleware. Pure, no imports, so Node can load it with type-stripping.
//
//   MODELVISIO_PROVIDER    gemini | grok. Unset: gemini if a Gemini key is
//                          set, else grok if XAI_API_KEY is set, else gemini.
//   GEMINI_API_KEY         (or GOOGLE_API_KEY) Gemini key
//   XAI_API_KEY            xAI Grok key
//   MODELVISIO_MODEL       Gemini model (default gemini-2.5-flash)
//   MODELVISIO_GROK_MODEL  Grok model   (default grok-4.7)

export type ChatProvider = "gemini" | "grok";

export const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";
export const DEFAULT_GROK_MODEL = "grok-4.7";

type Env = Record<string, string | undefined>;

export type ProviderConfig = {
  provider: ChatProvider;
  /** Undefined when the chosen provider has no key — answer with `missingKeyError`. */
  key: string | undefined;
  model: string;
  missingKeyError: string;
};

export function parseProvider(value: unknown): ChatProvider | null {
  const v = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (v === "gemini" || v === "google") return "gemini";
  if (v === "grok" || v === "xai") return "grok";
  return null;
}

export function chatProviderFromEnv(env: Env): ProviderConfig {
  const geminiKey = env.GEMINI_API_KEY || env.GOOGLE_API_KEY || undefined;
  const xaiKey = env.XAI_API_KEY || undefined;
  const provider = parseProvider(env.MODELVISIO_PROVIDER) ?? (geminiKey ? "gemini" : xaiKey ? "grok" : "gemini");
  if (provider === "grok") {
    return {
      provider,
      key: xaiKey,
      model: env.MODELVISIO_GROK_MODEL || DEFAULT_GROK_MODEL,
      missingKeyError: "XAI_API_KEY is not configured on the server (copilot provider: Grok). Get a key at https://console.x.ai .",
    };
  }
  return {
    provider,
    key: geminiKey,
    model: env.MODELVISIO_MODEL || DEFAULT_GEMINI_MODEL,
    missingKeyError: "GEMINI_API_KEY is not configured on the server. Free key: https://aistudio.google.com/apikey .",
  };
}
