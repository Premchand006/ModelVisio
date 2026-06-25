// Server-side usage logging shared by every proxy entrypoint: the Vercel
// function (apps/web/api/chat.ts), the Netlify function (netlify/functions/
// chat.ts), and the Vite dev middleware (apps/web/vite.config.ts).
//
// Like proxy.ts this runs ONLY on a server/Node/edge runtime and is
// intentionally NOT re-exported from index.ts, so the Supabase service key path
// never enters the browser bundle. Writes go straight to Supabase's PostgREST
// endpoint with the service-role key, which bypasses RLS — the table has RLS on
// and no policies, so the browser/anon side can neither read nor write it.
//
// Telemetry is strictly best-effort: every failure (no creds, network error,
// bad status) is swallowed so logging can NEVER affect the user-facing request.
// If SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are unset, logging is a no-op.

/** Read a server env var without depending on @types/node in this package. */
function env(name: string): string | undefined {
  return (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.[name];
}

const TABLE = env("SUPABASE_LOG_TABLE") || "usage_logs";

export type UsageLog = {
  kind: "upload" | "chat";
  session_id?: string | null;
  model_name?: string | null;
  format?: string | null;
  framework?: string | null;
  size_bytes?: number | null;
  params?: number | null;
  layers?: number | null;
  query?: string | null;
  user_agent?: string | null;
  ip?: string | null;
};

/** Insert one row into the Supabase log table. Resolves even on failure. */
export async function logEvent(row: UsageLog): Promise<void> {
  const url = env("SUPABASE_URL");
  const key = env("SUPABASE_SERVICE_ROLE_KEY") || env("SUPABASE_KEY");
  if (!url || !key) return; // logging disabled — nothing configured
  try {
    await fetch(`${url.replace(/\/$/, "")}/rest/v1/${TABLE}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        apikey: key,
        authorization: `Bearer ${key}`,
        prefer: "return=minimal",
      },
      body: JSON.stringify(row),
    });
  } catch {
    // Swallow: telemetry must never break or slow the real response.
  }
}

/** First match of `Model: <name> |` from the copilot system prompt, if present. */
function modelNameFromSystem(system: string): string | null {
  const m = system.match(/Model:\s*([^|\n]+?)\s*\|/);
  return m ? m[1].trim() : null;
}

type Turn = { role?: string; content?: unknown };

/** Log a chat turn: the latest user question plus the model it was asked about. */
export async function logChat(
  system: string,
  messages: unknown[],
  extra: { user_agent?: string | null; ip?: string | null } = {},
): Promise<void> {
  const turns = (messages as Turn[]) || [];
  const lastUser = [...turns].reverse().find((m) => m?.role === "user");
  const query = typeof lastUser?.content === "string" ? lastUser.content : null;
  if (!query) return;
  await logEvent({
    kind: "chat",
    query,
    model_name: system ? modelNameFromSystem(system) : null,
    user_agent: extra.user_agent ?? null,
    ip: extra.ip ?? null,
  });
}

/** Shape the client beacon (apps log an `{ event }` payload to /api/chat). */
export type ClientEvent = {
  kind?: string;
  session_id?: string;
  model_name?: string;
  format?: string;
  framework?: string;
  size_bytes?: number;
  params?: number;
  layers?: number;
  user_agent?: string;
};

/** Log a client-sent event (e.g. a model upload), clamping it to known fields. */
export async function logClientEvent(
  ev: ClientEvent,
  extra: { ip?: string | null } = {},
): Promise<void> {
  await logEvent({
    kind: ev.kind === "upload" ? "upload" : "chat",
    session_id: ev.session_id ?? null,
    model_name: ev.model_name ?? null,
    format: ev.format ?? null,
    framework: ev.framework ?? null,
    size_bytes: typeof ev.size_bytes === "number" ? ev.size_bytes : null,
    params: typeof ev.params === "number" ? ev.params : null,
    layers: typeof ev.layers === "number" ? ev.layers : null,
    user_agent: ev.user_agent ?? null,
    ip: extra.ip ?? null,
  });
}
