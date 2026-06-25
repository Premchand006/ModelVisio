import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, type PluginOption } from "vite";
import react from "@vitejs/plugin-react";
import { runChatProxy } from "@modelvisio/ai/proxy";
import { logChat, logClientEvent } from "@modelvisio/ai/log";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");

/**
 * Dev-only middleware that answers POST /api/chat during `pnpm dev`, so the AI
 * copilot works locally WITHOUT deploying or running `vercel dev`. It mirrors
 * the production serverless proxy (apps/web/api/chat.ts) and reads the key from
 * the repo-root `.env`. The key stays on the Node dev server — it is never sent
 * to the browser. In production the real serverless function handles this route.
 */
function chatProxyDev(key: string | undefined, model: string, webSearch: boolean, thinking: boolean): PluginOption {
  return {
    name: "modelvisio-chat-proxy-dev",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/api/chat", async (req, res) => {
        res.setHeader("content-type", "application/json");
        if (req.method !== "POST") { res.statusCode = 405; res.end(JSON.stringify({ error: "Method not allowed" })); return; }
        if (!key) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: "GEMINI_API_KEY is not set. Add it to .env at the repo root (free key: https://aistudio.google.com/apikey) and restart `pnpm dev`." }));
          return;
        }
        try {
          const chunks: Buffer[] = [];
          for await (const c of req) chunks.push(c as Buffer);
          const body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
          const ipHeader = req.headers["x-forwarded-for"];
          const ip = (Array.isArray(ipHeader) ? ipHeader[0] : ipHeader)?.split(",")[0].trim()
            || req.socket?.remoteAddress || null;
          // Client telemetry beacon (e.g. a model upload): record it and return.
          if (body.event) {
            await logClientEvent(body.event, { ip });
            res.statusCode = 200;
            res.end(JSON.stringify({ ok: true }));
            return;
          }
          const { system, messages } = body;
          const logP = logChat(system ?? "", messages ?? [], { ip, user_agent: req.headers["user-agent"] ?? null });
          const out = await runChatProxy({ key, system: system ?? "", messages: messages ?? [], model, webSearch, thinking });
          await logP;
          res.statusCode = 200;
          res.end(JSON.stringify(out));
        } catch (e) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: e instanceof Error ? e.message : "Proxy error" }));
        }
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  // Load all vars (no VITE_ prefix filter) from the repo-root .env for the dev proxy.
  const env = loadEnv(mode, repoRoot, "");
  const key = env.GEMINI_API_KEY || env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  const model = env.MODELVISIO_MODEL || process.env.MODELVISIO_MODEL || "gemini-2.5-flash";
  const webSearch = (env.MODELVISIO_WEB_SEARCH || process.env.MODELVISIO_WEB_SEARCH || "on") !== "off";
  const thinking = (env.MODELVISIO_THINKING || process.env.MODELVISIO_THINKING || "off") === "on";
  return {
    plugins: [react(), chatProxyDev(key, model, webSearch, thinking)],
    envDir: repoRoot,
    server: { port: 5173 },
  };
});
