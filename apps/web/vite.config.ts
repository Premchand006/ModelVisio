import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, type PluginOption } from "vite";
import react from "@vitejs/plugin-react";
import { runChatProxy } from "@modelvisio/ai/proxy";
import { runGrokChat } from "@modelvisio/ai/grok";
import { chatProviderFromEnv } from "@modelvisio/ai/providers";
import { chatWithWebResearch, freeSearchApplies, parseFreeSearchMode } from "@modelvisio/ai/search";
import { logChat, logClientEvent } from "@modelvisio/ai/log";
import { handleScrapeRequest, scrapeOptionsFromEnv, scrapeUrl } from "@modelvisio/ai/scrape";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");

/**
 * Dev-only middleware that answers POST /api/chat during `pnpm dev`, so the AI
 * copilot works locally WITHOUT deploying or running `vercel dev`. It mirrors
 * the production serverless proxy (apps/web/api/chat.ts) and reads the provider
 * key (GEMINI_API_KEY, or XAI_API_KEY for Grok) from the repo-root `.env`. The
 * key stays on the Node dev server and is never sent to the browser. In
 * production the real serverless function handles this route.
 */
function chatProxyDev(env: Record<string, string | undefined>): PluginOption {
  const { provider, key, model, missingKeyError } = chatProviderFromEnv(env);
  const webSearch = (env.MODELVISIO_WEB_SEARCH || "on") !== "off";
  const freeSearch = parseFreeSearchMode(env.MODELVISIO_FREE_SEARCH, "auto");
  const thinking = (env.MODELVISIO_THINKING || "off") === "on";
  const scrape = scrapeOptionsFromEnv(env);
  return {
    name: "modelvisio-chat-proxy-dev",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/api/chat", async (req, res) => {
        res.setHeader("content-type", "application/json");
        if (req.method !== "POST") { res.statusCode = 405; res.end(JSON.stringify({ error: "Method not allowed" })); return; }
        if (!key) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: `${missingKeyError} Add it to .env at the repo root and restart \`pnpm dev\`.` }));
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
          const system: string = body.system ?? "";
          const messages: unknown[] = body.messages ?? [];
          const logP = logChat(system, messages, { ip, user_agent: req.headers["user-agent"] ?? null });
          const run = (sys: string) =>
            provider === "grok"
              ? runGrokChat({ key, system: sys, messages, model, thinking })
              : runChatProxy({ key, system: sys, messages, model, webSearch, thinking });
          const out = freeSearchApplies(provider, webSearch, freeSearch)
            ? await chatWithWebResearch({
                system, messages, mode: freeSearch, searxngUrl: env.MODELVISIO_SEARXNG_URL || undefined,
                read: (url, limits) => scrapeUrl({ url, ...scrape, ...limits }),
                onError: (m) => server.config.logger.warn(`[modelvisio] ${m}`),
              }, run)
            : await run(system);
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

/**
 * Dev-only middleware that answers POST /api/scrape during `pnpm dev`, mirroring
 * the production serverless scrape proxy (apps/web/api/scrape.ts and netlify/
 * functions/scrape.ts). Delegates to the shared SSRF-guarded, size-capped
 * scraper in @modelvisio/ai/scrape. No scraper code enters the browser bundle.
 */
function scrapeProxyDev(env: Record<string, string | undefined>): PluginOption {
  return {
    name: "modelvisio-scrape-proxy-dev",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/api/scrape", async (req, res) => {
        res.setHeader("content-type", "application/json");
        if (req.method !== "POST") { res.statusCode = 405; res.end(JSON.stringify({ error: "Method not allowed" })); return; }
        let body: unknown;
        try {
          const chunks: Buffer[] = [];
          for await (const c of req) chunks.push(c as Buffer);
          body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
        } catch {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: "Request body must be JSON." }));
          return;
        }
        const out = await handleScrapeRequest(body, env);
        res.statusCode = out.status;
        res.end(JSON.stringify(out.body));
      });
    },
  };
}

/**
 * Keep onnxruntime-web's GPU binary (ort-wasm-simd-threaded.asyncify.wasm,
 * ~27 MB) out of the build: it is over Cloudflare Pages' 25 MiB per-file limit.
 * Core's GPU benchmark never lets ORT resolve that URL (it downloads the binary
 * from a pinned CDN URL, checks its SHA-256 and passes the bytes), so hiding
 * the `new URL(…, import.meta.url)` from Vite's asset handling only stops the
 * copy. The CPU build's 14 MB binary is still emitted and served locally.
 */
function ortGpuWasmExternal(): PluginOption {
  return {
    name: "modelvisio:ort-gpu-wasm-external",
    enforce: "pre",
    transform(code, id) {
      if (!/onnxruntime-web[\/]dist[\/][^\/]+\.mjs$/.test(id.split("?")[0])) return null;
      const out = code.replace(
        /new URL\((["'])(ort-wasm-simd-threaded\.(?:asyncify|jsep|jspi)\.wasm)\1\s*,\s*import\.meta\.url\)/g,
        "new URL($1$2$1, self.location.href)",
      );
      return out === code ? null : { code: out, map: null };
    },
  };
}

export default defineConfig(({ mode }) => {
  // Load all vars (no VITE_ prefix filter) from the repo-root .env for the dev proxy.
  const env = loadEnv(mode, repoRoot, "");
  // Repo-root .env wins over the shell env; a blank .env entry falls back to it.
  const merged = { ...process.env, ...Object.fromEntries(Object.entries(env).filter(([, v]) => v)) };
  return {
    plugins: [
      react(),
      chatProxyDev(merged),
      scrapeProxyDev(merged),
    ],
    envDir: repoRoot,
    server: { port: 5173 },
    // onnxruntime-web (core's benchmark worker) locates its .wasm with
    // new URL(…, import.meta.url); dev pre-bundling moves the JS to .vite/deps
    // without the .wasm, so serve it from node_modules unbundled.
    optimizeDeps: { exclude: ["onnxruntime-web"] },
    worker: { plugins: () => [ortGpuWasmExternal()] },
  };
});
