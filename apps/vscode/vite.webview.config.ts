import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// The webview's Worker shim (webview/workerShim.ts) prefetches each worker
// script and runs it from a blob, so it needs the emitted worker file names —
// which carry a content hash only known at the end of the build. Splice them
// into the placeholder in webview.js, and fail the build if a worker chunk
// isn't self-contained: a blob worker can't load further scripts under the
// webview's nonce-only script-src.
const PLACEHOLDER = /(["'`])__MODELVISIO_WORKER_ASSETS__\1/g;
function workerAssets(): Plugin {
  return {
    name: "modelvisio:worker-assets",
    apply: "build",
    enforce: "post",
    generateBundle(_opts, bundle) {
      const workers = Object.values(bundle).filter((f) => f.type === "asset" && f.fileName.endsWith(".js"));
      for (const w of workers) {
        const src = typeof w.source === "string" ? w.source : new TextDecoder().decode(w.source);
        if (/^\s*(import|export)\b[\s{*"'`]|\bimport\s*\(|\bimportScripts\s*\(/m.test(src)) {
          this.error(`${w.fileName} is not self-contained (imports another module); the webview Worker shim can't load it.`);
        }
      }
      const list = JSON.stringify(JSON.stringify(workers.map((w) => w.fileName)));
      for (const f of Object.values(bundle)) {
        if (f.type === "chunk") f.code = f.code.replace(PLACEHOLDER, list);
      }
    },
  };
}

// Bundles the WebView into a single nonce-loadable IIFE script (+ worker chunk)
// under media/, which the extension host injects into the custom editor.
export default defineConfig({
  plugins: [react(), workerAssets()],
  // Lib mode leaves process.env.NODE_ENV in place (React reads it), and a
  // webview has no `process` — the bundle would die with a ReferenceError.
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  // Emit asset URLs in JS as plain "./assets/…" so core's
  // `new URL(…, import.meta.url)` resolves against webview.js (→ media/assets/…).
  // The default base ("/") would resolve to the CDN origin's root, and base "./"
  // makes vite read document.currentScript lazily — null by the time the worker
  // is created, so it falls back to the vscode-webview:// document URL.
  // import.meta.url itself is safe: rollup's IIFE output captures currentScript
  // once, at load.
  experimental: {
    renderBuiltUrl: (filename, { hostType }) => (hostType === "js" ? `./${filename}` : undefined),
  },
  build: {
    outDir: "media",
    emptyOutDir: true,
    cssCodeSplit: false,
    lib: {
      entry: fileURLToPath(new URL("./webview/main.tsx", import.meta.url)),
      formats: ["iife"],
      name: "ModelVisioWebview",
      fileName: () => "webview.js",
    },
  },
  worker: {
    format: "es",
  },
});
