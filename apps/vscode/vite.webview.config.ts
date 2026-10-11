import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

// core's live benchmark runs onnxruntime-web in a worker. Lib (IIFE) mode
// inlines every asset, which would put ORT's .wasm into that worker as
// base64. Instead: ortWasmRef hides ORT's `new URL("<x>.wasm", import.meta.url)`
// from Vite's asset handling (core never takes that path: it fetches the
// binary on the main thread and passes the bytes to ORT), workerAssets emits
// the CPU build's binary (~14 MB) once as media/assets/<name>, and the define
// tells core where it is relative to webview.js. The GPU build's ~27 MB binary
// is not packaged: core downloads it from a pinned CDN URL on first use.
const ORT_WASM = "ort-wasm-simd-threaded.wasm";
const coreRequire = createRequire(fileURLToPath(new URL("../../packages/core/package.json", import.meta.url)));
function ortWasmRef(): Plugin {
  return {
    name: "modelvisio:ort-wasm-ref",
    enforce: "pre",
    transform(code, id) {
      if (!/onnxruntime-web[\\/]dist[\\/][^\\/]+\.mjs$/.test(id.split("?")[0])) return null;
      const out = code.replace(/new URL\((["'])(ort-wasm[^"']*\.wasm)\1\s*,\s*import\.meta\.url\)/g, "new URL($1$2$1, self.location.href)");
      return out === code ? null : { code: out, map: null };
    },
  };
}

// The webview's Worker shim (webview/workerShim.ts) prefetches each worker
// script and runs it from a blob, so it needs the emitted worker file names,
// which carry a content hash only known at the end of the build. Splice them
// into the placeholder in webview.js, and fail the build if a worker chunk
// isn't self-contained: a blob worker can't load further scripts under the
// webview's nonce-only script-src. Only string-literal `import("…")` counts as
// loading a module: a dynamic import of a runtime variable (onnxruntime-web's
// optional loader for an external .mjs, which core never triggers) loads
// nothing at build time.
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
        if (/^\s*(import|export)\b[\s{*"'`]|\bimport\s*\(\s*["'`]|\bimportScripts\s*\(/m.test(src)) {
          this.error(`${w.fileName} is not self-contained (imports another module); the webview Worker shim can't load it.`);
        }
      }
      if (workers.some((w) => String(w.source).includes("ONNX Runtime Web"))) {
        this.emitFile({ type: "asset", fileName: `assets/${ORT_WASM}`, source: readFileSync(coreRequire.resolve(`onnxruntime-web/${ORT_WASM}`)) });
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
  // webview has no `process`, so the bundle would die with a ReferenceError.
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
    // Read by core's bench client (packages/core/src/bench/benchClient.ts).
    __MODELVISIO_ORT_WASM__: JSON.stringify(`./assets/${ORT_WASM}`),
  },
  // Emit asset URLs in JS as plain "./assets/…" so core's
  // `new URL(…, import.meta.url)` resolves against webview.js (→ media/assets/…).
  // The default base ("/") would resolve to the CDN origin's root, and base "./"
  // makes vite read document.currentScript lazily, and it is null by the time the worker
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
    plugins: () => [ortWasmRef()],
  },
});
