import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Bundles the WebView into a single nonce-loadable IIFE script (+ worker chunk)
// under media/, which the extension host injects into the custom editor.
export default defineConfig({
  plugins: [react()],
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
