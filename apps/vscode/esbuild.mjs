// Bundles the extension host (src/extension.ts) into one CommonJS file.
// A bundler, not plain tsc, because the workspace packages it uses
// (@modelvisio/ai/proxy, /scrape) ship as ESM TypeScript sources that a CJS
// `require` can't load. Everything except `vscode` (provided by the editor at
// runtime) is inlined, so the .vsix needs no node_modules.
//
//   node esbuild.mjs            one-off build
//   node esbuild.mjs --watch    rebuild on change
//   node esbuild.mjs --minify   smaller output for release
import { build, context } from "esbuild";
import { rmSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const watch = process.argv.includes("--watch");

/** @type {import("esbuild").BuildOptions} */
const options = {
  absWorkingDir: root,
  entryPoints: ["src/extension.ts"],
  outfile: "out/extension.js",
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "node18",
  external: ["vscode"],
  sourcemap: true,
  sourcesContent: false,
  minify: process.argv.includes("--minify"),
  logLevel: "info",
};

// Clear stale per-file output from the old tsc build so it can't be packaged.
rmSync(new URL("./out", import.meta.url), { recursive: true, force: true });

if (watch) {
  const ctx = await context(options);
  await ctx.watch();
} else {
  await build(options);
}
