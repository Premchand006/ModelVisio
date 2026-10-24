// Bundles the CLI into one self-contained CommonJS file (dist/modelvisio.cjs).
// A bundler rather than tsc because the workspace packages it uses ship as ESM
// TypeScript sources with extensionless imports, which Node can't load as-is.
// Everything (parsers, protobufjs, the pure core modules) is inlined, so the
// output runs with plain `node` on Windows, Linux and macOS, with no install.
//
//   node build.mjs            build dist/modelvisio.cjs
//   node build.mjs --minify   smaller output
import { build } from "esbuild";
import { fileURLToPath } from "node:url";

const options = {
  absWorkingDir: fileURLToPath(new URL(".", import.meta.url)),
  entryPoints: ["src/main.ts"],
  outfile: "dist/modelvisio.cjs",
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "node18",
  banner: { js: "#!/usr/bin/env node" },
  sourcemap: false,
  minify: process.argv.includes("--minify"),
  logLevel: "info",
};

await build(options);
