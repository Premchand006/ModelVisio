#!/usr/bin/env node
// Tell IndexNow search engines (Bing, which also feeds Copilot, plus Yandex,
// Seznam, Naver and others) that the website's pages changed. Run it after a
// production deploy:
//
//   pnpm indexnow            # submit every URL in the live sitemap
//   pnpm indexnow --dry-run  # show what would be submitted
//
// The key is public by design: IndexNow verifies it by fetching
// https://<host>/<key>.txt, which ships from apps/web/public/.

import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = "https://modelvisio.netlify.app";
const dryRun = process.argv.includes("--dry-run");

const keyFile = readdirSync(join(dirname(fileURLToPath(import.meta.url)), "../apps/web/public")).find((f) => /^[0-9a-f]{32}\.txt$/.test(f));
if (!keyFile) throw new Error("No IndexNow key file (32 hex chars + .txt) in apps/web/public/");
const key = keyFile.slice(0, -4);

const liveKey = await fetch(`${SITE}/${keyFile}`).then((r) => (r.ok ? r.text() : ""));
if (liveKey.trim() !== key) {
  console.error(`The key file is not live at ${SITE}/${keyFile} yet. Deploy first, then rerun.`);
  process.exit(1);
}

const sitemap = await fetch(`${SITE}/sitemap.xml`).then((r) => r.text());
const urlList = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
if (!urlList.length) throw new Error("The live sitemap lists no URLs.");

console.log(`${dryRun ? "Would submit" : "Submitting"} ${urlList.length} URLs:\n  ${urlList.join("\n  ")}`);
if (dryRun) process.exit(0);

const res = await fetch("https://api.indexnow.org/indexnow", {
  method: "POST",
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify({ host: new URL(SITE).host, key, keyLocation: `${SITE}/${keyFile}`, urlList }),
});
// 200 = accepted, 202 = accepted (key validation pending).
console.log(`IndexNow: HTTP ${res.status}${res.ok ? " (accepted)" : ` ${await res.text()}`}`);
process.exit(res.ok ? 0 : 1);
