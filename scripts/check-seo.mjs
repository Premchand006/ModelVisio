#!/usr/bin/env node
// Checks the built web app (apps/web/dist) for the SEO and discovery essentials:
// title/description length, canonical, absolute social image of the right size,
// valid JSON-LD, crawlable static H1, and robots.txt / sitemap.xml / llms.txt /
// manifest / _headers. Run after `pnpm build`; exits non-zero on any failure.
//
//   node scripts/check-seo.mjs [distDir]

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

const dist = resolve(process.argv[2] ?? "apps/web/dist");
const SITE = "https://modelvisio.netlify.app/";
const failures = [];
const check = (ok, msg) => { if (!ok) failures.push(msg); };
const read = (f) => readFileSync(join(dist, f), "utf8");

for (const f of ["index.html", "robots.txt", "sitemap.xml", "llms.txt", "manifest.webmanifest", "_headers", "404.html", "og-image.png"]) {
  check(existsSync(join(dist, f)), `missing ${f}`);
}
if (failures.length) finish();

const html = read("index.html");
const meta = (attr, name) =>
  html.match(new RegExp(`<meta\\s+${attr}="${name}"\\s+content="([^"]*)"`, "i"))?.[1];
const decode = (s) => s.replace(/&amp;/g, "&");

const title = decode(html.match(/<title>([^<]*)<\/title>/i)?.[1] ?? "");
check(title.length >= 20 && title.length <= 60, `<title> should be 20-60 chars (is ${title.length}): "${title}"`);
check(/ModelVisio/.test(title), "<title> should contain the brand name");

const desc = meta("name", "description") ?? "";
check(desc.length >= 70 && desc.length <= 160, `meta description should be 70-160 chars (is ${desc.length})`);

check(html.includes(`<link rel="canonical" href="${SITE}"`), `canonical should be ${SITE}`);

for (const [attr, name] of [["property", "og:image"], ["name", "twitter:image"]]) {
  const v = meta(attr, name) ?? "";
  check(v.startsWith("https://"), `${name} must be an absolute https URL (is "${v}")`);
}
check(meta("name", "twitter:card") === "summary_large_image", "twitter:card should be summary_large_image");

// PNG IHDR: width and height are big-endian uint32 at bytes 16 and 20.
const png = readFileSync(join(dist, "og-image.png"));
const [w, h] = [png.readUInt32BE(16), png.readUInt32BE(20)];
check(w === 1200 && h === 630, `og-image.png should be 1200x630 (is ${w}x${h})`);
check(png.length < 5 * 1024 * 1024, "og-image.png should be under 5 MB");

const blocks = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
check(blocks.length > 0, "no JSON-LD block in index.html");
for (const [, json] of blocks) {
  try {
    const data = JSON.parse(json);
    const nodes = data["@graph"] ?? [data];
    const ids = new Set(nodes.map((n) => n["@id"]).filter(Boolean));
    const refs = JSON.stringify(data).match(/"@id":"[^"]+"/g) ?? [];
    for (const r of refs) check(ids.has(r.slice(7, -1)), `JSON-LD reference ${r} has no matching node`);
    check(nodes.some((n) => /Application$/.test(n["@type"])), "JSON-LD should describe the app (WebApplication/SoftwareApplication)");
  } catch (e) {
    failures.push(`JSON-LD does not parse: ${e.message}`);
  }
}

// Crawlers that don't run JS must still see a heading and real text.
const root = html.match(/<div id="root">([\s\S]*)<\/div>\s*<\/body>/)?.[1] ?? "";
check(/<h1[\s>]/.test(root), "static #root content should contain an <h1>");
check(root.replace(/<[^>]+>/g, " ").split(/\s+/).filter(Boolean).length >= 150, "static #root content should have at least 150 words");

const robots = read("robots.txt");
check(/^Sitemap:\s*https:\/\/\S+\/sitemap\.xml$/m.test(robots), "robots.txt should point to the absolute sitemap URL");
check(!/^Disallow:\s*\/\s*$/m.test(robots), "robots.txt must not disallow the whole site");

const sitemap = read("sitemap.xml");
check(sitemap.includes(`<loc>${SITE}</loc>`), `sitemap.xml should list ${SITE}`);

// Every other sitemap URL is a generated reference page (apps/web/seo/pages.ts):
// it must exist, be self-canonical, carry a unique title and description, an
// H1, real content, valid JSON-LD, and link back to the app.
const titles = new Set([title]);
const locs = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]).filter((u) => u !== SITE);
check(locs.length > 0, "sitemap.xml should list the generated reference pages");
for (const url of locs) {
  const path = url.slice(SITE.length);
  const file = join(dist, path, "index.html");
  if (!existsSync(file)) { failures.push(`${path}: listed in sitemap.xml but ${file} is missing`); continue; }
  const page = readFileSync(file, "utf8");
  const t = decode(page.match(/<title>([^<]*)<\/title>/i)?.[1] ?? "");
  check(t.length >= 20 && t.length <= 60 && t.includes("ModelVisio"), `${path}: <title> should be 20-60 chars with the brand (is ${t.length}): "${t}"`);
  check(!titles.has(t), `${path}: duplicate <title> "${t}"`);
  titles.add(t);
  const d = decode(page.match(/<meta name="description" content="([^"]*)"/i)?.[1] ?? "");
  check(d.length >= 70 && d.length <= 160, `${path}: meta description should be 70-160 chars (is ${d.length})`);
  check(page.includes(`<link rel="canonical" href="${url}"`), `${path}: canonical should be ${url}`);
  check((page.match(/<h1[\s>]/g) ?? []).length === 1, `${path}: should have exactly one <h1>`);
  const main = page.match(/<main>([\s\S]*)<\/main>/)?.[1] ?? "";
  const words = main.replace(/<[^>]+>/g, " ").split(/\s+/).filter(Boolean).length;
  check(words >= 250, `${path}: main content should have at least 250 words (has ${words})`);
  check(/href="\/"/.test(page), `${path}: should link back to the app`);
  try {
    const ld = JSON.parse(page.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)?.[1] ?? "");
    check(ld["@graph"]?.some((n) => n["@type"] === "BreadcrumbList"), `${path}: JSON-LD should include a BreadcrumbList`);
  } catch (e) {
    failures.push(`${path}: JSON-LD does not parse: ${e.message}`);
  }
}

check(/^# ModelVisio\s*$/m.test(read("llms.txt")), "llms.txt should start with an H1");

try {
  const m = JSON.parse(read("manifest.webmanifest"));
  check(m.name && m.short_name && m.icons?.some((i) => i.sizes === "512x512"), "manifest needs name, short_name and a 512px icon");
} catch (e) {
  failures.push(`manifest.webmanifest does not parse: ${e.message}`);
}

check(/application\/manifest\+json/.test(read("_headers")), "_headers should serve the manifest as application/manifest+json");
check(/<meta name="robots" content="noindex"/.test(read("404.html")), "404.html should be noindex");

finish();

function finish() {
  if (failures.length) {
    console.error(`SEO check failed (${failures.length}):\n` + failures.map((f) => `  - ${f}`).join("\n"));
    process.exit(1);
  }
  console.log(`SEO check passed: ${dist}`);
  process.exit(0);
}
