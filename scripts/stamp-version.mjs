// Stamp a release version into the desktop app's build config so the produced
// installers are named like `ModelVisio_1.2.3_x64-setup.exe` instead of
// `ModelVisio_0.0.0_...`. Called by the release workflow with the tag version:
//
//   node scripts/stamp-version.mjs 1.2.3
//
// Updates apps/desktop/src-tauri/tauri.conf.json (the version tauri-action reads
// for bundle naming) and apps/desktop/src-tauri/Cargo.toml (the crate version).
// Idempotent and cross-platform (runs the same on the Windows/macOS/Linux CI
// runners). Validates the version is plain semver so a bad tag can't silently
// produce a broken bundle name.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

const raw = process.argv[2];
if (!raw) {
  console.error("usage: node scripts/stamp-version.mjs <version>");
  process.exit(1);
}
// Accept an optional leading `v` / `desktop-v`; keep only the semver core.
const version = raw.replace(/^.*?v/, "").trim();
if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
  console.error(`error: "${raw}" is not a valid semver version (got "${version}")`);
  process.exit(1);
}

const confPath = join(root, "apps/desktop/src-tauri/tauri.conf.json");
const conf = JSON.parse(readFileSync(confPath, "utf8"));
conf.version = version;
writeFileSync(confPath, JSON.stringify(conf, null, 2) + "\n");

const cargoPath = join(root, "apps/desktop/src-tauri/Cargo.toml");
let cargo = readFileSync(cargoPath, "utf8");
// Replace only the first `version = "..."` (the [package] one at the top).
cargo = cargo.replace(/^version = ".*"/m, `version = "${version}"`);
writeFileSync(cargoPath, cargo);

console.log(`stamped version ${version} into tauri.conf.json and Cargo.toml`);
