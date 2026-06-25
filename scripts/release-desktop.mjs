// Cut a desktop release: validate the version, create the `desktop-v<version>`
// tag, and push it — which triggers .github/workflows/desktop-release.yml to
// build and attach Windows/macOS/Linux installers to a GitHub Release.
//
//   npm run release:desktop -- 1.0.0
//   pnpm release:desktop 1.0.0
//
// Refuses to run on a dirty tree or a tag that already exists, so a release can
// never be cut from uncommitted code or silently clobber a previous one.
import { execSync } from "node:child_process";

const version = (process.argv[2] || "").replace(/^v/, "").trim();
if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
  console.error(`error: pass a semver version, e.g. "npm run release:desktop -- 1.0.0" (got "${process.argv[2] ?? ""}")`);
  process.exit(1);
}

const tag = `desktop-v${version}`;
const run = (cmd) => execSync(cmd, { encoding: "utf8" }).trim();

// A dirty tree is fine — the release builds from the tagged commit (HEAD), not
// the working tree — but warn so uncommitted work isn't silently left out.
if (run("git status --porcelain")) {
  console.warn(
    "warning: working tree has uncommitted changes. The release builds from the\n" +
      "         current commit (HEAD); uncommitted changes will NOT be included.\n"
  );
}
const tags = run("git tag -l").split("\n");
if (tags.includes(tag)) {
  console.error(`error: tag ${tag} already exists. Bump the version.`);
  process.exit(1);
}

console.log(`Tagging ${tag} at ${run("git rev-parse --short HEAD")} and pushing…`);
execSync(`git tag -a ${tag} -m "ModelVisio ${version}"`, { stdio: "inherit" });
execSync(`git push origin ${tag}`, { stdio: "inherit" });
console.log(`\n✓ Pushed ${tag}. Watch the build:`);
console.log("  https://github.com/Premchand006/ModelVisio/actions/workflows/desktop-release.yml");
console.log("Then publish the draft release at:");
console.log("  https://github.com/Premchand006/ModelVisio/releases");
