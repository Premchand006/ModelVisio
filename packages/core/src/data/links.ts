// Distribution links shown on the landing page. Override at build time with
// VITE_DESKTOP_URL / VITE_VSCODE_URL (e.g. set them in the Vercel/Netlify project
// env).
//
// Desktop: a STABLE, version-independent GitHub Release asset — the release
//   workflow (.github/workflows/desktop-release.yml) uploads the Windows
//   installer under the fixed name `ModelVisio-Windows-x64-setup.exe`, and
//   `/releases/latest/download/<name>` always resolves to the newest published
//   release. So the button one-click-downloads the current installer, and the
//   URL never changes between versions. (It 404s until the first release is
//   PUBLISHED — publish the draft that `pnpm release:desktop` creates.)
// VS Code: the Marketplace item URL once the extension is published (vsce publish).

const env = import.meta.env;

export const DESKTOP_DOWNLOAD_URL: string =
  env?.VITE_DESKTOP_URL ||
  "https://github.com/Premchand006/ModelVisio/releases/latest/download/ModelVisio-Windows-x64-setup.exe";

export const VSCODE_MARKETPLACE_URL: string =
  env?.VITE_VSCODE_URL || "https://marketplace.visualstudio.com/items?itemName=modelvisio.modelvisio-vscode";

/** Public source repository. */
export const GITHUB_URL: string =
  env?.VITE_GITHUB_URL || "https://github.com/Premchand006/ModelVisio";

/** Project author / copyright holder (GitHub handle). */
export const AUTHOR = "Premchand006";
