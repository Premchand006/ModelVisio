// Distribution links shown on the landing page. Override at build time with
// VITE_DESKTOP_URL / VITE_VSCODE_URL (e.g. set them in the Vercel/Netlify project
// env). These defaults are PLACEHOLDERS — replace the org/publisher before going
// live, or the buttons will 404.
//
// Desktop: points at the GitHub Releases page produced by
//   .github/workflows/desktop-release.yml (tauri-action) when you push a tag.
// VS Code: the Marketplace item URL once the extension is published (vsce publish).

const env = import.meta.env;

export const DESKTOP_DOWNLOAD_URL: string =
  env?.VITE_DESKTOP_URL || "https://github.com/YOUR-ORG/modelvisio/releases/latest";

export const VSCODE_MARKETPLACE_URL: string =
  env?.VITE_VSCODE_URL || "https://marketplace.visualstudio.com/items?itemName=modelvisio.modelvisio-vscode";
