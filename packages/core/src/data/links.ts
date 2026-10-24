// Distribution links shown on the landing page. Each is overridable at build
// time with a Vite env var (e.g. set in the Vercel/Netlify project env):
//   VITE_DESKTOP_MAC_URL / VITE_DESKTOP_WIN_URL / VITE_DESKTOP_LINUX_URL,
//   VITE_DESKTOP_RELEASES_URL, VITE_VSCODE_URL, VITE_GITHUB_URL.
// The legacy VITE_DESKTOP_URL is still honored as the Windows override.
//
// Desktop: STABLE, version-independent GitHub Release assets. The release
//   workflow uploads each installer under a fixed name, and
//   `/releases/latest/download/<name>` always resolves to the newest PUBLISHED
//   release, so the landing page one-click-downloads the right installer for the
//   visitor's OS and the URLs never change between versions:
//     macOS   ModelVisio-macOS-universal.dmg      (one build: Apple Silicon + Intel)
//     Windows ModelVisio-Windows-x64-setup.exe
//     Linux   ModelVisio-Linux-x86_64.AppImage
//   Everything else (.msi, .deb, .rpm) is on the releases page. The asset URLs
//   404 until the first release is published (publish the draft that
//   `pnpm release:desktop` creates).
// VS Code: the Marketplace item URL once the extension is published (vsce publish).

const env = import.meta.env;

const RELEASES = "https://github.com/Premchand006/ModelVisio/releases";
const latestAsset = (name: string) => `${RELEASES}/latest/download/${name}`;

/** Stable release asset file names (uploaded by the desktop release workflow). */
export const DESKTOP_ASSET_NAMES = {
  mac: "ModelVisio-macOS-universal.dmg",
  windows: "ModelVisio-Windows-x64-setup.exe",
  linux: "ModelVisio-Linux-x86_64.AppImage",
} as const;

export const DESKTOP_MAC_URL: string =
  env?.VITE_DESKTOP_MAC_URL || latestAsset(DESKTOP_ASSET_NAMES.mac);

export const DESKTOP_WIN_URL: string =
  env?.VITE_DESKTOP_WIN_URL || env?.VITE_DESKTOP_URL || latestAsset(DESKTOP_ASSET_NAMES.windows);

export const DESKTOP_LINUX_URL: string =
  env?.VITE_DESKTOP_LINUX_URL || latestAsset(DESKTOP_ASSET_NAMES.linux);

/** Latest release page with every installer (.msi, .deb, .rpm, …). */
export const DESKTOP_RELEASES_URL: string =
  env?.VITE_DESKTOP_RELEASES_URL || `${RELEASES}/latest`;

/** One downloadable desktop build, as listed in the landing page's download panel. */
export type DesktopBuild = { os: "mac" | "windows" | "linux"; label: string; url: string; file: string; arch: string };

/** Desktop builds in display order. */
export const DESKTOP_BUILDS: readonly DesktopBuild[] = [
  { os: "mac", label: "macOS", url: DESKTOP_MAC_URL, file: DESKTOP_ASSET_NAMES.mac, arch: "Universal: Apple Silicon & Intel" },
  { os: "windows", label: "Windows", url: DESKTOP_WIN_URL, file: DESKTOP_ASSET_NAMES.windows, arch: "x64 installer · .msi on Releases" },
  { os: "linux", label: "Linux", url: DESKTOP_LINUX_URL, file: DESKTOP_ASSET_NAMES.linux, arch: "x86_64 AppImage · .deb/.rpm on Releases" },
];

/** @deprecated Windows installer; use the per-platform constants above. */
export const DESKTOP_DOWNLOAD_URL: string = DESKTOP_WIN_URL;

export const VSCODE_MARKETPLACE_URL: string =
  env?.VITE_VSCODE_URL || "https://marketplace.visualstudio.com/items?itemName=modelvisio.modelvisio-vscode";

/** Public source repository. */
export const GITHUB_URL: string =
  env?.VITE_GITHUB_URL || "https://github.com/Premchand006/ModelVisio";

/** Step-by-step macOS install guide (docs/install-macos.md). */
export const MAC_INSTALL_GUIDE_URL = `${GITHUB_URL}/blob/main/docs/install-macos.md`;

/** Project author / copyright holder (GitHub handle). */
export const AUTHOR = "Premchand006";
