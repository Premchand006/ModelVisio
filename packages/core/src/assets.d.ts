// Asset imports resolved by the consuming app's bundler (Vite). Declared here so
// `tsc` in the standalone core package accepts them.
declare module "*.png" {
  const src: string;
  export default src;
}
declare module "*.svg" {
  const src: string;
  export default src;
}

// Build-time env (Vite). Declared so core typechecks standalone; merges with
// vite/client's ImportMetaEnv where that's present (web / vscode webview builds).
interface ImportMetaEnv {
  /** Legacy Windows-installer override (same as VITE_DESKTOP_WIN_URL). */
  readonly VITE_DESKTOP_URL?: string;
  readonly VITE_DESKTOP_MAC_URL?: string;
  readonly VITE_DESKTOP_WIN_URL?: string;
  readonly VITE_DESKTOP_LINUX_URL?: string;
  readonly VITE_DESKTOP_RELEASES_URL?: string;
  readonly VITE_VSCODE_URL?: string;
  readonly VITE_GITHUB_URL?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
