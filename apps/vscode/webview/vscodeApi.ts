import type { WebviewToHost } from "../src/protocol";

// Minimal VS Code WebView API surface we use. acquireVsCodeApi() may only be
// called once per webview, so it is wrapped in a lazy singleton.
type VsCodeApi = { postMessage: (msg: unknown) => void };
declare function acquireVsCodeApi(): VsCodeApi;

let api: VsCodeApi | null = null;

/** Typed sender for webview → host messages. Injected into the bridges so tests can fake it. */
export type Post = (msg: WebviewToHost) => void;

export const post: Post = (msg) => {
  api ??= acquireVsCodeApi();
  api.postMessage(msg);
};
