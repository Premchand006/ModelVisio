// WebView HTML + Content-Security-Policy. Pure (no `vscode` import) so the exact
// policy — which the WebView bundle depends on — is pinned by a unit test.

import { randomBytes } from "node:crypto";

/** Fresh per-render nonce; the only way a <script> may run in the WebView. */
export function makeNonce(): string {
  return randomBytes(18).toString("base64");
}

/**
 * The editor's CSP. Notes on the non-obvious sources:
 *  - connect-src cspSource: the WebView fetches its parse-worker chunk from
 *    media/assets/ and starts it from a blob: URL (worker-src blob:), since a
 *    vscode-webview origin can't construct a Worker from a resource URL.
 *  - connect-src https:: core's copilot may fetch public assets; the AI and
 *    scraper calls themselves go through the host bridge, never the network.
 *  - style-src 'unsafe-inline': React inline styles + injected CSS.
 *  - script-src 'unsafe-eval': protobufjs (ONNX parser) compiles its decoders
 *    with Function(); a blob: worker inherits this policy, so without it ONNX
 *    fails to parse both in the worker and on the main-thread fallback. Script
 *    *loading* stays nonce-only, so this does not admit injected <script>s.
 */
export function buildCsp(cspSource: string, nonce: string): string {
  return [
    `default-src 'none'`,
    `img-src ${cspSource} data: blob:`,
    `script-src 'nonce-${nonce}' 'unsafe-eval'`,
    `style-src ${cspSource} 'unsafe-inline' https://fonts.googleapis.com`,
    `font-src ${cspSource} https://fonts.gstatic.com data:`,
    `connect-src ${cspSource} https:`,
    `worker-src ${cspSource} blob:`,
  ].join("; ");
}

const escapeAttr = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export type HtmlArgs = {
  cspSource: string;
  nonce: string;
  /** webview URI of media/webview.js */
  scriptUri: string;
  /** webview URIs of stylesheets to link, if the build emitted any. */
  styleUris?: string[];
};

export function buildHtml({ cspSource, nonce, scriptUri, styleUris = [] }: HtmlArgs): string {
  const links = styleUris.map((u) => `\n    <link rel="stylesheet" href="${escapeAttr(u)}" />`).join("");
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="${escapeAttr(buildCsp(cspSource, nonce))}" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />${links}
    <style>html,body,#root{height:100%;margin:0;padding:0}</style>
  </head>
  <body>
    <div id="root"></div>
    <script nonce="${escapeAttr(nonce)}" src="${escapeAttr(scriptUri)}"></script>
  </body>
</html>`;
}
