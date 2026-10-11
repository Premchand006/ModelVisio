import { describe, expect, it } from "vitest";
import { buildCsp, buildHtml, makeNonce } from "../src/html";

const SRC = "https://file+.vscode-resource.vscode-cdn.net";

describe("CSP", () => {
  it("is exactly the policy the WebView bundle depends on", () => {
    expect(buildCsp(SRC, "N0nce")).toBe(
      `default-src 'none'; img-src ${SRC} data: blob:; script-src 'nonce-N0nce' 'unsafe-eval'; ` +
        `style-src ${SRC} 'unsafe-inline' https://fonts.googleapis.com; ` +
        `font-src ${SRC} https://fonts.gstatic.com data:; connect-src ${SRC} https:; worker-src ${SRC} blob:`,
    );
  });
});

describe("makeNonce", () => {
  it("is random and CSP-safe base64", () => {
    const a = makeNonce();
    expect(a).toMatch(/^[A-Za-z0-9+/]{24}$/);
    expect(makeNonce()).not.toBe(a);
  });
});

describe("buildHtml", () => {
  const html = buildHtml({
    cspSource: SRC,
    nonce: "abc",
    scriptUri: `${SRC}/ext/media/webview.js`,
    styleUris: [`${SRC}/ext/media/style.css`],
  });

  it("loads the bundle with the nonce into a bare root", () => {
    expect(html).toContain(`<script nonce="abc" src="${SRC}/ext/media/webview.js"></script>`);
    expect(html).toContain(`<div id="root"></div>`);
    expect(html).toContain(`<link rel="stylesheet" href="${SRC}/ext/media/style.css" />`);
    expect(html).toContain(`content="${buildCsp(SRC, "abc")}"`);
  });

  it("has no inline scripts", () => {
    expect(html.match(/<script/g)).toHaveLength(1);
  });
});
