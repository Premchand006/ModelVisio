import { describe, it, expect, afterEach, vi } from "vitest";
import { scrapeUrl, htmlToText, isPrivateHost, extractUrls } from "../src/scrape";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; vi.restoreAllMocks(); });

function mockFetch(body: string, init: { status?: number; contentType?: string; url?: string } = {}) {
  globalThis.fetch = vi.fn(async () => {
    return new Response(body, {
      status: init.status ?? 200,
      headers: { "content-type": init.contentType ?? "text/html; charset=utf-8" },
    }) as Response & { url: string };
  }) as unknown as typeof fetch;
}

describe("isPrivateHost", () => {
  it("blocks loopback, RFC1918, link-local, IPv6 loopback/ULA", () => {
    for (const h of [
      "localhost", "app.localhost", "metadata.google.internal",
      "127.0.0.1", "127.13.1.7", "10.0.0.1", "172.16.5.4", "172.31.0.1",
      "192.168.1.1", "169.254.169.254", "0.0.0.0", "::1", "fe80::1", "fd12:abcd::1",
    ]) expect(isPrivateHost(h)).toBe(true);
  });
  it("allows normal public hosts", () => {
    for (const h of ["example.com", "docs.nvidia.com", "8.8.8.8", "172.32.0.1"])
      expect(isPrivateHost(h)).toBe(false);
  });
});

describe("htmlToText", () => {
  it("extracts title and drops script/style", () => {
    const { title, text } = htmlToText(
      `<html><head><title>Op Support</title><style>b{}</style></head>` +
      `<body><script>alert(1)</script><h1>Ops</h1><p>Conv2D &amp; ReLU</p></body></html>`,
    );
    expect(title).toBe("Op Support");
    expect(text).not.toMatch(/alert/);
    expect(text).toMatch(/Ops/);
    expect(text).toMatch(/Conv2D & ReLU/);
  });
  it("decodes numeric and named entities", () => {
    const { text } = htmlToText("<p>&#8734; &amp; &nbsp;x&hellip;</p>");
    expect(text).toContain("∞");
    expect(text).toContain("&");
    expect(text).toContain("x...");
  });
});

describe("extractUrls", () => {
  it("finds URLs and trims trailing punctuation", () => {
    expect(extractUrls("see https://docs.nvidia.com/tensorrt/, and https://hailo.ai/x.")).toEqual([
      "https://docs.nvidia.com/tensorrt/",
      "https://hailo.ai/x",
    ]);
  });
  it("dedupes", () => {
    expect(extractUrls("a https://x.io b https://x.io")).toEqual(["https://x.io"]);
  });
});

describe("scrapeUrl", () => {
  it("refuses non-http(s) schemes", async () => {
    await expect(scrapeUrl({ url: "file:///etc/passwd" })).rejects.toThrow(/non-http/);
  });
  it("refuses private hosts (SSRF guard)", async () => {
    await expect(scrapeUrl({ url: "http://127.0.0.1/admin" })).rejects.toThrow(/private/i);
    await expect(scrapeUrl({ url: "http://169.254.169.254/latest/meta-data/" })).rejects.toThrow(/private/i);
  });
  it("refuses disallowed content-type", async () => {
    mockFetch("PK\x03\x04binary", { contentType: "application/zip" });
    await expect(scrapeUrl({ url: "https://example.com/a.zip" })).rejects.toThrow(/content-type/i);
  });
  it("enforces allowlist", async () => {
    mockFetch("<html><title>ok</title><body>hi</body></html>");
    await expect(
      scrapeUrl({ url: "https://evil.com/", allowlist: ["docs.nvidia.com"] }),
    ).rejects.toThrow(/allowlist/i);
  });
  it("returns readable text on a good response", async () => {
    mockFetch("<html><title>TensorRT Ops</title><body><p>Conv, Gemm, Softmax</p></body></html>");
    const r = await scrapeUrl({ url: "https://docs.nvidia.com/tensorrt/" });
    expect(r.title).toBe("TensorRT Ops");
    expect(r.text).toMatch(/Conv, Gemm, Softmax/);
    expect(r.contentType).toMatch(/text\/html/);
    expect(r.truncated).toBe(false);
  });
  it("truncates when maxChars is exceeded", async () => {
    const big = "<html><body>" + "abc ".repeat(5000) + "</body></html>";
    mockFetch(big);
    const r = await scrapeUrl({ url: "https://example.com/", maxChars: 200 });
    expect(r.text.length).toBe(200);
    expect(r.truncated).toBe(true);
  });
  it("passes through JSON, pretty-printed", async () => {
    mockFetch(`{"a":1,"b":[2,3]}`, { contentType: "application/json" });
    const r = await scrapeUrl({ url: "https://api.example.com/x" });
    expect(r.text).toContain("\"a\": 1");
  });
});
