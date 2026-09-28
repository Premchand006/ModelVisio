import { describe, it, expect, afterEach, vi } from "vitest";
import {
  scrapeUrl, htmlToText, isPrivateHost, extractUrls, handleScrapeRequest, scrapeOptionsFromEnv, ScrapeError,
} from "../src/scrape";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; vi.restoreAllMocks(); });

// Every hostname "resolves" to a public address unless a test says otherwise,
// so no test depends on real DNS.
const publicDns = async () => ["93.184.216.34"];

type MockResp = { body?: BodyInit | null; status?: number; contentType?: string; location?: string };

/** Queue responses for successive fetch calls; returns the mock to inspect calls. */
function mockFetch(...responses: (MockResp | string)[]) {
  const queue = responses.map((r) => (typeof r === "string" ? { body: r } : r));
  const fn = vi.fn(async () => {
    const r = queue.shift() ?? { status: 500, body: "no more mocked responses" };
    const headers: Record<string, string> = { "content-type": r.contentType ?? "text/html; charset=utf-8" };
    if (r.location) headers.location = r.location;
    return new Response(r.body ?? null, { status: r.status ?? 200, headers });
  });
  globalThis.fetch = fn as unknown as typeof fetch;
  return fn;
}

const requestedUrls = (fn: ReturnType<typeof mockFetch>) =>
  fn.mock.calls.map((c) => String((c as unknown[])[0]));

describe("isPrivateHost", () => {
  it("blocks loopback, RFC1918, link-local, IPv6 loopback/ULA", () => {
    for (const h of [
      "localhost", "app.localhost", "metadata.google.internal",
      "127.0.0.1", "127.13.1.7", "10.0.0.1", "172.16.5.4", "172.31.0.1",
      "192.168.1.1", "169.254.169.254", "0.0.0.0", "::1", "fe80::1", "fd12:abcd::1",
    ]) expect(isPrivateHost(h), h).toBe(true);
  });
  it("blocks CGNAT, unspecified, multicast, trailing-dot names and IPv4 embedded in IPv6", () => {
    for (const h of [
      "100.64.0.1", "100.127.255.254", "198.18.0.1", "::", "[::]", "ff02::1", "localhost.",
      "::ffff:127.0.0.1", "::ffff:7f00:1", "[::ffff:a9fe:a9fe]", "0:0:0:0:0:ffff:0a00:0001",
      "64:ff9b::a00:1", "2002:c0a8:0101::1", "::127.0.0.1", "fe80::1%eth0",
    ]) expect(isPrivateHost(h), h).toBe(true);
  });
  it("refuses malformed IPv6-looking hosts", () => {
    expect(isPrivateHost("1:2:3:4:5:6:7:8:9")).toBe(true);
    expect(isPrivateHost("::g")).toBe(true);
  });
  it("allows normal public hosts", () => {
    for (const h of [
      "example.com", "docs.nvidia.com", "8.8.8.8", "172.32.0.1", "100.128.0.1",
      "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808",
    ]) expect(isPrivateHost(h), h).toBe(false);
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
    const { text } = htmlToText("<p>&#8734; &amp; &nbsp;x&hellip; 3&times;3</p>");
    expect(text).toContain("∞");
    expect(text).toContain("&");
    expect(text).toContain("x...");
    expect(text).toContain("3×3");
  });
  it("keeps line breaks and table cells readable", () => {
    const { text } = htmlToText("<p>a<br>b</p><table><tr><td>Conv</td><td>yes</td></tr></table>");
    expect(text).toMatch(/a\nb/);
    expect(text).toMatch(/Conv \| yes/);
  });
  it("does not treat <header> as <head>", () => {
    const { text } = htmlToText("<header>Nav</header><p>Body</p>");
    expect(text).toMatch(/Nav/);
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
  it("keeps balanced parens but drops an unbalanced closing one", () => {
    expect(extractUrls("https://en.wikipedia.org/wiki/Mish_(function)")).toEqual([
      "https://en.wikipedia.org/wiki/Mish_(function)",
    ]);
    expect(extractUrls("(see https://x.io/a).")).toEqual(["https://x.io/a"]);
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
  it("refuses alternate encodings of loopback that URL parsing normalises", async () => {
    const fn = mockFetch("<p>should never be fetched</p>");
    for (const url of ["http://2130706433/", "http://0x7f.1/", "http://[::ffff:127.0.0.1]/", "http://localhost./"]) {
      await expect(scrapeUrl({ url, resolveHost: publicDns }), url).rejects.toThrow(/private/i);
    }
    expect(fn).not.toHaveBeenCalled();
  });
  it("refuses a hostname that resolves to a private address, without fetching", async () => {
    const fn = mockFetch("<p>internal</p>");
    await expect(
      scrapeUrl({ url: "https://127.0.0.1.nip.io/", resolveHost: async () => ["93.184.216.34", "10.0.0.5"] }),
    ).rejects.toThrow(/resolves to private address 10\.0\.0\.5/);
    expect(fn).not.toHaveBeenCalled();
  });
  it("reports unresolvable hosts", async () => {
    await expect(
      scrapeUrl({ url: "https://nope.invalid/", resolveHost: async () => { throw new Error("ENOTFOUND"); } }),
    ).rejects.toThrow(/Could not resolve host/);
  });
  it("refuses disallowed content-type", async () => {
    mockFetch({ body: "PK\x03\x04binary", contentType: "application/zip" });
    await expect(scrapeUrl({ url: "https://example.com/a.zip", resolveHost: publicDns })).rejects.toThrow(/content-type/i);
  });
  it("enforces allowlist", async () => {
    mockFetch("<html><title>ok</title><body>hi</body></html>");
    await expect(
      scrapeUrl({ url: "https://evil.com/", allowlist: ["docs.nvidia.com"], resolveHost: publicDns }),
    ).rejects.toThrow(/allowlist/i);
  });
  it("returns readable text on a good response", async () => {
    mockFetch("<html><title>TensorRT Ops</title><body><p>Conv, Gemm, Softmax</p></body></html>");
    const r = await scrapeUrl({ url: "https://docs.nvidia.com/tensorrt/", resolveHost: publicDns });
    expect(r.title).toBe("TensorRT Ops");
    expect(r.text).toMatch(/Conv, Gemm, Softmax/);
    expect(r.contentType).toMatch(/text\/html/);
    expect(r.truncated).toBe(false);
    expect(r.finalUrl).toBe("https://docs.nvidia.com/tensorrt/");
  });
  it("truncates when maxChars is exceeded", async () => {
    mockFetch("<html><body>" + "abc ".repeat(5000) + "</body></html>");
    const r = await scrapeUrl({ url: "https://example.com/", maxChars: 200, resolveHost: publicDns });
    expect(r.text.length).toBe(200);
    expect(r.truncated).toBe(true);
  });
  it("stops reading at maxBytes", async () => {
    mockFetch({ body: "x".repeat(10_000), contentType: "text/plain" });
    const r = await scrapeUrl({ url: "https://example.com/big.txt", maxBytes: 1000, resolveHost: publicDns });
    expect(r.bytes).toBe(1000);
    expect(r.truncated).toBe(true);
  });
  it("passes through JSON, pretty-printed", async () => {
    mockFetch({ body: `{"a":1,"b":[2,3]}`, contentType: "application/json" });
    const r = await scrapeUrl({ url: "https://api.example.com/x", resolveHost: publicDns });
    expect(r.text).toContain("\"a\": 1");
  });
});

describe("scrapeUrl redirects", () => {
  it("follows a redirect to a public host and reports the final URL", async () => {
    const fn = mockFetch(
      { status: 301, location: "/v2/ops" },
      "<html><title>Ops v2</title><body>Conv</body></html>",
    );
    const r = await scrapeUrl({ url: "https://docs.example.com/ops", resolveHost: publicDns });
    expect(r.finalUrl).toBe("https://docs.example.com/v2/ops");
    expect(r.title).toBe("Ops v2");
    expect(requestedUrls(fn)).toEqual(["https://docs.example.com/ops", "https://docs.example.com/v2/ops"]);
  });
  it("refuses a redirect to a private host BEFORE requesting it", async () => {
    const fn = mockFetch({ status: 302, location: "http://169.254.169.254/latest/meta-data/" }, "<p>secrets</p>");
    await expect(scrapeUrl({ url: "https://evil.example/", resolveHost: publicDns }))
      .rejects.toThrow(/redirect to private/i);
    expect(fn).toHaveBeenCalledTimes(1);
  });
  it("refuses a redirect to a non-allowlisted host", async () => {
    const fn = mockFetch({ status: 302, location: "https://evil.com/" }, "<p>x</p>");
    await expect(
      scrapeUrl({ url: "https://docs.nvidia.com/a", allowlist: ["nvidia.com"], resolveHost: publicDns }),
    ).rejects.toThrow(/allowlist.*via redirect/i);
    expect(fn).toHaveBeenCalledTimes(1);
  });
  it("refuses a redirect to a non-http scheme", async () => {
    mockFetch({ status: 302, location: "file:///etc/passwd" });
    await expect(scrapeUrl({ url: "https://example.com/", resolveHost: publicDns })).rejects.toThrow(/non-http/);
  });
  it("gives up after maxRedirects", async () => {
    mockFetch(...Array.from({ length: 5 }, (_, i) => ({ status: 302, location: `/hop${i + 1}` })));
    const err = await scrapeUrl({ url: "https://example.com/", maxRedirects: 3, resolveHost: publicDns }).catch((e) => e);
    expect(err).toBeInstanceOf(ScrapeError);
    expect(err.message).toMatch(/Too many redirects/);
    expect(err.status).toBe(502);
  });
});

describe("scrapeUrl deadline + decoding", () => {
  it("times out a body that never finishes (slow-drip origin)", async () => {
    const stalled = new ReadableStream<Uint8Array>({
      start(c) { c.enqueue(new TextEncoder().encode("<p>first chunk")); }, // never closes
    });
    mockFetch({ body: stalled });
    const err = await scrapeUrl({ url: "https://slow.example/", timeoutMs: 50, resolveHost: publicDns }).catch((e) => e);
    expect(err).toBeInstanceOf(ScrapeError);
    expect(err.message).toMatch(/timed out after 50ms/);
    expect(err.status).toBe(502);
  });
  it("decodes the Content-Type charset", async () => {
    mockFetch({ body: new Uint8Array([0x63, 0x61, 0x66, 0xe9]), contentType: "text/plain; charset=windows-1252" });
    const r = await scrapeUrl({ url: "https://example.com/", resolveHost: publicDns });
    expect(r.text).toBe("café");
  });
  it("falls back to an HTML <meta charset>", async () => {
    const head = new TextEncoder().encode('<html><head><meta charset="iso-8859-1"><title>x</title></head><body>');
    const body = new Uint8Array([...head, 0x6e, 0x61, 0xef, 0x76, 0x65]);
    mockFetch({ body, contentType: "text/html" });
    const r = await scrapeUrl({ url: "https://example.com/", resolveHost: publicDns });
    expect(r.text).toContain("naïve");
  });
});

describe("handleScrapeRequest", () => {
  it("400s on a missing url", async () => {
    expect(await handleScrapeRequest({}, {})).toEqual({ status: 400, body: { error: expect.stringMatching(/Missing `url`/) } });
    expect((await handleScrapeRequest(null, {})).status).toBe(400);
  });
  it("400s on a blocked host and 502s on an origin 5xx", async () => {
    expect((await handleScrapeRequest({ url: "http://10.0.0.1/" }, {})).status).toBe(400);
    mockFetch({ status: 503, body: "down" });
    // IP-literal URL: skips DNS, so the default resolver never runs in tests.
    const out = await handleScrapeRequest({ url: "https://93.184.216.34/" }, {});
    expect(out.status).toBe(502);
    expect(out.body).toEqual({ error: expect.stringMatching(/HTTP 503/) });
  });
  it("applies env limits", async () => {
    mockFetch({ body: "y".repeat(500), contentType: "text/plain" });
    const out = await handleScrapeRequest({ url: "https://93.184.216.34/" }, { MODELVISIO_SCRAPE_MAX_CHARS: "100" });
    expect(out.status).toBe(200);
    expect((out.body as { text: string }).text.length).toBe(100);
  });
});

describe("scrapeOptionsFromEnv", () => {
  it("parses the allowlist and numeric limits, ignoring junk", () => {
    expect(scrapeOptionsFromEnv({
      MODELVISIO_SCRAPE_ALLOWLIST: " docs.nvidia.com, ,hailo.ai ",
      MODELVISIO_SCRAPE_MAX_BYTES: "2048",
      MODELVISIO_SCRAPE_MAX_CHARS: "abc",
      MODELVISIO_SCRAPE_TIMEOUT_MS: "-5",
    })).toEqual({ allowlist: ["docs.nvidia.com", "hailo.ai"], maxBytes: 2048, maxChars: undefined, timeoutMs: undefined });
    expect(scrapeOptionsFromEnv({}).allowlist).toBeUndefined();
  });
});
