// Server-side web scraper for the ModelVisio copilot. Fetches a URL, strips it
// to readable text, and returns a compact snippet the AI can ground answers on
// (vendor op-support pages, MLPerf tables, GitHub issues, changelogs).
//
// Runs ONLY on a server/Node/edge runtime — NOT re-exported from index.ts so no
// scraper code enters the browser bundle. Companion to runChatProxy: Gemini's
// google_search tool covers "search the web"; this covers "read THIS url".
//
// Safeguards, in order of priority:
//  1. SSRF: refuse non-http(s) schemes and any host that looks private/link-local
//     (localhost, 127/8, 10/8, 172.16/12, 192.168/16, 169.254/16, ::1, fc00::/7,
//     fe80::/10, and hostnames like "localhost"/"metadata.google.internal").
//  2. Size cap: streams the body but stops reading at MAX_BYTES so a hostile
//     server can't OOM the function.
//  3. Time cap: AbortController deadline so a slow origin can't hold the
//     serverless slot until the platform kills it.
//  4. Content-type: only text/html, text/plain, application/xhtml+xml, and
//     application/json are read; binaries are refused with a clear error.
//  5. Optional allowlist: MODELVISIO_SCRAPE_ALLOWLIST is a comma-separated list
//     of host suffixes; if set, only matching hosts are fetched.

export type ScrapeResult = {
  url: string;
  finalUrl: string;
  title: string;
  text: string;
  truncated: boolean;
  bytes: number;
  contentType: string;
};

export type ScrapeArgs = {
  url: string;
  /** Max bytes read from the body. Default 512 KiB. */
  maxBytes?: number;
  /** Max readable characters returned (post extraction). Default 12 000. */
  maxChars?: number;
  /** Total wall-clock budget in ms. Default 8000. */
  timeoutMs?: number;
  /** Host suffixes (e.g. "docs.nvidia.com"). Empty = no restriction. */
  allowlist?: string[];
  /** User-Agent for the outbound request. */
  userAgent?: string;
};

const DEFAULT_UA =
  "ModelVisio-Scraper/1.0 (+https://github.com/Premchand006/ModelVisio; contact: repo owner)";

const ALLOWED_CT = /^(text\/(html|plain)|application\/(xhtml\+xml|json))\b/i;

/** Recognise obviously non-routable hosts so we never make an SSRF request. */
export function isPrivateHost(host: string): boolean {
  const h = host.toLowerCase().replace(/^\[|\]$/g, "");
  if (!h) return true;
  const blockedNames = new Set([
    "localhost",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",
    "metadata",
  ]);
  if (blockedNames.has(h)) return true;
  if (h.endsWith(".localhost") || h.endsWith(".internal") || h.endsWith(".local")) return true;
  // IPv6 loopback / link-local / ULA
  if (h === "::1" || h === "0:0:0:0:0:0:0:1") return true;
  if (/^fe[89ab][0-9a-f]:/i.test(h)) return true; // fe80::/10
  if (/^f[cd][0-9a-f]{2}:/i.test(h)) return true; // fc00::/7
  // IPv4 literals
  const v4 = h.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (v4) {
    const [a, b] = [parseInt(v4[1], 10), parseInt(v4[2], 10)];
    if (a === 10) return true;
    if (a === 127) return true;
    if (a === 0) return true;
    if (a === 169 && b === 254) return true;
    if (a === 172 && b >= 16 && b <= 31) return true;
    if (a === 192 && b === 168) return true;
    if (a >= 224) return true; // multicast + reserved
  }
  return false;
}

function hostAllowed(host: string, allowlist: string[] | undefined): boolean {
  if (!allowlist || allowlist.length === 0) return true;
  const h = host.toLowerCase();
  return allowlist.some((suffix) => {
    const s = suffix.toLowerCase().trim().replace(/^\./, "");
    return s !== "" && (h === s || h.endsWith("." + s));
  });
}

/** Extract readable text from an HTML string. Deliberately lightweight — no
 *  DOM: strips <script>/<style>/<noscript>/<template>, unwraps other tags,
 *  decodes the handful of named entities that matter, collapses whitespace. */
export function htmlToText(html: string): { title: string; text: string } {
  let s = html;
  // Title first, before scripts are stripped.
  const titleMatch = s.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const title = titleMatch ? decodeEntities(titleMatch[1]).trim().replace(/\s+/g, " ") : "";
  // Drop non-content blocks entirely.
  s = s.replace(/<!--[\s\S]*?-->/g, " ");
  s = s.replace(/<(script|style|noscript|template|svg|iframe)[^>]*>[\s\S]*?<\/\1>/gi, " ");
  // Turn block-level breaks into newlines so paragraphs survive.
  s = s.replace(/<\/(p|div|section|article|li|tr|h[1-6]|br)\s*>/gi, "\n");
  s = s.replace(/<br\s*\/?>(?=[^<]*<)/gi, "\n");
  // Drop remaining tags.
  s = s.replace(/<[^>]+>/g, " ");
  s = decodeEntities(s);
  // Collapse whitespace.
  s = s.replace(/[ \t\f\v]+/g, " ");
  s = s.replace(/ *\n */g, "\n");
  s = s.replace(/\n{3,}/g, "\n\n");
  return { title, text: s.trim() };
}

const NAMED_ENTITIES: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " ",
  ndash: "-",
  mdash: "-",
  hellip: "...",
  copy: "(c)",
  reg: "(R)",
  trade: "(TM)",
};

function decodeEntities(s: string): string {
  return s.replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (m, body: string) => {
    if (body[0] === "#") {
      const code = body[1] === "x" || body[1] === "X"
        ? parseInt(body.slice(2), 16)
        : parseInt(body.slice(1), 10);
      if (Number.isFinite(code) && code > 0 && code < 0x110000) {
        try { return String.fromCodePoint(code); } catch { return m; }
      }
      return m;
    }
    return NAMED_ENTITIES[body.toLowerCase()] ?? m;
  });
}

/** Read at most `maxBytes` of a Response body. Works on Node 18+ (Undici),
 *  Vercel, Netlify and Vite's dev server. */
async function readBounded(res: Response, maxBytes: number): Promise<{ buf: Uint8Array; truncated: boolean }> {
  const body = res.body;
  if (!body) {
    const text = await res.text();
    const buf = new TextEncoder().encode(text);
    return { buf: buf.slice(0, maxBytes), truncated: buf.byteLength > maxBytes };
  }
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  let truncated = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    if (!value) continue;
    if (total + value.byteLength > maxBytes) {
      chunks.push(value.slice(0, maxBytes - total));
      total = maxBytes;
      truncated = true;
      try { await reader.cancel(); } catch { /* ignore */ }
      break;
    }
    chunks.push(value);
    total += value.byteLength;
  }
  const buf = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
  return { buf, truncated };
}

/**
 * Fetch a URL and return a compact, readable snippet safe to feed to an LLM.
 * Never throws for a normal HTTP failure — returns a rejected Promise with a
 * short human-readable message that the proxy relays to the client.
 */
export async function scrapeUrl(args: ScrapeArgs): Promise<ScrapeResult> {
  const {
    url,
    maxBytes = 512 * 1024,
    maxChars = 12_000,
    timeoutMs = 8000,
    allowlist,
    userAgent = DEFAULT_UA,
  } = args;

  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`Invalid URL: ${url}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`Refusing non-http(s) scheme: ${parsed.protocol}`);
  }
  if (isPrivateHost(parsed.hostname)) {
    throw new Error(`Refusing private/loopback host: ${parsed.hostname}`);
  }
  if (!hostAllowed(parsed.hostname, allowlist)) {
    throw new Error(`Host not in MODELVISIO_SCRAPE_ALLOWLIST: ${parsed.hostname}`);
  }

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(parsed.toString(), {
      method: "GET",
      redirect: "follow",
      signal: ctrl.signal,
      headers: {
        "user-agent": userAgent,
        accept: "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1",
        "accept-language": "en",
      },
    });
  } catch (e) {
    clearTimeout(timer);
    if (e instanceof Error && e.name === "AbortError") {
      throw new Error(`Scrape timed out after ${timeoutMs}ms: ${url}`);
    }
    throw new Error(`Scrape fetch failed: ${e instanceof Error ? e.message : String(e)}`);
  }
  clearTimeout(timer);

  // If redirected, re-check the final host: the origin may have bounced us
  // toward something private (SSRF via redirect).
  const finalUrl = res.url || parsed.toString();
  try {
    const finalHost = new URL(finalUrl).hostname;
    if (isPrivateHost(finalHost)) throw new Error(`Refusing redirect to private host: ${finalHost}`);
    if (!hostAllowed(finalHost, allowlist)) throw new Error(`Redirect host not allowlisted: ${finalHost}`);
  } catch (e) {
    throw e instanceof Error ? e : new Error(String(e));
  }

  if (!res.ok) {
    throw new Error(`Scrape returned HTTP ${res.status} for ${finalUrl}`);
  }
  const contentType = res.headers.get("content-type") || "";
  if (!ALLOWED_CT.test(contentType)) {
    throw new Error(`Refusing content-type "${contentType || "(none)"}" for ${finalUrl}`);
  }

  const { buf, truncated } = await readBounded(res, maxBytes);
  const raw = new TextDecoder("utf-8", { fatal: false }).decode(buf);

  let title = "";
  let text = "";
  if (/html|xhtml/i.test(contentType)) {
    ({ title, text } = htmlToText(raw));
  } else if (/json/i.test(contentType)) {
    try {
      text = JSON.stringify(JSON.parse(raw), null, 2);
    } catch {
      text = raw;
    }
  } else {
    text = raw;
  }

  let outTruncated = truncated;
  if (text.length > maxChars) {
    text = text.slice(0, maxChars);
    outTruncated = true;
  }

  return {
    url,
    finalUrl,
    title,
    text,
    truncated: outTruncated,
    bytes: buf.byteLength,
    contentType,
  };
}

/** Extract http(s) URLs from a free-form message. Used by the copilot's
 *  research step: any URL the user pastes can be scraped and grounded on. */
export function extractUrls(text: string): string[] {
  const rx = /https?:\/\/[^\s<>"'\)\]]+/gi;
  const out: string[] = [];
  const seen = new Set<string>();
  for (const m of text.matchAll(rx)) {
    let u = m[0];
    // Trim trailing punctuation people commonly type after a URL.
    u = u.replace(/[.,;:!?]+$/, "");
    if (!seen.has(u)) { seen.add(u); out.push(u); }
  }
  return out;
}
