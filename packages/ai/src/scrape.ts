// Server-side web scraper for the ModelVisio copilot. Fetches a URL, strips it
// to readable text, and returns a compact snippet the AI can ground answers on
// (vendor op-support pages, MLPerf tables, GitHub issues, changelogs).
//
// Runs ONLY on a server/Node runtime — NOT imported by index.ts so no scraper
// code enters the browser bundle (shared browser-safe bits live in urls.ts).
// Companion to runChatProxy: Gemini's google_search tool covers "search the
// web"; this covers "read THIS url".
//
// Safeguards, in order of priority:
//  1. SSRF: refuse non-http(s) schemes and any host that is, or RESOLVES TO, a
//     private/loopback/link-local/CGNAT/multicast address — including IPv4
//     embedded in IPv6 (::ffff:127.0.0.1, 64:ff9b::/96, 2002::/16). Redirects
//     are followed manually so every hop is checked BEFORE it is requested.
//     Residual risk: DNS rebinding between our lookup and fetch's own lookup;
//     closing that needs a pinned-IP dispatcher, which global fetch can't take.
//  2. Size cap: streams the body but stops reading at maxBytes so a hostile
//     server can't OOM the function.
//  3. Time cap: one deadline covers DNS, every redirect hop AND the body read,
//     so a slow-drip origin can't hold the serverless slot until it's killed.
//  4. Content-type: only text/html, text/plain, application/xhtml+xml, and
//     application/json are read; binaries are refused with a clear error.
//  5. Optional allowlist: MODELVISIO_SCRAPE_ALLOWLIST is a comma-separated list
//     of host suffixes; if set, only matching hosts (every hop) are fetched.

import type { ScrapedPage } from "./urls";

export { extractUrls } from "./urls";
export type ScrapeResult = ScrapedPage;

/** Resolves a hostname to its IP addresses (A + AAAA). */
export type HostResolver = (host: string) => Promise<string[]>;

export type ScrapeArgs = {
  url: string;
  /** Max bytes read from the body. Default 512 KiB. */
  maxBytes?: number;
  /** Max readable characters returned (post extraction). Default 12 000. */
  maxChars?: number;
  /** Total wall-clock budget in ms (DNS + redirects + body). Default 8000. */
  timeoutMs?: number;
  /** Host suffixes (e.g. "docs.nvidia.com"). Empty = no restriction. */
  allowlist?: string[];
  /** User-Agent for the outbound request. */
  userAgent?: string;
  /** Max redirect hops followed. Default 5. */
  maxRedirects?: number;
  /**
   * DNS resolver used to reject hostnames that resolve to private addresses.
   * Default: node:dns lookup. `null` skips resolution (literal checks only) —
   * only for runtimes without DNS access, and for tests.
   */
  resolveHost?: HostResolver | null;
};

/** A scrape failure with the HTTP status the proxy should relay:
 *  400 = caller's fault (bad/blocked URL, wrong content-type, origin 4xx),
 *  502 = origin failure (timeout, network error, origin 5xx, redirect loop). */
export class ScrapeError extends Error {
  constructor(message: string, readonly status: 400 | 502 = 400) {
    super(message);
    this.name = "ScrapeError";
  }
}

const DEFAULT_UA =
  "ModelVisio-Scraper/1.0 (+https://github.com/Premchand006/ModelVisio; contact: repo owner)";

const ALLOWED_CT = /^(text\/(html|plain)|application\/(xhtml\+xml|json))\b/i;

// ─── Address classification ──────────────────────────────────────────────────

function isPrivateIPv4(a: number, b: number, c: number): boolean {
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 100 && b >= 64 && b <= 127) return true; // 100.64/10 CGNAT
  if (a === 169 && b === 254) return true; // link-local + cloud metadata
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 192 && b === 0 && (c === 0 || c === 2)) return true; // IETF + TEST-NET-1
  if (a === 198 && (b === 18 || b === 19)) return true; // benchmarking
  if (a === 198 && b === 51 && c === 100) return true; // TEST-NET-2
  if (a === 203 && b === 0 && c === 113) return true; // TEST-NET-3
  return a >= 224; // multicast + reserved + broadcast
}

function parseIPv4(h: string): [number, number, number, number] | null {
  const m = h.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (!m) return null;
  const p = m.slice(1).map((x) => parseInt(x, 10));
  return p.every((x) => x <= 255) ? (p as [number, number, number, number]) : null;
}

/** Parse an IPv6 literal into 8 hextets (handles "::" and a dotted-quad tail). */
function parseIPv6(h: string): number[] | null {
  if (!h.includes(":")) return null;
  let s = h.split("%")[0]; // drop zone id
  const tail = s.match(/:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (tail) {
    const v4 = parseIPv4(tail[1]);
    if (!v4) return null;
    s = s.slice(0, -tail[1].length)
      + ((v4[0] << 8) | v4[1]).toString(16) + ":" + ((v4[2] << 8) | v4[3]).toString(16);
  }
  const halves = s.split("::");
  if (halves.length > 2) return null;
  const part = (x: string) => (x === "" ? [] : x.split(":"));
  const head = part(halves[0]);
  const rest = halves.length === 2 ? part(halves[1]) : [];
  const fill = halves.length === 2 ? 8 - head.length - rest.length : 0;
  if (fill < 0 || (halves.length === 1 && head.length !== 8)) return null;
  const groups = [...head, ...Array(fill).fill("0"), ...rest];
  if (groups.length !== 8 || !groups.every((g) => /^[0-9a-f]{1,4}$/i.test(g))) return null;
  return groups.map((g) => parseInt(g, 16));
}

function isPrivateIPv6(x: number[]): boolean {
  const zeros = (from: number, to: number) => x.slice(from, to).every((v) => v === 0);
  const v4 = (hi: number, lo: number) => isPrivateIPv4(hi >> 8, hi & 0xff, lo >> 8);
  if (zeros(0, 8)) return true; // ::
  if (zeros(0, 7) && x[7] === 1) return true; // ::1
  if ((x[0] & 0xffc0) === 0xfe80) return true; // fe80::/10 link-local
  if ((x[0] & 0xffc0) === 0xfec0) return true; // fec0::/10 site-local (deprecated)
  if ((x[0] & 0xfe00) === 0xfc00) return true; // fc00::/7 ULA
  if ((x[0] & 0xff00) === 0xff00) return true; // ff00::/8 multicast
  if (zeros(0, 5) && (x[5] === 0xffff || x[5] === 0)) return v4(x[6], x[7]); // ::ffff:a.b.c.d, ::a.b.c.d
  if (x[0] === 0x64 && x[1] === 0xff9b && zeros(2, 6)) return v4(x[6], x[7]); // NAT64
  if (x[0] === 0x2002) return v4(x[1], x[2]); // 6to4
  return false;
}

/** Recognise non-routable hosts (names or IP literals) so we never make an
 *  SSRF request. Also used on DNS results, which are plain IP strings. */
export function isPrivateHost(host: string): boolean {
  const h = host.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.$/, "");
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
  const v4 = parseIPv4(h);
  if (v4) return isPrivateIPv4(v4[0], v4[1], v4[2]);
  const v6 = parseIPv6(h);
  if (v6) return isPrivateIPv6(v6);
  // A colon that didn't parse as IPv6 is malformed — refuse rather than guess.
  return h.includes(":");
}

function hostAllowed(host: string, allowlist: string[] | undefined): boolean {
  if (!allowlist || allowlist.length === 0) return true;
  const h = host.toLowerCase().replace(/\.$/, "");
  return allowlist.some((suffix) => {
    const s = suffix.toLowerCase().trim().replace(/^\./, "");
    return s !== "" && (h === s || h.endsWith("." + s));
  });
}

type LookupFn = (host: string, opts: { all: true; verbatim: boolean }) => Promise<{ address: string }[]>;

/** Default resolver: node:dns. The specifier is non-literal so this package
 *  typechecks without @types/node and bundlers leave the builtin alone. */
const nodeResolve: HostResolver = async (host) => {
  const spec = "node:dns/promises";
  const { lookup } = (await import(/* @vite-ignore */ spec)) as { lookup: LookupFn };
  return (await lookup(host, { all: true, verbatim: true })).map((a) => a.address);
};

/** Throw unless `url` is safe to request: http(s), public host, allowlisted,
 *  and (for hostnames) every resolved address public. */
async function assertSafeTarget(
  url: URL,
  allowlist: string[] | undefined,
  resolveHost: HostResolver | null,
  hop: number,
): Promise<void> {
  const what = hop === 0 ? "" : "redirect to ";
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new ScrapeError(`Refusing ${what}non-http(s) scheme: ${url.protocol}`);
  }
  const host = url.hostname;
  if (isPrivateHost(host)) throw new ScrapeError(`Refusing ${what}private/loopback host: ${host}`);
  if (!hostAllowed(host, allowlist)) {
    throw new ScrapeError(`Host not in MODELVISIO_SCRAPE_ALLOWLIST: ${host}${hop ? " (via redirect)" : ""}`);
  }
  const bare = host.replace(/^\[|\]$/g, "");
  if (!resolveHost || parseIPv4(bare) || parseIPv6(bare)) return;
  let addrs: string[];
  try {
    addrs = await resolveHost(host);
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") throw e;
    throw new ScrapeError(`Could not resolve host: ${host}`);
  }
  if (addrs.length === 0) throw new ScrapeError(`Could not resolve host: ${host}`);
  const bad = addrs.find(isPrivateHost);
  if (bad) throw new ScrapeError(`Refusing ${what}host ${host}: resolves to private address ${bad}`);
}

// ─── HTML → text ─────────────────────────────────────────────────────────────

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
  s = s.replace(/<(script|style|noscript|template|svg|iframe|head)\b[^>]*>[\s\S]*?<\/\1>/gi, " ");
  // Turn block-level breaks into newlines so paragraphs survive.
  s = s.replace(/<\/(p|div|section|article|li|tr|h[1-6]|pre|blockquote|table)\s*>/gi, "\n");
  s = s.replace(/<br\s*\/?>/gi, "\n");
  s = s.replace(/<\/t[dh]\s*>/gi, " | ");
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
  times: "×",
  micro: "µ",
  plusmn: "±",
  le: "≤",
  ge: "≥",
  rarr: "→",
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

// ─── Body reading ────────────────────────────────────────────────────────────

const abortError = () => Object.assign(new Error("aborted"), { name: "AbortError" });

/** Reject as soon as `signal` aborts, even if `p` never settles (a mocked or
 *  non-cooperative body stream must not outlive the deadline). */
function abortable<T>(p: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError());
    signal.addEventListener("abort", onAbort, { once: true });
    p.then(
      (v) => { signal.removeEventListener("abort", onAbort); resolve(v); },
      (e) => { signal.removeEventListener("abort", onAbort); reject(e); },
    );
  });
}

/** Read at most `maxBytes` of a Response body, aborting at the deadline. Works
 *  on Node 18+ (Undici), Vercel, Netlify and Vite's dev server. */
async function readBounded(
  res: Response,
  maxBytes: number,
  signal: AbortSignal,
): Promise<{ buf: Uint8Array; truncated: boolean }> {
  const body = res.body;
  if (!body) {
    const buf = new Uint8Array(await abortable(res.arrayBuffer(), signal));
    return { buf: buf.slice(0, maxBytes), truncated: buf.byteLength > maxBytes };
  }
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  let truncated = false;
  try {
    for (;;) {
      const { value, done } = await abortable(reader.read(), signal);
      if (done) break;
      if (!value) continue;
      if (total + value.byteLength > maxBytes) {
        chunks.push(value.slice(0, maxBytes - total));
        total = maxBytes;
        truncated = true;
        break;
      }
      chunks.push(value);
      total += value.byteLength;
    }
  } finally {
    if (truncated || signal.aborted) reader.cancel().catch(() => { /* ignore */ });
  }
  const buf = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
  return { buf, truncated };
}

/** Decode using the Content-Type charset, else an HTML <meta charset>, else
 *  UTF-8. Unknown labels fall back to UTF-8 rather than failing the scrape. */
function decodeBody(buf: Uint8Array, contentType: string): string {
  let label = contentType.match(/charset\s*=\s*"?([\w.:-]+)/i)?.[1];
  if (!label && /html/i.test(contentType)) {
    // <meta> must appear in the first 1024 bytes per the HTML spec; ASCII-safe sniff.
    const head = new TextDecoder("latin1").decode(buf.subarray(0, 1024));
    label = head.match(/<meta[^>]+charset\s*=\s*["']?([\w.:-]+)/i)?.[1];
  }
  try {
    return new TextDecoder(label || "utf-8", { fatal: false }).decode(buf);
  } catch {
    return new TextDecoder("utf-8", { fatal: false }).decode(buf);
  }
}

// ─── Scrape ──────────────────────────────────────────────────────────────────

/**
 * Fetch a URL and return a compact, readable snippet safe to feed to an LLM.
 * Every failure rejects with a ScrapeError whose short message and `status`
 * the proxy relays to the client.
 */
export async function scrapeUrl(args: ScrapeArgs): Promise<ScrapeResult> {
  const {
    url,
    maxBytes = 512 * 1024,
    maxChars = 12_000,
    timeoutMs = 8000,
    allowlist,
    userAgent = DEFAULT_UA,
    maxRedirects = 5,
    resolveHost = nodeResolve,
  } = args;

  let current: URL;
  try {
    current = new URL(url);
  } catch {
    throw new ScrapeError(`Invalid URL: ${url}`);
  }

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    // Follow redirects by hand so each hop passes the SSRF/allowlist checks
    // BEFORE a request is sent to it (redirect:"follow" would already have
    // hit the private host by the time we saw res.url).
    let res!: Response;
    for (let hop = 0; ; hop++) {
      await abortable(assertSafeTarget(current, allowlist, resolveHost, hop), ctrl.signal);
      res = await fetch(current.toString(), {
        method: "GET",
        redirect: "manual",
        signal: ctrl.signal,
        headers: {
          "user-agent": userAgent,
          accept: "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1",
          "accept-language": "en",
        },
      });
      const location = res.status >= 300 && res.status < 400 ? res.headers.get("location") : null;
      if (!location) break;
      res.body?.cancel().catch(() => { /* ignore */ });
      if (hop >= maxRedirects) throw new ScrapeError(`Too many redirects (>${maxRedirects}) for ${url}`, 502);
      try {
        current = new URL(location, current);
      } catch {
        throw new ScrapeError(`Invalid redirect location "${location}" from ${current}`, 502);
      }
    }
    const finalUrl = current.toString();

    if (!res.ok) {
      throw new ScrapeError(`Scrape returned HTTP ${res.status} for ${finalUrl}`, res.status >= 500 ? 502 : 400);
    }
    const contentType = res.headers.get("content-type") || "";
    if (!ALLOWED_CT.test(contentType)) {
      res.body?.cancel().catch(() => { /* ignore */ });
      throw new ScrapeError(`Refusing content-type "${contentType || "(none)"}" for ${finalUrl}`);
    }

    const { buf, truncated } = await readBounded(res, maxBytes, ctrl.signal);
    const raw = decodeBody(buf, contentType);

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

    return { url, finalUrl, title, text, truncated: outTruncated, bytes: buf.byteLength, contentType };
  } catch (e) {
    if (e instanceof ScrapeError) throw e;
    if (ctrl.signal.aborted || (e instanceof Error && e.name === "AbortError")) {
      throw new ScrapeError(`Scrape timed out after ${timeoutMs}ms: ${url}`, 502);
    }
    throw new ScrapeError(`Scrape fetch failed: ${e instanceof Error ? e.message : String(e)}`, 502);
  } finally {
    clearTimeout(timer);
  }
}

// ─── Proxy entrypoint (shared by Vercel, Netlify and the Vite dev server) ────

type Env = Record<string, string | undefined>;

/** Scraper limits from MODELVISIO_SCRAPE_* env vars (unset/invalid → defaults). */
export function scrapeOptionsFromEnv(env: Env): Omit<ScrapeArgs, "url"> {
  const num = (k: string) => {
    const n = Number(env[k]);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  };
  const allowlist = (env.MODELVISIO_SCRAPE_ALLOWLIST || "").split(",").map((s) => s.trim()).filter(Boolean);
  return {
    allowlist: allowlist.length ? allowlist : undefined,
    maxBytes: num("MODELVISIO_SCRAPE_MAX_BYTES"),
    maxChars: num("MODELVISIO_SCRAPE_MAX_CHARS"),
    timeoutMs: num("MODELVISIO_SCRAPE_TIMEOUT_MS"),
  };
}

/** Validate a POST /api/scrape body, run the scraper, and return the JSON
 *  response + status. Never throws — each platform handler just relays it. */
export async function handleScrapeRequest(
  body: unknown,
  env: Env,
): Promise<{ status: number; body: ScrapeResult | { error: string } }> {
  const url = (body as { url?: unknown } | null)?.url;
  if (typeof url !== "string" || !url) {
    return { status: 400, body: { error: "Missing `url` (string) in request body." } };
  }
  try {
    return { status: 200, body: await scrapeUrl({ url, ...scrapeOptionsFromEnv(env) }) };
  } catch (e) {
    const status = e instanceof ScrapeError ? e.status : 500;
    return { status, body: { error: e instanceof Error ? e.message : "Scrape error" } };
  }
}
