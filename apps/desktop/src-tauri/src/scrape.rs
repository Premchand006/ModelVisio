//! URL reader for the desktop copilot: the Rust port of
//! packages/ai/src/scrape.ts. The web build POSTs `{ url }` to /api/scrape; in
//! the desktop shell apps/web/src/tauri.ts routes that POST to the `scrape`
//! command (lib.rs), which calls `scrape_url` here. The reply has the same JSON
//! shape as the web proxy: a `ScrapedPage` on success, `{ error, status }` on
//! failure (status 400 = the URL's fault, 502 = the origin's).
//!
//! Keep the behaviour in step with scrape.ts: same limits, same env knobs
//! (MODELVISIO_SCRAPE_*), same error messages, same HTML-to-text rules, so a
//! page reads the same on the web and on the desktop.
//!
//! Safeguards, as in scrape.ts:
//!  1. SSRF: only http(s); refuse any host that is, or resolves to, a private /
//!     loopback / link-local / CGNAT / multicast / unspecified / metadata
//!     address, including IPv4 embedded in IPv6. Redirects are followed by hand
//!     so every hop is checked before it is requested. Unlike the Node version,
//!     the connection is pinned to the addresses we checked (a custom reqwest
//!     resolver that only answers for that host), so DNS rebinding between the
//!     check and the connect can't reach a private address. No proxy is used,
//!     since a proxy would do its own lookup.
//!  2. Size cap: stop reading the body at `max_bytes`.
//!  3. Time cap: one deadline covers DNS, every redirect hop and the body.
//!  4. Content-type: only text/html, text/plain, application/xhtml+xml and
//!     application/json are read.
//!  5. Optional allowlist: MODELVISIO_SCRAPE_ALLOWLIST host suffixes, every hop.

use std::borrow::Cow;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use regex::{Captures, Regex};
use serde::Serialize;
use url::{Host, Url};

const DEFAULT_UA: &str =
    "ModelVisio-Scraper/1.0 (+https://github.com/Premchand006/ModelVisio; contact: repo owner)";
const ACCEPT: &str = "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1";

/// Readable result of scraping one URL. Same fields as `ScrapedPage` in
/// packages/ai/src/urls.ts (the /api/scrape response body).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScrapedPage {
    pub url: String,
    pub final_url: String,
    pub title: String,
    pub text: String,
    pub truncated: bool,
    pub bytes: usize,
    pub content_type: String,
}

/// A scrape failure. `error` is the message the chat shows ("Couldn't read
/// <url>: ..."); `status` is the HTTP status the web proxy would have answered
/// with, which tauri.ts uses for the synthetic Response.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScrapeError {
    pub error: String,
    pub status: u16,
}

impl ScrapeError {
    /// The caller's fault: bad/blocked URL, wrong content-type, origin 4xx.
    fn bad(msg: impl Into<String>) -> Self {
        Self { error: msg.into(), status: 400 }
    }
    /// The origin's fault: timeout, network error, origin 5xx, redirect loop.
    fn upstream(msg: impl Into<String>) -> Self {
        Self { error: msg.into(), status: 502 }
    }
}

/// Scraper limits. Defaults match scrape.ts.
#[derive(Clone)]
pub struct ScrapeOptions {
    pub max_bytes: usize,
    pub max_chars: usize,
    pub timeout_ms: u64,
    pub allowlist: Vec<String>,
    pub user_agent: String,
    pub max_redirects: usize,
    /// The SSRF address check. Always `is_private_ip` in the app; tests swap it
    /// to let a loopback test server through while still blocking the rest.
    pub is_blocked_ip: fn(IpAddr) -> bool,
}

impl Default for ScrapeOptions {
    fn default() -> Self {
        Self {
            max_bytes: 512 * 1024,
            max_chars: 12_000,
            timeout_ms: 8000,
            allowlist: Vec::new(),
            user_agent: DEFAULT_UA.to_string(),
            max_redirects: 5,
            is_blocked_ip: is_private_ip,
        }
    }
}

impl ScrapeOptions {
    /// Limits from the MODELVISIO_SCRAPE_* environment variables of the process
    /// that launched the app (unset/invalid → defaults), like
    /// `scrapeOptionsFromEnv` in scrape.ts.
    pub fn from_env() -> Self {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    fn from_vars(get: impl Fn(&str) -> Option<String>) -> Self {
        let num = |k: &str| {
            get(k)
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|n| n.is_finite() && *n > 0.0)
        };
        let mut o = Self::default();
        o.allowlist = get("MODELVISIO_SCRAPE_ALLOWLIST")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if let Some(n) = num("MODELVISIO_SCRAPE_MAX_BYTES") {
            o.max_bytes = n as usize;
        }
        if let Some(n) = num("MODELVISIO_SCRAPE_MAX_CHARS") {
            o.max_chars = n as usize;
        }
        if let Some(n) = num("MODELVISIO_SCRAPE_TIMEOUT_MS") {
            o.timeout_ms = n.ceil() as u64;
        }
        o
    }
}

// ─── Address classification ──────────────────────────────────────────────────

fn is_private_v4_octets(a: u8, b: u8, c: u8) -> bool {
    if a == 0 || a == 10 || a == 127 {
        return true;
    }
    if a == 100 && (64..=127).contains(&b) {
        return true; // 100.64/10 CGNAT
    }
    if a == 169 && b == 254 {
        return true; // link-local + cloud metadata
    }
    if a == 172 && (16..=31).contains(&b) {
        return true;
    }
    if a == 192 && b == 168 {
        return true;
    }
    if a == 192 && b == 0 && (c == 0 || c == 2) {
        return true; // IETF + TEST-NET-1
    }
    if a == 198 && (b == 18 || b == 19) {
        return true; // benchmarking
    }
    if a == 198 && b == 51 && c == 100 {
        return true; // TEST-NET-2
    }
    if a == 203 && b == 0 && c == 113 {
        return true; // TEST-NET-3
    }
    a >= 224 // multicast + reserved + broadcast
}

pub fn is_private_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    is_private_v4_octets(a, b, c)
}

pub fn is_private_ipv6(ip: Ipv6Addr) -> bool {
    let x = ip.segments();
    let zeros = |from: usize, to: usize| x[from..to].iter().all(|&v| v == 0);
    let v4 = |hi: u16, lo: u16| is_private_v4_octets((hi >> 8) as u8, (hi & 0xff) as u8, (lo >> 8) as u8);
    if zeros(0, 8) {
        return true; // ::
    }
    if zeros(0, 7) && x[7] == 1 {
        return true; // ::1
    }
    if (x[0] & 0xffc0) == 0xfe80 {
        return true; // fe80::/10 link-local
    }
    if (x[0] & 0xffc0) == 0xfec0 {
        return true; // fec0::/10 site-local (deprecated)
    }
    if (x[0] & 0xfe00) == 0xfc00 {
        return true; // fc00::/7 unique-local
    }
    if (x[0] & 0xff00) == 0xff00 {
        return true; // ff00::/8 multicast
    }
    if zeros(0, 5) && (x[5] == 0xffff || x[5] == 0) {
        return v4(x[6], x[7]); // ::ffff:a.b.c.d, ::a.b.c.d
    }
    if x[0] == 0x64 && x[1] == 0xff9b && zeros(2, 6) {
        return v4(x[6], x[7]); // NAT64 64:ff9b::/96
    }
    if x[0] == 0x2002 {
        return v4(x[1], x[2]); // 6to4
    }
    false
}

/// True for any address the scraper must never connect to.
pub fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_private_ipv4(v4),
        IpAddr::V6(v6) => is_private_ipv6(v6),
    }
}

/// Host names that are internal by name, whatever they resolve to.
pub fn is_blocked_name(host: &str) -> bool {
    let h = host.trim_end_matches('.').to_ascii_lowercase();
    if h.is_empty() {
        return true;
    }
    const NAMES: &[&str] = &[
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
        "metadata",
    ];
    NAMES.contains(&h.as_str())
        || h.ends_with(".localhost")
        || h.ends_with(".internal")
        || h.ends_with(".local")
}

fn host_allowed(host: &str, allowlist: &[String]) -> bool {
    if allowlist.is_empty() {
        return true;
    }
    let h = host.trim_end_matches('.').to_ascii_lowercase();
    allowlist.iter().any(|suffix| {
        let s = suffix.trim().trim_start_matches('.').to_ascii_lowercase();
        !s.is_empty() && (h == s || h.ends_with(&format!(".{s}")))
    })
}

/// The address set a request is pinned to: `host` (as it appears in the URL,
/// without IPv6 brackets) may only connect to `addrs`.
#[derive(Debug, Clone)]
struct Pin {
    host: String,
    addrs: Vec<SocketAddr>,
}

/// Reject a hostname whose resolved addresses include a blocked one (the
/// message names the first offender, like scrape.ts).
fn check_resolved(
    host: &str,
    addrs: &[IpAddr],
    redirect: bool,
    is_blocked: fn(IpAddr) -> bool,
) -> Result<(), ScrapeError> {
    if addrs.is_empty() {
        return Err(ScrapeError::bad(format!("Could not resolve host: {host}")));
    }
    if let Some(bad) = addrs.iter().find(|ip| is_blocked(**ip)) {
        let what = if redirect { "redirect to " } else { "" };
        return Err(ScrapeError::bad(format!(
            "Refusing {what}host {host}: resolves to private address {bad}"
        )));
    }
    Ok(())
}

/// The checks that need no network: scheme, literal/named private hosts,
/// allowlist. `hop` > 0 marks a redirect target in the messages.
fn check_static(url: &Url, opts: &ScrapeOptions, hop: usize) -> Result<(), ScrapeError> {
    let what = if hop == 0 { "" } else { "redirect to " };
    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(ScrapeError::bad(format!("Refusing {what}non-http(s) scheme: {scheme}:")));
    }
    let shown = url.host_str().unwrap_or("");
    let private = match url.host() {
        None => true,
        Some(Host::Domain(d)) => is_blocked_name(d),
        Some(Host::Ipv4(ip)) => (opts.is_blocked_ip)(IpAddr::V4(ip)),
        Some(Host::Ipv6(ip)) => (opts.is_blocked_ip)(IpAddr::V6(ip)),
    };
    if private {
        return Err(ScrapeError::bad(format!("Refusing {what}private/loopback host: {shown}")));
    }
    if !host_allowed(shown, &opts.allowlist) {
        let via = if hop > 0 { " (via redirect)" } else { "" };
        return Err(ScrapeError::bad(format!(
            "Host not in MODELVISIO_SCRAPE_ALLOWLIST: {shown}{via}"
        )));
    }
    Ok(())
}

/// Every check for one hop, then the addresses to pin the connection to.
async fn assert_safe_target(url: &Url, opts: &ScrapeOptions, hop: usize) -> Result<Pin, ScrapeError> {
    check_static(url, opts, hop)?;
    let port = url.port_or_known_default().unwrap_or(80);
    match url.host() {
        Some(Host::Ipv4(ip)) => Ok(Pin { host: ip.to_string(), addrs: vec![SocketAddr::new(ip.into(), port)] }),
        Some(Host::Ipv6(ip)) => Ok(Pin { host: ip.to_string(), addrs: vec![SocketAddr::new(ip.into(), port)] }),
        Some(Host::Domain(d)) => {
            let not_resolved = || ScrapeError::bad(format!("Could not resolve host: {d}"));
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((d, port))
                .await
                .map_err(|_| not_resolved())?
                .collect();
            let ips: Vec<IpAddr> = addrs.iter().map(|a| a.ip()).collect();
            check_resolved(d, &ips, hop > 0, opts.is_blocked_ip)?;
            Ok(Pin { host: d.to_string(), addrs })
        }
        None => Err(ScrapeError::bad("Refusing private/loopback host: ")),
    }
}

/// reqwest resolver that only answers for the pinned host, with the addresses
/// that passed the SSRF check. Any other lookup fails, so there is no path to
/// an unchecked address (and no second, rebindable DNS query).
struct PinnedResolver(Pin);

impl reqwest::dns::Resolve for PinnedResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let norm = |s: &str| s.trim_end_matches('.').trim_matches(['[', ']']).to_ascii_lowercase();
        let result: Result<reqwest::dns::Addrs, Box<dyn std::error::Error + Send + Sync>> =
            if norm(name.as_str()) == norm(&self.0.host) {
                Ok(Box::new(self.0.addrs.clone().into_iter()))
            } else {
                Err(format!("lookup of {} refused (not the checked host)", name.as_str()).into())
            };
        Box::pin(async move { result })
    }
}

fn client_for(pin: Pin) -> Result<reqwest::Client, ScrapeError> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .dns_resolver(Arc::new(PinnedResolver(pin)))
        .build()
        .map_err(|e| ScrapeError::upstream(format!("Scrape fetch failed: {}", error_chain(&e))))
}

/// "outer: inner: root" so a connect/TLS failure says why, not just "error
/// sending request".
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        let msg = s.to_string();
        if !out.contains(&msg) {
            out.push_str(": ");
            out.push_str(&msg);
        }
        src = s.source();
    }
    out
}

// ─── HTML → text ─────────────────────────────────────────────────────────────

macro_rules! re {
    ($pat:expr) => {{
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new($pat).expect("valid regex"))
    }};
}

/// Remove `<script>…</script>` and friends in one left-to-right pass, the same
/// way scrape.ts's `/<(script|…)\b[^>]*>[\s\S]*?<\/\1>/gi` does. An opening tag
/// with no matching close is left for the generic tag stripper. Once a tag name
/// has no close after some point it can't have one later either, so each name
/// is searched to the end at most once (linear, unlike the backtracking regex).
fn strip_blocks(s: &str) -> String {
    let open = re!(r"(?i)<(script|style|noscript|template|svg|iframe|head)\b[^>]*>");
    let lower = s.to_ascii_lowercase(); // same byte offsets as `s`
    let mut unclosed: Vec<String> = Vec::new();
    let mut out = String::with_capacity(s.len());
    let mut copied = 0;
    let mut at = 0;
    while let Some(m) = open.captures_at(s, at) {
        let whole = m.get(0).expect("match");
        let name = m[1].to_ascii_lowercase();
        let close = format!("</{name}>");
        let found = if unclosed.contains(&name) {
            None
        } else {
            lower[whole.end()..].find(&close).map(|i| whole.end() + i + close.len())
        };
        match found {
            Some(end) => {
                out.push_str(&s[copied..whole.start()]);
                out.push(' ');
                copied = end;
                at = end;
            }
            None => {
                if !unclosed.contains(&name) {
                    unclosed.push(name);
                }
                at = whole.start() + 1; // '<' is one byte
            }
        }
    }
    out.push_str(&s[copied..]);
    out
}

fn named_entity(name: &str) -> Option<&'static str> {
    Some(match name.to_ascii_lowercase().as_str() {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => " ",
        "ndash" => "-",
        "mdash" => "-",
        "hellip" => "...",
        "copy" => "(c)",
        "reg" => "(R)",
        "trade" => "(TM)",
        "times" => "×",
        "micro" => "µ",
        "plusmn" => "±",
        "le" => "≤",
        "ge" => "≥",
        "rarr" => "→",
        _ => return None,
    })
}

fn decode_entities(s: &str) -> Cow<'_, str> {
    re!(r"(?i)&(#x?[0-9a-f]+|[a-z]+);").replace_all(s, |c: &Captures| {
        let body = &c[1];
        if let Some(num) = body.strip_prefix('#') {
            // Like JS parseInt: hex after x/X, else the leading decimal digits.
            let code = match num.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => {
                    let digits: String = num.chars().take_while(|ch| ch.is_ascii_digit()).collect();
                    digits.parse::<u32>().ok()
                }
            };
            return match code.filter(|&n| n > 0 && n < 0x110000).and_then(char::from_u32) {
                Some(ch) => ch.to_string(),
                None => c[0].to_string(),
            };
        }
        named_entity(body).map_or_else(|| c[0].to_string(), str::to_string)
    })
}

/// Extract readable text from HTML: the page `<title>`, and the body with
/// script/style/noscript/template/svg/iframe/head removed, block breaks kept as
/// newlines, table cells joined with " | ", entities decoded and whitespace
/// collapsed. Same rules (and output) as `htmlToText` in scrape.ts.
pub fn html_to_text(html: &str) -> (String, String) {
    let title = re!(r"(?is)<title[^>]*>(.*?)</title>")
        .captures(html)
        .map(|c| {
            let t = decode_entities(&c[1]).trim().to_string();
            re!(r"\s+").replace_all(&t, " ").into_owned()
        })
        .unwrap_or_default();

    let s = re!(r"<!--(?s:.*?)-->").replace_all(html, " ");
    let s = strip_blocks(&s);
    let s = re!(r"(?i)</(p|div|section|article|li|tr|h[1-6]|pre|blockquote|table)\s*>").replace_all(&s, "\n");
    let s = re!(r"(?i)<br\s*/?>").replace_all(&s, "\n");
    let s = re!(r"(?i)</t[dh]\s*>").replace_all(&s, " | ");
    let s = re!(r"<[^>]+>").replace_all(&s, " ");
    let s = decode_entities(&s);
    let s = re!(r"[ \t\x0C\x0B]+").replace_all(&s, " ");
    let s = re!(r" *\n *").replace_all(&s, "\n");
    let s = re!(r"\n{3,}").replace_all(&s, "\n\n");
    (title, s.trim().to_string())
}

/// Pretty-print valid JSON with 2-space indents, keeping key order (what
/// `JSON.stringify(JSON.parse(raw), null, 2)` gives on the web). None if
/// `raw` isn't JSON.
fn pretty_json(raw: &str) -> Option<String> {
    serde_json::from_str::<serde::de::IgnoredAny>(raw).ok()?;
    let mut out = String::with_capacity(raw.len() * 2);
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    let mut chars = raw.chars().peekable();
    let newline = |out: &mut String, depth: usize| {
        out.push('\n');
        out.extend(std::iter::repeat("  ").take(depth));
    };
    while let Some(c) = chars.next() {
        if in_str {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                out.push(c);
            }
            '{' | '[' => {
                while chars.peek().is_some_and(|w| w.is_ascii_whitespace()) {
                    chars.next();
                }
                let close = if c == '{' { '}' } else { ']' };
                out.push(c);
                if chars.peek() == Some(&close) {
                    chars.next();
                    out.push(close); // {} / [] stay on one line
                } else {
                    depth += 1;
                    newline(&mut out, depth);
                }
            }
            '}' | ']' => {
                depth = depth.saturating_sub(1);
                newline(&mut out, depth);
                out.push(c);
            }
            ',' => {
                out.push(',');
                newline(&mut out, depth);
            }
            ':' => out.push_str(": "),
            c if c.is_ascii_whitespace() => {}
            c => out.push(c),
        }
    }
    Some(out)
}

/// Decode using the Content-Type charset, else an HTML `<meta charset>` in the
/// first 1024 bytes, else UTF-8. Unknown labels fall back to UTF-8.
fn decode_body(buf: &[u8], content_type: &str) -> String {
    let mut label = re!(r#"(?i)charset\s*=\s*"?([A-Za-z0-9_.:-]+)"#)
        .captures(content_type)
        .map(|c| c[1].to_string());
    if label.is_none() && content_type.to_ascii_lowercase().contains("html") {
        let head: String = buf[..buf.len().min(1024)].iter().map(|&b| b as char).collect();
        label = re!(r#"(?i)<meta[^>]+charset\s*=\s*["']?([A-Za-z0-9_.:-]+)"#)
            .captures(&head)
            .map(|c| c[1].to_string());
    }
    let enc = label
        .and_then(|l| encoding_rs::Encoding::for_label(l.as_bytes()))
        .filter(|e| *e != encoding_rs::REPLACEMENT)
        .unwrap_or(encoding_rs::UTF_8);
    enc.decode_with_bom_removal(buf).0.into_owned()
}

/// Cut `s` to at most `max` UTF-16 code units (JS string length, so the limit
/// means the same as on the web) without splitting a character.
fn truncate_utf16(s: &mut String, max: usize) -> bool {
    let mut units = 0;
    for (i, ch) in s.char_indices() {
        units += ch.len_utf16();
        if units > max {
            s.truncate(i);
            return true;
        }
    }
    false
}

// ─── Scrape ──────────────────────────────────────────────────────────────────

/// Fetch `url` and return a compact, readable snippet safe to feed to an LLM.
pub async fn scrape_url(url: &str, opts: &ScrapeOptions) -> Result<ScrapedPage, ScrapeError> {
    match tokio::time::timeout(Duration::from_millis(opts.timeout_ms), scrape_inner(url, opts)).await {
        Ok(result) => result,
        Err(_) => Err(ScrapeError::upstream(format!(
            "Scrape timed out after {}ms: {url}",
            opts.timeout_ms
        ))),
    }
}

async fn scrape_inner(url: &str, opts: &ScrapeOptions) -> Result<ScrapedPage, ScrapeError> {
    let mut current = Url::parse(url).map_err(|_| ScrapeError::bad(format!("Invalid URL: {url}")))?;
    let fetch_failed = |e: &dyn std::error::Error| ScrapeError::upstream(format!("Scrape fetch failed: {}", error_chain(e)));

    // Follow redirects by hand so each hop passes the SSRF/allowlist checks
    // BEFORE a request is sent to it.
    let mut hop = 0;
    let mut res = loop {
        let pin = assert_safe_target(&current, opts, hop).await?;
        let res = client_for(pin)?
            .get(current.clone())
            .header(reqwest::header::USER_AGENT, &opts.user_agent)
            .header(reqwest::header::ACCEPT, ACCEPT)
            .header(reqwest::header::ACCEPT_LANGUAGE, "en")
            .send()
            .await
            .map_err(|e| fetch_failed(&e))?;
        let status = res.status().as_u16();
        let location = if (300..400).contains(&status) {
            res.headers()
                .get(reqwest::header::LOCATION)
                .map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
                .filter(|l| !l.is_empty())
        } else {
            None
        };
        let Some(location) = location else { break res };
        drop(res);
        if hop >= opts.max_redirects {
            return Err(ScrapeError::upstream(format!(
                "Too many redirects (>{}) for {url}",
                opts.max_redirects
            )));
        }
        current = current.join(&location).map_err(|_| {
            ScrapeError::upstream(format!("Invalid redirect location \"{location}\" from {current}"))
        })?;
        hop += 1;
    };
    let final_url = current.to_string();

    let status = res.status().as_u16();
    if !(200..300).contains(&status) {
        let msg = format!("Scrape returned HTTP {status} for {final_url}");
        return Err(if status >= 500 { ScrapeError::upstream(msg) } else { ScrapeError::bad(msg) });
    }
    let content_type = res
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
        .unwrap_or_default();
    if !re!(r"(?i)^(text/(html|plain)|application/(xhtml\+xml|json))\b").is_match(&content_type) {
        let shown = if content_type.is_empty() { "(none)" } else { content_type.as_str() };
        return Err(ScrapeError::bad(format!("Refusing content-type \"{shown}\" for {final_url}")));
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = res.chunk().await.map_err(|e| fetch_failed(&e))? {
        let room = opts.max_bytes - buf.len();
        if chunk.len() > room {
            buf.extend_from_slice(&chunk[..room]);
            truncated = true;
            break;
        }
        buf.extend_from_slice(&chunk);
    }
    let raw = decode_body(&buf, &content_type);

    let ct = content_type.to_ascii_lowercase();
    let (title, mut text) = if ct.contains("html") {
        html_to_text(&raw)
    } else if ct.contains("json") {
        (String::new(), pretty_json(&raw).unwrap_or(raw))
    } else {
        (String::new(), raw)
    };
    if truncate_utf16(&mut text, opts.max_chars) {
        truncated = true;
    }

    Ok(ScrapedPage {
        url: url.to_string(),
        final_url,
        title,
        text,
        truncated,
        bytes: buf.len(),
        content_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn blocked(h: &str) -> bool {
        // Mirrors isPrivateHost(): an IP literal is classified, a name is
        // checked against the blocked names. URL parsing does the rest (see
        // the `loopback_encodings` test).
        let bare = h.trim_start_matches('[').trim_end_matches(']');
        let bare = bare.split('%').next().unwrap_or(bare);
        match bare.parse::<IpAddr>() {
            Ok(ip) => is_private_ip(ip),
            Err(_) => is_blocked_name(h),
        }
    }

    #[test]
    fn blocks_loopback_rfc1918_link_local_and_v6_local() {
        for h in [
            "localhost", "app.localhost", "metadata.google.internal", "metadata", "printer.local",
            "127.0.0.1", "127.13.1.7", "10.0.0.1", "172.16.5.4", "172.31.0.1", "192.168.1.1",
            "169.254.169.254", "0.0.0.0", "::1", "fe80::1", "fd12:abcd::1", "fec0::1",
        ] {
            assert!(blocked(h), "{h}");
        }
    }

    #[test]
    fn blocks_cgnat_multicast_and_v4_embedded_in_v6() {
        for h in [
            "100.64.0.1", "100.127.255.254", "198.18.0.1", "224.0.0.1", "255.255.255.255",
            "192.0.2.1", "198.51.100.7", "203.0.113.9", "::", "[::]", "ff02::1", "localhost.",
            "::ffff:127.0.0.1", "::ffff:7f00:1", "[::ffff:a9fe:a9fe]", "0:0:0:0:0:ffff:0a00:0001",
            "64:ff9b::a00:1", "2002:c0a8:0101::1", "::127.0.0.1", "fe80::1%eth0",
        ] {
            assert!(blocked(h), "{h}");
        }
    }

    #[test]
    fn allows_public_hosts() {
        for h in [
            "example.com", "docs.nvidia.com", "8.8.8.8", "172.32.0.1", "100.128.0.1",
            "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808", "93.184.216.34",
        ] {
            assert!(!blocked(h), "{h}");
        }
    }

    #[test]
    fn loopback_encodings_are_normalised_then_refused() {
        let opts = ScrapeOptions::default();
        for u in [
            "http://2130706433/", "http://0x7f.1/", "http://[::ffff:127.0.0.1]/", "http://localhost./",
            "http://127.1/", "http://[::1]:8080/", "http://169.254.169.254/latest/meta-data/",
        ] {
            let err = check_static(&Url::parse(u).unwrap(), &opts, 0).unwrap_err();
            assert!(err.error.contains("private/loopback"), "{u}: {}", err.error);
            assert_eq!(err.status, 400);
        }
    }

    #[test]
    fn refuses_non_http_schemes_and_marks_redirects() {
        let opts = ScrapeOptions::default();
        let err = check_static(&Url::parse("file:///etc/passwd").unwrap(), &opts, 0).unwrap_err();
        assert_eq!(err.error, "Refusing non-http(s) scheme: file:");
        let err = check_static(&Url::parse("http://10.0.0.1/").unwrap(), &opts, 1).unwrap_err();
        assert_eq!(err.error, "Refusing redirect to private/loopback host: 10.0.0.1");
    }

    #[test]
    fn resolved_private_address_is_refused() {
        let ips: Vec<IpAddr> = vec!["93.184.216.34".parse().unwrap(), "10.0.0.5".parse().unwrap()];
        let err = check_resolved("127.0.0.1.nip.io", &ips, false, is_private_ip).unwrap_err();
        assert_eq!(err.error, "Refusing host 127.0.0.1.nip.io: resolves to private address 10.0.0.5");
        let mapped: Vec<IpAddr> = vec!["::ffff:192.168.0.1".parse().unwrap()];
        assert!(check_resolved("x.example", &mapped, true, is_private_ip).is_err());
        assert!(check_resolved("x.example", &[], false, is_private_ip)
            .unwrap_err()
            .error
            .starts_with("Could not resolve host"));
        let public: Vec<IpAddr> = vec!["8.8.8.8".parse().unwrap(), "2606:4700::1111".parse().unwrap()];
        assert!(check_resolved("dns.example", &public, false, is_private_ip).is_ok());
    }

    #[test]
    fn allowlist_matches_suffixes_and_marks_redirects() {
        let opts = ScrapeOptions { allowlist: vec!["nvidia.com".into()], ..Default::default() };
        assert!(check_static(&Url::parse("https://docs.nvidia.com/a").unwrap(), &opts, 0).is_ok());
        assert!(check_static(&Url::parse("https://nvidia.com/").unwrap(), &opts, 0).is_ok());
        let err = check_static(&Url::parse("https://evilnvidia.com/").unwrap(), &opts, 1).unwrap_err();
        assert_eq!(err.error, "Host not in MODELVISIO_SCRAPE_ALLOWLIST: evilnvidia.com (via redirect)");
    }

    #[test]
    fn env_options_parse_and_ignore_junk() {
        let vars = |k: &str| {
            match k {
                "MODELVISIO_SCRAPE_ALLOWLIST" => Some(" docs.nvidia.com, ,hailo.ai ".to_string()),
                "MODELVISIO_SCRAPE_MAX_BYTES" => Some("2048".to_string()),
                "MODELVISIO_SCRAPE_MAX_CHARS" => Some("abc".to_string()),
                "MODELVISIO_SCRAPE_TIMEOUT_MS" => Some("-5".to_string()),
                _ => None,
            }
        };
        let o = ScrapeOptions::from_vars(vars);
        assert_eq!(o.allowlist, vec!["docs.nvidia.com", "hailo.ai"]);
        assert_eq!(o.max_bytes, 2048);
        assert_eq!(o.max_chars, 12_000);
        assert_eq!(o.timeout_ms, 8000);
    }

    #[test]
    fn html_title_and_script_style_removed() {
        let (title, text) = html_to_text(
            "<html><head><title>Op Support</title><style>b{}</style></head>\
             <body><script>alert(1)</script><h1>Ops</h1><p>Conv2D &amp; ReLU</p></body></html>",
        );
        assert_eq!(title, "Op Support");
        assert!(!text.contains("alert"));
        assert!(text.contains("Ops"));
        assert!(text.contains("Conv2D & ReLU"));
    }

    #[test]
    fn html_entities_decoded() {
        let (_, text) = html_to_text("<p>&#8734; &amp; &nbsp;x&hellip; 3&times;3 &#x41;&#X42; &bogus; &#0;</p>");
        assert!(text.contains('∞'));
        assert!(text.contains('&'));
        assert!(text.contains("x..."));
        assert!(text.contains("3×3"));
        assert!(text.contains("AB"));
        assert!(text.contains("&bogus;"));
        assert!(text.contains("&#0;"));
    }

    #[test]
    fn html_keeps_breaks_and_table_cells() {
        let (_, text) = html_to_text("<p>a<br>b</p><table><tr><td>Conv</td><td>yes</td></tr></table>");
        assert!(text.contains("a\nb"), "{text:?}");
        assert!(text.contains("Conv | yes"), "{text:?}");
    }

    #[test]
    fn html_header_is_not_head_and_comments_go() {
        let (_, text) = html_to_text("<header>Nav</header><!-- secret --><p>Body</p>");
        assert!(text.contains("Nav"));
        assert!(!text.contains("secret"));
    }

    #[test]
    fn html_unclosed_script_falls_back_to_tag_strip() {
        let (_, text) = html_to_text("<p>keep</p><script src=x><p>after</p>");
        assert_eq!(text, "keep\nafter");
        // Case-insensitive close, and an unclosed tag before a closed one.
        let (_, text) = html_to_text("<SVG><p>a</p><svg>drawing</SVG> tail");
        assert_eq!(text, "tail");
    }

    #[test]
    fn html_whitespace_collapsed() {
        let (title, text) =
            html_to_text("<head><title>\n  A \t  B\n</title></head><p>x   \t y</p>\n\n\n\n<p>z</p>");
        assert_eq!(title, "A B");
        assert_eq!(text, "x y\n\nz");
    }

    #[test]
    fn json_pretty_printed_in_order() {
        assert_eq!(
            pretty_json(r#" {"b":1,"a":[2,3],"e":{},"s":"x,{y}:\"z\""} "#).unwrap(),
            "{\n  \"b\": 1,\n  \"a\": [\n    2,\n    3\n  ],\n  \"e\": {},\n  \"s\": \"x,{y}:\\\"z\\\"\"\n}"
        );
        assert!(pretty_json("{not json").is_none());
    }

    #[test]
    fn charset_decoding() {
        assert_eq!(decode_body(&[0x63, 0x61, 0x66, 0xe9], "text/plain; charset=windows-1252"), "café");
        let mut html = b"<html><head><meta charset=\"iso-8859-1\"><title>x</title></head><body>".to_vec();
        html.extend_from_slice(&[0x6e, 0x61, 0xef, 0x76, 0x65]);
        assert!(decode_body(&html, "text/html").contains("naïve"));
        assert_eq!(decode_body("ok".as_bytes(), "text/plain; charset=nonsense"), "ok");
    }

    #[test]
    fn truncation_counts_utf16_units() {
        let mut s = "abc".repeat(10);
        assert!(truncate_utf16(&mut s, 7));
        assert_eq!(s, "abcabca");
        let mut s = "a😀b".to_string(); // 😀 is 2 UTF-16 units
        assert!(truncate_utf16(&mut s, 2));
        assert_eq!(s, "a");
        let mut s = "short".to_string();
        assert!(!truncate_utf16(&mut s, 5));
    }

    // ── End to end against a loopback server ────────────────────────────────

    /// Allows the loopback test server; blocks everything else the real guard
    /// blocks, so redirects to e.g. 169.254.169.254 are still refused.
    fn allow_loopback(ip: IpAddr) -> bool {
        !ip.is_loopback() && is_private_ip(ip)
    }

    /// Serve canned raw HTTP responses, one per connection, in order. Returns
    /// the base URL and a handle yielding the request lines received.
    fn serve(responses: Vec<String>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for resp in responses {
                let Ok((mut sock, _)) = listener.accept() else { break };
                let mut req = Vec::new();
                let mut b = [0u8; 1024];
                while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                    match sock.read(&mut b) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => req.extend_from_slice(&b[..n]),
                    }
                }
                let req = String::from_utf8_lossy(&req).into_owned();
                seen.push(req.lines().next().unwrap_or("").to_string());
                let _ = sock.write_all(resp.as_bytes());
            }
            seen
        });
        (base, handle)
    }

    fn http(status: &str, headers: &[(&str, &str)], body: &str) -> String {
        let mut s = format!("HTTP/1.1 {status}\r\nconnection: close\r\ncontent-length: {}\r\n", body.len());
        for (k, v) in headers {
            s.push_str(&format!("{k}: {v}\r\n"));
        }
        s.push_str("\r\n");
        s.push_str(body);
        s
    }

    fn test_opts() -> ScrapeOptions {
        ScrapeOptions { is_blocked_ip: allow_loopback, timeout_ms: 5000, ..Default::default() }
    }

    fn run<F: std::future::Future>(f: F) -> F::Output {
        tauri::async_runtime::block_on(f)
    }

    #[test]
    fn e2e_reads_html_and_follows_a_redirect() {
        let (base, server) = serve(vec![
            http("301 Moved", &[("location", "/v2/ops")], ""),
            http(
                "200 OK",
                &[("content-type", "text/html; charset=utf-8")],
                "<html><head><title>Ops v2</title></head><body><p>Conv, Gemm</p></body></html>",
            ),
        ]);
        let page = run(scrape_url(&format!("{base}/ops"), &test_opts())).unwrap();
        assert_eq!(page.title, "Ops v2");
        assert_eq!(page.text, "Conv, Gemm");
        assert_eq!(page.final_url, format!("{base}/v2/ops"));
        assert_eq!(page.url, format!("{base}/ops"));
        assert!(!page.truncated);
        assert!(page.content_type.starts_with("text/html"));
        assert_eq!(server.join().unwrap(), vec!["GET /ops HTTP/1.1", "GET /v2/ops HTTP/1.1"]);
    }

    #[test]
    fn e2e_refuses_redirect_to_metadata_before_requesting_it() {
        let (base, server) = serve(vec![http(
            "302 Found",
            &[("location", "http://169.254.169.254/latest/meta-data/")],
            "",
        )]);
        let err = run(scrape_url(&format!("{base}/"), &test_opts())).unwrap_err();
        assert_eq!(err.error, "Refusing redirect to private/loopback host: 169.254.169.254");
        assert_eq!(err.status, 400);
        assert_eq!(server.join().unwrap().len(), 1);
    }

    #[test]
    fn e2e_redirect_limit_and_bad_content_type_and_5xx() {
        let hops = (1..=4).map(|i| http("302 Found", &[("location", &format!("/hop{i}"))], "")).collect();
        let (base, _s) = serve(hops);
        let opts = ScrapeOptions { max_redirects: 3, ..test_opts() };
        let err = run(scrape_url(&format!("{base}/"), &opts)).unwrap_err();
        assert_eq!(err.error, format!("Too many redirects (>3) for {base}/"));
        assert_eq!(err.status, 502);

        let (base, _s) = serve(vec![http("200 OK", &[("content-type", "application/zip")], "PK")]);
        let err = run(scrape_url(&format!("{base}/a.zip"), &test_opts())).unwrap_err();
        assert_eq!(err.error, format!("Refusing content-type \"application/zip\" for {base}/a.zip"));
        assert_eq!(err.status, 400);

        let (base, _s) = serve(vec![http("503 Service Unavailable", &[("content-type", "text/plain")], "down")]);
        let err = run(scrape_url(&format!("{base}/"), &test_opts())).unwrap_err();
        assert!(err.error.starts_with("Scrape returned HTTP 503"), "{}", err.error);
        assert_eq!(err.status, 502);
    }

    #[test]
    fn e2e_caps_bytes_and_chars_and_pretty_prints_json() {
        let big = "x".repeat(10_000);
        let (base, _s) = serve(vec![http("200 OK", &[("content-type", "text/plain")], &big)]);
        let opts = ScrapeOptions { max_bytes: 1000, ..test_opts() };
        let page = run(scrape_url(&format!("{base}/big.txt"), &opts)).unwrap();
        assert_eq!(page.bytes, 1000);
        assert!(page.truncated);

        let (base, _s) = serve(vec![http("200 OK", &[("content-type", "text/plain")], &big)]);
        let opts = ScrapeOptions { max_chars: 200, ..test_opts() };
        let page = run(scrape_url(&format!("{base}/big.txt"), &opts)).unwrap();
        assert_eq!(page.text.len(), 200);
        assert!(page.truncated);

        let (base, _s) = serve(vec![http("200 OK", &[("content-type", "application/json")], r#"{"a":1,"b":[2,3]}"#)]);
        let page = run(scrape_url(&format!("{base}/x"), &test_opts())).unwrap();
        assert!(page.text.contains("\"a\": 1"), "{}", page.text);
    }

    #[test]
    fn e2e_times_out_a_stalled_origin() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let _hold = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut b = [0u8; 1024];
            let _ = sock.read(&mut b); // take the request before answering
            let _ = sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: 999\r\n\r\n<p>first");
            std::thread::sleep(Duration::from_secs(3));
        });
        let opts = ScrapeOptions { timeout_ms: 200, ..test_opts() };
        let err = run(scrape_url(&format!("{base}/"), &opts)).unwrap_err();
        assert_eq!(err.error, format!("Scrape timed out after 200ms: {base}/"));
        assert_eq!(err.status, 502);
    }

    #[test]
    fn real_guard_refuses_the_loopback_test_server() {
        let err = run(scrape_url("http://127.0.0.1:9/", &ScrapeOptions::default())).unwrap_err();
        assert!(err.error.contains("private/loopback"));
    }

    #[test]
    fn invalid_url() {
        let err = run(scrape_url("not a url", &ScrapeOptions::default())).unwrap_err();
        assert_eq!(err, ScrapeError { error: "Invalid URL: not a url".into(), status: 400 });
    }
}
