// Browser-safe helpers shared by the scrape client (index.ts) and the server
// scraper (scrape.ts). No Node imports and no network code — index.ts imports
// from here, never from scrape.ts, so the scraper stays out of the browser bundle.

/** Readable result of scraping one URL (the /api/scrape response body). */
export type ScrapedPage = {
  url: string;
  finalUrl: string;
  title: string;
  text: string;
  truncated: boolean;
  bytes: number;
  contentType: string;
};

/** Extract http(s) URLs from a free-form message. Used by the copilot's
 *  research step: any URL the user pastes can be scraped and grounded on. */
export function extractUrls(text: string): string[] {
  const rx = /https?:\/\/[^\s<>"'`\]]+/gi;
  const out: string[] = [];
  const seen = new Set<string>();
  for (const m of text.matchAll(rx)) {
    let u = m[0];
    // Trim trailing punctuation people commonly type after a URL, and a closing
    // paren only when it's unbalanced — keeps wiki-style ".../Foo_(bar)" intact
    // while "(see https://x.io)" still loses its ")".
    for (;;) {
      const before = u;
      u = u.replace(/[.,;:!?]+$/, "");
      if (u.endsWith(")") && count(u, "(") < count(u, ")")) u = u.slice(0, -1);
      if (u === before) break;
    }
    if (!seen.has(u)) { seen.add(u); out.push(u); }
  }
  return out;
}

function count(s: string, ch: string): number {
  let n = 0;
  for (const c of s) if (c === ch) n++;
  return n;
}
