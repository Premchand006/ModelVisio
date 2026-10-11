// Locate a JSON object embedded in an arbitrary binary container (e.g. the
// Keras `model_config` attribute inside an HDF5 file) without parsing the
// container: search for a start marker, then brace-balance with awareness of
// string literals and escapes, and validate with JSON.parse + a predicate.

const td = new TextDecoder();

/** End index (exclusive) of the balanced object starting at `start` ('{'), or -1. */
export function balancedObjectEnd(buf: Uint8Array, start: number, maxLen = 64 << 20): number {
  let depth = 0;
  let inStr = false;
  const end = Math.min(buf.length, start + maxLen);
  for (let i = start; i < end; i++) {
    const c = buf[i];
    if (inStr) {
      if (c === 0x5c) i++; // backslash escapes the next byte
      else if (c === 0x22) inStr = false;
      continue;
    }
    if (c === 0x22) inStr = true;
    else if (c === 0x7b || c === 0x5b) depth++;
    else if (c === 0x7d || c === 0x5d) {
      depth--;
      if (depth === 0) return i + 1;
    } else if (c === 0) return -1; // NUL never appears inside JSON text
  }
  return -1;
}

function indexOfBytes(buf: Uint8Array, pat: Uint8Array, from: number): number {
  const first = pat[0];
  outer: for (let i = buf.indexOf(first, from); i !== -1 && i <= buf.length - pat.length; i = buf.indexOf(first, i + 1)) {
    for (let j = 1; j < pat.length; j++) if (buf[i + j] !== pat[j]) continue outer;
    return i;
  }
  return -1;
}

/**
 * First JSON object in `buf` that starts with one of `markers` (e.g.
 * '{"class_name"') and satisfies `accept`. Earliest match wins, so an outer
 * object is preferred over objects nested inside it.
 */
export function findJsonObject(
  buf: Uint8Array, markers: string[], accept: (o: unknown) => boolean,
): unknown | null {
  const pats = markers.map((m) => new TextEncoder().encode(m));
  const hits: number[] = [];
  for (const p of pats) {
    for (let i = indexOfBytes(buf, p, 0); i !== -1; i = indexOfBytes(buf, p, i + 1)) hits.push(i);
  }
  hits.sort((a, b) => a - b);
  for (const start of hits) {
    const end = balancedObjectEnd(buf, start);
    if (end < 0) continue;
    try {
      const o = JSON.parse(td.decode(buf.subarray(start, end)));
      if (accept(o)) return o;
    } catch {
      // not valid JSON at this marker; keep scanning
    }
  }
  return null;
}
