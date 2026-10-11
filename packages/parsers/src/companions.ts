// Companion files: formats that keep weights next to the graph file
// (OpenVINO .xml + .bin, ncnn .param + .bin, Paddle .pdmodel + .pdiparams,
// TF.js model.json + shards, SavedModel saved_model.pb + variables/).
//
// Hosts pass whatever extra files they have as a name → bytes map. Hosts that
// can read the disk (VS Code, desktop, CLI) ask companionNames() which
// siblings to load; the browser lets the user pick or drop them together.

export type Companions = Record<string, Uint8Array | ArrayBuffer>;

const td = new TextDecoder();

const base = (p: string) => p.replace(/\\/g, "/").split("/").pop() ?? p;
const stem = (name: string) => base(name).replace(/\.[^.]+$/, "");
const lower = (s: string) => s.toLowerCase();
const bytesOf = (b: Uint8Array | ArrayBuffer) => (b instanceof Uint8Array ? b : new Uint8Array(b));

/**
 * Find a companion by exact (case-insensitive) file name, by relative path
 * suffix, or, failing that, the only file with a wanted extension.
 */
export function findCompanion(c: Companions | undefined, wanted: string[], exts: string[] = []): Uint8Array | undefined {
  if (!c) return undefined;
  const keys = Object.keys(c);
  for (const w of wanted) {
    const lw = lower(w.replace(/\\/g, "/"));
    const hit = keys.find((k) => {
      const lk = lower(k.replace(/\\/g, "/"));
      return lk === lw || lk.endsWith("/" + lw) || lower(base(k)) === lower(base(w));
    });
    if (hit) return bytesOf(c[hit]);
  }
  if (exts.length) {
    const byExt = keys.filter((k) => exts.some((e) => lower(k).endsWith(e)));
    if (byExt.length === 1) return bytesOf(c[byExt[0]]);
  }
  return undefined;
}

/** Names of all companions (for "which file did you mean" messages). */
export const companionKeys = (c: Companions | undefined) => Object.keys(c ?? {});

/** A manifest path that stays inside the model's folder (no "..", not absolute). */
const isLocalPath = (p: string) =>
  typeof p === "string" && p.length > 0 && !/^([a-z]:|[\\/])/i.test(p) && !p.split(/[\\/]/).some((c) => c === ".." || c === "");

/** TF.js weight shard paths listed in a model.json, in manifest order (folder-local only). */
export function tfjsShardPaths(json: string): string[] {
  try {
    const j = JSON.parse(json) as { weightsManifest?: { paths?: string[] }[] };
    return (j.weightsManifest ?? []).flatMap((g) => g.paths ?? []).filter(isLocalPath);
  } catch {
    return [];
  }
}

/**
 * Sibling files (relative to the graph file's folder) a host should load
 * alongside `name` for full weight statistics. Empty when the format keeps
 * its weights inline.
 */
export function companionNames(name: string, head?: Uint8Array): string[] {
  const b = lower(base(name));
  const s = stem(name);
  if (b.endsWith(".xml")) return [`${s}.bin`];
  if (b.endsWith(".param.bin")) return [`${b.slice(0, -".param.bin".length)}.bin`];
  if (b.endsWith(".param")) return [`${s}.bin`];
  if (b.endsWith(".pdmodel")) return [`${s}.pdiparams`];
  if (b === "saved_model.pb") return ["variables/variables.index", "variables/variables.data-00000-of-00001"];
  if (b.endsWith(".json") && head) return tfjsShardPaths(td.decode(head));
  return [];
}

/** Extensions that only ever hold weights for another file; never the file to parse. */
const WEIGHT_ONLY = /\.(bin|pdiparams|pdiparams\.info|index|data-\d{5}-of-\d{5})$/i;

/**
 * From a set of files the user opened together, choose the one to parse; the
 * rest become its companions. `.param.bin` (binary ncnn graph) counts as a graph.
 */
export function pickPrimary(names: string[]): string | undefined {
  const isGraph = (n: string) => /\.param\.bin$/i.test(n) || !WEIGHT_ONLY.test(n);
  return names.find(isGraph) ?? names[0];
}
