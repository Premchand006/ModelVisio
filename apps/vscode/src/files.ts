// File-name and size helpers for loading and saving. Pure (no `vscode` import)
// so they are unit tested.

/** Default for `modelvisio.maxFileSizeMB`. */
export const DEFAULT_MAX_FILE_MB = 2048;

/** Human-readable byte count ("12.3 MB"). */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "? B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return i === 0 ? `${v} B` : `${v.toFixed(1)} ${units[i]}`;
}

/** The limit in MB from the raw setting; garbage falls back to the default. */
export function normalizeMaxMB(raw: unknown): number {
  return typeof raw === "number" && Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_MAX_FILE_MB;
}

/** Friendly error when a file exceeds the limit, else null. */
export function sizeLimitError(name: string, sizeBytes: number, maxMB: number): string | null {
  if (sizeBytes <= maxMB * 1024 * 1024) return null;
  return (
    `${name} is ${formatBytes(sizeBytes)}, over the ${maxMB} MB limit. ` +
    `Raise "modelvisio.maxFileSizeMB" in Settings to open it — very large models can exhaust the editor's memory.`
  );
}

/**
 * A file name from the WebView made safe to use as the save dialog's default:
 * path components are stripped (no "../" escapes), characters Windows forbids
 * are replaced, and an empty result falls back to "export".
 */
export function safeFilename(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? "";
  // eslint-disable-next-line no-control-regex
  const cleaned = base.replace(/[<>:"|?*\u0000-\u001f]/g, "_").replace(/^[.\s]+|[.\s]+$/g, "");
  return cleaned || "export";
}

/** `showSaveDialog` filters for a file name: its own extension, then anything. */
export function saveFilters(filename: string): Record<string, string[]> {
  const dot = filename.lastIndexOf(".");
  const ext = dot > 0 ? filename.slice(dot + 1) : "";
  return ext ? { [`${ext.toUpperCase()} file`]: [ext], "All files": ["*"] } : { "All files": ["*"] };
}

/** POSIX directory of a URI path ("/a/b/model.onnx" → "/a/b"). */
export function dirnamePosix(p: string): string {
  const i = p.lastIndexOf("/");
  return i <= 0 ? "/" : p.slice(0, i);
}

/**
 * Glob that matches `name` for a file watcher. VS Code globs have no escape
 * syntax, so each special character becomes `?` (any one char): the pattern
 * may over-match a sibling, which the caller filters out by exact URI.
 */
export function watchGlobFor(name: string): string {
  return name.replace(/[*?[\]{}!]/g, "?");
}

/**
 * The bytes as a plain Uint8Array that owns its whole buffer, ready for
 * webview.postMessage. Node often hands back a Buffer that is a window into a
 * larger shared pool (small files); transferring that would ship the pool.
 * Large reads own their buffer, so they are re-viewed without copying.
 */
export function toTransferable(bytes: Uint8Array): Uint8Array {
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.constructor === Uint8Array ? bytes : new Uint8Array(bytes.buffer, 0, bytes.byteLength);
  }
  return new Uint8Array(bytes);
}
