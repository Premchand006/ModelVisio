// Read and parse a model file from disk.

import { existsSync, readFileSync, statSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { FORMAT_SUPPORT, companionNames, detectFormat, parseModel, type Companions, type Model } from "@modelvisio/parsers";

export type LoadedModel = {
  path: string;
  fileName: string;
  bytes: Uint8Array;
  model: Model;
  /** False for formats ModelVisio detects but does not graph-parse (metadata only). */
  graphParsed: boolean;
};

export class CliError extends Error {
  constructor(message: string, readonly exitCode = 2) {
    super(message);
  }
}

export function loadModel(file: string, cwd: string): LoadedModel {
  const path = resolve(cwd, file);
  let st;
  try { st = statSync(path); } catch { throw new CliError(`File not found: ${file}`); }
  if (!st.isFile()) throw new CliError(`Not a file: ${file}`);
  const bytes = new Uint8Array(readFileSync(path));
  const fileName = basename(path);
  // Weights kept next to the graph (OpenVINO .bin, Paddle .pdiparams, ...).
  const companions: Companions = {};
  for (const rel of companionNames(fileName, bytes)) {
    const p = join(dirname(path), rel);
    if (existsSync(p)) companions[rel] = new Uint8Array(readFileSync(p));
  }
  let model: Model;
  try {
    model = parseModel(bytes, fileName, companions);
  } catch (e) {
    throw new CliError(`Could not parse ${file}: ${e instanceof Error ? e.message : String(e)}`);
  }
  const status = FORMAT_SUPPORT.find((f) => f.id === detectFormat(fileName, bytes))?.status;
  return { path, fileName, bytes, model: { ...model, sizeBytes: bytes.byteLength }, graphParsed: status === "full" && model.layers.length > 0 };
}

// ── formatting ──────────────────────────────────────────────────────────────

export function human(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (a >= 1e9) return (n / 1e9).toFixed(2) + "G";
  if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n * 100) / 100);
}

export function bytesStr(b: number): string {
  const a = Math.abs(b);
  if (a >= 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (a >= 1e6) return (b / 1e6).toFixed(2) + " MB";
  if (a >= 1e3) return (b / 1e3).toFixed(1) + " KB";
  return b + " B";
}

export const shapeStr = (s: number[]) => (s.length ? `[${s.map((d) => (d > 0 ? d : "?")).join(", ")}]` : "?");

export const pctStr = (p: number | null) => (p == null ? "n/a" : `${p >= 0 ? "+" : ""}${p.toFixed(1)}%`);

export function escapeXml(s: string): string {
  return s.replace(/[<>&"']/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&apos;" })[c]!)
    // Strip characters XML 1.0 cannot carry.
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, "");
}

export function escapeMd(s: string): string {
  return s.replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
}
