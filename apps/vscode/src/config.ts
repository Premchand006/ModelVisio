import * as vscode from "vscode";
import { normalizeMaxMB } from "./files";
import { sanitizeGeminiModel } from "./geminiModel";

/** Typed snapshot of the `modelvisio.*` settings. Read fresh per request so
 *  changes apply without reloading the window. */
export function settings() {
  const cfg = vscode.workspace.getConfiguration("modelvisio");
  const allowlist = cfg.get<unknown>("scrapeAllowlist");
  return {
    geminiModel: sanitizeGeminiModel(cfg.get<unknown>("geminiModel")),
    webSearch: cfg.get<boolean>("webSearch") ?? false,
    thinking: cfg.get<boolean>("thinking") ?? false,
    scrapeAllowlist: Array.isArray(allowlist)
      ? allowlist.filter((s): s is string => typeof s === "string" && s.trim() !== "").map((s) => s.trim())
      : [],
    maxFileSizeMB: normalizeMaxMB(cfg.get<unknown>("maxFileSizeMB")),
  };
}
