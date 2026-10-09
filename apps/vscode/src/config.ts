import * as vscode from "vscode";
import { parseProvider } from "@modelvisio/ai/providers";
import { parseFreeSearchMode } from "@modelvisio/ai/search";
import { normalizeMaxMB } from "./files";
import { sanitizeGeminiModel, sanitizeGrokModel } from "./geminiModel";

/** Typed snapshot of the `modelvisio.*` settings. Read fresh per request so
 *  changes apply without reloading the window. */
export function settings() {
  const cfg = vscode.workspace.getConfiguration("modelvisio");
  const allowlist = cfg.get<unknown>("scrapeAllowlist");
  return {
    provider: parseProvider(cfg.get<unknown>("provider")) ?? "gemini",
    geminiModel: sanitizeGeminiModel(cfg.get<unknown>("geminiModel")),
    grokModel: sanitizeGrokModel(cfg.get<unknown>("grokModel")),
    webSearch: cfg.get<boolean>("webSearch") ?? false,
    freeWebSearch: parseFreeSearchMode(cfg.get<unknown>("freeWebSearch"), "auto"),
    thinking: cfg.get<boolean>("thinking") ?? false,
    scrapeAllowlist: Array.isArray(allowlist)
      ? allowlist.filter((s): s is string => typeof s === "string" && s.trim() !== "").map((s) => s.trim())
      : [],
    maxFileSizeMB: normalizeMaxMB(cfg.get<unknown>("maxFileSizeMB")),
  };
}
