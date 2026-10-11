import * as vscode from "vscode";
import { lookup } from "node:dns/promises";
import { runChatProxy } from "@modelvisio/ai/proxy";
import { runGrokChat } from "@modelvisio/ai/grok";
import { chatWithWebResearch, freeSearchApplies } from "@modelvisio/ai/search";
import { scrapeUrl, type HostResolver } from "@modelvisio/ai/scrape";
import type { HostToWebview, WebviewToHost } from "./protocol";
import { settings } from "./config";
import { safeFilename, saveFilters } from "./files";
import { messageId, saveBytes, scrapeFailure } from "./messages";
import { missingKeyMessage } from "./apiKey";
import { notifyMissingKey, type ApiKeys } from "./secrets";
import { errMsg, log } from "./log";

/** Not serverless, so no hard platform limit — but a hung model request
 *  must still end so the chat UI can recover. */
const CHAT_TIME_BUDGET_MS = 60_000;

/** Static node:dns import instead of the scraper's default dynamic import, so
 *  the bundle resolves the builtin at build time. */
const resolveHost: HostResolver = async (host) =>
  (await lookup(host, { all: true, verbatim: true })).map((a) => a.address);

export type BridgeContext = {
  post: (msg: HostToWebview) => void;
  keys: ApiKeys;
  /** The model file this editor shows; the save dialog defaults next to it. */
  modelUri: vscode.Uri;
};

/** Handle one validated WebView request (everything except `ready`, which the
 *  editor provider owns because it drives the panel's own lifecycle). */
export async function handleRequest(msg: WebviewToHost, ctx: BridgeContext): Promise<void> {
  switch (msg.type) {
    case "chat":
      return handleChat(msg, ctx);
    case "scrape":
      return handleScrape(msg, ctx);
    case "save":
      return handleSave(msg, ctx);
    case "notify":
      return handleNotify(msg);
    case "ready":
      return;
  }
}

/** Reply to a malformed request that still carried an id, so the WebView's
 *  pending promise settles instead of hanging forever. */
export function rejectMalformed(raw: unknown, post: BridgeContext["post"]): void {
  const m = messageId(raw);
  log.warn(`Dropped malformed WebView message${m ? ` (${m.type} #${m.id})` : ""}.`);
  if (!m) return;
  const error = "Malformed request from the ModelVisio view.";
  if (m.type === "chat") post({ type: "chatError", id: m.id, error });
  else if (m.type === "scrape") post({ type: "scrapeError", id: m.id, error, status: 400 });
  else if (m.type === "save") post({ type: "saveResult", id: m.id, saved: false, error });
}

async function handleChat(msg: Extract<WebviewToHost, { type: "chat" }>, ctx: BridgeContext): Promise<void> {
  // Usage beacons (logUpload) arrive as chat calls with no messages; they must
  // never become a billed model request.
  if (msg.messages.length === 0) {
    log.debug(`Ignored empty chat request #${msg.id}.`);
    ctx.post({ type: "chatError", id: msg.id, error: "Empty chat request — nothing to send." });
    return;
  }
  const s = settings();
  const store = ctx.keys[s.provider];
  const found = await store.resolve();
  if (!found) {
    log.warn(`Chat request without a ${store.label} API key.`);
    ctx.post({ type: "chatError", id: msg.id, error: missingKeyMessage(s.provider) });
    notifyMissingKey(s.provider);
    return;
  }
  const { key } = found;
  const model = s.provider === "grok" ? s.grokModel : s.geminiModel;
  const run = (system: string, timeBudgetMs: number) =>
    s.provider === "grok"
      ? runGrokChat({ key, system, messages: msg.messages, model, thinking: s.thinking, timeBudgetMs })
      : runChatProxy({
          key, system, messages: msg.messages, model, webSearch: s.webSearch, thinking: s.thinking, timeBudgetMs,
        });
  const started = Date.now();
  try {
    // Free web research (search + read top results) when the provider isn't
    // already searching natively — Grok, or Gemini with grounding off.
    const out = freeSearchApplies(s.provider, s.webSearch, s.freeWebSearch)
      ? await chatWithWebResearch({
          system: msg.system,
          messages: msg.messages,
          mode: s.freeWebSearch,
          timeBudgetMs: CHAT_TIME_BUDGET_MS,
          read: (url, limits) => scrapeUrl({
            url, allowlist: s.scrapeAllowlist.length ? s.scrapeAllowlist : undefined, resolveHost, ...limits,
          }),
          onError: (m) => log.warn(m),
        }, run)
      : await run(msg.system, CHAT_TIME_BUDGET_MS);
    log.info(`Chat #${msg.id}: ${model} answered in ${Date.now() - started} ms (key from ${found.source}).`);
    ctx.post({ type: "chatResult", id: msg.id, text: out.text, sources: out.sources });
  } catch (e) {
    const error =
      errMsg(e) === "TIMEOUT" ? `${store.label} did not answer within ${CHAT_TIME_BUDGET_MS / 1000}s — try again.` : errMsg(e);
    log.error(`Chat #${msg.id} failed: ${error}`);
    ctx.post({ type: "chatError", id: msg.id, error });
  }
}

async function handleScrape(msg: Extract<WebviewToHost, { type: "scrape" }>, ctx: BridgeContext): Promise<void> {
  const { scrapeAllowlist } = settings();
  try {
    const page = await scrapeUrl({
      url: msg.url,
      allowlist: scrapeAllowlist.length ? scrapeAllowlist : undefined,
      resolveHost,
    });
    log.info(`Scraped ${page.finalUrl} (${page.bytes} bytes${page.truncated ? ", truncated" : ""}).`);
    ctx.post({ type: "scrapeResult", id: msg.id, page });
  } catch (e) {
    const { error, status } = scrapeFailure(e);
    log.warn(`Scrape ${msg.url} failed (${status}): ${error}`);
    ctx.post({ type: "scrapeError", id: msg.id, error, status });
  }
}

async function handleSave(msg: Extract<WebviewToHost, { type: "save" }>, ctx: BridgeContext): Promise<void> {
  const filename = safeFilename(msg.filename);
  let target: vscode.Uri | undefined;
  try {
    const dir = saveDirFor(ctx.modelUri);
    target = await vscode.window.showSaveDialog({
      defaultUri: dir && vscode.Uri.joinPath(dir, filename),
      filters: saveFilters(filename),
      saveLabel: "Save",
      title: `Save ${filename}`,
    });
    if (!target) {
      ctx.post({ type: "saveResult", id: msg.id, saved: false });
      return;
    }
    await vscode.workspace.fs.writeFile(target, saveBytes(msg.data));
  } catch (e) {
    log.error(`Save ${filename} failed: ${errMsg(e)}`);
    ctx.post({ type: "saveResult", id: msg.id, saved: false, error: errMsg(e) });
    return;
  }
  const path = target.scheme === "file" ? target.fsPath : target.toString();
  log.info(`Saved ${path}.`);
  // Reply first: the notification below waits on the user and must not hold
  // up the WebView's promise.
  ctx.post({ type: "saveResult", id: msg.id, saved: true, path });
  void announceSaved(target);
}

/** Where the Save dialog starts: next to the model, unless its file system is
 *  read-only (e.g. a git: or zip-backed virtual FS), then the first writable
 *  workspace folder, else undefined so VS Code picks its own default.
 *  `isWritableFileSystem` returns undefined for unknown schemes — assume writable. */
function saveDirFor(modelUri: vscode.Uri): vscode.Uri | undefined {
  const writable = (u: vscode.Uri) => vscode.workspace.fs.isWritableFileSystem(u.scheme) !== false;
  if (writable(modelUri)) return vscode.Uri.joinPath(modelUri, "..");
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri;
  return folder && writable(folder) ? folder : undefined;
}

async function announceSaved(target: vscode.Uri): Promise<void> {
  const reveal = "Reveal";
  const open = "Open";
  const name = target.path.split("/").pop() ?? "file";
  const choice = await vscode.window.showInformationMessage(`ModelVisio: saved ${name}.`, reveal, open);
  if (choice === reveal) {
    // The OS file manager only exists for local files; otherwise use the Explorer.
    await vscode.commands.executeCommand(target.scheme === "file" ? "revealFileInOS" : "revealInExplorer", target);
  } else if (choice === open) {
    await vscode.commands.executeCommand("vscode.open", target);
  }
}

function handleNotify(msg: Extract<WebviewToHost, { type: "notify" }>): void {
  const text = `ModelVisio: ${msg.message}`;
  if (msg.level === "error") {
    log.error(msg.message);
    void vscode.window.showErrorMessage(text);
  } else if (msg.level === "warn") {
    log.warn(msg.message);
    void vscode.window.showWarningMessage(text);
  } else {
    log.info(msg.message);
    void vscode.window.showInformationMessage(text);
  }
}
