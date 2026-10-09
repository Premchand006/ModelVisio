import * as vscode from "vscode";
import { existsSync } from "node:fs";
import type { HostToWebview } from "./protocol";
import { handleRequest, rejectMalformed } from "./bridge";
import { settings } from "./config";
import { dirnamePosix, formatBytes, sizeLimitError, toTransferable, watchGlobFor } from "./files";
import { basename, VIEW_TYPE, VIEW_TYPE_OPTION } from "./formats";
import { buildHtml, makeNonce } from "./html";
import { parseWebviewMessage } from "./messages";
import type { ApiKeys } from "./secrets";
import { errMsg, log } from "./log";

/** Coalesces the burst of events one save produces (truncate + write, or an
 *  atomic rename's delete + create) into a single reload. */
const RELOAD_DEBOUNCE_MS = 300;

/**
 * Opens model files in a WebView running the @modelvisio/core app. Read-only:
 * we never write the model back. File bytes are read on the extension host and
 * pushed to the WebView as a Uint8Array, which parses them in its worker.
 * Serves both viewTypes (default + "Reopen With…" option) with one instance.
 */
export class ModelEditorProvider implements vscode.CustomReadonlyEditorProvider {
  static register(context: vscode.ExtensionContext, keys: ApiKeys): vscode.Disposable[] {
    const provider = new ModelEditorProvider(context, keys);
    const options = {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: false,
    };
    return [VIEW_TYPE, VIEW_TYPE_OPTION].map((viewType) =>
      vscode.window.registerCustomEditorProvider(viewType, provider, options),
    );
  }

  private constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly keys: ApiKeys,
  ) {}

  openCustomDocument(uri: vscode.Uri): vscode.CustomDocument {
    return { uri, dispose: () => undefined };
  }

  async resolveCustomEditor(document: vscode.CustomDocument, panel: vscode.WebviewPanel): Promise<void> {
    const { uri } = document;
    const name = basename(uri.path) || "model";
    const webview = panel.webview;
    webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")],
    };
    webview.html = this.getHtml(webview);

    let disposed = false;
    const post = (msg: HostToWebview) => {
      if (disposed) return;
      try {
        void webview.postMessage(msg).then(undefined, (e) => log.debug(`postMessage failed: ${errMsg(e)}`));
      } catch (e) {
        log.debug(`postMessage failed: ${errMsg(e)}`);
      }
    };
    const postTheme = () => post({ type: "theme", theme: currentTheme() });
    const sendModel = this.modelSender(uri, name, post, () => disposed);

    const subs: vscode.Disposable[] = [];
    subs.push(
      webview.onDidReceiveMessage((raw: unknown) => {
        const msg = parseWebviewMessage(raw);
        if (!msg) return rejectMalformed(raw, post);
        if (msg.type === "ready") {
          // Also re-sent if the WebView reloads itself (e.g. after a crash).
          postTheme();
          void sendModel(false);
          return;
        }
        handleRequest(msg, { post, keys: this.keys, modelUri: uri }).catch((e) =>
          log.error(`Unhandled error in ${msg.type} handler: ${errMsg(e)}`),
        );
      }),
      vscode.window.onDidChangeActiveColorTheme(postTheme),
      ...this.watch(uri, () => void sendModel(true)),
    );
    panel.onDidDispose(() => {
      disposed = true;
      subs.forEach((s) => s.dispose());
    });
  }

  /**
   * Reads the file and posts it. A sequence number drops results that a newer
   * (re)load overtook, so a slow read can't replace fresher bytes.
   */
  private modelSender(
    uri: vscode.Uri,
    name: string,
    post: (msg: HostToWebview) => void,
    isDisposed: () => boolean,
  ): (reload: boolean) => Promise<void> {
    let seq = 0;
    let offeredSizeSetting = false;
    return async (reload) => {
      const mine = ++seq;
      const started = Date.now();
      try {
        const stat = await vscode.workspace.fs.stat(uri);
        if (stat.type & vscode.FileType.Directory) throw new Error(`${name} is a folder, not a model file.`);
        const tooBig = sizeLimitError(name, stat.size, settings().maxFileSizeMB);
        if (tooBig) {
          if (mine !== seq || isDisposed()) return;
          log.warn(tooBig);
          post({ type: "modelError", name, error: tooBig });
          // Toast once per panel, on open only: a file being rewritten fires a
          // reload per save, and those must not stack identical warnings.
          if (!reload && !offeredSizeSetting) {
            offeredSizeSetting = true;
            void offerSizeSetting(tooBig);
          }
          return;
        }
        const bytes = await vscode.workspace.fs.readFile(uri);
        if (mine !== seq || isDisposed()) return;
        post(reload ? { type: "model", name, bytes: toTransferable(bytes), reload: true } : { type: "model", name, bytes: toTransferable(bytes) });
        log.info(`${reload ? "Reloaded" : "Loaded"} ${name} (${formatBytes(bytes.byteLength)}) in ${Date.now() - started} ms.`);
      } catch (e) {
        if (mine !== seq || isDisposed()) return;
        const error =
          e instanceof vscode.FileSystemError && e.code === "FileNotFound"
            ? `${name} was deleted or moved.`
            : `Could not read ${name}: ${errMsg(e)}`;
        log.error(error);
        post({ type: "modelError", name, error });
      }
    };
  }

  /**
   * Live reload: watch exactly this file. Every event (change, create, delete)
   * funnels into one debounced re-read — a delete then surfaces as modelError
   * from the read, while an atomic save (delete + create) just reloads.
   */
  private watch(uri: vscode.Uri, reload: () => void): vscode.Disposable[] {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const target = uri.toString();
    const onEvent = (changed: vscode.Uri) => {
      if (changed.toString() !== target) return; // the glob may over-match siblings
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = undefined;
        reload();
      }, RELOAD_DEBOUNCE_MS);
    };
    try {
      const base = uri.with({ path: dirnamePosix(uri.path) });
      const watcher = vscode.workspace.createFileSystemWatcher(
        new vscode.RelativePattern(base, watchGlobFor(basename(uri.path))),
      );
      return [
        watcher,
        watcher.onDidChange(onEvent),
        watcher.onDidCreate(onEvent),
        watcher.onDidDelete(onEvent),
        { dispose: () => timer && clearTimeout(timer) },
      ];
    } catch (e) {
      log.warn(`Live reload unavailable for ${uri.toString()}: ${errMsg(e)}`);
      return [];
    }
  }

  private getHtml(webview: vscode.Webview): string {
    const media = vscode.Uri.joinPath(this.context.extensionUri, "media");
    // Vite emits a stylesheet only if the WebView build stops injecting CSS
    // from JS; link it when present so either build mode renders styled.
    const styleUris =
      media.scheme === "file"
        ? ["webview.css", "style.css"]
            .map((f) => vscode.Uri.joinPath(media, f))
            .filter((u) => existsSync(u.fsPath))
            .map((u) => webview.asWebviewUri(u).toString())
        : [];
    return buildHtml({
      cspSource: webview.cspSource,
      nonce: makeNonce(),
      scriptUri: webview.asWebviewUri(vscode.Uri.joinPath(media, "webview.js")).toString(),
      styleUris,
    });
  }
}

async function offerSizeSetting(message: string): Promise<void> {
  const open = "Open Settings";
  if ((await vscode.window.showWarningMessage(message, open)) === open) {
    await vscode.commands.executeCommand("workbench.action.openSettings", "modelvisio.maxFileSizeMB");
  }
}

function currentTheme(): "dark" | "light" {
  const k = vscode.window.activeColorTheme.kind;
  return k === vscode.ColorThemeKind.Light || k === vscode.ColorThemeKind.HighContrastLight ? "light" : "dark";
}
