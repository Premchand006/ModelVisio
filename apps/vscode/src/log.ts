import * as vscode from "vscode";

/**
 * The one shared "ModelVisio" output channel (View → Output, or the
 * "ModelVisio: Show Logs" command). A LogOutputChannel adds timestamps and
 * levels, and honours the user's log-level choice for the channel.
 */
let channel: vscode.LogOutputChannel | undefined;

export function initLog(context: vscode.ExtensionContext): vscode.LogOutputChannel {
  if (!channel) {
    channel = vscode.window.createOutputChannel("ModelVisio", { log: true });
    context.subscriptions.push(channel, { dispose: () => (channel = undefined) });
  }
  return channel;
}

/** Logging that is safe to call before activation finished (or in tests). */
export const log = {
  info: (msg: string) => channel?.info(msg),
  warn: (msg: string) => channel?.warn(msg),
  error: (msg: string) => channel?.error(msg),
  debug: (msg: string) => channel?.debug(msg),
};

export function showLogs(): void {
  channel?.show(true);
}

/** Message text of any thrown value. */
export function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
