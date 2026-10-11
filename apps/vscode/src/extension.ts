import * as vscode from "vscode";
import { ModelEditorProvider } from "./modelEditorProvider";
import { ApiKeyStore, clearApiKey, promptForApiKey } from "./secrets";
import { openDialogFilters, viewTypeFor } from "./formats";
import { errMsg, initLog, log, showLogs } from "./log";

export function activate(context: vscode.ExtensionContext) {
  initLog(context);
  const keys = new ApiKeyStore(context.secrets, context.globalState);

  context.subscriptions.push(
    ...ModelEditorProvider.register(context, keys),
    vscode.commands.registerCommand("modelvisio.setApiKey", () => promptForApiKey(keys)),
    vscode.commands.registerCommand("modelvisio.clearApiKey", () => clearApiKey(keys)),
    vscode.commands.registerCommand("modelvisio.showLogs", showLogs),
    vscode.commands.registerCommand("modelvisio.openModel", openModel),
    vscode.commands.registerCommand("modelvisio.openWith", openWith),
  );

  // Fire-and-forget: the migration prompt must not delay opening the editor
  // that triggered activation.
  keys.migrateLegacySetting().catch((e) => log.warn(`API key migration failed: ${errMsg(e)}`));
}

export function deactivate() {
  // no-op: everything is disposed through context.subscriptions
}

/** "ModelVisio: Open Model File…" — pick a file, open it in our editor. */
async function openModel(): Promise<void> {
  const picked = await vscode.window.showOpenDialog({
    canSelectMany: false,
    canSelectFolders: false,
    openLabel: "Open in ModelVisio",
    filters: openDialogFilters(),
  });
  if (picked?.[0]) await openInModelVisio(picked[0]);
}

/**
 * "Open with ModelVisio" from the Explorer / editor-tab context menus, which
 * pass (clicked, allSelected). From the Command Palette there is no argument,
 * so fall back to the active tab, then to the file picker.
 */
async function openWith(clicked?: vscode.Uri, selected?: vscode.Uri[]): Promise<void> {
  const targets = selected?.length ? selected : clicked instanceof vscode.Uri ? [clicked] : [];
  if (!targets.length) {
    const active = activeTabUri();
    if (!active) return openModel();
    targets.push(active);
  }
  for (const uri of targets) await openInModelVisio(uri);
}

async function openInModelVisio(uri: vscode.Uri): Promise<void> {
  try {
    await vscode.commands.executeCommand("vscode.openWith", uri, viewTypeFor(uri.path));
  } catch (e) {
    log.error(`Could not open ${uri.toString()}: ${errMsg(e)}`);
    void vscode.window.showErrorMessage(`ModelVisio could not open the file: ${errMsg(e)}`);
  }
}

function activeTabUri(): vscode.Uri | undefined {
  const input = vscode.window.tabGroups.activeTabGroup.activeTab?.input;
  if (input instanceof vscode.TabInputText || input instanceof vscode.TabInputCustom) return input.uri;
  return undefined;
}
