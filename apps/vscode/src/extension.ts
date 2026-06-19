import * as vscode from "vscode";
import { ModelEditorProvider } from "./modelEditorProvider";

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(ModelEditorProvider.register(context));
}

export function deactivate() {
  // no-op
}
