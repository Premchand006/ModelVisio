import * as vscode from "vscode";
import {
  API_KEY_URL,
  MISSING_KEY_MESSAGE,
  resolveApiKey,
  shouldMigrate,
  userSettingKey,
  validateApiKeyInput,
  type KeySource,
} from "./apiKey";
import { errMsg, log } from "./log";

/** SecretStorage slot. Same id as the legacy setting, for discoverability. */
const SECRET_ID = "modelvisio.geminiApiKey";
const LEGACY_SETTING = "geminiApiKey";
/** globalState flag: the legacy setting has been dealt with (migrated, or
 *  superseded by a key the user stored or cleared) — never migrate again. */
const MIGRATED_FLAG = "modelvisio.keyMigrated";

/** The legacy setting's user-level value; workspace values are never used. */
function legacyUserKey(): string | undefined {
  return userSettingKey(vscode.workspace.getConfiguration("modelvisio").inspect<string>(LEGACY_SETTING));
}

/**
 * Where the Gemini key lives. The key only ever leaves the extension host as
 * the x-goog-api-key header to Google — it is never posted to the WebView.
 */
export class ApiKeyStore {
  constructor(
    private readonly secrets: vscode.SecretStorage,
    private readonly state: vscode.Memento,
  ) {}

  /** Secret may be unreadable (e.g. no OS keyring on Linux): treat as unset. */
  private async readSecret(): Promise<string | undefined> {
    try {
      return await this.secrets.get(SECRET_ID);
    } catch (e) {
      log.warn(`Secret Storage unavailable: ${errMsg(e)}`);
      return undefined;
    }
  }

  /** SecretStorage → legacy `modelvisio.geminiApiKey` *user* setting → GEMINI_API_KEY env. */
  async resolve(): Promise<{ key: string; source: KeySource } | null> {
    return resolveApiKey({
      secret: await this.readSecret(),
      setting: legacyUserKey(),
      env: process.env.GEMINI_API_KEY,
    });
  }

  async store(key: string): Promise<void> {
    await this.secrets.store(SECRET_ID, key.trim());
  }

  /** Also marks migration done, so the next activation can't copy the legacy
   *  setting back into Secret Storage and silently undo the clear. */
  async clear(): Promise<void> {
    await this.secrets.delete(SECRET_ID);
    await this.markMigrated();
  }

  private async markMigrated(): Promise<void> {
    await this.state.update(MIGRATED_FLAG, true);
  }

  /**
   * One-shot move of the legacy plain-text user setting into SecretStorage,
   * then offer to delete the plain-text copy. Only the *user* (global) value is
   * migrated: a workspace-level value comes from a checked-in/shared
   * .vscode/settings.json and must not be silently adopted as the user's own key.
   */
  async migrateLegacySetting(): Promise<void> {
    if (this.state.get<boolean>(MIGRATED_FLAG)) return;
    const legacy = legacyUserKey();
    const secret = await this.readSecret();
    if (!shouldMigrate(secret, legacy)) {
      // A stored key already supersedes the setting; nothing left to migrate.
      if (secret?.trim()) await this.markMigrated();
      return;
    }
    try {
      await this.store(legacy!);
    } catch (e) {
      log.warn(`Could not migrate the Gemini API key to Secret Storage: ${errMsg(e)}`);
      return; // not marked: retry on the next activation
    }
    // Mark before prompting: whether the user removes, keeps or dismisses the
    // plain-text copy, it must not be migrated again (e.g. after a Clear).
    await this.markMigrated();
    log.info("Migrated the Gemini API key from settings into Secret Storage.");
    const remove = "Remove from Settings";
    const choice = await vscode.window.showInformationMessage(
      "ModelVisio moved your Gemini API key into encrypted Secret Storage. Remove the plain-text copy from your settings?",
      remove,
      "Keep",
    );
    if (choice === remove) {
      await vscode.workspace
        .getConfiguration("modelvisio")
        .update(LEGACY_SETTING, undefined, vscode.ConfigurationTarget.Global);
    }
  }
}

/** "ModelVisio: Set Gemini API Key" — password InputBox with a "get a key" link button. */
export async function promptForApiKey(store: ApiKeyStore): Promise<boolean> {
  const input = vscode.window.createInputBox();
  input.title = "ModelVisio: Gemini API Key";
  input.prompt = `Paste your Google Gemini API key. Get a free one at ${API_KEY_URL} (button above).`;
  input.placeholder = "AIza…";
  input.password = true;
  input.ignoreFocusOut = true;
  input.buttons = [{ iconPath: new vscode.ThemeIcon("link-external"), tooltip: `Get a free key at ${API_KEY_URL}` }];

  const value = await new Promise<string | undefined>((resolve) => {
    const subs = [
      input.onDidChangeValue((v) => (input.validationMessage = v ? validateApiKeyInput(v) ?? undefined : undefined)),
      input.onDidTriggerButton(() => void vscode.env.openExternal(vscode.Uri.parse(API_KEY_URL))),
      input.onDidAccept(() => {
        const err = validateApiKeyInput(input.value);
        if (err) {
          input.validationMessage = err;
          return;
        }
        resolve(input.value.trim());
        input.hide();
      }),
      input.onDidHide(() => {
        resolve(undefined);
        subs.forEach((s) => s.dispose());
        input.dispose();
      }),
    ];
    input.show();
  });
  if (!value) return false;

  try {
    await store.store(value);
  } catch (e) {
    void vscode.window.showErrorMessage(`Could not save the API key: ${errMsg(e)}`);
    return false;
  }
  log.info("Gemini API key saved to Secret Storage.");
  void vscode.window.showInformationMessage("ModelVisio: Gemini API key saved securely.");
  return true;
}

/** "ModelVisio: Clear Gemini API Key". Mentions any fallback key still in effect. */
export async function clearApiKey(store: ApiKeyStore): Promise<void> {
  try {
    await store.clear();
  } catch (e) {
    void vscode.window.showErrorMessage(`Could not clear the API key: ${errMsg(e)}`);
    return;
  }
  log.info("Gemini API key removed from Secret Storage.");
  const still = await store.resolve();
  const tail =
    still?.source === "setting"
      ? ' A key is still set in your user settings ("modelvisio.geminiApiKey") and will still be used.'
      : still?.source === "env"
        ? " GEMINI_API_KEY from the environment will still be used."
        : "";
  void vscode.window.showInformationMessage(`ModelVisio: Gemini API key removed from Secret Storage.${tail}`);
}

let missingKeyNoticeOpen = false;

/** Notification with a "Set API Key" button; at most one on screen at a time
 *  so several queued chat turns don't stack identical toasts. */
export function notifyMissingKey(): void {
  if (missingKeyNoticeOpen) return;
  missingKeyNoticeOpen = true;
  const set = "Set API Key";
  void vscode.window.showWarningMessage(MISSING_KEY_MESSAGE, set).then((choice) => {
    missingKeyNoticeOpen = false;
    if (choice === set) void vscode.commands.executeCommand("modelvisio.setApiKey");
  });
}
