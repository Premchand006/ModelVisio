import * as vscode from "vscode";
import {
  PROVIDERS,
  missingKeyMessage,
  resolveApiKey,
  shouldMigrate,
  userSettingKey,
  validateApiKeyInput,
  type ChatProvider,
  type KeySource,
} from "./apiKey";
import { errMsg, log } from "./log";

/** Legacy plain-text Gemini key setting (Grok never had one). */
const LEGACY_SETTING = "geminiApiKey";
/** globalState flag: the legacy setting has been dealt with (migrated, or
 *  superseded by a key the user stored or cleared) — never migrate again. */
const MIGRATED_FLAG = "modelvisio.keyMigrated";

/** The legacy setting's user-level value; workspace values are never used. */
function legacyUserKey(): string | undefined {
  return userSettingKey(vscode.workspace.getConfiguration("modelvisio").inspect<string>(LEGACY_SETTING));
}

/**
 * Where one provider's API key lives. The key only ever leaves the extension
 * host as the request header to that provider (x-goog-api-key for Gemini, a
 * Bearer token for xAI) — it is never posted to the WebView.
 */
export class ApiKeyStore {
  readonly provider: ChatProvider;
  private readonly secrets: vscode.SecretStorage;
  private readonly state: vscode.Memento;

  constructor(secrets: vscode.SecretStorage, state: vscode.Memento, provider: ChatProvider = "gemini") {
    this.secrets = secrets;
    this.state = state;
    this.provider = provider;
  }

  get label(): string {
    return PROVIDERS[this.provider].label;
  }

  private get secretId(): string {
    return PROVIDERS[this.provider].secretId;
  }

  /** Secret may be unreadable (e.g. no OS keyring on Linux): treat as unset. */
  private async readSecret(): Promise<string | undefined> {
    try {
      return await this.secrets.get(this.secretId);
    } catch (e) {
      log.warn(`Secret Storage unavailable: ${errMsg(e)}`);
      return undefined;
    }
  }

  /** SecretStorage → (Gemini only) legacy `modelvisio.geminiApiKey` *user*
   *  setting → GEMINI_API_KEY / XAI_API_KEY env. */
  async resolve(): Promise<{ key: string; source: KeySource } | null> {
    return resolveApiKey({
      secret: await this.readSecret(),
      setting: this.provider === "gemini" ? legacyUserKey() : undefined,
      env: process.env[PROVIDERS[this.provider].envVar],
    });
  }

  async store(key: string): Promise<void> {
    await this.secrets.store(this.secretId, key.trim());
  }

  /** For Gemini, also marks migration done, so the next activation can't copy
   *  the legacy setting back into Secret Storage and silently undo the clear. */
  async clear(): Promise<void> {
    await this.secrets.delete(this.secretId);
    if (this.provider === "gemini") await this.markMigrated();
  }

  private async markMigrated(): Promise<void> {
    await this.state.update(MIGRATED_FLAG, true);
  }

  /**
   * One-shot move of the legacy plain-text Gemini user setting into
   * SecretStorage, then offer to delete the plain-text copy. Only the *user*
   * (global) value is migrated: a workspace-level value comes from a
   * checked-in/shared .vscode/settings.json and must not be silently adopted as
   * the user's own key. No-op for Grok.
   */
  async migrateLegacySetting(): Promise<void> {
    if (this.provider !== "gemini" || this.state.get<boolean>(MIGRATED_FLAG)) return;
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

/** One key store per copilot provider. */
export type ApiKeys = Record<ChatProvider, ApiKeyStore>;

export function createApiKeys(secrets: vscode.SecretStorage, state: vscode.Memento): ApiKeys {
  return {
    gemini: new ApiKeyStore(secrets, state, "gemini"),
    grok: new ApiKeyStore(secrets, state, "grok"),
  };
}

/** "ModelVisio: Set Gemini / Grok API Key" — password InputBox with a "get a key" link button. */
export async function promptForApiKey(store: ApiKeyStore): Promise<boolean> {
  const info = PROVIDERS[store.provider];
  const input = vscode.window.createInputBox();
  input.title = `ModelVisio: ${info.label} API Key`;
  input.prompt = `Paste your ${info.label} API key. Get one at ${info.keyUrl} (button above).`;
  input.placeholder = info.placeholder;
  input.password = true;
  input.ignoreFocusOut = true;
  input.buttons = [{ iconPath: new vscode.ThemeIcon("link-external"), tooltip: `Get a key at ${info.keyUrl}` }];

  const value = await new Promise<string | undefined>((resolve) => {
    const subs = [
      input.onDidChangeValue((v) => (input.validationMessage = v ? validateApiKeyInput(v) ?? undefined : undefined)),
      input.onDidTriggerButton(() => void vscode.env.openExternal(vscode.Uri.parse(info.keyUrl))),
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
  log.info(`${info.label} API key saved to Secret Storage.`);
  void vscode.window.showInformationMessage(`ModelVisio: ${info.label} API key saved securely.`);
  return true;
}

/** "ModelVisio: Clear Gemini / Grok API Key". Mentions any fallback key still in effect. */
export async function clearApiKey(store: ApiKeyStore): Promise<void> {
  const info = PROVIDERS[store.provider];
  try {
    await store.clear();
  } catch (e) {
    void vscode.window.showErrorMessage(`Could not clear the API key: ${errMsg(e)}`);
    return;
  }
  log.info(`${info.label} API key removed from Secret Storage.`);
  const still = await store.resolve();
  const tail =
    still?.source === "setting"
      ? ' A key is still set in your user settings ("modelvisio.geminiApiKey") and will still be used.'
      : still?.source === "env"
        ? ` ${info.envVar} from the environment will still be used.`
        : "";
  void vscode.window.showInformationMessage(`ModelVisio: ${info.label} API key removed from Secret Storage.${tail}`);
}

const missingKeyNoticeOpen = new Set<ChatProvider>();

/** Notification with a "Set API Key" button; at most one per provider on
 *  screen at a time so several queued chat turns don't stack identical toasts. */
export function notifyMissingKey(provider: ChatProvider = "gemini"): void {
  if (missingKeyNoticeOpen.has(provider)) return;
  missingKeyNoticeOpen.add(provider);
  const set = "Set API Key";
  void vscode.window.showWarningMessage(missingKeyMessage(provider), set).then((choice) => {
    missingKeyNoticeOpen.delete(provider);
    if (choice === set) void vscode.commands.executeCommand(PROVIDERS[provider].setCommand);
  });
}

/** "ModelVisio: Select AI Provider" — switch the copilot between Gemini and
 *  Grok (user setting), then offer to add a key if the choice has none. */
export async function selectProvider(keys: ApiKeys): Promise<void> {
  const current = vscode.workspace.getConfiguration("modelvisio").get<string>("provider") ?? "gemini";
  const items = (Object.keys(PROVIDERS) as ChatProvider[]).map((id) => ({
    id,
    label: PROVIDERS[id].label,
    description: id === current ? "current" : undefined,
    detail: id === "gemini"
      ? "Google Gemini — free key from Google AI Studio."
      : "xAI Grok — key from console.x.ai. Live answers via the free web search.",
  }));
  const picked = await vscode.window.showQuickPick(items, { title: "ModelVisio: AI copilot provider" });
  if (!picked) return;
  await vscode.workspace.getConfiguration("modelvisio").update("provider", picked.id, vscode.ConfigurationTarget.Global);
  log.info(`Copilot provider set to ${picked.label}.`);
  if (!(await keys[picked.id].resolve())) await promptForApiKey(keys[picked.id]);
}
