# @modelvisio/desktop — Tauri 2 shell

Wraps the `apps/web` build in a native window (~10MB, Rust + system WebView).
Thin shell only: all UI is `@modelvisio/core`; this package adds native glue
(window config, the **File → Open Model…** menu, OS file dialog).

## Prerequisites

Tauri needs the **Rust toolchain**:

- Install Rust: https://rustup.rs  (`rustc`, `cargo`)
- Platform deps: https://tauri.app/start/prerequisites/
  - **Windows:** Microsoft C++ Build Tools + WebView2 (preinstalled on Win 11)
  - **macOS:** Xcode Command Line Tools
  - **Linux:** `libwebkit2gtk-4.1-dev`, `librsvg2-dev`, `patchelf`, etc.

## Run / build

```bash
# from the repo root
pnpm --filter @modelvisio/desktop dev     # launches the native dev window
pnpm --filter @modelvisio/desktop build   # produces installers in src-tauri/target/release/bundle
```

`dev` runs `tauri dev`, which starts the web dev server (`devUrl`
http://localhost:5173) — stop any other dev server on that port first.
`build` runs the web production build then bundles the native app.

## How native file-open works

The native menu (Rust, `src-tauri/src/lib.rs`) opens an OS dialog filtered to
every supported model extension (`MODEL_EXTENSIONS`, mirrored from
`packages/parsers/src/registry.ts`), reads the file bytes, and emits a
`model-opened` event. The web layer
(`apps/web/src/tauri.ts`) listens via the global Tauri API (`withGlobalTauri`)
and hands a `File` to `core`'s `App` through the `onReady` → `openFile` handle.
The in-window drag-drop and "Browse Files" button also work (the WebView's HTML
file picker is itself native; `dragDropEnabled: false` lets the HTML5 drop fire).

## Icons

Generated from `app-icon.png` via `pnpm tauri icon app-icon.png`. Re-run after
changing the source to regenerate `src-tauri/icons/`.

## Release

Cut a cross-platform release (like Netron's Releases page) from the repo root:

```bash
pnpm release:desktop 1.0.0     # tags desktop-v1.0.0 and pushes it
```

That triggers `.github/workflows/desktop-release.yml`, which on macOS / Windows /
Linux runners:

1. Stamps the version (`1.0.0`) into `tauri.conf.json` + `Cargo.toml` via
   `scripts/stamp-version.mjs`, so installers are named `ModelVisio_1.0.0_...`.
2. Builds installers via `tauri-action` — Windows `.exe`/`.msi`, macOS `.dmg`
   (Apple Silicon + Intel), Linux `.AppImage`/`.deb`/`.rpm`.
3. Attaches them to a **draft** GitHub Release named `ModelVisio 1.0.0`. Review
   it on the Releases page and click **Publish** (draft = a single failed
   platform can never publish a half-finished release).

To verify the build compiles **without** cutting a release, run the
**Desktop Build Check** workflow (`workflow_dispatch`; also runs on PRs touching
the app) — it builds on all three OSes without publishing anything. It merges
`src-tauri/tauri.ci.conf.json` to turn off updater artifacts, so it needs **no
signing secrets** and works on PRs from forks (which GitHub never gives secrets).

Code-signing certs (Apple Developer, Windows EV) are optional and added later as
CI secrets; unsigned installers still run (with an OS "unidentified developer"
prompt the user clicks through).

### Auto-update

The app self-updates via `tauri-plugin-updater`: on launch it checks the latest
published release's `latest.json` and, when a newer **signed** build exists,
asks the user ("Update now" / "Later") before installing and restarting
(`check_for_updates` in `src-tauri/src/lib.rs`). Updates are verified
against the public key in `tauri.conf.json` (`plugins.updater.pubkey`).

CI signs the update artifacts with a private key held in two repo secrets,
`TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` (already
configured). The matching public key is committed; **never commit the private
key**. The release workflow's first step checks the secret is a real Tauri
private key and fails in seconds otherwise.

To rotate the keypair:

```bash
pnpm --filter @modelvisio/desktop exec tauri signer generate -w ~/.tauri/modelvisio.key
gh secret set TAURI_SIGNING_PRIVATE_KEY < ~/.tauri/modelvisio.key   # the whole file, not the path
gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD                    # the password you chose
```

then paste the contents of `~/.tauri/modelvisio.key.pub` into
`plugins.updater.pubkey` in `tauri.conf.json`. Rotating means apps installed with
the old key can no longer auto-update; users must reinstall once.

## AI chat

The copilot is wired through Tauri's Rust side. The frontend's `/api/chat` POST
is intercepted (`apps/web/src/tauri.ts` → `installTauriChatProxy`) and forwarded
to the `chat` command in `src-tauri/src/lib.rs`, which calls the Gemini API.

Bring-your-own-key: click the key icon in the AI Copilot and paste a free key
from <https://aistudio.google.com/apikey>. It is stored only on this device and
no key ships in the app. For local dev, `GEMINI_API_KEY` in the launching
environment is used as a fallback. Optional: `MODELVISIO_MODEL`
(default `gemini-2.5-flash`), `MODELVISIO_WEB_SEARCH` (Google Search grounding is
on by default; set to `off` to disable — the call degrades to ungrounded if
grounding isn't available on your key).
