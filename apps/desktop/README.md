# @modelvisio/desktop (Tauri 2 app)

Wraps the `apps/web` build in a native window (~10MB, Rust + system WebView).
All UI comes from `@modelvisio/core`. This package only adds the native parts:
window config, the **File > Open Model…** and **Open Recent** menus, the OS file
dialog, and Rust-side proxies for the AI copilot (Gemini chat, reading pasted URLs).

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
http://localhost:5173), so stop any other dev server on that port first.
`build` runs the web production build then bundles the native app.

## How native file-open works

The native menu (Rust, `src-tauri/src/lib.rs`) opens an OS dialog filtered to
every supported model extension (`MODEL_EXTENSIONS`, mirrored from
`packages/parsers/src/registry.ts`), reads the file bytes, and emits a
`model-opened` event. The web layer
(`apps/web/src/tauri.ts`) listens via the global Tauri API (`withGlobalTauri`)
and hands a `File` to `core`'s `App` through the `onReady`/`openFile` handle.
The in-window drag-drop and "Browse Files" button also work (the WebView's HTML
file picker is itself native; `dragDropEnabled: false` lets the HTML5 drop fire).

### Open Recent

**File > Open Recent** lists the last 10 models opened through the native side
(File > Open Model… or Open Recent itself), most recent first, each shown as
`file.onnx (folder)`. The list is saved to `recent-models.json` in the app config
directory (`%APPDATA%\com.modelvisio.app` on Windows,
`~/Library/Application Support/com.modelvisio.app` on macOS,
`~/.config/com.modelvisio.app` on Linux). Clicking an entry reads the file and
emits the same `model-opened` event as the Open dialog. If the file has been moved
or deleted it is dropped from the list and a dialog says so. **Clear Recently
Opened** empties the list.

Files dropped on the window or picked with "Browse Files" go through the WebView,
which never sees a file path, so they aren't added to the list.

The same list is available to the web layer (typed wrappers in
`apps/web/src/tauri.ts`): `getRecentModels()` (`recent_models` command, returns
`[{ path, name }]`), `openRecentModel(path)` (`open_recent`; only listed paths
are accepted), `clearRecentModels()` (`clear_recent_models`) and
`onRecentModelsChanged(cb)` (the `recent-models-changed` event, sent on every
change).

## Icons

Generated from `app-icon.png` via `pnpm tauri icon app-icon.png`. Re-run after
changing the source to regenerate `src-tauri/icons/`.

## Release

To cut a release for all three platforms, run this from the repo root:

```bash
pnpm release:desktop 1.0.0     # tags desktop-v1.0.0 and pushes it
```

That triggers `.github/workflows/desktop-release.yml`, which on macOS / Windows /
Linux runners:

1. Stamps the version (`1.0.0`) into `tauri.conf.json` + `Cargo.toml` via
   `scripts/stamp-version.mjs`, so installers are named `ModelVisio_1.0.0_...`.
2. Builds installers via `tauri-action`: Windows `.exe`/`.msi`, one
   **universal** macOS `.dmg` (Apple Silicon + Intel in a single app,
   `--target universal-apple-darwin`), Linux `.AppImage`/`.deb`/`.rpm`.
3. Attaches them to a **draft** GitHub Release named `ModelVisio 1.0.0`. Review
   it on the Releases page and click **Publish**. Because it starts as a draft,
   one failed platform can't publish a half-finished release.
4. Uploads copies with fixed names that the website links to, which resolve to the
   newest **published** release:
   `/releases/latest/download/ModelVisio-macOS-universal.dmg`,
   `ModelVisio-Windows-x64-setup.exe`, `ModelVisio-Linux-x86_64.AppImage`.
   The landing page detects the visitor's OS and offers the matching one
   (`packages/core/src/data/links.ts`).

### macOS (.dmg)

Step-by-step: [docs/release-macos.md](../../docs/release-macos.md) (build, test on a Mac,
publish, verify links) and the user-facing [docs/install-macos.md](../../docs/install-macos.md).

`.dmg` files can only be built on macOS, so they come from CI (or `pnpm --filter
@modelvisio/desktop build -- --target universal-apple-darwin` on a Mac with both
`aarch64-apple-darwin` and `x86_64-apple-darwin` Rust targets installed).

- **Installer window:** `src-tauri/dmg/background.png` (660×400, matching
  `bundle.macOS.dmg.windowSize`) shows a "drag to Applications" arrow between
  the app and Applications icons. Keep it light, because Finder draws the icon
  labels in dark text.
- **Signing:** with no Apple Developer account the app is **ad-hoc signed**
  (`bundle.macOS.signingIdentity: "-"`). Without at least an ad-hoc signature,
  Apple Silicon Macs call a downloaded app "damaged". The release fails if
  `codesign --verify` doesn't pass on the built `.app`.
- **First launch (not notarized):** macOS blocks it once. The user opens the app,
  then **System Settings > Privacy & Security > Open Anyway** (macOS 15+; on 14
  and older, Control-click and choose Open). Terminal alternative:
  `xattr -dr com.apple.quarantine /Applications/ModelVisio.app`. The website's
  download panel and the release notes show these steps.
- **Native menus:** macOS gets the standard app / File / Edit / View / Window
  menus. The Edit menu is what makes ⌘C / ⌘V / ⌘A work in text fields. The
  WebView doesn't handle those shortcuts without it.
- **Try a build without releasing:** run the **Desktop Build Check** workflow and
  download the `ModelVisio-macOS-universal-dmg` artifact from the run.

**Developer ID signing + notarization (optional, removes the Open Anyway
step):** add these repo secrets and the release picks them up automatically:
`APPLE_CERTIFICATE` (base64 `.p12`), `APPLE_CERTIFICATE_PASSWORD`,
`APPLE_SIGNING_IDENTITY` (e.g. `Developer ID Application: Name (TEAMID)`), and for
notarization `APPLE_ID`, `APPLE_PASSWORD` (an app-specific password),
`APPLE_TEAM_ID`. They're only exported when set: Tauri treats an *empty*
variable as present, which would break the build.

To verify the build compiles **without** cutting a release, run the
**Desktop Build Check** workflow (`workflow_dispatch`; also runs on PRs touching
the app). It builds on all three OSes without publishing anything. It merges
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
is intercepted (`installTauriChatProxy` in `apps/web/src/tauri.ts`) and forwarded
to the `chat` command in `src-tauri/src/lib.rs`, which calls the Gemini API.

You bring your own key: click the key icon in the AI Copilot and paste a free key
from <https://aistudio.google.com/apikey>. It is stored only on this device and
no key ships in the app. For local dev, `GEMINI_API_KEY` in the launching
environment is used as a fallback. Optional: `MODELVISIO_MODEL`
(default `gemini-2.5-flash`), `MODELVISIO_WEB_SEARCH` (Google Search grounding is
on by default; set to `off` to disable it. If your key can't use grounding, the call
falls back to an ungrounded answer).

### Reading pasted URLs

Paste a link into the copilot (a vendor op-support page, a GitHub issue, a
changelog) and it reads the page before answering, as on the web. The chat's
`/api/scrape` POST is routed to the `scrape` command (`src-tauri/src/scrape.rs`),
a Rust port of the web scraper (`packages/ai/src/scrape.ts`) with the same
limits and replies:

- Only `http(s)` URLs. Hosts that are or **resolve to** loopback, private,
  link-local, unique-local, CGNAT, multicast or cloud-metadata addresses are
  refused (IPv4 embedded in IPv6 included). Redirects are followed by hand (up
  to 5) and every hop is checked before it is requested.
- The connection is pinned to the addresses that passed the check, so a DNS
  answer that changes between check and connect can't reach a private address.
  For the same reason no HTTP proxy is used for these requests.
- Only HTML, plain text, XHTML and JSON are read: at most 512 KiB, 8 s in total,
  and 12 000 characters of extracted text (title + readable body; scripts,
  styles and the `<head>` are dropped).

The `MODELVISIO_SCRAPE_ALLOWLIST` (comma-separated host suffixes),
`MODELVISIO_SCRAPE_MAX_BYTES`, `MODELVISIO_SCRAPE_MAX_CHARS` and
`MODELVISIO_SCRAPE_TIMEOUT_MS` environment variables work as on the web, read
from the environment that launched the app.
