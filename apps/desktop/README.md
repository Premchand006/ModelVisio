# @modelvisio/desktop — Tauri 2 shell

Wraps the `apps/web` build in a native window (~10MB, Rust + system WebView).
Thin shell only: all UI is `@modelvisio/core`; this package adds native glue
(window config, the **File → Open Model…** menu, OS file dialog).

## Prerequisites

Tauri needs the **Rust toolchain** (not installed in this repo's environment):

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

The native menu (Rust, `src-tauri/src/lib.rs`) opens an OS dialog, reads the
file bytes, and emits a `model-opened` event. The web layer
(`apps/web/src/tauri.ts`) listens via the global Tauri API (`withGlobalTauri`)
and hands a `File` to `core`'s `App` through the `onReady` → `openFile` handle.
The in-window drag-drop and "Browse Files" button also work (the WebView's HTML
file picker is itself native; `dragDropEnabled: false` lets the HTML5 drop fire).

## Status

Scaffold complete and the web/TS integration is type-checked and built. The
Rust (`lib.rs`) has **not been compiled here** (no Rust toolchain in this env);
verify with `pnpm --filter @modelvisio/desktop dev` once Rust is installed.

## Icons

Generated from `app-icon.png` via `pnpm tauri icon app-icon.png`. Re-run after
changing the source to regenerate `src-tauri/icons/`.

## Release

Push a `desktop-v*` tag to trigger `.github/workflows/desktop-release.yml`,
which builds macOS / Windows / Linux installers via `tauri-action`.
Code-signing certs (Apple Developer, Windows EV) are added later as CI secrets.

## AI chat

The copilot is wired through Tauri's Rust side. The frontend's `/api/chat` POST
is intercepted (`apps/web/src/tauri.ts` → `installTauriChatProxy`) and forwarded
to the `chat` command in `src-tauri/src/lib.rs`, which holds the key and calls
the Gemini API — the key never reaches the WebView.

Set `GEMINI_API_KEY` in the environment that launches the app (free key at
<https://aistudio.google.com/apikey>). Optional: `MODELVISIO_MODEL`
(default `gemini-2.5-flash`), `MODELVISIO_WEB_SEARCH` (Google Search grounding is
on by default; set to `off` to disable — the call degrades to ungrounded if
grounding isn't available on your key).
