# Changelog

All notable changes to the ModelVisio VS Code extension are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- xAI **Grok** as a second AI copilot provider next to Gemini, through the shared
  `@modelvisio/ai/grok` client (OpenAI-compatible chat completions, low reasoning effort unless
  `modelvisio.thinking` is on).
- Settings: `modelvisio.provider` (`gemini` | `grok`, default `gemini`) and
  `modelvisio.grokModel` (default `grok-4.7`; only plain model ids, anything else falls back to
  the default).
- Commands: **Set Grok (xAI) API Key**, **Clear Grok (xAI) API Key**, **Select AI Provider
  (Gemini / Grok)**. The Grok key is kept in Secret Storage (`modelvisio.grokApiKey`), with the
  `XAI_API_KEY` environment variable as fallback.
- Free, keyless web search for the copilot (`modelvisio.freeWebSearch`: `auto` | `always` |
  `off`, default `auto`): searches DuckDuckGo, reads the top results through the SSRF-guarded
  scraper (honoring `modelvisio.scrapeAllowlist`) and cites them. Runs with Grok, and with Gemini
  when `modelvisio.webSearch` is off. A failed search never fails the chat.

### Changed

- `modelvisio.thinking` now also applies to Grok (default reasoning effort instead of low).
- `modelvisio.webSearch` is documented as Gemini only.
- Untrusted workspaces: `modelvisio.provider`, `modelvisio.grokModel` and
  `modelvisio.freeWebSearch` are also read only from user settings.

## [0.1.0]

First Marketplace release.

### Added

- Custom editor **ModelVisio** (`modelvisio.modelViewer`) that opens model files as the
  interactive graph, inspector, compiler pre-flight, hardware scoring and converter from
  `@modelvisio/core`. It is the default editor for 33 model extensions (`.onnx`, `.tflite`,
  `.pt`, `.safetensors`, `.gguf`, …).
- Opt-in editor (`modelvisio.modelViewerOption`) for 27 generic extensions (`.json`, `.bin`,
  `.xml`, `.pb`, `.h5`, `.har`, …), offered via **Open with ModelVisio** and **Reopen Editor
  With…** without replacing the text editor. `.har` is opt-in so browser HTTP Archives keep
  opening as JSON.
- Commands: **Open Model File…**, **Open with ModelVisio** (Explorer and editor-tab context
  menus), **Set Gemini API Key**, **Clear Gemini API Key**, **Show Logs**.
- AI copilot through the extension host using the shared `@modelvisio/ai` chat proxy. The
  Gemini key is kept in VS Code Secret Storage. Lookup order: Secret Storage → legacy
  `modelvisio.geminiApiKey` *user* setting (workspace values are never used) →
  `GEMINI_API_KEY`. A key in user settings is migrated to Secret Storage once; clearing the
  stored key afterwards is not undone by a later migration.
- `modelvisio.geminiModel` accepts only plain model ids; anything else falls back to
  `gemini-2.5-flash`.
- URL reading in the copilot through the shared SSRF-guarded scraper, with the
  `modelvisio.scrapeAllowlist` setting.
- Exports and downloads (SVG/PNG, converted models, deploy scripts) open a native Save dialog.
- Live reload when the model file changes on disk.
- `modelvisio.maxFileSizeMB` setting (default 2048) with a clear error for larger files.
- "ModelVisio" output channel for diagnostics.
- Limited support for untrusted workspaces: the API key, model, web-search, thinking and
  scrape-allowlist settings are never read from workspace settings there.
- Typed host/view message protocol. Model bytes are sent as a `Uint8Array` instead of
  base64.

### Fixed (compared with the pre-release shell)

- The view crashed at load with `ReferenceError: process is not defined`, because the webview
  bundle kept React's `process.env.NODE_ENV` check.
- Usage beacons sent on every model load were forwarded as chat requests and triggered
  Gemini calls. They are now answered locally and never leave the view.
- Exports silently did nothing, because `<a download>` has no effect in a webview.
- Models were parsed on the UI thread, because the webview couldn't start the parse worker
  from a cross-origin URL, so large models froze the view. The worker now starts from a
  `blob:` URL.
- ONNX parsing failed under the webview CSP. protobufjs needs `'unsafe-eval'`. Script loading
  still requires the nonce.
- The packaged extension no longer depends on `node_modules`: the host is bundled with esbuild.

[Unreleased]: https://github.com/Premchand006/ModelVisio/compare/vscode-v0.1.0...HEAD
[0.1.0]: https://github.com/Premchand006/ModelVisio/releases/tag/vscode-v0.1.0
