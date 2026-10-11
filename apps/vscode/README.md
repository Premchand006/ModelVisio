# ModelVisio for VS Code

Open neural-network model files right in the editor. Instead of a wall of
binary, `.onnx`, `.tflite`, `.pt`, `.safetensors`, `.gguf` and 30+ other
formats open as an **interactive graph** with a per-layer **inspector**, a
**compiler pre-flight** for edge accelerators, **hardware-fit scoring**, and an
optional **AI copilot** that answers questions about *your* model.

It is the same app as the [ModelVisio](https://github.com/Premchand006/ModelVisio)
website and desktop app, running inside a VS Code custom editor.

![Graph + inspector](https://raw.githubusercontent.com/Premchand006/ModelVisio/main/docs/screenshots/graph.png)

## Features

- **Graph + inspector** — Netron-style layered graph; shapes, weights, FLOPs/MACs and
  quantization sensitivity per layer; SVG/PNG export.
- **Compiler pre-flight** — per-target op-support and memory warnings (Coral, RKNN, Hailo,
  Kneron, …) surfaced on the graph, with reversible auto-fixes.
- **Hardware-fit scoring** — roofline estimates across 21 edge accelerators: FPS, FPS/W,
  compute- vs memory-bound, and a memory-fit guard.
- **Converter + deploy recipes** — Graph-JSON / Layers-CSV / Safetensors / NumPy exports and
  TensorRT / HailoRT / RKNN scripts. Every export opens a native **Save** dialog.
- **AI copilot** (optional, bring your own Gemini or xAI Grok key) — grounded in the model's
  computed analysis; paste a URL and it reads the page, and a free, keyless web search cites live
  sources when you ask about docs, versions, benchmarks or op support.
- **Live reload** — the view re-parses when the file changes on disk (e.g. re-exported from a
  training script).
- **Follows your theme** — light / dark switches with the VS Code color theme.
- Parsing runs in a background Web Worker, so large models don't freeze the editor.

## Supported formats

| Opens by default | Graph | Detected (metadata) |
| --- | --- | --- |
| `.onnx` `.tflite` `.lite` `.tfl` `.safetensors` `.gguf` `.ggml` `.npy` `.npz` `.pt` `.pth` `.ckpt` | ✔ | |
| `.ort` `.ptl` `.torchscript` `.pt2` `.pte` `.mlmodel` `.keras` `.caffemodel` `.pdmodel` `.pdparams` `.mnn` `.tnnmodel` `.tnnproto` `.rknn` `.uff` `.cntk` `.mge` `.nntxt` `.mlnet` `.bigdl` `.cbm` | | ✔ |

Generic extensions that other tools also use — `.json` `.xml` `.bin` `.cfg` `.pb` `.pbtxt`
`.meta` `.h5` `.hdf5` `.params` `.param` `.nb` `.weights` `.nn` `.plan` `.engine` `.trt`
`.prototxt` `.pkl` `.pickle` `.joblib` `.mlir` `.dnn` `.om` `.tm` `.hn` `.har` — are **not**
taken over. Open them with right-click → **Open with ModelVisio**, or **Reopen Editor With… →
ModelVisio**. (`.cfg` Darknet and `.bin` PyTorch files get a full graph. `.har` is opt-in
because the extension is far more often a browser HTTP Archive than a Hailo archive.)

"Detected" formats are identified and shown as metadata; their graph parsers are still in
progress. Folder bundles (`.mlpackage`, `.mlmodelc`) can't be opened by a custom editor.

## Commands

| Command | Where | What it does |
| --- | --- | --- |
| **ModelVisio: Open Model File…** | Command Palette | Pick a file and open it in ModelVisio |
| **ModelVisio: Open with ModelVisio** | Command Palette, Explorer and editor-tab context menus | Open the selected file(s), or the active tab, in ModelVisio |
| **ModelVisio: Set Gemini API Key** | Command Palette | Store your key in VS Code's encrypted Secret Storage |
| **ModelVisio: Clear Gemini API Key** | Command Palette | Remove the stored key |
| **ModelVisio: Set Grok (xAI) API Key** | Command Palette | Store your xAI key in Secret Storage |
| **ModelVisio: Clear Grok (xAI) API Key** | Command Palette | Remove the stored xAI key |
| **ModelVisio: Select AI Provider (Gemini / Grok)** | Command Palette | Switch the copilot's provider (user setting); offers to add a key if the choice has none |
| **ModelVisio: Show Logs** | Command Palette | Open the **ModelVisio** output channel |

## Settings

| Setting | Default | Description |
| --- | --- | --- |
| `modelvisio.maxFileSizeMB` | `2048` | Largest file the viewer will load. Bigger files show an error with a link to this setting. |
| `modelvisio.provider` | `gemini` | Which model answers in the copilot: `gemini` or `grok`. |
| `modelvisio.geminiModel` | `gemini-2.5-flash` | Gemini model used by the copilot (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`). A value that isn't a plain model id (letters, digits, `.`, `-`) falls back to `gemini-2.5-flash`. |
| `modelvisio.grokModel` | `grok-4.7` | xAI Grok model used when the provider is `grok`. Same plain-id rule; anything else falls back to `grok-4.7`. |
| `modelvisio.webSearch` | `false` | Gemini only: Google Search grounding, so answers cite live links. Needs a paid-tier key; with it off, the free web search below covers live answers. |
| `modelvisio.freeWebSearch` | `auto` | Free, keyless web search (DuckDuckGo): reads the top results through the same guarded reader as pasted URLs and cites them. `auto` searches only when the question asks for something live (docs, links, versions, releases, benchmarks, op support); `always` every question; `off` never. Used with Grok, and with Gemini when `webSearch` is off. |
| `modelvisio.thinking` | `false` | Let the model think before answering — `gemini-2.5` thinking, or Grok at its default reasoning effort instead of low (slower, sometimes better). |
| `modelvisio.scrapeAllowlist` | `[]` | Host suffixes the copilot may read — pasted URLs and free web search results (e.g. `docs.nvidia.com`). Empty = any public host. |
| `modelvisio.geminiApiKey` | `""` | **Deprecated** plain-text Gemini key. Use **Set Gemini API Key** instead. (Grok has no plain-text setting.) |

## AI copilot setup

1. Get a free Gemini key at <https://aistudio.google.com/apikey>, or an xAI Grok key at
   <https://console.x.ai>.
2. Run **ModelVisio: Set Gemini API Key** (or **Set Grok (xAI) API Key**) and paste it. It is
   stored in VS Code's Secret Storage (the OS keychain), not in `settings.json`.
3. For Grok, run **ModelVisio: Select AI Provider** and pick **Grok (xAI)** (or set
   `modelvisio.provider` to `grok`).
4. Open a model and switch to the **AI** tab.

The Grok key is looked up in Secret Storage, then the `XAI_API_KEY` environment variable of the
process that launched VS Code. The Gemini key is looked up in this order: Secret Storage → the legacy `modelvisio.geminiApiKey`
setting in your *user* settings → the `GEMINI_API_KEY` environment variable of the process that
launched VS Code. A `modelvisio.geminiApiKey` in a workspace or folder `.vscode/settings.json` is
never used, so a shared repository can't make you call Gemini with its key. A key found in your
user settings is moved into Secret Storage once, on first run, and you're offered to delete the
plain-text copy. That migration never runs again, so **Clear Gemini API Key** stays cleared (a
key you kept in user settings still applies until you remove it; the command tells you so).

## Privacy

- Model files are read and parsed **locally**. Nothing is uploaded to open or analyze a model.
- API keys stay in the extension host. The view never sees them; the host calls the Gemini or
  xAI API directly.
- The copilot sends a request only when you send a chat message. That request goes to the
  provider you picked (Google's Gemini API or xAI's Grok API) and contains your messages and a
  summary of the model's computed analysis (layer stats, hardware scores, compiler issues), not
  the model file.
- With `modelvisio.freeWebSearch` on (`auto` by default), a question that asks for something
  live also sends a search query — your latest message, with URLs removed — to DuckDuckGo, and
  the host reads the top results like pasted URLs. Set it to `off` to never search. The usage beacon the website
  logs on each upload (model name, format, size) is answered inside the extension and never
  sent anywhere.
- Pasted URLs and search results are fetched by the extension host. Private, loopback and
  cloud-metadata addresses are always refused, including hostnames that resolve to them and
  redirects to them. `modelvisio.scrapeAllowlist` narrows reading to the hosts you list. Page
  text is passed to the model as untrusted data, never as instructions.
- The view loads its fonts from Google Fonts.
- **Untrusted workspaces:** the extension runs in Restricted Mode, but `modelvisio.provider`,
  `modelvisio.geminiApiKey`, `modelvisio.geminiModel`, `modelvisio.grokModel`,
  `modelvisio.webSearch`, `modelvisio.freeWebSearch`, `modelvisio.thinking` and
  `modelvisio.scrapeAllowlist` are read only from your user settings, never from the
  workspace's `.vscode/settings.json` — a repository can't switch your key to another provider
  or a pricier model, or turn on billed features or web search. (The API key setting is ignored at workspace level even in
  trusted workspaces.)

## Troubleshooting

- **Nothing happens / blank view:** run **ModelVisio: Show Logs** and check the output.
- **"over the … MB limit":** raise `modelvisio.maxFileSizeMB`. Very large models can use a lot of
  memory.
- **"No Gemini API key configured"** / **"No Grok (xAI) API key configured":** run **ModelVisio:
  Set Gemini API Key** / **Set Grok (xAI) API Key**, or **Select AI Provider** to switch to the
  provider you have a key for.
- **Answers say live results were unavailable:** the free web search failed (often DuckDuckGo
  rate-limiting); the reason is in **Show Logs**. The copilot still answers from the model's
  own knowledge.
- Report issues at <https://github.com/Premchand006/ModelVisio/issues>.

---

## Development

The extension is a thin shell in the [ModelVisio monorepo](https://github.com/Premchand006/ModelVisio).
All UI and analysis is `@modelvisio/core` (React) and `@modelvisio/parsers`. This package adds
the editor registration, a typed host/view message channel, the AI and scraper bridge, and
native save dialogs.

### Architecture

```text
extension host (Node, out/extension.js)        webview (media/webview.js)
──────────────────────────────────────         ─────────────────────────────────
ModelEditorProvider ── model bytes ──────────▶ modelSink → core App.openFile
  workspace.fs.readFile + file watcher           └─ parse Web Worker (blob: URL)
  theme on open / on change ── theme ────────▶ themeOverride
bridge.ts ◀── chat / scrape / save / notify ── fetchBridge (/api/chat, /api/scrape)
  runChatProxy (@modelvisio/ai/proxy)          save.ts (core setSaveHandler)
   or runGrokChat (@modelvisio/ai/grok)
   in chatWithWebResearch (@modelvisio/ai/search)
  scrapeUrl    (@modelvisio/ai/scrape)
  showSaveDialog + workspace.fs.writeFile
```

#### Host (`src/`)

- `extension.ts` — activation; registers the editors and the commands.
- `modelEditorProvider.ts` — one `CustomReadonlyEditorProvider` serving two view types:
  `modelvisio.modelViewer` (`priority: default`, unambiguous model extensions) and
  `modelvisio.modelViewerOption` (`priority: option`, generic extensions). It reads the
  file with `workspace.fs` (enforcing `maxFileSizeMB`), posts the bytes as a `Uint8Array`, and
  watches the file to re-send it after changes (debounced).
- `protocol.ts` — the typed message contract in both directions; `messages.ts` validates every
  incoming message and drops anything malformed.
- `bridge.ts` — handles `chat` (dispatches to the shared `runChatProxy` for Gemini or
  `runGrokChat` for Grok per `modelvisio.provider`, wrapped in `chatWithWebResearch` when the
  free web search applies — Grok, or Gemini with `webSearch` off; empty usage beacons are
  rejected, never billed), `scrape` (shared SSRF-guarded `scrapeUrl` +
  `modelvisio.scrapeAllowlist`), `save` (native Save dialog) and `notify`.
- `secrets.ts` / `apiKey.ts` — one Secret Storage key store per provider (`modelvisio.geminiApiKey`,
  `modelvisio.grokApiKey`), lookup order, legacy Gemini-setting migration, the provider picker.
- `html.ts` — webview HTML and CSP. Scripts load only with a per-render nonce. `'unsafe-eval'` is
  allowed because protobufjs (ONNX) compiles decoders with `Function()`; it does not let
  injected `<script>` tags run.
- `formats.ts` — extension lists for both editors (tests keep them in sync with `package.json`
  and the parser registry). `files.ts`, `config.ts` / `geminiModel.ts` (Gemini and Grok model-id
  sanitizing), `log.ts` — helpers, settings, output channel.

#### Webview (`webview/`)

- `main.tsx` — installs the platform glue, then mounts core's `App`.
- `rpc.ts` — id-correlated request/response over `postMessage`.
- `fetchBridge.ts` — intercepts the copilot's `POST /api/chat` and `/api/scrape` and forwards
  them to the host, so core's chat code runs unchanged.
- `workerShim.ts` — a webview can't start a Worker from its cross-origin resource URL, so the
  parse-worker chunk is prefetched and started from a `blob:` URL. If that fails, core parses
  on the main thread.
- `save.ts` — routes core downloads (`setSaveHandler`) to the host's Save dialog.
- `modelSink.ts`, `ErrorBanner.tsx`, `vscodeApi.ts` — buffering, error UI, API handle.

#### Bundling

- Host: `esbuild.mjs` → `out/extension.js` (CommonJS, everything inlined except `vscode`, so
  the `.vsix` needs no `node_modules`).
- Webview: `vite.webview.config.ts` → `media/webview.js` (IIFE) + `media/assets/parse.worker-*.js`.
  The build fails if the worker chunk isn't self-contained.
- `.vscodeignore` is an allowlist: `package.json`, `out/extension.js`, `media/**`, `icon.png`,
  `README.md`, `CHANGELOG.md`, `LICENSE`; source maps are excluded.

### Build, run, test

From the repo root, after `pnpm install`:

```bash
pnpm --filter modelvisio-vscode build       # media/ (webview) + out/ (host)
pnpm --filter modelvisio-vscode typecheck
pnpm --filter modelvisio-vscode test        # vitest: host logic + webview glue
```

To debug, open **`apps/vscode` as the workspace folder** and press **F5**. The *Run Extension*
launch config runs the `build-vscode-extension` task (`pnpm build`) first, then starts an
Extension Development Host. Open any model file there.

### Package and install

```bash
pnpm --filter modelvisio-vscode package      # → apps/vscode/modelvisio.vsix
code --install-extension apps/vscode/modelvisio.vsix
```

Releases: pushing a `vscode-v<version>` tag runs `.github/workflows/vscode-release.yml`, which
attaches the `.vsix` to a GitHub Release and publishes to the Marketplace when the `VSCE_PAT`
secret is set. Bump `version` in `package.json` and add a `CHANGELOG.md` entry first.
