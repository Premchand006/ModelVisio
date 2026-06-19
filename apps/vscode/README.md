# ModelVisio — VS Code extension

Registers a **Custom Editor** so opening a model file (`.onnx`, `.tflite`,
`.pt`, `.pth`, `.mlmodel`, `.mlpackage`, `.gguf`, `.safetensors`) in the
Explorer renders it as the ModelVisio graph + inspector — instead of
showing binary garbage.

Thin shell: all UI is `@modelvisio/core`. This package adds the editor
registration, reads file bytes on the extension host, pushes them to the
WebView, and syncs the editor's color theme into the app.

## Architecture

- `src/extension.ts` — activates and registers the custom editor.
- `src/modelEditorProvider.ts` — `CustomReadonlyEditorProvider`: builds the
  WebView HTML (nonce + CSP), reads the file with `workspace.fs`, and
  message-passes `{ name, b64 }` to the WebView. Sends `{ theme }` on open and
  on `onDidChangeActiveColorTheme`.
- `webview/main.tsx` — mounts `core`'s `App`, decodes the bytes into a `File`,
  and drives it through the `onReady → openFile` handle. Theme arrives as
  `themeOverride` (live-synced).

The WebView's parse Worker may be blocked by the editor CSP; `core`'s
`parseModelInWorker` automatically falls back to main-thread parsing, so models
still load.

## Develop

```bash
pnpm --filter modelvisio-vscode build   # builds media/ (webview) + out/ (extension)
```

Then press **F5** in VS Code (uses `.vscode/launch.json`) to open an Extension
Development Host, and open any `.onnx` file from the Explorer.

> Register a `build-vscode-extension` task (or run `pnpm --filter
> modelvisio-vscode build` manually) before launching.

## Package

```bash
pnpm --filter modelvisio-vscode package   # → modelvisio.vsix
```

`vsce publish` puts it on the Marketplace (needs a publisher + PAT).

## AI chat

The copilot is wired through the extension host. The webview's `/api/chat`
request is intercepted and forwarded over the message channel to the host, which
holds the key and calls the Gemini API — the key is never exposed to the webview.

Set it up: **Settings → Extensions → ModelVisio → "Gemini API Key"** (free
key at <https://aistudio.google.com/apikey>), or export `GEMINI_API_KEY` in the
environment that launched VS Code. Optional: `modelvisio.geminiModel`,
`modelvisio.webSearch` (grounding; needs a paid-tier key).
