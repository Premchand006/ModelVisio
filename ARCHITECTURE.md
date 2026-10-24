# ModelVisio architecture and conventions

Neural-network model analyzer for edge deployment: a Netron-style graph viewer, an edge hardware advisor and an AI copilot in one tool. It ships as a website, a VS Code extension and a desktop app from one shared core.

## Architecture

This is a pnpm monorepo. One core package is shared by three apps. Don't duplicate UI logic across apps.

```
packages/
  core/      React component library. ALL app UI lives here (graph, inspector, compiler checker, converters, hardware report, AI chat).
  parsers/   Real model-format parsing → normalized graph objects. Pure TS, no React.
  ai/        Copilot prompt templates + client (used by core's Chat, investigator and deployment agent), plus server-only modules: Gemini proxy (proxy.ts), xAI Grok (grok.ts), provider selection (providers.ts), free keyless web search (search.ts), SSRF-guarded URL scraper (scrape.ts).
  cli/       `modelvisio` Node CLI for CI. Reuses parsers + core's pure subpath exports (no React).
apps/
  web/       Vite + React + TS. Imports core. Deploys to Vercel/Netlify.
  desktop/   Tauri 2. Wraps the web build in a native WebView.
  vscode/    VS Code extension. CustomEditor opens core in a WebView for model files.
```

Features go in `packages/core`. The three apps only mount core and add platform code (file pickers, hosting, WebView bridges), so a fix in core reaches all three.

## Stack decisions (don't change without discussion)

- Package manager: **pnpm workspaces** (Turborepo optional for caching)
- Bundler: **Vite** (web), wrapped by Tauri and the VS Code WebView
- Desktop: **Tauri 2** (Rust, ~10MB), not Electron unless a local Node backend becomes necessary
- Language: **TypeScript** everywhere
- Styling: Tailwind + the existing theme-context system (dark/light)
- ONNX parsing: `onnxruntime-web` + `protobufjs`
- AI: Google Gemini (default; free key from Google AI Studio) or xAI Grok via a **server-side proxy** (see Security). Live grounding from Google Search (Gemini) or a free, keyless web search (DuckDuckGo / self-hosted SearXNG) read through the guarded scraper.

## Normalized model object

Every parser MUST output this shape so the existing graph renderer works unchanged:

```ts
type ModelLayer = {
  id: number; name: string; type: string; op: string;
  shape: string; dt: string;
  params: number; flops: number; macs: number; mem: number;
  group: "input"|"backbone"|"neck"|"head"|"output";
  ins: { n: string; s?: number[] }[];
  outs: { n: string; s?: number[] }[];
  attr: Record<string, unknown>;
  w: { shape: number[]|string; size?: number; min?: number; max?: number; mean?: number; std?: number; sparse?: number } | null;
  math: string; insight: string;
  qSens: number;                 // quantization sensitivity 0..1
  compIssues: { target: string; severity: "warn"|"error"; msg: string }[];
  constFolded?: boolean;         // shape arithmetic converters fold away (never on the device)
};
type Model = {
  name: string; format: string; framework: string; opset: number;
  sizeBytes: number; inputShape: number[]; outputShape: number[];
  layers: ModelLayer[]; edges: [number, number][];
  producer?: string; irVersion?: number; doc?: string;
  notes?: string[];              // caveats shown to the user (assumed input size, missing weight file)
};
```

`parseModel(bytes, name, companions?)` takes sibling weight files (`companionNames()` lists
what a format wants) and fills `compIssues` with the rule-based checks in
`packages/parsers/src/compat.ts` for every format, so parsers themselves leave it empty.

## Supported formats (parser targets, priority order)

1. ONNX  2. TFLite  3. PyTorch (.pt/.pth) / TorchScript / torch.export / ExecuTorch
4. TensorFlow / Keras  5. Core ML  6. OpenVINO  7. Safetensors / GGUF / NumPy
8. RKNN / ncnn / MNN  9. PaddlePaddle  10. Caffe / Darknet / MLIR / JAX / scikit-learn

Build ONNX fully end-to-end before starting others. Each parser needs tests against a real small model file.

## Security

- **Never ship the AI provider key to the browser.** The web app calls a serverless proxy (`api/chat` on Vercel, `netlify/functions/chat` on Netlify) that holds `GEMINI_API_KEY` (or `XAI_API_KEY` for Grok, handled the same way; `MODELVISIO_PROVIDER` picks) as an env var. Locally, a dev-only Vite middleware (`apps/web/vite.config.ts`) serves `/api/chat` from the repo-root `.env`. The desktop app proxies through Tauri's Rust side. The VS Code extension proxies through the extension host (keys in Secret Storage). Shared call logic: `@modelvisio/ai/proxy` (Gemini), `@modelvisio/ai/grok`, `@modelvisio/ai/providers`. These server modules are never re-exported from the browser entry (`index.ts`).
- **Crawled text is untrusted.** Pasted URLs and free-web-search result pages (`@modelvisio/ai/search`; only the fixed search endpoint itself is fetched directly) are read only through `@modelvisio/ai/scrape` (blocks private/loopback/metadata hosts incl. DNS results and redirects; size/time caps; `MODELVISIO_SCRAPE_ALLOWLIST`) and fenced in the prompt as data, never instructions. Pasted-URL reading (`/api/scrape`) is SSRF-guarded on every surface; on desktop it is the Rust `scrape` command (`apps/desktop/src-tauri/src/scrape.rs`), which also pins connections to the checked addresses.
- Parse uploaded files in a **Web Worker** so the main thread never blocks. Large models (100MB+) must stream nodes into the graph progressively (`parseModelStream` in `packages/parsers/src/stream.ts`; ONNX and TFLite stream per node, other formats fall back to one chunk).

## Build order

1. Monorepo skeleton; `pnpm dev` serves apps/web rendering a component from packages/core
2. Move existing app into packages/core, split into per-component files
3. **ONNX parser** in packages/parsers, wired to the live graph (replaces demo data)
4. apps/web production hardening, Web Worker parsing and the serverless AI proxy; deploy to Vercel
5. apps/desktop (Tauri): native file dialogs, recent-models list, multi-OS installers via tauri-action
6. apps/vscode: CustomEditor for model file extensions, WebView message-passing, theme sync
7. Remaining parsers, one at a time, each with tests

## Conventions

- One component per file in packages/core. Keep the theme context shared.
- Pure functions in parsers/ and ai/, with no React imports.
- Every parser change ships with a test using a real fixture model.
- Commit after each working milestone; keep PRs scoped to one build-order step.

## Product roadmap

Phase 1 (done): Graph + Inspector + Compiler compatibility pre-flight + Auto-fix engine + Deploy recipe generator
Phase 2 (done): Quantization heatmap + real on-device benchmarking + hardware-aware model advisor
Phase 3 (done): AI performance investigator + cross-compiler optimization search
Phase 4 (done): AI deployment agent + fleet simulation + model registry / CI-CD

Where each lives in packages/core/src: quant/, advisor/, bench/ (onnxruntime-web, own worker, lazy-loaded), investigate/, optsearch/, deploy/, fleet/, registry/. AI-backed features always have a deterministic fallback so they work without a key.
