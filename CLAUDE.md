# ModelVisio — Project Context

AI-native neural-network model analyzer for edge deployment. Think "Netron + TensorRT advisor + AI copilot" in one tool. Ships as a **website**, **VS Code extension**, and **desktop app** from a single shared core.

## Architecture (read this first)

This is a **pnpm monorepo**. One engine, three shells — never duplicate UI logic across apps.

```
packages/
  core/      React component library — ALL app UI lives here (graph, inspector, compiler checker, converters, hardware report, AI chat). This is the product.
  parsers/   Real model-format parsing → normalized graph objects. Pure TS, no React.
  ai/        Gemini API prompt templates + client. Used by core's Chat.
apps/
  web/       Vite + React + TS. Imports core. Deploys to Vercel/Netlify.
  desktop/   Tauri 2. Wraps the web build in a native WebView.
  vscode/    VS Code extension. CustomEditor opens core in a WebView for model files.
```

**Golden rule:** features go in `packages/core`. The three apps are thin shells that mount core + provide platform glue (file pickers, hosting, WebView bridges). A fix in core must benefit all three surfaces automatically.

## Stack decisions (locked — don't swap without discussion)

- Package manager: **pnpm workspaces** (Turborepo optional for caching)
- Bundler: **Vite** (web), wrapped by Tauri and the VS Code WebView
- Desktop: **Tauri 2** (Rust, ~10MB) — NOT Electron, unless a local Node backend becomes necessary
- Language: **TypeScript** everywhere
- Styling: Tailwind + the existing theme-context system (dark/light)
- ONNX parsing: `onnxruntime-web` + `protobufjs`
- AI: Google Gemini API via a **server-side proxy** (see Security). Free key from Google AI Studio.

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
};
type Model = {
  name: string; format: string; framework: string; opset: number;
  sizeBytes: number; inputShape: number[]; outputShape: number[];
  layers: ModelLayer[]; edges: [number, number][];
};
```

## Supported formats (parser targets, priority order)

1. ONNX  2. TFLite  3. PyTorch (.pt/.pth) / TorchScript / torch.export / ExecuTorch
4. TensorFlow / Keras  5. Core ML  6. OpenVINO  7. Safetensors / GGUF / NumPy
8. RKNN / ncnn / MNN  9. PaddlePaddle  10. Caffe / Darknet / MLIR / JAX / scikit-learn

Build ONNX fully end-to-end before starting others. Each parser needs tests against a real small model file.

## Security (non-negotiable)

- **Never ship the Gemini API key to the browser.** The web app calls a serverless proxy (`api/chat` on Vercel, `netlify/functions/chat` on Netlify) that holds `GEMINI_API_KEY` as an env var. Locally, a dev-only Vite middleware (`apps/web/vite.config.ts`) serves `/api/chat` from the repo-root `.env`. The desktop app proxies through Tauri's Rust side. The VS Code extension proxies through the extension host. Shared call logic: `@modelvisio/ai/proxy`.
- Parse uploaded files in a **Web Worker** — never block the main thread. Large models (100MB+) must stream nodes into the graph progressively.

## Build order

1. Monorepo skeleton; `pnpm dev` serves apps/web rendering a component from packages/core
2. Move existing app into packages/core, split into per-component files
3. **ONNX parser** in packages/parsers, wired to the live graph (replaces demo data)
4. apps/web production hardening + Web Worker parsing + serverless AI proxy → deploy to Vercel
5. apps/desktop (Tauri): native file dialogs, recent-models list, multi-OS installers via tauri-action
6. apps/vscode: CustomEditor for model file extensions, WebView message-passing, theme sync
7. Remaining parsers, one at a time, each with tests

## Conventions

- One component per file in packages/core. Keep the theme context shared.
- Pure functions in parsers/ and ai/ — no React imports.
- Every parser change ships with a test using a real fixture model.
- Commit after each working milestone; keep PRs scoped to one build-order step.

## Product roadmap (informs priorities)

Phase 1: Graph + Inspector + Compiler compatibility pre-flight + Auto-fix engine + Deploy recipe generator
Phase 2: Quantization heatmap + real on-device benchmarking + hardware-aware model advisor
Phase 3: AI performance investigator + cross-compiler optimization search
Phase 4: AI deployment agent + fleet simulation + model registry / CI-CD
