// @modelvisio/ai — Gemini API prompt templates + client.
// Pure functions, no React. The API key NEVER lives here or in the browser;
// this calls the per-shell server-side proxy (see Security in CLAUDE.md).

/** Endpoint of the server-side proxy that holds GEMINI_API_KEY. */
export const DEFAULT_CHAT_ENDPOINT = "/api/chat";

/** Endpoint of the server-side scrape proxy (SSRF-guarded, size-capped). */
export const DEFAULT_SCRAPE_ENDPOINT = "/api/scrape";

// Re-export the pure URL extractor from the scrape module; the actual scrapeUrl
// implementation is server-side only and NOT re-exported here so scraper code
// never enters the browser bundle. Consumers call fetchScraped() below, which
// POSTs to the /api/scrape proxy.
export { extractUrls } from "./scrape";
export type { ScrapeResult } from "./scrape";

export type ChatMessage = { role: "user" | "assistant"; content: string };
export type Source = { title: string; url: string };
export type ChatResponse = { text: string; sources: Source[] };

/**
 * System prompt for the ModelVisio copilot — a deep edge-ML deployment
 * expert. The proxy enables Google Search grounding, so it can cite real,
 * current reference links; this prompt tells it the domain and when to cite.
 */
export function buildSystemPrompt(modelSummary: string): string {
  return [
    "You are ModelVisio's copilot — a world-class expert in neural-network",
    "architecture and edge/on-device deployment. You know, in depth:",
    "",
    "- ARCHITECTURES: CNNs (ResNet, MobileNet, EfficientNet), detectors (YOLO v5–v11/v26,",
    "  SSD, RetinaNet, DETR), transformers & ViT, segmentation (U-Net, DeepLab, SAM),",
    "  LLMs/SLMs (Llama, Qwen, Phi, Gemma), diffusion, RNN/LSTM, GNNs.",
    "- FORMATS: ONNX, TFLite/LiteRT, PyTorch (.pt/.pth)/TorchScript/torch.export/ExecuTorch,",
    "  TensorFlow/Keras SavedModel, Core ML, OpenVINO IR, TensorRT engines, Safetensors,",
    "  GGUF/GGML, NumPy, RKNN, ncnn, MNN, PaddlePaddle, Caffe, Darknet, MLIR, StableHLO.",
    "- LAYERS/OPS: conv variants (depthwise, grouped, transposed, dilated), attention,",
    "  normalization (BN/LN/GN/RMSNorm), activations (ReLU/SiLU/GELU/HardSwish), pooling,",
    "  residual/skip, SPPF, C2f, and how each maps to / fuses on compilers and runtimes.",
    "- QUANTIZATION: PTQ vs QAT, INT8/INT4/FP16/BF16/FP8, per-tensor vs per-channel,",
    "  calibration, mixed precision, quant-sensitivity, accuracy/latency tradeoffs.",
    "- HARDWARE: NVIDIA Jetson (Orin/Nano/AGX, TensorRT), Google Coral Edge TPU, Hailo-8/8L,",
    "  Qualcomm (SNPE/QNN), Rockchip RKNN, Intel (OpenVINO/Movidius/NPU), Apple ANE/Core ML,",
    "  ARM (CMSIS-NN, Ethos), MCUs (ESP32, STM32), and their memory/throughput envelopes.",
    "- COMPILERS & RUNTIMES: TensorRT, OpenVINO, TVM, XLA, ONNX Runtime, TFLite delegates,",
    "  CoreMLtools, and per-target op-support gaps + fusion behavior.",
    "",
    "RULES:",
    "1. Lead with the direct answer in 1-2 sentences, THEN the specifics. Be concise and to",
    "   the point — no filler, no restating the question, no generic 101 explanations. Prefer",
    "   compact bullets and short code/CLI over long prose.",
    "2. Be technically precise and actionable: name the exact ops, shapes, flags, CLI/Python,",
    "   precision, and target. Give a concrete recommendation, not a list of options.",
    "3. The app has ALREADY run a roofline-based hardware-fit analysis for THIS model — scores,",
    "   estimated FPS, FPS/W, the compute- vs memory-bound regime, memory-fit, and op/CPU-",
    "   fallback coverage are in the context below. USE it: cite the specific devices and",
    "   numbers, explain WHY (the binding bottleneck), and tailor optimization advice to this",
    "   model's actual bottleneck layers and quantization-sensitive layers — don't give advice",
    "   that ignores the computed analysis.",
    "4. When the user asks for references, links, docs, papers, or current/version-specific",
    "   facts (latest releases, benchmarks, op support), rely on Google Search grounding",
    "   (enabled) and cite the real sources it returns. Never invent a URL. Without grounded",
    "   results, only cite canonical stable docs you are certain exist.",
    "",
    "MODEL + COMPUTED ANALYSIS (ground every answer in this):",
    modelSummary,
  ].join("\n");
}

/** Stable-per-tab id so a session's upload + chat rows can be correlated. */
function sessionId(): string {
  try {
    const k = "mv_sid";
    let v = sessionStorage.getItem(k);
    if (!v) {
      v = (globalThis.crypto?.randomUUID?.() ?? String(Date.now()) + Math.random().toString(36).slice(2));
      sessionStorage.setItem(k, v);
    }
    return v;
  } catch {
    return "anon";
  }
}

export type UploadMeta = {
  modelName: string;
  format: string;
  framework: string;
  sizeBytes: number;
  params: number;
  layers: number;
};

/**
 * Fire-and-forget usage beacon for a loaded model. POSTs to the same proxy
 * endpoint as chat (so no extra/identifiable request shows up), where the server
 * records it. Never throws and never blocks the UI — all errors are swallowed.
 */
export function logUpload(meta: UploadMeta, endpoint: string = DEFAULT_CHAT_ENDPOINT): void {
  try {
    void fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      keepalive: true,
      body: JSON.stringify({
        event: {
          kind: "upload",
          session_id: sessionId(),
          model_name: meta.modelName,
          format: meta.format,
          framework: meta.framework,
          size_bytes: meta.sizeBytes,
          params: meta.params,
          layers: meta.layers,
          user_agent: typeof navigator !== "undefined" ? navigator.userAgent : undefined,
        },
      }),
    }).catch(() => {});
  } catch {
    // ignore — best-effort only
  }
}

export type SendChatArgs = {
  endpoint?: string;
  messages: ChatMessage[];
  modelSummary: string;
  signal?: AbortSignal;
};

/**
 * POSTs a chat turn to the server-side proxy and returns the assistant text +
 * any cited reference links. The proxy holds the API key, runs Claude with the
 * web-search tool, and extracts citations.
 */
export async function sendChat({
  endpoint = DEFAULT_CHAT_ENDPOINT,
  messages,
  modelSummary,
  signal,
}: SendChatArgs): Promise<ChatResponse> {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ system: buildSystemPrompt(modelSummary), messages }),
    signal,
  });
  const data = (await res.json().catch(() => ({}))) as { text?: string; sources?: Source[]; error?: string };
  if (!res.ok || data.error) {
    throw new Error(data.error || `Chat proxy returned ${res.status}. Is the serverless function running?`);
  }
  return { text: data.text ?? "(empty response)", sources: Array.isArray(data.sources) ? data.sources : [] };
}

// ─── Web scraping (real-time URL grounding) ──────────────────────────────────
// Companion to Gemini's google_search grounding. The user (or a copilot step)
// can pin a specific URL — a vendor op-support page, an MLPerf result table, a
// changelog — and get its readable text back to feed into the chat context, so
// answers stay grounded in the actual document rather than the model's prior.

export type ScrapedPage = {
  url: string;
  finalUrl: string;
  title: string;
  text: string;
  truncated: boolean;
  bytes: number;
  contentType: string;
};

export type FetchScrapedArgs = {
  url: string;
  endpoint?: string;
  signal?: AbortSignal;
};

/**
 * POST a URL to the /api/scrape proxy and return the page's readable text +
 * title. The proxy enforces SSRF guards, size caps, timeouts, and the optional
 * MODELVISIO_SCRAPE_ALLOWLIST — the browser never fetches cross-origin itself.
 */
export async function fetchScraped({
  url,
  endpoint = DEFAULT_SCRAPE_ENDPOINT,
  signal,
}: FetchScrapedArgs): Promise<ScrapedPage> {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
    signal,
  });
  const data = (await res.json().catch(() => ({}))) as Partial<ScrapedPage> & { error?: string };
  if (!res.ok || data.error) {
    throw new Error(data.error || `Scrape proxy returned ${res.status}.`);
  }
  return {
    url: data.url ?? url,
    finalUrl: data.finalUrl ?? url,
    title: data.title ?? "",
    text: data.text ?? "",
    truncated: !!data.truncated,
    bytes: data.bytes ?? 0,
    contentType: data.contentType ?? "",
  };
}

/**
 * Compact a scraped page into a system-prompt-ready block. Keeps the source
 * URL visible so the model can cite it in the reply.
 */
export function formatScrapedForPrompt(pages: ScrapedPage[]): string {
  if (pages.length === 0) return "";
  const parts = pages.map((p, i) => {
    const head = `[Source ${i + 1}] ${p.title || "(untitled)"} — ${p.finalUrl}`;
    return `${head}\n${p.text}${p.truncated ? "\n…(truncated)" : ""}`;
  });
  return [
    "REAL-TIME SCRAPED SOURCES (grounding for this turn):",
    "Use these verbatim over your prior knowledge. Cite the source URL when you draw on it.",
    "",
    parts.join("\n\n---\n\n"),
  ].join("\n");
}
