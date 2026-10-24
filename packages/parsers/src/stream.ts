import type { Model } from "./types";
import { detectFormat, parseModel } from "./registry";
import { headerOf, type ModelStream, type ParseChunk } from "./common/chunks";
import { onnxStream } from "./onnx/parse";
import { tfliteStream } from "./formats/tflite";
import { ggufStream } from "./formats/gguf";
import { safetensorsStream } from "./formats/safetensors";
import { annotateCompat, annotateLayers } from "./compat";
import type { Companions } from "./companions";

// Progressive (streaming) parsing: the registry-level counterpart of
// parseModel. Large models (100MB+) stream layers into the graph as they are
// parsed instead of arriving all at once.

export type { ModelHeader, ModelStream, ParseChunk } from "./common/chunks";

/** Files at or above this size are parsed progressively by the UI. */
export const STREAM_THRESHOLD_BYTES = 100 * 1024 * 1024;

/** Default number of layers per streamed chunk. */
export const DEFAULT_STREAM_BATCH = 256;

export type StreamOptions = { batchSize?: number; companions?: Companions };

/** Formats with a real incremental path (others arrive as one final chunk). */
export const STREAMING_FORMATS: readonly string[] = ["onnx", "tflite", "gguf", "safetensors"];

/**
 * Parse any supported model file as a generator: yields ParseChunks as layers
 * are produced and returns the final Model, identical to parseModel's result.
 * Formats without an incremental path yield one chunk holding the whole model.
 */
export function* parseModelStream(
  input: ArrayBuffer | Uint8Array,
  name: string,
  opts: StreamOptions = {},
): ModelStream {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const batch = Math.max(1, Math.floor(opts.batchSize ?? DEFAULT_STREAM_BATCH));
  switch (detectFormat(name, buf)) {
    case "onnx": return yield* withCompat(onnxStream(buf, { name }, batch));
    case "tflite": return yield* withCompat(tfliteStream(buf, name, batch));
    case "gguf": return yield* withCompat(ggufStream(buf, name, batch));
    case "safetensors": return yield* withCompat(safetensorsStream(buf, name, batch));
    default: {
      const model = parseModel(buf, name, opts.companions);
      yield { layers: model.layers, edges: model.edges, header: headerOf(model), progress: 1 };
      return model;
    }
  }
}

/** Annotate compatibility issues on each streamed chunk, matching parseModel's output. */
function* withCompat(stream: ModelStream): ModelStream {
  for (;;) {
    const r = stream.next();
    if (r.done) return annotateCompat(r.value);
    annotateLayers(r.value.layers);
    yield r.value;
  }
}

/**
 * Synchronous driver for parseModelStream: calls onChunk for every chunk and
 * returns the final Model. Suited to a Worker, where blocking is fine.
 */
export function parseModelProgressive(
  input: ArrayBuffer | Uint8Array,
  name: string,
  onChunk: (chunk: ParseChunk) => void,
  opts?: StreamOptions,
): Model {
  const it = parseModelStream(input, name, opts);
  for (;;) {
    const r = it.next();
    if (r.done) return r.value;
    onChunk(r.value);
  }
}
