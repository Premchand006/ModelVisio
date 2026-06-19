import { convertWeights, parseModel, type ConvertResult, type ConvTarget, type Model, type Precision } from "@modelvisio/parsers";

// Client for the parse/convert worker. Spawns one lazily, reuses it, and falls
// back to main-thread work if Workers aren't available (e.g. a strict CSP in
// the VS Code WebView).
type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };

let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, Pending>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./parse.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent) => {
      const { id, model, result, error } = e.data as { id: number; model?: Model; result?: ConvertResult; error?: string };
      const p = pending.get(id);
      if (!p) return;
      pending.delete(id);
      if (error) p.reject(new Error(error));
      else p.resolve(result ?? model);
    };
    worker.onerror = (e) => {
      const err = new Error(e.message || "Worker crashed");
      pending.forEach((p) => p.reject(err));
      pending.clear();
    };
  }
  return worker;
}

/** Parse an already-read buffer off the main thread. */
export async function parseBufferInWorker(buffer: ArrayBuffer, name: string): Promise<Model> {
  // Fall back to the main thread ONLY when a Worker can't be created (e.g. a
  // strict CSP). Real parse errors must propagate, not silently re-run here
  // (which would double the work and freeze the UI on large models).
  let w: Worker;
  try {
    w = getWorker();
  } catch {
    return parseModel(buffer, name);
  }
  const id = ++seq;
  return new Promise<Model>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    w.postMessage({ kind: "parse", id, buffer, name });
  });
}

/** Parse a model file off the main thread, returning the normalized Model. */
export async function parseModelInWorker(file: File): Promise<Model> {
  return parseBufferInWorker(await file.arrayBuffer(), file.name);
}

/** Convert a model's weights to safetensors off the main thread. */
export async function convertWeightsInWorker(
  buffer: ArrayBuffer,
  name: string,
  format: string,
  target: ConvTarget,
  precision: Precision,
): Promise<ConvertResult> {
  let w: Worker;
  try {
    w = getWorker();
  } catch {
    return convertWeights(buffer, name, { format, target, precision });
  }
  const id = ++seq;
  return new Promise<ConvertResult>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    w.postMessage({ kind: "convert", id, buffer, name, format, target, precision });
  });
}
