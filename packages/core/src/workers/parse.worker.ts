/// <reference lib="webworker" />
import { convertWeights, parseModel } from "@modelvisio/parsers";

// Off-main-thread model parsing + weight conversion. Large models decode/convert
// here without freezing the UI.
const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent) => {
  const msg = e.data as
    | { kind?: "parse"; id: number; buffer: ArrayBuffer; name: string }
    | { kind: "convert"; id: number; buffer: ArrayBuffer; name: string; format: string; target: "safetensors" | "npz"; precision: "keep" | "f16" | "f32" };
  try {
    if (msg.kind === "convert") {
      const r = convertWeights(msg.buffer, msg.name, { format: msg.format, target: msg.target, precision: msg.precision });
      ctx.postMessage({ id: msg.id, result: r });
    } else {
      const model = parseModel(msg.buffer, msg.name);
      ctx.postMessage({ id: msg.id, model });
    }
  } catch (err) {
    ctx.postMessage({ id: msg.id, error: err instanceof Error ? err.message : String(err) });
  }
};
