/// <reference lib="webworker" />
import { convertWeights, parseModel, parseModelProgressive, type Companions } from "@modelvisio/parsers";
import { ChunkCoalescer } from "./streaming";

// Off-main-thread model parsing + weight conversion. Large models decode/convert
// here without freezing the UI.
const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent) => {
  const msg = e.data as
    | { kind?: "parse"; id: number; buffer: ArrayBuffer; name: string; companions?: Companions }
    | { kind: "parseStream"; id: number; buffer: ArrayBuffer; name: string; companions?: Companions }
    | { kind: "convert"; id: number; buffer: ArrayBuffer; name: string; format: string; target: "safetensors" | "npz"; precision: "keep" | "f16" | "f32" };
  try {
    if (msg.kind === "convert") {
      const r = convertWeights(msg.buffer, msg.name, { format: msg.format, target: msg.target, precision: msg.precision });
      ctx.postMessage({ id: msg.id, result: r });
    } else if (msg.kind === "parseStream") {
      // Progressive parse: {id, chunk} messages (coalesced to ~400 layers or
      // 50ms each), then the authoritative {id, model}.
      const co = new ChunkCoalescer((chunk) => ctx.postMessage({ id: msg.id, chunk }));
      const model = parseModelProgressive(msg.buffer, msg.name, (c) => co.push(c), { companions: msg.companions });
      co.flush();
      ctx.postMessage({ id: msg.id, model });
    } else {
      const model = parseModel(msg.buffer, msg.name, msg.companions);
      ctx.postMessage({ id: msg.id, model });
    }
  } catch (err) {
    ctx.postMessage({ id: msg.id, error: err instanceof Error ? err.message : String(err) });
  }
};
