// Build a registry snapshot (stats, layer fingerprint, compat matrix, hardware
// estimates) from a parsed Model. Pure apart from the async SHA-256, which uses
// Web Crypto (`crypto.subtle`), available in browsers, WebViews and Node 18+.

import type { Model } from "@modelvisio/parsers";
import { scoreDevice } from "../scoring/scoreDevice";
import { COMPAT_TARGETS, REFERENCE_DEVICES, compatMatrix, findDevice, type CompatTarget } from "./compat";
import type { DeviceEstimate, LayerFingerprint, ModelSnapshot, ModelStats } from "./types";

/** Hex SHA-256 of the bytes. */
export async function sha256Hex(bytes: ArrayBuffer | Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("Web Crypto (crypto.subtle) is unavailable; cannot hash the model file.");
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  // Copy into a fresh ArrayBuffer: digest() rejects SharedArrayBuffer-backed views.
  const digest = await subtle.digest("SHA-256", view.slice().buffer as ArrayBuffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export function modelStats(model: Model): ModelStats {
  let params = 0, flops = 0, macs = 0, memBytes = 0;
  for (const l of model.layers) {
    params += l.params || 0;
    flops += l.flops || 0;
    macs += l.macs || 0;
    memBytes += l.mem || 0;
  }
  return { layers: model.layers.length, params, flops, macs, memBytes };
}

export function layerFingerprints(model: Model): LayerFingerprint[] {
  return model.layers.map((l) => ({ name: l.name, op: l.op, shape: l.shape, params: l.params || 0, flops: l.flops || 0 }));
}

export function deviceEstimates(model: Model, deviceIds: string[] = REFERENCE_DEVICES): DeviceEstimate[] {
  // No compute info (e.g. the PyTorch module-tree parser reports no FLOPs): a
  // latency estimate would be dispatch overhead only, so store none.
  if (model.layers.every((l) => !l.flops && !l.macs)) return [];
  const out: DeviceEstimate[] = [];
  for (const id of deviceIds) {
    const dev = findDevice(id);
    if (!dev) continue;
    const s = scoreDevice(model, dev);
    out.push({
      deviceId: dev.id, deviceName: dev.n, overall: s.overall,
      fps: round(s.fps, 2), latencyMs: Number.isFinite(s.latencyMs) ? round(s.latencyMs, 3) : -1,
      fits: s.memory.fits, compileFail: s.opSupport.compileFail,
      precision: s.precision, confidence: s.confidence,
    });
  }
  return out;
}

export type SnapshotOptions = {
  /** Original file name; defaults to model.name. */
  fileName?: string;
  /** Original bytes: used for the hash and size. Without them sha256 is "" and size is model.sizeBytes. */
  bytes?: ArrayBuffer | Uint8Array | null;
  targets?: CompatTarget[];
  devices?: string[];
};

/** Analyze a parsed model into a registry snapshot. */
export async function snapshotModel(model: Model, opts: SnapshotOptions = {}): Promise<ModelSnapshot> {
  const bytes = opts.bytes ?? null;
  return {
    fileName: opts.fileName ?? model.name,
    sizeBytes: bytes ? bytes.byteLength : model.sizeBytes,
    sha256: bytes ? await sha256Hex(bytes) : "",
    format: model.format,
    framework: model.framework,
    opset: model.opset,
    producer: model.producer,
    inputShape: [...model.inputShape],
    outputShape: [...model.outputShape],
    stats: modelStats(model),
    layers: layerFingerprints(model),
    compat: compatMatrix(model, opts.targets ?? COMPAT_TARGETS),
    estimates: deviceEstimates(model, opts.devices ?? REFERENCE_DEVICES),
  };
}

function round(n: number, d: number): number {
  const f = 10 ** d;
  return Math.round(n * f) / f;
}
