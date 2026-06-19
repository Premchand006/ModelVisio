import type { Model, ModelLayer } from "../types";

export type TensorInfo = {
  name: string;
  shape: number[];
  dtype: string;
  bytes: number;
};

/** Classify a tensor into a graph group from its name. */
function groupFor(name: string): ModelLayer["group"] {
  const n = name.toLowerCase();
  if (/embed|embd|patch_embed|tok_embeddings|wte|wpe/.test(n)) return "input";
  if (/lm_head|classifier|logits|\.head\.|output|fc_out|score/.test(n)) return "output";
  if (/norm|ln_f|layernorm|bn|batch_norm/.test(n)) return "neck";
  return "backbone";
}

/**
 * Build a normalized Model from a flat collection of stored tensors (the shape
 * of weight-only formats: safetensors, GGUF, NumPy). There is no true compute
 * graph, so layers are chained in file order — the insight notes this. The
 * Layers panel and Inspector remain fully useful.
 */
export function weightsToModel(
  meta: { name: string; format: string; framework: string; sizeBytes: number; note?: string },
  tensors: TensorInfo[],
): Model {
  const layers: ModelLayer[] = tensors.map((t, i) => {
    const params = t.shape.reduce((a, b) => a * b, t.shape.length ? 1 : 1);
    return {
      id: i,
      name: t.name,
      type: "Tensor",
      op: "Weight",
      shape: t.shape.length ? t.shape.join("×") : "scalar",
      dt: t.dtype,
      params,
      flops: 0,
      macs: 0,
      mem: t.bytes,
      group: groupFor(t.name),
      ins: [],
      outs: [{ n: t.name, s: t.shape }],
      attr: {},
      w: { shape: t.shape, size: params },
      math: `${t.dtype}${t.shape.length ? " " + t.shape.join("×") : ""}`,
      insight: meta.note || "Stored weight tensor (file order; not a compute edge).",
      qSens: 0.3,
      compIssues: [],
    };
  });
  // A light spine so the graph is readable; clearly file order, not data flow.
  const edges: [number, number][] = [];
  for (let i = 1; i < layers.length; i++) edges.push([i - 1, i]);

  const first = tensors[0]?.shape ?? [];
  const last = tensors[tensors.length - 1]?.shape ?? [];
  return {
    name: meta.name,
    format: meta.format,
    framework: meta.framework,
    opset: 0,
    sizeBytes: meta.sizeBytes,
    inputShape: first,
    outputShape: last,
    layers,
    edges,
  };
}
