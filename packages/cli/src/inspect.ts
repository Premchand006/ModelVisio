// `modelvisio inspect`: model summary.

import type { Model } from "@modelvisio/parsers";
import { modelStats } from "@modelvisio/core/registry";
import { bytesStr, escapeMd, human, shapeStr, type LoadedModel } from "./load";

export type IoTensor = { name: string; shape: number[] };

export type InspectSummary = {
  file: string;
  format: string;
  framework: string;
  opset: number;
  producer?: string;
  irVersion?: number;
  sizeBytes: number;
  layers: number;
  params: number;
  flops: number;
  macs: number;
  /** False for formats that are detected but not graph-parsed (metadata only). */
  graphParsed: boolean;
  inputs: IoTensor[];
  outputs: IoTensor[];
  ops: { op: string; count: number }[];
};

function ioTensors(model: Model): { inputs: IoTensor[]; outputs: IoTensor[] } {
  const produced = new Set<string>();
  const consumed = new Set<string>();
  for (const l of model.layers) {
    for (const o of l.outs) produced.add(o.n);
    for (const i of l.ins) consumed.add(i.n);
  }
  const inputs: IoTensor[] = [];
  const outputs: IoTensor[] = [];
  const seenIn = new Set<string>();
  const seenOut = new Set<string>();
  for (const l of model.layers) {
    if (l.group === "input") {
      for (const o of l.outs) if (!seenIn.has(o.n)) { seenIn.add(o.n); inputs.push({ name: o.n, shape: o.s ?? [] }); }
    }
  }
  // Graph outputs: tensors produced but never consumed.
  for (const l of model.layers) for (const o of l.outs) {
    if (!consumed.has(o.n) && !seenOut.has(o.n)) { seenOut.add(o.n); outputs.push({ name: o.n, shape: o.s ?? [] }); }
  }
  // Fall back to graph structure: tensors consumed but never produced / produced but never consumed.
  if (inputs.length === 0) {
    for (const l of model.layers) for (const i of l.ins) {
      if (!produced.has(i.n) && i.s && i.s.length > 1 && !seenIn.has(i.n)) { seenIn.add(i.n); inputs.push({ name: i.n, shape: i.s }); }
    }
  }
  if (inputs.length === 0 && model.inputShape.length) inputs.push({ name: "input", shape: model.inputShape });
  if (outputs.length === 0 && model.outputShape.length) outputs.push({ name: "output", shape: model.outputShape });
  return { inputs: inputs.slice(0, 16), outputs: outputs.slice(0, 16) };
}

export function summarize(loaded: LoadedModel): InspectSummary {
  const { model } = loaded;
  const s = modelStats(model);
  const counts = new Map<string, number>();
  for (const l of model.layers) counts.set(l.op, (counts.get(l.op) ?? 0) + 1);
  const ops = [...counts].map(([op, count]) => ({ op, count })).sort((a, b) => b.count - a.count || a.op.localeCompare(b.op));
  return {
    file: loaded.fileName,
    format: model.format,
    framework: model.framework,
    opset: model.opset,
    producer: model.producer,
    irVersion: model.irVersion,
    sizeBytes: loaded.bytes.byteLength,
    layers: s.layers,
    params: s.params,
    flops: s.flops,
    macs: s.macs,
    graphParsed: loaded.graphParsed,
    ...ioTensors(model),
    ops,
  };
}

export function inspectText(s: InspectSummary): string {
  const rows: [string, string][] = [
    ["File", s.file],
    ["Format", `${s.format}${s.graphParsed ? "" : " (metadata only, no graph)"}`],
    ["Framework", s.framework || "-"],
    ["Opset", s.opset ? String(s.opset) : "-"],
    ...(s.producer ? [["Producer", s.producer] as [string, string]] : []),
    ["Size", bytesStr(s.sizeBytes)],
    ["Layers", String(s.layers)],
    ["Params", `${human(s.params)} (${s.params.toLocaleString("en-US")})`],
    ["FLOPs", s.flops || s.macs ? human(s.flops) : "n/a (not computed for this format)"],
    ["MACs", s.flops || s.macs ? human(s.macs) : "n/a"],
  ];
  const w = Math.max(...rows.map(([k]) => k.length));
  const out = rows.map(([k, v]) => `${k.padEnd(w)}  ${v}`);
  out.push("Inputs:");
  for (const t of s.inputs) out.push(`  ${t.name}  ${shapeStr(t.shape)}`);
  if (!s.inputs.length) out.push("  -");
  out.push("Outputs:");
  for (const t of s.outputs) out.push(`  ${t.name}  ${shapeStr(t.shape)}`);
  if (!s.outputs.length) out.push("  -");
  if (s.ops.length) out.push("Top ops: " + s.ops.slice(0, 10).map((o) => `${o.op}×${o.count}`).join(", "));
  return out.join("\n") + "\n";
}

export function inspectMarkdown(s: InspectSummary): string {
  const io = (ts: IoTensor[]) => ts.map((t) => `\`${escapeMd(t.name)}\` ${shapeStr(t.shape)}`).join("<br>") || "-";
  return [
    `### ${escapeMd(s.file)}`,
    "",
    "| Format | Layers | Params | FLOPs | Size | Inputs | Outputs |",
    "|---|---:|---:|---:|---:|---|---|",
    `| ${escapeMd(s.format)}${s.graphParsed ? "" : " (metadata only)"} | ${s.layers} | ${human(s.params)} | ${human(s.flops)} | ${bytesStr(s.sizeBytes)} | ${io(s.inputs)} | ${io(s.outputs)} |`,
    "",
  ].join("\n");
}
