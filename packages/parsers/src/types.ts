// Normalized model object — the contract every parser MUST emit so the core
// graph renderer works unchanged. Do not diverge from this shape.

export type ModelLayer = {
  id: number;
  name: string;
  type: string;
  op: string;
  shape: string;
  dt: string;
  params: number;
  flops: number;
  macs: number;
  mem: number;
  group: "input" | "backbone" | "neck" | "head" | "output";
  ins: { n: string; s?: number[] }[];
  outs: { n: string; s?: number[] }[];
  attr: Record<string, unknown>;
  w: {
    shape: number[] | string;
    size?: number;
    min?: number;
    max?: number;
    mean?: number;
    std?: number;
    sparse?: number;
  } | null;
  math: string;
  insight: string;
  qSens: number; // quantization sensitivity 0..1
  compIssues: { target: string; severity: "warn" | "error"; msg: string }[];
};

export type Model = {
  name: string;
  format: string;
  framework: string;
  opset: number;
  sizeBytes: number;
  inputShape: number[];
  outputShape: number[];
  layers: ModelLayer[];
  edges: [number, number][];
  // Optional provenance, surfaced in the UI footer when available.
  producer?: string;
  irVersion?: number;
  doc?: string;
};
