import dagre from "@dagrejs/dagre";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { fmt } from "../../utils/format";

export type LineKind = "name" | "weight" | "shape" | "metric";
export type NodeLine = { text: string; kind: LineKind };
export type NodeView = { id: number; x: number; y: number; w: number; h: number; layer: ModelLayer; lines: NodeLine[] };
export type EdgeView = {
  from: number; to: number; points: { x: number; y: number }[]; label: string;
  /** Collision-free label center, reserved by the layout (null when no label). */
  lx: number | null; ly: number | null;
};
export type Layout = { nodes: NodeView[]; edges: EdgeView[]; width: number; height: number };

export type LayoutOpts = {
  direction: "TB" | "LR";
  showWeights: boolean;
  showNames: boolean;
  showMetrics: boolean;
};

const HEADER = 22, LINE = 15, PADX = 12, MINW = 96, MAXW = 320, MINH = 30;
// Edge shape-labels: rendered at 8.5px mono. We register them with dagre as real
// label boxes so the layout reserves space and never lets them land on a node.
const LBL_PX = 8.5, LBL_H = 13, LBL_MAX = 22;
const labelW = (s: string) => Math.max(14, textW(s, LBL_PX) + 8);
const edgeLabel = (l: ModelLayer | undefined) => {
  const s = l?.shape ?? "";
  return s && s !== "?" ? trunc(s, LBL_MAX) : "";
};

const trunc = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const shapeStr = (w: ModelLayer["w"]) =>
  w ? (typeof w.shape === "string" ? w.shape : w.shape.join("×")) : "";

function nodeLines(l: ModelLayer, o: LayoutOpts): NodeLine[] {
  const out: NodeLine[] = [];
  if (o.showNames && l.name && l.name !== l.op) out.push({ text: trunc(l.name, 40), kind: "name" });
  if (o.showWeights && l.w) {
    out.push({ text: `W ⟨${shapeStr(l.w)}⟩`, kind: "weight" });
    if (l.w.size != null && l.params > l.w.size) out.push({ text: `B ⟨${l.params - l.w.size}⟩`, kind: "weight" });
  }
  if (o.showMetrics) {
    if (l.shape && l.shape !== "?") out.push({ text: l.shape, kind: "shape" });
    if (l.flops > 0) out.push({ text: `${fmt(l.flops)} FLOPs`, kind: "metric" });
  }
  return out;
}

// Monospace-ish width estimate for layout sizing.
const textW = (s: string, px: number) => s.length * px * 0.62;

/** Compute a Netron-style layered layout with dagre. */
export function computeGraph(model: Model, o: LayoutOpts): Layout {
  const byId = new Map(model.layers.map((l) => [l.id, l] as const));
  const g = new dagre.graphlib.Graph();
  // ranksep is kept modest because each labelled edge inserts a label rank
  // (≈ 2·ranksep + label height) between its endpoints.
  g.setGraph({ rankdir: o.direction, nodesep: 26, ranksep: 24, marginx: 28, marginy: 28, ranker: "tight-tree" });
  g.setDefaultEdgeLabel(() => ({}));

  const lineMap = new Map<number, NodeLine[]>();
  for (const l of model.layers) {
    const lines = nodeLines(l, o);
    lineMap.set(l.id, lines);
    const bodyW = Math.max(textW(l.op, 11.5), ...lines.map((x) => textW(x.text, 10)), 0);
    const w = Math.min(MAXW, Math.max(MINW, bodyW + PADX * 2));
    const h = Math.max(MINH, HEADER + lines.length * LINE + (lines.length ? 8 : 6));
    g.setNode(String(l.id), { width: w, height: h });
  }
  for (const [a, b] of model.edges) {
    if (!g.hasNode(String(a)) || !g.hasNode(String(b))) continue;
    // Register the shape label as a real, centered label box so dagre reserves
    // space for it and keeps it off every node — no overlap on any model.
    const text = edgeLabel(byId.get(a));
    g.setEdge(String(a), String(b), text ? { width: labelW(text), height: LBL_H, labelpos: "c", labeloffset: 0 } : {});
  }

  dagre.layout(g);

  const nodes: NodeView[] = model.layers.map((l) => {
    const n = g.node(String(l.id));
    return { id: l.id, x: n.x, y: n.y, w: n.width, h: n.height, layer: l, lines: lineMap.get(l.id) ?? [] };
  });
  const edges: EdgeView[] = model.edges
    .filter(([a, b]) => g.hasEdge(String(a), String(b)))
    .map(([a, b]) => {
      const e = g.edge(String(a), String(b)) as { points: { x: number; y: number }[]; x?: number; y?: number };
      const label = edgeLabel(byId.get(a));
      return {
        from: a, to: b, points: e.points,
        label,
        lx: label && typeof e.x === "number" ? e.x : null,
        ly: label && typeof e.y === "number" ? e.y : null,
      };
    });
  const gg = g.graph();
  return { nodes, edges, width: gg.width ?? 0, height: gg.height ?? 0 };
}

/** Smooth SVG path through dagre's routed edge points. */
export function edgePath(points: { x: number; y: number }[]): string {
  if (points.length === 0) return "";
  if (points.length < 3) return points.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
  let d = `M${points[0].x},${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const p = points[i], n = points[i + 1];
    const mx = (p.x + n.x) / 2, my = (p.y + n.y) / 2;
    d += ` Q${p.x},${p.y} ${mx},${my}`;
  }
  const last = points[points.length - 1];
  d += ` L${last.x},${last.y}`;
  return d;
}
