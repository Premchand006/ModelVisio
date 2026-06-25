import { describe, expect, it } from "vitest";
import { computeGraph, type LayoutOpts } from "../src/components/Graph/dagreLayout";
import { demoModel } from "../src/demo/demoModel";
import type { Model, ModelLayer } from "@modelvisio/parsers";

type Rect = { x1: number; y1: number; x2: number; y2: number };
const overlap = (a: Rect, b: Rect, tol = 1) =>
  a.x1 < b.x2 - tol && a.x2 > b.x1 + tol && a.y1 < b.y2 - tol && a.y2 > b.y1 + tol;

// Mirrors DAGGraph's rendered label box (width = label.length*5.3 + 8, height 14).
function labelRects(lay: ReturnType<typeof computeGraph>): Rect[] {
  return lay.edges.flatMap((e) => {
    if (!e.label || e.lx == null || e.ly == null) return [];
    const w = e.label.length * 5.3 + 8;
    return [{ x1: e.lx - w / 2, y1: e.ly - 7, x2: e.lx + w / 2, y2: e.ly + 7 }];
  });
}
const nodeRects = (lay: ReturnType<typeof computeGraph>): Rect[] =>
  lay.nodes.map((n) => ({ x1: n.x - n.w / 2, y1: n.y - n.h / 2, x2: n.x + n.w / 2, y2: n.y + n.h / 2 }));

function assertNoLabelOverlap(model: Model, o: LayoutOpts) {
  const lay = computeGraph(model, o);
  const nodes = nodeRects(lay);
  for (const lr of labelRects(lay)) {
    const hit = nodes.find((nr) => overlap(lr, nr));
    expect(hit, `edge label at ${JSON.stringify(lr)} overlaps a node (${o.direction})`).toBeUndefined();
  }
}

// A deep chain of tiny activation nodes with long shape labels — the worst case
// for label/node collisions (short edges, big labels), like the reported Sigmoid.
function chainModel(n: number): Model {
  const layers: ModelLayer[] = Array.from({ length: n }, (_, i) => ({
    id: i, name: `block.${i}.act`, type: "Mul", op: i % 2 ? "Mul(Sigmoid)" : "Conv",
    shape: "1×16×320×320", dt: "float32", params: 0, flops: 0, macs: 0, mem: 0,
    group: "backbone", ins: [], outs: [], attr: {}, w: null, math: "", insight: "",
    qSens: 0, compIssues: [],
  }));
  const edges: [number, number][] = Array.from({ length: n - 1 }, (_, i) => [i, i + 1]);
  return {
    name: "chain", format: "ONNX", framework: "test", opset: 18, sizeBytes: 0,
    inputShape: [1, 3, 640, 640], outputShape: [1, 16, 320, 320], layers, edges,
  };
}

const OPT_COMBOS: LayoutOpts[] = [
  { direction: "TB", showWeights: false, showNames: false, showMetrics: false },
  { direction: "TB", showWeights: true, showNames: true, showMetrics: true },
  { direction: "LR", showWeights: false, showNames: false, showMetrics: false },
  { direction: "LR", showWeights: true, showNames: true, showMetrics: true },
];

describe("graph layout — edge labels never overlap nodes", () => {
  for (const o of OPT_COMBOS) {
    it(`demo model: no label/node overlap (${o.direction}, metrics=${o.showMetrics})`, () => {
      assertNoLabelOverlap(demoModel, o);
    });
  }

  for (const o of OPT_COMBOS) {
    it(`deep activation chain: no label/node overlap (${o.direction})`, () => {
      assertNoLabelOverlap(chainModel(40), o);
    });
  }

  it("every labelled edge has a reserved, collision-free position", () => {
    const lay = computeGraph(demoModel, OPT_COMBOS[0]);
    const labelled = lay.edges.filter((e) => e.label);
    expect(labelled.length).toBeGreaterThan(0);
    expect(labelled.every((e) => e.lx != null && e.ly != null)).toBe(true);
  });
});
