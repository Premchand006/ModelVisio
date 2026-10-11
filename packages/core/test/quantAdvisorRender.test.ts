import { describe, expect, it } from "vitest";
import { createElement as h, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { Model } from "@modelvisio/parsers";
import { ThemeCtx } from "../src/theme/ThemeContext";
import { TH } from "../src/theme/theme";
import { demoModel } from "../src/demo/demoModel";
import { QuantHeatmap } from "../src/components/QuantHeatmap";
import { ModelAdvisor } from "../src/components/ModelAdvisor";
import { GraphPanel } from "../src/components/Graph/GraphPanel";

// Offline render smoke tests (no DOM), in the style of render.test.ts.
const themed = (node: ReactElement, mode: "dark" | "light" = "dark") => h(ThemeCtx.Provider, { value: TH[mode] }, node);

const noWeights: Model = {
  name: "bare", format: "ONNX", framework: "t", opset: 17, sizeBytes: 0, inputShape: [], outputShape: [],
  layers: [
    { id: 0, name: "x", type: "Tensor", op: "Input", shape: "", dt: "float32", params: 0, flops: 0, macs: 0, mem: 0, group: "input", ins: [], outs: [{ n: "x" }], attr: {}, w: null, math: "", insight: "", qSens: 0, compIssues: [] },
    { id: 1, name: "add", type: "Add", op: "Add", shape: "", dt: "float32", params: 0, flops: 0, macs: 0, mem: 0, group: "backbone", ins: [{ n: "x" }], outs: [{ n: "y" }], attr: {}, w: null, math: "", insight: "", qSens: 0, compIssues: [] },
  ],
  edges: [[0, 1]],
};

describe("quantization + advisor render smoke", () => {
  it("QuantHeatmap renders summary, table and exports for the demo model (both themes)", () => {
    for (const mode of ["dark", "light"] as const) {
      const html = renderToStaticMarkup(themed(h(QuantHeatmap, { model: demoModel, onSelectLayer: () => {}, onShowOnGraph: () => {} }), mode));
      expect(html).toContain("Quantization heatmap");
      expect(html).toContain("Safe for INT8");
      expect(html).toContain("Plan JSON");
      expect(html).toContain("ONNX Runtime script");
      expect(html).toContain("TFLite script");
      expect(html).toContain("stem.conv");
      expect(html).toContain("not measured accuracy");
    }
  });

  it("QuantHeatmap handles a model without weights", () => {
    const html = renderToStaticMarkup(themed(h(QuantHeatmap, { model: noWeights })));
    expect(html).toContain("Quantization heatmap");
  });

  it("ModelAdvisor renders ranked recommendations and the bottleneck table", () => {
    const html = renderToStaticMarkup(themed(h(ModelAdvisor, { model: demoModel, initialDeviceId: "coral", onSelectLayer: () => {} })));
    expect(html).toContain("Model advisor");
    expect(html).toContain("Recommendations");
    expect(html).toContain("Full INT8 quantization");
    expect(html).toContain("Per-layer bottleneck");
    expect(html).toMatch(/memory-bound|compute/);
  });

  it("ModelAdvisor picks a default device and survives a weightless model", () => {
    expect(renderToStaticMarkup(themed(h(ModelAdvisor, { model: demoModel })))).toContain("Target");
    expect(renderToStaticMarkup(themed(h(ModelAdvisor, { model: noWeights }), "light"))).toContain("Model advisor");
  });

  it("GraphPanel offers the color-by selector and keeps the default look", () => {
    const html = renderToStaticMarkup(themed(h(GraphPanel, { model: demoModel, selected: null, onSelect: () => {} })));
    expect(html).toContain("Color: None");
    expect(html).toContain("Color: Quant sensitivity (INT8)");
    expect(html).not.toContain("not measured accuracy"); // no legend by default
  });

  it("GraphPanel renders the legend in a controlled quant mode", () => {
    const html = renderToStaticMarkup(themed(h(GraphPanel, { model: demoModel, selected: null, onSelect: () => {}, colorMode: "q8" })));
    expect(html).toContain("INT8 risk (estimate)");
    expect(html).toContain("linear-gradient");
  });
});
