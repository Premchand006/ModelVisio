import { describe, expect, it } from "vitest";
import { createElement as h, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ThemeCtx } from "../src/theme/ThemeContext";
import { TH } from "../src/theme/theme";
import { demoModel } from "../src/demo/demoModel";
import { Benchmark } from "../src/components/Benchmark";

// Offline render smoke tests for the live-benchmark tab (no DOM, no runtime).
const themed = (node: ReactElement, theme: "dark" | "light" = "dark") => h(ThemeCtx.Provider, { value: TH[theme] }, node);

describe("Benchmark render smoke", () => {
  it("asks for a model when none is loaded", () => {
    const html = renderToStaticMarkup(themed(h(Benchmark, { model: null, modelBytes: null })));
    expect(html).toContain("Live benchmark");
    expect(html).toContain("Load a model");
  });

  it("explains that non-ONNX models need conversion instead of faking numbers", () => {
    const tflite = { ...demoModel, format: "TFLite" };
    const html = renderToStaticMarkup(themed(h(Benchmark, { model: tflite, modelBytes: new ArrayBuffer(8) })));
    expect(html).toContain("needs an <b>ONNX</b> file");
    expect(html).toContain("Convert");
    expect(html).not.toContain("Run benchmark");
  });

  it("asks for the real file when only the demo model is loaded", () => {
    const html = renderToStaticMarkup(themed(h(Benchmark, { model: { ...demoModel, format: "ONNX" }, modelBytes: null })));
    expect(html).toContain("original file isn&#x27;t available");
  });

  it("renders the settings form, dynamic-dim inputs and the prediction for an ONNX model", () => {
    const g = globalThis as { Worker?: unknown };
    const hadWorker = "Worker" in g;
    if (!hadWorker) g.Worker = function () {};
    try {
      const model = { ...demoModel, format: "ONNX" };
      for (const theme of ["dark", "light"] as const) {
        const html = renderToStaticMarkup(themed(h(Benchmark, { model, modelBytes: new ArrayBuffer(8) }), theme));
        expect(html).toContain("Run benchmark");
        expect(html).toContain("WebAssembly (CPU)");
        expect(html).toContain("Measured vs predicted");
        expect(html).toContain("Roofline prediction on");
        expect(html).toContain("Per-op profile");
      }
    } finally {
      if (!hadWorker) delete g.Worker;
    }
  });
});
