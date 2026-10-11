import { describe, expect, it } from "vitest";
import { createElement as h, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ThemeCtx } from "../src/theme/ThemeContext";
import { TH } from "../src/theme/theme";
import { demoModel } from "../src/demo/demoModel";
import { DeployAgent } from "../src/components/DeployAgent";
import { FleetSim } from "../src/components/FleetSim";
import { DeployRecipe } from "../src/components/DeployRecipe";

// Offline render smoke tests (no DOM) for the Phase 4 panels.
const themed = (node: ReactElement, mode: "dark" | "light" = "dark") => h(ThemeCtx.Provider, { value: TH[mode] }, node);

describe("Phase 4 render smoke", () => {
  it("DeployAgent renders the goal input and quick-goal chips", () => {
    const html = renderToStaticMarkup(themed(h(DeployAgent, { model: demoModel })));
    expect(html).toContain("AI Deployment Agent");
    expect(html).toContain("Run agent");
    expect(html).toContain("Ship to Android phones via TFLite");
    expect(html).toContain('aria-label="Deployment goal"');
  });

  it("DeployAgent without a model asks for one", () => {
    expect(renderToStaticMarkup(themed(h(DeployAgent, { model: null })))).toContain("Load a model");
  });

  it("FleetSim renders the default preset, results and the Monte-Carlo histogram", () => {
    const html = renderToStaticMarkup(themed(h(FleetSim, { model: demoModel }), "light"));
    expect(html).toContain("Fleet Simulation");
    expect(html).toContain("Retail edge boxes (Jetson + NUC)");
    expect(html).toContain("Meeting SLO");
    expect(html).toContain("Fleet p95");
    expect(html).toContain("Monte-Carlo trials");
    expect(html).toContain("<svg");
  });

  it("FleetSim without a model asks for one", () => {
    expect(renderToStaticMarkup(themed(h(FleetSim, { model: null })))).toContain("Load a model");
  });

  it("DeployRecipe still renders the extracted quick-start recipe", () => {
    const html = renderToStaticMarkup(themed(h(DeployRecipe, { model: demoModel })));
    expect(html).toContain("Deployment Recipe Generator");
    expect(html).toContain("trtexec --onnx=yolo26n.onnx --saveEngine=model.engine --fp16 --workspace=4096");
  });
});
