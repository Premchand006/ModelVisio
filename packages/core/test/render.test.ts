import { describe, expect, it } from "vitest";
import { createElement as h, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ThemeCtx } from "../src/theme/ThemeContext";
import { TH } from "../src/theme/theme";
import { demoModel } from "../src/demo/demoModel";
import { App } from "../src/App";
import { CompilerChecker } from "../src/components/CompilerChecker";
import { HWReport } from "../src/components/HWReport";
import { Inspector } from "../src/components/Inspector";
import { DeployRecipe } from "../src/components/DeployRecipe";

// Offline render smoke tests — render to a static HTML string (no DOM, no
// browser). Catches crashes in component render paths across the product.
const themed = (node: ReactElement) => h(ThemeCtx.Provider, { value: TH.dark }, node);

describe("component render smoke (offline, no DOM)", () => {
  it("App renders the landing screen without throwing", () => {
    const html = renderToStaticMarkup(h(App));
    expect(html).toContain("Drop your model file");
  });

  it("CompilerChecker renders the working auto-fix engine for the demo model", () => {
    const html = renderToStaticMarkup(themed(h(CompilerChecker, { model: demoModel })));
    expect(html).toContain("Auto-Fix Engine");
    expect(html).toContain("Apply Fix");
  });

  it("HWReport renders scored device rows with FPS", () => {
    const html = renderToStaticMarkup(themed(h(HWReport, { model: demoModel })));
    expect(html).toContain("Hardware");
    expect(html).toMatch(/FPS/);
  });

  it("Inspector renders a selected layer's details", () => {
    const layer = demoModel.layers.find((l) => l.op === "Conv")!;
    const html = renderToStaticMarkup(themed(h(Inspector, { layer, model: demoModel })));
    expect(html).toContain(layer.name);
  });

  it("DeployRecipe renders", () => {
    const html = renderToStaticMarkup(themed(h(DeployRecipe, { model: demoModel })));
    expect(html).toContain("Deployment Recipe");
  });
});
