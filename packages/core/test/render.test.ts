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
import { DownloadPanel } from "../src/components/DownloadPanel";

// Offline render smoke tests. Render to a static HTML string (no DOM, no
// browser). Catches crashes in component render paths across the product.
const themed = (node: ReactElement) => h(ThemeCtx.Provider, { value: TH.dark }, node);

describe("component render smoke (offline, no DOM)", () => {
  it("App renders the landing screen without throwing", () => {
    const html = renderToStaticMarkup(h(App));
    expect(html).toContain("Drop your model file");
  });

  it("App landing hides the download buttons inside the VS Code WebView, keeps GitHub", () => {
    const g = globalThis as { acquireVsCodeApi?: unknown };
    expect(renderToStaticMarkup(h(App))).toContain("Download Desktop App");
    g.acquireVsCodeApi = () => ({});
    try {
      const html = renderToStaticMarkup(h(App));
      expect(html).not.toContain("Download Desktop App");
      expect(html).not.toContain("macOS · Windows · Linux");
      expect(html).not.toContain("VS Code Extension");
      expect(html).toContain("GitHub");
    } finally {
      delete g.acquireVsCodeApi;
    }
  });

  it("App landing (SSR, no browser navigator) offers the generic desktop download", () => {
    const html = renderToStaticMarkup(h(App));
    expect(html).toContain("Download Desktop App");
    expect(html).toContain("https://github.com/Premchand006/ModelVisio/releases/latest");
    expect(html).not.toContain("Download for ");
    // Panel is closed until "Other platforms" is clicked.
    expect(html).not.toContain("Download ModelVisio Desktop");
  });

  it("DownloadPanel lists every platform build plus macOS first-launch help", () => {
    const html = renderToStaticMarkup(themed(h(DownloadPanel, { detected: "mac", onClose: () => {} })));
    expect(html).toContain('role="dialog"');
    expect(html).toContain("ModelVisio-macOS-universal.dmg");
    expect(html).toContain("ModelVisio-Windows-x64-setup.exe");
    expect(html).toContain("ModelVisio-Linux-x86_64.AppImage");
    expect(html).toContain("Universal: Apple Silicon &amp; Intel");
    expect(html).toContain("Open Anyway");
    expect(html).toContain("docs/install-macos.md");
    expect(html).toContain("xattr -dr com.apple.quarantine /Applications/ModelVisio.app");
    expect(html).toContain("Your system");
    expect(html).toContain('aria-label="Close download panel"');
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
