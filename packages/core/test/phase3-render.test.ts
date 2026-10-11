import { describe, expect, it } from "vitest";
import { createElement as h, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ThemeCtx } from "../src/theme/ThemeContext";
import { TH } from "../src/theme/theme";
import { demoModel } from "../src/demo/demoModel";
import { Investigator } from "../src/components/Investigator";
import { OptimizationSearch } from "../src/components/OptimizationSearch";

// Offline render smoke tests for the Phase 3 panels (no DOM, no network).
const themed = (node: ReactElement) => h(ThemeCtx.Provider, { value: TH.dark }, node);

describe("Phase 3 panels render (offline)", () => {
  it("Investigator shows evidence charts and the rule-based diagnosis immediately", () => {
    const html = renderToStaticMarkup(themed(h(Investigator, { model: demoModel, defaultDeviceId: "coral", onSelectLayer: () => {} })));
    expect(html).toContain("Performance Investigator");
    expect(html).toContain("Investigate");
    expect(html).toContain("Rule-based");
    expect(html).toContain("<svg");
    expect(html).toContain("Unsupported ops");
    expect(html).toContain("Jump to layer");
  });
  it("Investigator shows the set-key guidance on desktop without a key", () => {
    // isDesktop() checks `"__TAURI__" in window`; Node has no window, so stub one.
    const g = globalThis as { window?: unknown };
    const prev = g.window;
    g.window = { __TAURI__: {} };
    try {
      const html = renderToStaticMarkup(themed(h(Investigator, { model: demoModel })));
      expect(html).toContain("Gemini API key");
      expect(html).toContain("Rule-based");
    } finally {
      if (prev === undefined) delete g.window; else g.window = prev;
    }
  });
  it("OptimizationSearch renders the device picker, constraints and run button", () => {
    const html = renderToStaticMarkup(themed(h(OptimizationSearch, { model: demoModel, onApplyFix: () => {} })));
    expect(html).toContain("Optimization Search");
    expect(html).toContain("Run search");
    expect(html).toContain("Max latency");
    expect(html).toContain("Hailo-8");
  });
});
