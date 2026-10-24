import { describe, expect, it } from "vitest";
import { createElement as h, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ThemeCtx } from "../src/theme/ThemeContext";
import { TH } from "../src/theme/theme";
import { ALL_TOOLS, TOOL_GROUPS, groupOf } from "../src/data/tools";
import { ToolStages, ToolTabs } from "../src/components/ToolNav";
import { CommandPalette, filterCommands, type Command } from "../src/components/CommandPalette";

const themed = (node: ReactElement) => h(ThemeCtx.Provider, { value: TH.dark }, node);
const cmd = (section: Command["section"], label: string, hint?: string): Command =>
  ({ id: `${section}:${label}`, section, label, hint, icon: "action", run: () => {} });

describe("tool groups", () => {
  it("cover every sidebar panel exactly once", () => {
    const ids = ALL_TOOLS.map((tl) => tl.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.sort()).toEqual([
      "advisor", "agent", "bench", "chat", "compiler", "convert", "deploy", "fleet",
      "hardware", "inspector", "investigate", "optimize", "quant", "registry",
    ]);
  });

  it("groupOf maps a tool to its stage and falls back to the first", () => {
    expect(groupOf("bench").id).toBe("optimize");
    expect(groupOf("registry").id).toBe("ship");
    expect(groupOf("nope").id).toBe(TOOL_GROUPS[0].id);
  });
});

describe("ToolNav", () => {
  it("marks the active stage and shows only its tools", () => {
    const stages = renderToStaticMarkup(themed(h(ToolStages, { active: "deploy", onChange: () => {} })));
    expect(stages.match(/aria-pressed="true"/g)).toHaveLength(1);
    expect(stages).toMatch(/aria-pressed="true"[^>]*>.*?Ship/);
    const tabs = renderToStaticMarkup(themed(h(ToolTabs, { active: "deploy", onChange: () => {} })));
    expect(tabs).toContain("Registry");
    expect(tabs).not.toContain("Inspector");
    expect(tabs.match(/aria-selected="true"/g)).toHaveLength(1);
  });
});

describe("filterCommands", () => {
  const cmds = [
    cmd("Layers", "model.0.conv", "Conv · backbone"),
    cmd("Tools", "Compiler", "Analyze"),
    cmd("Tools", "Convert", "Ship"),
    cmd("Tools", "Inspector", "Analyze"),
    cmd("Actions", "Close model"),
  ];

  it("keeps everything, grouped Tools, Actions, Layers, for an empty query", () => {
    expect(filterCommands(cmds, "  ").map((c) => c.label)).toEqual(["Compiler", "Convert", "Inspector", "Close model", "model.0.conv"]);
  });

  it("ranks prefix over word-start over substring, and searches hints", () => {
    expect(filterCommands(cmds, "co").map((c) => c.label)).toEqual(["Compiler", "Convert", "model.0.conv"]);
    expect(filterCommands(cmds, "mod").map((c) => c.label)).toEqual(["Close model", "model.0.conv"]);
    expect(filterCommands(cmds, "conv").map((c) => c.label)).toEqual(["Convert", "model.0.conv"]);
    expect(filterCommands(cmds, "analyze").map((c) => c.label)).toEqual(["Compiler", "Inspector"]);
    expect(filterCommands(cmds, "zzz")).toEqual([]);
  });

  it("caps each section so huge models stay fast", () => {
    const many = Array.from({ length: 500 }, (_, i) => cmd("Layers", `layer_${i}`));
    expect(filterCommands(many, "layer")).toHaveLength(40);
  });

  it("renders as a labelled modal dialog with a combobox", () => {
    const html = renderToStaticMarkup(themed(h(CommandPalette, { commands: cmds, onClose: () => {} })));
    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-modal="true"');
    expect(html).toContain('role="combobox"');
    expect(html).toContain("Compiler");
  });
});
