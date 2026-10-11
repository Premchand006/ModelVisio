import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { FORMAT_SUPPORT } from "@modelvisio/parsers";
import {
  ALL_EXTS,
  DEFAULT_EXTS,
  DIRECTORY_EXTS,
  OPTION_EXTS,
  VIEW_TYPE,
  VIEW_TYPE_OPTION,
  extOf,
  openDialogFilters,
  selectorFor,
  viewTypeFor,
} from "../src/formats";

type Editor = { viewType: string; priority: string; selector: { filenamePattern: string }[] };
const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
  contributes: { customEditors: Editor[] };
};
const editor = (viewType: string) => pkg.contributes.customEditors.find((e) => e.viewType === viewType)!;

describe("file associations", () => {
  it("package.json selectors match formats.ts exactly", () => {
    expect(editor(VIEW_TYPE).priority).toBe("default");
    expect(editor(VIEW_TYPE).selector).toEqual(selectorFor(DEFAULT_EXTS));
    expect(editor(VIEW_TYPE_OPTION).priority).toBe("option");
    expect(editor(VIEW_TYPE_OPTION).selector).toEqual(selectorFor(OPTION_EXTS));
  });

  it("default and option lists are disjoint and duplicate-free", () => {
    expect(new Set(ALL_EXTS).size).toBe(ALL_EXTS.length);
  });

  it("covers every file extension the parser registry supports", () => {
    const registry = new Set(FORMAT_SUPPORT.flatMap((f) => f.exts));
    const contributed = new Set([...ALL_EXTS, ...DIRECTORY_EXTS]);
    expect([...registry].filter((e) => !contributed.has(e))).toEqual([]);
    expect([...contributed].filter((e) => !registry.has(e))).toEqual([]);
  });

  it("never contributes directory bundles", () => {
    for (const e of DIRECTORY_EXTS) expect(ALL_EXTS).not.toContain(e);
  });

  it("picks the editor by extension", () => {
    expect(viewTypeFor("/a/model.ONNX")).toBe(VIEW_TYPE);
    expect(viewTypeFor("C:\m\yolo.pt")).toBe(VIEW_TYPE);
    expect(viewTypeFor("/a/config.json")).toBe(VIEW_TYPE_OPTION);
    expect(viewTypeFor("/a/README")).toBe(VIEW_TYPE_OPTION);
  });

  it("extOf ignores dotfiles and folders with dots", () => {
    expect(extOf("/a.b/model")).toBe("");
    expect(extOf(".bashrc")).toBe("");
    expect(extOf("x.tar.GZ")).toBe("gz");
  });

  it("open dialog lists all model extensions then a catch-all", () => {
    const f = openDialogFilters();
    expect(f["Model files"]).toEqual([...ALL_EXTS]);
    expect(f["All files"]).toEqual(["*"]);
  });
});
