import { describe, expect, it } from "vitest";
import {
  DEFAULT_MAX_FILE_MB,
  dirnamePosix,
  formatBytes,
  normalizeMaxMB,
  safeFilename,
  saveFilters,
  sizeLimitError,
  toTransferable,
  watchGlobFor,
} from "../src/files";

describe("size limit", () => {
  it("allows files up to the limit and rejects larger ones with a friendly message", () => {
    const mb = 1024 * 1024;
    expect(sizeLimitError("m.onnx", 10 * mb, 10)).toBeNull();
    const err = sizeLimitError("m.onnx", 10 * mb + 1, 10);
    expect(err).toContain("m.onnx");
    expect(err).toContain("10 MB limit");
    expect(err).toContain("modelvisio.maxFileSizeMB");
  });

  it("normalizes the setting", () => {
    expect(normalizeMaxMB(512)).toBe(512);
    for (const bad of [0, -1, NaN, Infinity, "100", undefined, null]) {
      expect(normalizeMaxMB(bad)).toBe(DEFAULT_MAX_FILE_MB);
    }
  });

  it("formats bytes", () => {
    expect(formatBytes(12)).toBe("12 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.0 GB");
  });
});

describe("safeFilename", () => {
  it("strips directories so a save can't escape the default folder", () => {
    expect(safeFilename("../../etc/passwd")).toBe("passwd");
    expect(safeFilename("C:\\Windows\\evil.bat")).toBe("evil.bat");
  });

  it("replaces characters Windows forbids and trims dots/spaces", () => {
    expect(safeFilename('a<b>:c"d|e?f*.json')).toBe("a_b__c_d_e_f_.json");
    expect(safeFilename("  report.md. ")).toBe("report.md");
  });

  it("falls back when nothing usable remains", () => {
    expect(safeFilename("")).toBe("export");
    expect(safeFilename("...")).toBe("export");
    expect(safeFilename("dir/")).toBe("export");
  });
});

describe("saveFilters", () => {
  it("offers the file's own extension first", () => {
    expect(saveFilters("graph.svg")).toEqual({ "SVG file": ["svg"], "All files": ["*"] });
    expect(saveFilters("Makefile")).toEqual({ "All files": ["*"] });
  });
});

describe("dirnamePosix", () => {
  it("returns the parent path", () => {
    expect(dirnamePosix("/a/b/m.onnx")).toBe("/a/b");
    expect(dirnamePosix("/m.onnx")).toBe("/");
  });
});

describe("watchGlobFor", () => {
  it("neutralizes glob syntax without an escape character", () => {
    expect(watchGlobFor("model.onnx")).toBe("model.onnx");
    expect(watchGlobFor("m[1]{a,b}*?!.pt")).toBe("m?1??a,b????.pt");
  });
});

describe("toTransferable", () => {
  it("passes a plain full-buffer Uint8Array through untouched", () => {
    const u8 = new Uint8Array([1, 2, 3]);
    expect(toTransferable(u8)).toBe(u8);
  });

  it("re-views a full-buffer Buffer as a plain Uint8Array without copying", () => {
    const ab = new ArrayBuffer(4);
    const buf = Buffer.from(ab);
    const out = toTransferable(buf);
    expect(out.constructor).toBe(Uint8Array);
    expect(out.buffer).toBe(ab);
  });

  it("copies a window into a larger buffer", () => {
    const big = new Uint8Array([0, 1, 2, 3, 4]);
    const out = toTransferable(big.subarray(1, 3));
    expect(Array.from(out)).toEqual([1, 2]);
    expect(out.buffer.byteLength).toBe(2);
  });
});
