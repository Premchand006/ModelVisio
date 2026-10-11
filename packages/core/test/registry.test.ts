import { describe, expect, it } from "vitest";
import { createElement as h } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ThemeCtx } from "../src/theme/ThemeContext";
import { TH } from "../src/theme/theme";
import { demoModel } from "../src/demo/demoModel";
import { ModelRegistry } from "../src/components/ModelRegistry";
import {
  MemoryRegistryStore, REGISTRY_SCHEMA, checkCompat, exportRegistry, importRegistry, nextVersion, normalizeTags,
  openRegistryStore, parseRegistryFile, registerModel, resolveTarget, searchModels, serializeRegistry, sha256Hex,
  snapshotModel, storeFromFile,
} from "../src/registry";

const bytes = (...b: number[]) => new Uint8Array(b).buffer as ArrayBuffer;

describe("registry helpers", () => {
  it("sha256Hex matches the known digest of 'abc'", async () => {
    expect(await sha256Hex(new TextEncoder().encode("abc")))
      .toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });

  it("nextVersion bumps the highest semver minor and skips labels", () => {
    expect(nextVersion([])).toBe("1.0.0");
    expect(nextVersion(["1.0.0"])).toBe("1.1.0");
    expect(nextVersion(["1.0.0", "int8-experiment", "1.4.2", "v2.0.0"])).toBe("2.1.0");
    expect(nextVersion(["my-label"])).toBe("1.0.0");
  });

  it("normalizeTags trims, lowercases, dedupes, sorts", () => {
    expect(normalizeTags(" Detector, int8,detector ,, ")).toEqual(["detector", "int8"]);
  });

  it("resolveTarget accepts compat targets and hardware device ids", () => {
    expect(resolveTarget("coral")).toMatchObject({ id: "coral", deviceId: "coral", match: "coral" });
    expect(resolveTarget("AGX_ORIN")).toMatchObject({ id: "agx_orin", deviceId: "agx_orin" });
    expect(resolveTarget("nope")).toBeUndefined();
  });

  it("checkCompat collects op-level issues for a target from compIssues", () => {
    const r = checkCompat(demoModel, resolveTarget("coral")!);
    expect(r.issues.filter((i) => i.source === "op").length).toBeGreaterThan(0);
    expect(r.errors + r.warnings).toBe(r.issues.length);
  });
});

describe("registerModel + MemoryRegistryStore", () => {
  it("creates an entry, auto-versions, dedupes identical bytes", async () => {
    const s = new MemoryRegistryStore();
    const r1 = await registerModel(s, demoModel, { bytes: bytes(1, 2, 3), fileName: "yolo.onnx", tags: ["Detector"], now: "2026-01-01T00:00:00.000Z" });
    expect(r1.duplicate).toBe(false);
    expect(r1.entry.name).toBe("yolo");
    expect(r1.entry.tags).toEqual(["detector"]);
    expect(r1.version.version).toBe("1.0.0");
    expect(r1.version.sha256).toHaveLength(64);
    expect(r1.version.sizeBytes).toBe(3);
    expect(r1.version.compat.map((c) => c.target)).toEqual(["coral", "rk3588", "kneron", "hailo"]);
    expect(r1.version.estimates.length).toBeGreaterThan(0);
    expect(r1.version.layers).toHaveLength(demoModel.layers.length);

    const dup = await registerModel(s, demoModel, { bytes: bytes(1, 2, 3), fileName: "yolo.onnx" });
    expect(dup.duplicate).toBe(true);
    expect(await s.listVersions(r1.entry.id)).toHaveLength(1);

    const r2 = await registerModel(s, demoModel, { bytes: bytes(4, 5), fileName: "yolo.onnx", notes: "pruned", now: "2026-01-01T00:00:00.000Z" });
    expect(r2.entry.id).toBe(r1.entry.id); // matched by name
    expect(r2.version.version).toBe("1.1.0");
    expect(r2.version.createdAt > r1.version.createdAt).toBe(true); // strictly increasing
    const r3 = await registerModel(s, demoModel, { bytes: bytes(6), modelId: r1.entry.id, label: "int8" });
    expect(r3.version.version).toBe("int8");
    await expect(registerModel(s, demoModel, { bytes: bytes(7), modelId: r1.entry.id, label: "int8" })).rejects.toThrow(/already exists/);
    expect((await s.listVersions(r1.entry.id)).map((v) => v.version)).toEqual(["1.0.0", "1.1.0", "int8"]);
  });

  it("stores the file only when asked, and deletes cascade", async () => {
    const s = new MemoryRegistryStore();
    const a = await registerModel(s, demoModel, { bytes: bytes(9, 9), fileName: "a.onnx" });
    expect(a.version.fileStored).toBeFalsy();
    expect(await s.getBlob(a.version.id)).toBeUndefined();
    const b = await registerModel(s, demoModel, { bytes: bytes(8, 8), fileName: "a.onnx", storeFile: true });
    expect(b.version.fileStored).toBe(true);
    expect(new Uint8Array((await s.getBlob(b.version.id))!)).toEqual(new Uint8Array([8, 8]));
    await s.deleteModel(a.entry.id);
    expect(await s.listModels()).toEqual([]);
    expect(await s.getBlob(b.version.id)).toBeUndefined();
  });

  it("export → JSON → import round-trips and merge skips duplicates", async () => {
    const s = new MemoryRegistryStore();
    await registerModel(s, demoModel, { bytes: bytes(1), fileName: "m1.onnx", storeFile: true });
    await registerModel(s, demoModel, { bytes: bytes(2), fileName: "m1.onnx" });
    await registerModel(s, demoModel, { bytes: bytes(3), fileName: "m2.onnx", tags: ["edge"] });
    const file = await exportRegistry(s, "2026-02-02T00:00:00.000Z");
    expect(file.schema).toBe(REGISTRY_SCHEMA);
    expect(file.models.map((m) => [m.name, m.versions.length])).toEqual([["m1", 2], ["m2", 1]]);
    expect(file.models[0].versions.every((v) => !("fileStored" in v))).toBe(true);

    const text = serializeRegistry(file);
    const s2 = await storeFromFile(parseRegistryFile(text));
    expect(await exportRegistry(s2, file.exportedAt)).toEqual(file);

    const again = await importRegistry(s2, parseRegistryFile(text));
    expect(again).toEqual({ models: 0, versions: 0, skipped: 3 });
  });

  it("parseRegistryFile rejects bad input with readable errors", () => {
    expect(() => parseRegistryFile("{")).toThrow(/not valid JSON/);
    expect(() => parseRegistryFile({ schema: "x", models: [] })).toThrow(/Unsupported registry schema/);
    expect(() => parseRegistryFile({ schema: REGISTRY_SCHEMA, models: [{ id: 1 }] })).toThrow(/malformed model/);
  });

  it("searchModels filters by text and tag", async () => {
    const s = new MemoryRegistryStore();
    await registerModel(s, demoModel, { bytes: bytes(1), name: "YOLO detector", tags: ["vision"] });
    await registerModel(s, demoModel, { bytes: bytes(2), name: "Keyword spotter", tags: ["audio"], description: "tiny KWS" });
    const all = await s.listModels();
    expect(searchModels(all, "yolo").map((m) => m.name)).toEqual(["YOLO detector"]);
    expect(searchModels(all, "kws").map((m) => m.name)).toEqual(["Keyword spotter"]);
    expect(searchModels(all, "", "audio").map((m) => m.name)).toEqual(["Keyword spotter"]);
    expect(searchModels(all, "")).toHaveLength(2);
  });

  it("snapshot without bytes has no hash and uses the model's size", async () => {
    const snap = await snapshotModel(demoModel);
    expect(snap.sha256).toBe("");
    expect(snap.sizeBytes).toBe(demoModel.sizeBytes);
  });

  it("openRegistryStore falls back to memory when IndexedDB is unavailable (Node)", async () => {
    const s = await openRegistryStore();
    expect(s.kind).toBe("memory");
    expect(s.persistent).toBe(false);
  });
});

describe("ModelRegistry render (offline)", () => {
  const themed = (props: Parameters<typeof ModelRegistry>[0]) => renderToStaticMarkup(h(ThemeCtx.Provider, { value: TH.dark }, h(ModelRegistry, props)));

  it("renders with no model", () => {
    const html = themed({ model: null, modelBytes: null, store: new MemoryRegistryStore() });
    expect(html).toContain("Model Registry");
    expect(html).toContain("Load a model to register it.");
  });

  it("renders the register form for a loaded model", () => {
    const html = themed({ model: demoModel, modelBytes: bytes(1, 2), store: new MemoryRegistryStore() });
    expect(html).toContain("Register current model");
    expect(html).toContain("Export JSON");
    expect(html).toContain("memory only");
  });
});
