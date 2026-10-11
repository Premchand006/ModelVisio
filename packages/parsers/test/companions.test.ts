import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { companionNames, parseModel, pickPrimary } from "../src/index";

const fx = (dir: string, f: string) => new Uint8Array(readFileSync(join(__dirname, "fixtures", dir, f)));
const hasStats = (m: ReturnType<typeof parseModel>) => m.layers.some((l) => l.w?.std != null);

describe("companion weight files", () => {
  it("names the siblings each split-weight format needs", () => {
    expect(companionNames("tinynet.xml")).toEqual(["tinynet.bin"]);
    expect(companionNames("net.param")).toEqual(["net.bin"]);
    expect(companionNames("net.param.bin")).toEqual(["net.bin"]);
    expect(companionNames("inference.pdmodel")).toEqual(["inference.pdiparams"]);
    expect(companionNames("saved_model.pb")).toEqual(["variables/variables.index", "variables/variables.data-00000-of-00001"]);
    expect(companionNames("model.onnx")).toEqual([]);
    const manifest = (paths: string[]) => new TextEncoder().encode(JSON.stringify({ weightsManifest: [{ paths, weights: [] }] }));
    expect(companionNames("model.json", manifest(["group1-shard1of2.bin", "group1-shard2of2.bin"]))).toEqual(["group1-shard1of2.bin", "group1-shard2of2.bin"]);
  });

  it("never asks a host for files outside the model's folder", () => {
    const evil = new TextEncoder().encode(JSON.stringify({ weightsManifest: [{ paths: ["../../.ssh/id_rsa", "/etc/passwd", "C:\\x.bin", "a\\..\\..\\b", "ok.bin"] }] }));
    expect(companionNames("model.json", evil)).toEqual(["ok.bin"]);
  });

  it("picks the graph file out of a multi-file selection", () => {
    expect(pickPrimary(["tinynet.bin", "tinynet.xml"])).toBe("tinynet.xml");
    expect(pickPrimary(["net.bin", "net.param.bin"])).toBe("net.param.bin");
    expect(pickPrimary(["m.pdiparams", "m.pdmodel"])).toBe("m.pdmodel");
  });

  it("OpenVINO: notes the missing .bin, then reads stats when it is supplied", () => {
    const xml = fx("openvino", "tinynet.xml");
    const alone = parseModel(xml, "tinynet.xml");
    expect(hasStats(alone)).toBe(false);
    expect(alone.notes?.join(" ")).toMatch(/tinynet\.bin/);
    const both = parseModel(xml, "tinynet.xml", { "tinynet.bin": fx("openvino", "tinynet.bin") });
    expect(hasStats(both)).toBe(true);
    expect(both.notes).toBeUndefined();
  });

  it("ncnn: reads the .bin companion", () => {
    const param = fx("ncnn", "tinynet.ncnn.param");
    expect(hasStats(parseModel(param, "tinynet.ncnn.param"))).toBe(false);
    const both = parseModel(param, "tinynet.ncnn.param", { "tinynet.ncnn.bin": fx("ncnn", "tinynet.ncnn.bin") });
    expect(hasStats(both)).toBe(true);
    expect(both.notes).toBeUndefined();
  });

  it("leaves single-file formats without a note", () => {
    expect(parseModel(fx("keras", "tiny_functional.keras"), "tiny_functional.keras").notes).toBeUndefined();
  });
});
