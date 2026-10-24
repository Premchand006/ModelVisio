import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { isOpenvinoXml, parseOpenvino } from "../src/formats/openvino";
import { parseXml } from "../src/common/xml";

// Fixtures: TinyNet (see fixtures/scripts/tinynet.py) exported torch → ONNX →
// OpenVINO 2026.4 IR v11 by fixtures/scripts/make_openvino.py.
const dir = join(__dirname, "fixtures", "openvino");
const read = (f: string) => new Uint8Array(readFileSync(join(dir, f)));

// conv1 216+8, conv2 576+8, conv3 128+16, fc 160+10 (BN folded into conv1 by the exporter).
const TOTAL_PARAMS = 1122;

describe("xml", () => {
  it("parses attributes, text, entities, comments and self-closing tags", () => {
    const el = parseXml(`<?xml version="1.0"?><!-- c --><a x="1 &amp; 2"><b/><c>t&lt;</c><![CDATA[<raw>]]></a>`);
    expect(el.tag).toBe("a");
    expect(el.attrs.x).toBe("1 & 2");
    expect(el.children.map((c) => c.tag)).toEqual(["b", "c"]);
    expect(el.children[1].text).toBe("t<");
    expect(el.text).toBe("<raw>");
  });
  it("throws on mismatched tags", () => {
    expect(() => parseXml("<a><b></a>")).toThrow();
  });
});

describe("openvino", () => {
  const xml = read("tinynet.xml");
  const bin = read("tinynet.bin");

  it("sniffs IR vs other XML", () => {
    expect(isOpenvinoXml(xml)).toBe(true);
    expect(isOpenvinoXml(new TextEncoder().encode('<?xml version="1.0"?><svg><g/></svg>'))).toBe(false);
  });

  it("builds the graph with Const layers folded into their consumers", () => {
    const m = parseOpenvino(xml, "tinynet.xml", bin);
    expect(m.format).toBe("OpenVINO IR");
    expect(m.irVersion).toBe(11);
    expect(m.opset).toBe(8);
    expect(m.layers).toHaveLength(18); // 28 IR layers - 10 Const
    expect(m.layers.some((l) => l.type === "Const")).toBe(false);
    expect(m.edges).toHaveLength(18);
    expect(m.inputShape).toEqual([1, 3, 32, 32]);
    expect(m.outputShape).toEqual([1, 10]);

    const total = m.layers.reduce((a, l) => a + l.params, 0);
    expect(total).toBe(TOTAL_PARAMS);

    const types = m.layers.map((l) => l.type);
    expect(types.filter((t) => t === "Convolution")).toHaveLength(3);
    expect(types[0]).toBe("Parameter");
    expect(types[types.length - 1]).toBe("Result");
    expect(m.layers[0].group).toBe("input");
    expect(m.layers[m.layers.length - 1].group).toBe("output");

    const conv1 = m.layers.find((l) => l.name === "/conv1/Conv/WithoutBiases")!;
    expect(conv1.shape).toBe("1×8×32×32");
    expect(conv1.params).toBe(216);
    expect(conv1.macs).toBe(8 * 32 * 32 * 3 * 9);
    expect(conv1.flops).toBe(2 * conv1.macs);
    // Weight stats come from the .bin.
    expect(conv1.w?.shape).toEqual([8, 3, 3, 3]);
    expect(conv1.w?.std).toBeGreaterThan(0);

    const fc = m.layers.find((l) => l.name === "/fc/Gemm/WithoutBiases")!;
    expect(fc.type).toBe("MatMul");
    expect(fc.params).toBe(160);
    expect(fc.macs).toBe(160);
    expect(fc.group).toBe("head");

    // Residual add has two data inputs and both edges exist.
    const add = m.layers.find((l) => l.name === "/Add")!;
    expect(add.ins).toHaveLength(2);
    const pool = m.layers.find((l) => l.type === "MaxPool")!;
    expect(m.edges).toContainEqual([pool.id, add.id]);
  });

  it("works without the .bin (params from Const shapes, no value stats)", () => {
    const m = parseOpenvino(xml, "tinynet.xml");
    expect(m.layers.reduce((a, l) => a + l.params, 0)).toBe(TOTAL_PARAMS);
    const conv1 = m.layers.find((l) => l.type === "Convolution")!;
    expect(conv1.w?.size).toBe(216);
    expect(conv1.w?.std).toBeUndefined();
  });

  it("folds fp16 Const → Convert decompression chains", () => {
    const m = parseOpenvino(read("tinynet_fp16.xml"), "tinynet_fp16.xml", read("tinynet_fp16.bin"));
    expect(m.layers.some((l) => l.type === "Convert" || l.type === "Const")).toBe(false);
    expect(m.layers).toHaveLength(18);
    expect(m.layers.reduce((a, l) => a + l.params, 0)).toBe(TOTAL_PARAMS);
    const conv1 = m.layers.find((l) => l.type === "Convolution")!;
    expect(conv1.w?.std).toBeGreaterThan(0);
  });

  it("falls back to metadata for non-IR XML", () => {
    const m = parseOpenvino(new TextEncoder().encode("<project><item/></project>"), "pom.xml");
    expect(m.layers).toHaveLength(1);
    expect(m.layers[0].op).toBe("Detected");
    const bad = parseOpenvino(new TextEncoder().encode("<net><layers><layer"), "broken.xml");
    expect(bad.layers).toHaveLength(1);
  });
});
