import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isRknn, parseRknn } from "../src/formats/rknn";

// Real public RKNN samples (BSD-3-Clause, see fixtures/rknn/README.md).
const fx = (f: string) => readFileSync(fileURLToPath(new URL(`./fixtures/rknn/${f}`, import.meta.url)));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

describe("RKNN (rknn-toolkit2, embedded FlatBuffer graph)", () => {
  for (const [file, ver, target] of [
    ["multiple_input_demo_rk3566.rknn", "1.3.4", "rk3566"],
    ["multiple_input_demo_rk3562.rknn", "1.4.1", "rk3562"],
  ] as const) {
    it(`recovers the operator graph of ${file}`, () => {
      const buf = fx(file);
      expect(isRknn(buf)).toBe(true);
      const m = parseRknn(buf, file);
      expect(m.format).toBe(`RKNN v${ver}`);
      expect(m.producer).toContain(`target ${target}`);
      expect(m.producer).toContain("from ONNX");
      // 2 InputOperators + Conv + fused ConvAdd (OutputOperator folded into "output" group).
      expect(m.layers.map((l) => l.type)).toEqual(["Input", "Input", "Conv", "ConvAdd"]);
      expect(m.edges).toEqual([[1, 2], [0, 3], [2, 3]]);
      expect(m.inputShape).toEqual([1, 1, 128, 128]);
      expect(m.outputShape).toEqual([1, 8, 128, 128]);
      const [, , conv, convAdd] = m.layers;
      expect(conv.shape).toBe("1×8×128×128");
      expect(conv.w?.shape).toEqual([8, 3, 3, 3]);
      expect(conv.params).toBe(216 + 8);
      expect(convAdd.params).toBe(72 + 8);
      expect(convAdd.group).toBe("output");
      expect(conv.dt).toBe("int8");
      expect(sum(m.layers.map((l) => l.params))).toBe(304);
      expect(conv.macs).toBe(8 * 128 * 128 * 27);
      expect(convAdd.macs).toBe(8 * 128 * 128 * 9);
    });
  }
});

describe("RKNN (rknn-toolkit v1, JSON graph)", () => {
  it("parses nodes, connections and norm tensors", () => {
    const m = parseRknn(fx("resize_area.rknn"), "resize_area.rknn");
    expect(m.format).toBe("RKNN v1.3.0");
    expect(m.producer).toContain("from tensorflow");
    expect(m.layers.map((l) => l.type)).toEqual(["Input", "RESIZEAREA"]);
    expect(m.edges).toEqual([[0, 1]]);
    // ovxlib sizes are innermost-first; shapes are flipped to outermost-first.
    expect(m.inputShape).toEqual([1, 32, 32, 3]);
    expect(m.outputShape).toEqual([64, 64, 3]);
    expect(m.layers[1].attr.size).toEqual([64, 64]);
    expect(m.layers[1].dt).toBe("float16");
    expect(m.layers[1].group).toBe("output");
  });

  it("falls back to a brace-balanced JSON scan when header sizes are wrong", () => {
    const buf = new Uint8Array(fx("resize_area.rknn"));
    new DataView(buf.buffer).setBigUint64(16, 0xffffffn, true); // corrupt data_size
    const m = parseRknn(buf, "broken.rknn");
    expect(m.layers.map((l) => l.type)).toEqual(["Input", "RESIZEAREA"]);
  });
});

describe("RKNN edge cases", () => {
  it("reports encrypted models as metadata", () => {
    const buf = new Uint8Array(64);
    buf.set(new TextEncoder().encode("CYPTRKNN"));
    expect(isRknn(buf)).toBe(true);
    const m = parseRknn(buf, "enc.rknn");
    expect(m.format).toBe("RKNN (encrypted)");
    expect(m.layers).toHaveLength(1);
  });

  it("rejects non-RKNN data", () => {
    expect(isRknn(new Uint8Array(16))).toBe(false);
    expect(() => parseRknn(new Uint8Array(16), "x.rknn")).toThrow();
  });
});
