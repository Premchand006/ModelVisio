import { describe, expect, it } from "vitest";
import { modelProtoType } from "../src/onnx/proto";
import { parseOnnx } from "../src/index";

// Exporters (plain torch.onnx.export, many hub models) omit intermediate
// value_info. These graphs carry none, so every shape past the input has to
// be inferred; a gap would zero the FLOPs of everything downstream.

const f32 = (name: string, dims: number[]) => ({
  name, data_type: 1, dims, float_data: new Array(dims.reduce((a, b) => a * b, 1)).fill(0.01),
});
const i64 = (name: string, vals: number[]) => ({ name, data_type: 7, dims: [vals.length], int64_data: vals });
// dims null = shape not declared (how exporters leave outputs they couldn't infer).
const vi = (name: string, dims: (number | string)[] | null) => ({
  name,
  type: { tensor_type: { elem_type: 1, ...(dims && { shape: { dim: dims.map((d) => (typeof d === "number" ? { dim_value: d } : { dim_param: d })) } }) } },
});
const ints = (name: string, v: number[]) => ({ name, type: 7, ints: v });
const int = (name: string, v: number) => ({ name, type: 2, i: v });

function encode(graph: object, meta: { key: string; value: string }[] = []): Uint8Array {
  const M = modelProtoType();
  return M.encode(M.create({ ir_version: 8, producer_name: "test", opset_import: [{ domain: "", version: 17 }], graph, metadata_props: meta })).finish();
}

// X[1,8,16,16] → Conv1x1(8→8) → Split[4,4] → Conv3x3 on each half → Concat →
// Resize ×2 → Reshape (target built by Shape→Gather→Concat) → Conv1x1(8→4)
function c2fLike(input: (number | string)[], meta: { key: string; value: string }[] = []) {
  return encode({
    name: "c2f",
    input: [vi("X", input)],
    output: [vi("Y", null)],
    initializer: [
      f32("w0", [8, 8, 1, 1]), f32("wa", [4, 4, 3, 3]), f32("wb", [4, 4, 3, 3]), f32("w3", [4, 8, 1, 1]),
      i64("split", [4, 4]), f32("roi", []), { name: "scales", data_type: 1, dims: [4], float_data: [1, 1, 2, 2] },
      i64("idx", [0, 1]), i64("hw", [32, 32]),
    ],
    node: [
      { op_type: "Conv", input: ["X", "w0"], output: ["c0"], attribute: [ints("kernel_shape", [1, 1])] },
      { op_type: "Split", input: ["c0", "split"], output: ["a", "b"], attribute: [int("axis", 1)] },
      { op_type: "Conv", input: ["a", "wa"], output: ["ca"], attribute: [ints("kernel_shape", [3, 3]), ints("pads", [1, 1, 1, 1])] },
      { op_type: "Conv", input: ["b", "wb"], output: ["cb"], attribute: [ints("kernel_shape", [3, 3]), ints("pads", [1, 1, 1, 1])] },
      { op_type: "Concat", input: ["ca", "cb"], output: ["cat"], attribute: [int("axis", 1)] },
      { op_type: "Resize", input: ["cat", "roi", "scales"], output: ["up"], attribute: [{ name: "mode", type: 3, s: new TextEncoder().encode("nearest") }] },
      { op_type: "Shape", input: ["up"], output: ["shp"] },
      { op_type: "Gather", input: ["shp", "idx"], output: ["nc"], attribute: [int("axis", 0)] },
      { op_type: "Concat", input: ["nc", "hw"], output: ["tgt"], attribute: [int("axis", 0)] },
      { op_type: "Reshape", input: ["up", "tgt"], output: ["rs"] },
      { op_type: "Conv", input: ["rs", "w3"], output: ["Y"], attribute: [ints("kernel_shape", [1, 1])] },
    ],
  }, meta);
}

const convMacs = (m: ReturnType<typeof parseOnnx>) => m.layers.filter((l) => l.op === "Conv").map((l) => l.macs);

describe("ONNX shape inference without value_info", () => {
  it("infers through Split, Concat, Resize and a computed Reshape target", () => {
    const m = parseOnnx(c2fLike([1, 8, 16, 16]));
    const shape = (op: string, i = 0) => m.layers.filter((l) => l.op === op)[i].shape;
    expect(shape("Split")).toBe("1×4×16×16");
    expect(m.layers.find((l) => l.op === "Split")!.outs.map((o) => o.s)).toEqual([[1, 4, 16, 16], [1, 4, 16, 16]]);
    expect(shape("Concat", 0)).toBe("1×8×16×16");
    expect(shape("Resize")).toBe("1×8×32×32");
    expect(shape("Reshape")).toBe("1×8×32×32");
    expect(m.outputShape).toEqual([1, 4, 32, 32]);
    // Every conv is costed: out elements × (Cin/group · kH · kW).
    expect(convMacs(m)).toEqual([8 * 256 * 8, 4 * 256 * 36, 4 * 256 * 36, 4 * 1024 * 8]);
    expect(m.notes).toBeUndefined();
  });

  it("uses the imgsz metadata for a dynamic height/width and says so", () => {
    const m = parseOnnx(c2fLike(["batch", 8, "height", "width"], [{ key: "imgsz", value: "[16, 16]" }]));
    expect(convMacs(m)).toEqual(convMacs(parseOnnx(c2fLike([1, 8, 16, 16]))));
    expect(m.notes?.[0]).toMatch(/16×16 recorded in the model metadata/);
  });

  it("prefers an explicit inputShape over metadata", () => {
    const m = parseOnnx(c2fLike(["batch", 8, "height", "width"], [{ key: "imgsz", value: "[640, 640]" }]), { inputShape: [1, 8, 16, 16] });
    expect(m.layers.find((l) => l.op === "Resize")!.shape).toBe("?×8×32×32");
    expect(m.notes?.[0]).toMatch(/requested shape/);
  });

  it("flags FLOPs as unreliable when the input size cannot be resolved", () => {
    const m = parseOnnx(c2fLike(["batch", 8, "height", "width"]));
    expect(m.notes?.[0]).toMatch(/dynamic dims .* far too low/);
  });

  it("marks a Concat extent unknown instead of dropping an unknown input", () => {
    const bytes = encode({
      name: "cat",
      input: [vi("A", [1, 4, 8, 8]), vi("B", [1, 4, 8, 8])],
      output: [vi("Y", null)],
      node: [
        { op_type: "NonStandardOp", input: ["B"], output: ["b2"] },
        { op_type: "Concat", input: ["A", "b2"], output: ["Y"], attribute: [int("axis", 1)] },
      ],
    });
    expect(parseOnnx(bytes).layers.find((l) => l.op === "Concat")!.shape).toBe("1×?×8×8");
  });
});
