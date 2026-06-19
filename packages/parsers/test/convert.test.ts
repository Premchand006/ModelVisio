import { describe, expect, it } from "vitest";
import { crc32 } from "../src/common/crc32";
import { toHalf, f32BytesToF16Bytes } from "../src/common/float16";
import { writeZip } from "../src/common/zipwriter";
import { isZip, readZipEntries, extractStored } from "../src/common/zip";
import { writeSafetensors } from "../src/common/safetensorsWriter";
import { readNumpy, writeNpy, writeNpz } from "../src/common/numpy";
import { convertWeights, parseNumpy, parseSafetensors } from "../src/index";
import { modelProtoType } from "../src/onnx/proto";
import { extractOnnxTensors } from "../src/onnx/tensors";

const enc = (s: string) => new TextEncoder().encode(s);
const dec = (b: Uint8Array) => new TextDecoder().decode(b);

describe("crc32", () => {
  it("matches the canonical check value", () => {
    expect(crc32(enc("123456789")) >>> 0).toBe(0xcbf43926);
  });
});

describe("float16 encoder", () => {
  it("encodes known patterns", () => {
    expect(toHalf(1)).toBe(0x3c00);
    expect(toHalf(-2)).toBe(0xc000);
    expect(toHalf(0)).toBe(0x0000);
    expect(toHalf(-0)).toBe(0x8000);
    expect(toHalf(0.5)).toBe(0x3800);
    expect(toHalf(65504)).toBe(0x7bff); // max finite half
    expect(toHalf(1e5)).toBe(0x7c00);   // overflow → +Inf
    expect(toHalf(NaN) & 0x7c00).toBe(0x7c00);
    expect(toHalf(NaN) & 0x03ff).not.toBe(0); // NaN payload preserved
  });
  it("rounds ties to even (not half-up)", () => {
    expect(toHalf(1 + 1 / 2048)).toBe(0x3c00); // exact tie → down to even
    expect(toHalf(1 + 3 / 2048)).toBe(0x3c02); // exact tie → up to even
    expect(toHalf(65520)).toBe(0x7c00);        // ties to even → Inf
  });
  it("handles the subnormal round-to-even boundary", () => {
    expect(toHalf(Math.pow(2, -25))).toBe(0x0000);     // exact tie → 0 (even)
    expect(toHalf(Math.pow(2, -24))).toBe(0x0001);     // smallest subnormal
    expect(toHalf(Math.pow(2, -25) * 1.5)).toBe(0x0001);
  });
  it("rejects non-multiple-of-4 float32 buffers", () => {
    expect(() => f32BytesToF16Bytes(new Uint8Array(6))).toThrow();
  });
  it("converts a float32 byte buffer to half bytes", () => {
    const src = new Uint8Array(new Float32Array([1, -2]).buffer);
    const half = f32BytesToF16Bytes(src);
    expect(half.byteLength).toBe(4);
    const v = new DataView(half.buffer);
    expect(v.getUint16(0, true)).toBe(0x3c00);
    expect(v.getUint16(2, true)).toBe(0xc000);
  });
});

describe("zip writer", () => {
  it("produces an archive our reader can extract", () => {
    const zip = writeZip([
      { name: "a.txt", data: enc("hello") },
      { name: "dir/b.bin", data: new Uint8Array([1, 2, 3, 4]) },
    ]);
    expect(isZip(zip)).toBe(true);
    const entries = readZipEntries(zip);
    expect(entries.map((e) => e.name)).toEqual(["a.txt", "dir/b.bin"]);
    expect(dec(extractStored(zip, entries[0])!)).toBe("hello");
    expect([...extractStored(zip, entries[1])!]).toEqual([1, 2, 3, 4]);
  });
});

describe("safetensors writer", () => {
  it("round-trips through the safetensors reader", () => {
    const data = new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer); // 2×2
    const st = writeSafetensors([{ name: "w", dtype: "F32", shape: [2, 2], data }]);
    const model = parseSafetensors(st, "x.safetensors");
    const w = model.layers.find((l) => l.name === "w")!;
    expect(w.shape).toBe("2×2");
    expect(w.dt).toBe("float32");
    expect(w.params).toBe(4);
  });
});

// Minimal ONNX with a single float32 initializer "w" of shape [2,3].
function onnxWithInitializer(): Uint8Array {
  const ModelProto = modelProtoType();
  const obj = {
    ir_version: 8,
    opset_import: [{ domain: "", version: 17 }],
    graph: {
      name: "t",
      input: [], output: [], node: [],
      initializer: [{ name: "w", data_type: 1, dims: [2, 3], float_data: [1, 2, 3, 4, 5, 6] }],
    },
  };
  return ModelProto.encode(ModelProto.create(obj)).finish();
}

describe("convertWeights (ONNX → safetensors)", () => {
  const onnx = onnxWithInitializer();

  it("exports FP32 weights", () => {
    const r = convertWeights(onnx, "m.onnx", { format: "ONNX" });
    expect(r.filename).toBe("m.safetensors");
    const model = parseSafetensors(r.data, r.filename);
    const w = model.layers.find((l) => l.name === "w")!;
    expect(w.shape).toBe("2×3");
    expect(w.dt).toBe("float32");
    expect(w.params).toBe(6);
  });

  it("recasts to FP16 (half the bytes)", () => {
    const r = convertWeights(onnx, "m.onnx", { format: "ONNX", precision: "f16" });
    expect(r.filename).toBe("m.fp16.safetensors");
    const model = parseSafetensors(r.data, r.filename);
    const w = model.layers.find((l) => l.name === "w")!;
    expect(w.dt).toBe("float16");
    expect(w.params).toBe(6);
    expect(w.w?.size).toBe(6);
  });
});

// Build a minimal ONNX whose single initializer is supplied via a typed field.
function onnxTyped(init: Record<string, unknown>): Uint8Array {
  const ModelProto = modelProtoType();
  const obj = { ir_version: 8, opset_import: [{ domain: "", version: 17 }], graph: { name: "t", input: [], output: [], node: [], initializer: [init] } };
  return ModelProto.encode(ModelProto.create(obj)).finish();
}

describe("extractOnnxTensors (typed-field widths)", () => {
  it("packs FLOAT16 from int32_data at 2 bytes/elem", () => {
    const { tensors } = extractOnnxTensors(onnxTyped({ name: "h", data_type: 10, dims: [2], int32_data: [0x3c00, 0x4000] }));
    expect(tensors[0].bytes.byteLength).toBe(4);
    const v = new DataView(tensors[0].bytes.buffer, tensors[0].bytes.byteOffset, tensors[0].bytes.byteLength);
    expect(v.getUint16(0, true)).toBe(0x3c00);
    expect(v.getUint16(2, true)).toBe(0x4000);
  });
  it("packs UINT8 from int32_data at 1 byte/elem", () => {
    const { tensors } = extractOnnxTensors(onnxTyped({ name: "u", data_type: 2, dims: [3], int32_data: [10, 20, 30] }));
    expect([...tensors[0].bytes]).toEqual([10, 20, 30]);
  });
  it("preserves large int64 exactly (no double round-trip)", () => {
    const { tensors } = extractOnnxTensors(onnxTyped({ name: "i", data_type: 7, dims: [1], int64_data: ["9007199254740993"] }));
    const v = new DataView(tensors[0].bytes.buffer, tensors[0].bytes.byteOffset, tensors[0].bytes.byteLength);
    expect(v.getBigInt64(0, true)).toBe(9007199254740993n); // 2^53 + 1
  });
});

describe("numpy .npy/.npz", () => {
  it("round-trips a .npy through write→read", () => {
    const data = new Uint8Array(new Float32Array([1, 2, 3, 4, 5, 6]).buffer); // 2×3
    const npy = writeNpy({ name: "w", dtype: "F32", shape: [2, 3], data });
    const back = readNumpy(npy, "w.npy");
    expect(back).toHaveLength(1);
    expect(back[0].dtype).toBe("F32");
    expect(back[0].shape).toEqual([2, 3]);
    expect([...new Float32Array(back[0].data.slice().buffer)]).toEqual([1, 2, 3, 4, 5, 6]);
  });
  it("writes a .npz our parser reads", () => {
    const npz = writeNpz([
      { name: "a", dtype: "F32", shape: [2], data: new Uint8Array(new Float32Array([1, 2]).buffer) },
      { name: "b", dtype: "I32", shape: [3], data: new Uint8Array(new Int32Array([5, 6, 7]).buffer) },
    ]);
    const model = parseNumpy(npz, "x.npz");
    expect(model.layers.map((l) => l.name).sort()).toEqual(["a", "b"]);
  });
});

describe("convertWeights targets", () => {
  const onnx = (() => {
    const ModelProto = modelProtoType();
    const obj = { ir_version: 8, opset_import: [{ domain: "", version: 17 }], graph: { name: "t", input: [], output: [], node: [], initializer: [{ name: "w", data_type: 1, dims: [2, 2], float_data: [1, 2, 3, 4] }] } };
    return ModelProto.encode(ModelProto.create(obj)).finish();
  })();

  it("ONNX → NumPy .npz", () => {
    const r = convertWeights(onnx, "m.onnx", { format: "ONNX", target: "npz" });
    expect(r.filename).toBe("m.npz");
    const model = parseNumpy(r.data, r.filename);
    expect(model.layers.find((l) => l.name === "w")).toBeTruthy();
  });
  it("Safetensors → NumPy .npz and back upcasts FP16→FP32", () => {
    const st16 = convertWeights(onnx, "m.onnx", { format: "ONNX", target: "safetensors", precision: "f16" }).data;
    const up = convertWeights(st16, "m.safetensors", { format: "Safetensors", target: "safetensors", precision: "f32" });
    const model = parseSafetensors(up.data, up.filename);
    expect(model.layers.find((l) => l.name === "w")!.dt).toBe("float32");
    expect(up.filename).toBe("m.fp32.safetensors");
  });
  it("NumPy → Safetensors", () => {
    const npz = convertWeights(onnx, "m.onnx", { format: "ONNX", target: "npz" }).data;
    const st = convertWeights(npz, "m.npz", { format: "NumPy (.npz)", target: "safetensors" });
    const model = parseSafetensors(st.data, st.filename);
    expect(model.layers.find((l) => l.name === "w")!.shape).toBe("2×2");
  });
});

describe("convert validation", () => {
  it("rejects unsupported ONNX dtypes instead of mislabeling as F32", () => {
    const onnx = onnxTyped({ name: "c", data_type: 14, dims: [2], int32_data: [0, 0] }); // complex64
    expect(() => convertWeights(onnx, "m.onnx", { format: "ONNX" })).toThrow(/Unsupported ONNX dtype 14/);
  });
  it("rejects safetensors with out-of-range offsets", () => {
    const json = '{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}';
    const jb = new TextEncoder().encode(json);
    const buf = new Uint8Array(8 + jb.length + 4); // data block only 4 bytes, header claims 8
    new DataView(buf.buffer).setBigUint64(0, BigInt(jb.length), true);
    buf.set(jb, 8);
    expect(() => convertWeights(buf, "x.safetensors", { format: "Safetensors" })).toThrow();
  });
});
