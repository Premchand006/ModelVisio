import type { Model, ModelLayer } from "../types";

// TFLite models are FlatBuffers. Rather than vendor the full generated schema,
// we read the tables we need with a minimal FlatBuffer reader. Field indices
// below come from tensorflow/lite/schema/schema.fbs (stable ordering).

class FB {
  view: DataView;
  constructor(public buf: Uint8Array) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  u8(p: number) { return this.view.getUint8(p); }
  i8(p: number) { return this.view.getInt8(p); }
  u32(p: number) { return this.view.getUint32(p, true); }
  i32(p: number) { return this.view.getInt32(p, true); }
  root(): number { return this.u32(0); }
  /** Absolute position of a table field, or null if absent. */
  field(table: number, index: number): number | null {
    const vtable = table - this.i32(table);
    const vtSize = this.view.getUint16(vtable, true);
    const slot = 4 + index * 2;
    if (slot >= vtSize) return null;
    const voff = this.view.getUint16(vtable + slot, true);
    return voff === 0 ? null : table + voff;
  }
  indirect(p: number): number { return p + this.u32(p); } // sub-table/string/vector
  string(table: number, index: number): string {
    const f = this.field(table, index);
    if (f == null) return "";
    const p = this.indirect(f);
    const len = this.u32(p);
    return new TextDecoder().decode(this.buf.subarray(p + 4, p + 4 + len));
  }
  /** Vector position + length for an offset-typed field. */
  vec(table: number, index: number): { base: number; len: number } | null {
    const f = this.field(table, index);
    if (f == null) return null;
    const p = this.indirect(f);
    return { base: p + 4, len: this.u32(p) };
  }
  /** Element position of an offset-vector item (resolves the uoffset). */
  vecTable(base: number, i: number): number {
    const elem = base + i * 4;
    return elem + this.u32(elem);
  }
  intVec(table: number, index: number): number[] {
    const v = this.vec(table, index);
    if (!v) return [];
    const out: number[] = [];
    for (let i = 0; i < v.len; i++) out.push(this.i32(v.base + i * 4));
    return out;
  }
  scalarByte(table: number, index: number, dflt = 0): number {
    const f = this.field(table, index);
    return f == null ? dflt : this.u8(f);
  }
  scalarI32(table: number, index: number, dflt = 0): number {
    const f = this.field(table, index);
    return f == null ? dflt : this.i32(f);
  }
}

const TENSOR_TYPE: Record<number, string> = {
  0: "float32", 1: "float16", 2: "int32", 3: "uint8", 4: "int64", 5: "string",
  6: "bool", 7: "int16", 8: "complex64", 9: "int8", 10: "float64", 11: "complex128",
  12: "uint64", 15: "uint32", 16: "uint16", 17: "int4",
};

// Partial BuiltinOperator map (common ops). Unknown → "OP_<code>".
const BUILTIN: Record<number, string> = {
  0: "Add", 1: "AveragePool2D", 2: "Concatenation", 3: "Conv2D", 4: "DepthwiseConv2D",
  6: "Dequantize", 9: "FullyConnected", 14: "Logistic", 17: "MaxPool2D", 18: "Mul",
  19: "Relu", 21: "Relu6", 22: "Reshape", 23: "ResizeBilinear", 25: "Softmax",
  28: "Tanh", 34: "Pad", 40: "Mean", 41: "Sub", 49: "Split", 56: "ArgMax",
  83: "Pack", 88: "Slice", 97: "ResizeNearestNeighbor", 99: "LeakyRelu",
  102: "Unpack", 114: "Quantize", 117: "HardSwish", 124: "Transpose",
};

function group(op: string, isOutput: boolean): ModelLayer["group"] {
  if (isOutput) return "output";
  if (/Conv|Depthwise/.test(op)) return "backbone";
  if (/FullyConnected|Softmax|ArgMax/.test(op)) return "head";
  if (/Pool|Concat|Reshape|Resize|Pad|Transpose/.test(op)) return "neck";
  return "backbone";
}

export function parseTflite(input: ArrayBuffer | Uint8Array, name = "model.tflite"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  // Optional file identifier "TFL3" sits at bytes 4..8.
  const fb = new FB(buf);
  const model = fb.root();

  // operator_codes (field 1) → builtin/custom name per index.
  const opCodes: string[] = [];
  const ocVec = fb.vec(model, 1);
  if (ocVec) {
    for (let i = 0; i < ocVec.len; i++) {
      const oc = fb.vecTable(ocVec.base, i);
      const deprecated = fb.scalarByte(oc, 0, 0);
      const builtin = fb.scalarI32(oc, 3, 0);
      const code = Math.max(deprecated, builtin);
      const custom = fb.string(oc, 1);
      opCodes.push(code === 0 && custom ? custom : BUILTIN[code] ?? `OP_${code}`);
    }
  }

  // subgraphs (field 2) → use the first.
  const sgVec = fb.vec(model, 2);
  if (!sgVec || sgVec.len === 0) throw new Error("TFLite model has no subgraphs.");
  const sg = fb.vecTable(sgVec.base, 0);

  const tensorsVec = fb.vec(sg, 0);
  const tensorNames: string[] = [];
  const tensorShapes: number[][] = [];
  const tensorTypes: string[] = [];
  if (tensorsVec) {
    for (let i = 0; i < tensorsVec.len; i++) {
      const t = fb.vecTable(tensorsVec.base, i);
      tensorShapes.push(fb.intVec(t, 0));
      tensorTypes.push(TENSOR_TYPE[fb.scalarByte(t, 1, 0)] ?? "float32");
      tensorNames.push(fb.string(t, 3) || `tensor_${i}`);
    }
  }

  const inputs = fb.intVec(sg, 1);
  const outputs = fb.intVec(sg, 2);
  const outputSet = new Set(outputs);

  const layers: ModelLayer[] = [];
  const producer = new Map<number, number>(); // tensor index → layer id
  let id = 0;

  // Input layers for the graph inputs.
  for (const ti of inputs) {
    const lid = id++;
    producer.set(ti, lid);
    layers.push(mkLayer(lid, tensorNames[ti], "Input", "Input", tensorShapes[ti], tensorTypes[ti], "input", [], [ti], tensorNames, tensorShapes));
  }

  // One layer per operator.
  const opsVec = fb.vec(sg, 3);
  const edges: [number, number][] = [];
  type Op = { opName: string; ins: number[]; outs: number[] };
  const ops: Op[] = [];
  if (opsVec) {
    for (let i = 0; i < opsVec.len; i++) {
      const op = fb.vecTable(opsVec.base, i);
      const opcodeIndex = fb.scalarI32(op, 0, 0);
      ops.push({ opName: opCodes[opcodeIndex] ?? `OP_${opcodeIndex}`, ins: fb.intVec(op, 1), outs: fb.intVec(op, 2) });
    }
  }
  for (const op of ops) {
    const lid = id++;
    const outTi = op.outs[0];
    const isOut = op.outs.some((t) => outputSet.has(t));
    for (const t of op.outs) producer.set(t, lid);
    layers.push(mkLayer(lid, tensorNames[outTi] ?? op.opName, op.opName, op.opName, tensorShapes[outTi] ?? [], tensorTypes[outTi] ?? "float32", group(op.opName, isOut), op.ins, op.outs, tensorNames, tensorShapes));
  }
  // Edges from data flow (skip weight tensors with no producer).
  const seen = new Set<string>();
  for (let li = 0; li < layers.length; li++) {
    const l = layers[li];
    for (const inp of l.ins) {
      const ti = (inp as { ti?: number }).ti;
      if (ti == null) continue;
      const from = producer.get(ti);
      if (from == null || from === l.id) continue;
      const key = `${from}->${l.id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      edges.push([from, l.id]);
    }
  }

  const firstIn = inputs[0];
  const firstOut = outputs[0];
  return {
    name, format: "TFLite", framework: "TensorFlow Lite", opset: 3,
    sizeBytes: buf.byteLength,
    inputShape: firstIn != null ? tensorShapes[firstIn] ?? [] : [],
    outputShape: firstOut != null ? tensorShapes[firstOut] ?? [] : [],
    layers, edges,
  };
}

function mkLayer(
  lid: number, name: string, type: string, op: string, shape: number[], dt: string,
  grp: ModelLayer["group"], inTis: number[], outTis: number[],
  tensorNames: string[], tensorShapes: number[][],
): ModelLayer {
  const memEls = shape.reduce((a, b) => a * b, shape.length ? 1 : 0);
  return {
    id: lid, name, type, op,
    shape: shape.length ? shape.join("×") : "?", dt,
    params: 0, flops: 0, macs: 0, mem: memEls * 4, group: grp,
    // Carry the source tensor index on each input so edges can be resolved.
    ins: inTis.map((ti) => ({ n: tensorNames[ti] ?? `#${ti}`, s: tensorShapes[ti], ti } as { n: string; s?: number[]; ti: number })),
    outs: outTis.map((ti) => ({ n: tensorNames[ti] ?? `#${ti}`, s: tensorShapes[ti] })),
    attr: {}, w: null,
    math: op, insight: `TFLite ${op} operator.`, qSens: /Conv|FullyConnected/.test(op) ? 0.5 : 0.15,
    compIssues: [],
  };
}
