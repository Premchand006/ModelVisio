import type { Model, ModelLayer } from "../types";
import { ProtoReader, WT_I32, WT_I64, WT_LEN, WT_VARINT } from "../common/protobufWire";
import { extractStored, isZip, readZipEntries, type ZipEntry } from "../common/zip";
import { inflateRaw } from "../common/inflate";
import {
  DYN, EdgeSet, NO_COST, broadcast, cleanShape, elemCost, linearCost, makeLayer, numel, sampleBytes, weightStats,
  type Cost, type Shape, type WStat,
} from "../common/graphKit";
import { parseContainer } from "./container";

// Core ML (.mlmodel = binary protobuf `Model`; .mlpackage = a directory with
// Manifest.json + Data/com.apple.CoreML/model.mlmodel + weights/weight.bin,
// accepted here as a zip). Two graph flavours:
//  • NeuralNetwork / NeuralNetworkClassifier / NeuralNetworkRegressor: a list
//    of typed layers wired by tensor names, weights inline (WeightParams).
//  • ML Program (MIL): functions → blocks → operations with typed outputs, so
//    every tensor shape is explicit. Weights are `const` ops, either inline or
//    blobs in weight.bin; they are folded into the op that consumes them.
// Pipelines chain sub-models by feature name; other model kinds (GLM, trees,
// …) become a single node. Field numbers follow apple/coremltools
// mlmodel/format/{Model,NeuralNetwork,MIL,FeatureTypes}.proto.

// ── Schema-less field reader ─────────────────────────────────────────────────

type Entry = { wt: number; v: number | Uint8Array };
class Fields {
  readonly m = new Map<number, Entry[]>();
  constructor(r: ProtoReader) {
    for (let t = r.next(); t; t = r.next()) {
      let v: number | Uint8Array;
      if (t.wireType === WT_VARINT) v = r.int64();
      else if (t.wireType === WT_LEN) v = r.bytes();
      else if (t.wireType === WT_I32) { v = r.buf.subarray(r.pos, r.pos + 4); r.pos += 4; }
      else if (t.wireType === WT_I64) { v = r.buf.subarray(r.pos, r.pos + 8); r.pos += 8; }
      else { r.skip(t.wireType); continue; }
      let arr = this.m.get(t.field);
      if (!arr) this.m.set(t.field, (arr = []));
      arr.push({ wt: t.wireType, v });
    }
  }
  static of(b: Uint8Array): Fields { return new Fields(new ProtoReader(b)); }
  has(f: number): boolean { return this.m.has(f); }
  num(f: number, d = 0): number {
    const e = this.m.get(f)?.[0];
    return e && typeof e.v === "number" ? e.v : d;
  }
  bool(f: number): boolean { return this.num(f, 0) !== 0; }
  /** Repeated integer field, packed or not. */
  nums(f: number): number[] {
    const out: number[] = [];
    for (const e of this.m.get(f) ?? []) {
      if (typeof e.v === "number") out.push(e.v);
      else {
        const r = new ProtoReader(e.v);
        while (!r.eof) out.push(r.int64());
      }
    }
    return out;
  }
  float(f: number, d = 0): number {
    const e = this.m.get(f)?.[0];
    if (!e || typeof e.v === "number" || e.v.byteLength !== 4) return d;
    return new DataView(e.v.buffer, e.v.byteOffset, 4).getFloat32(0, true);
  }
  bytes(f: number): Uint8Array | undefined {
    const e = this.m.get(f)?.[0];
    return e && typeof e.v !== "number" ? e.v : undefined;
  }
  all(f: number): Uint8Array[] {
    return (this.m.get(f) ?? []).map((e) => e.v).filter((v): v is Uint8Array => typeof v !== "number");
  }
  str(f: number): string { const b = this.bytes(f); return b ? utf8.decode(b) : ""; }
  strs(f: number): string[] { return this.all(f).map((b) => utf8.decode(b)); }
  msg(f: number): Fields | undefined { const b = this.bytes(f); return b ? Fields.of(b) : undefined; }
  msgs(f: number): Fields[] { return this.all(f).map((b) => Fields.of(b)); }
  /** map<string, Message> entries (key = 1, value = 2). */
  map(f: number): [string, Fields][] {
    return this.msgs(f).map((e) => [e.str(1), e.msg(2) ?? Fields.of(new Uint8Array(0))]);
  }
  /** First field number present among `candidates` (oneof). */
  which(candidates: Iterable<number>): number | undefined {
    for (const c of candidates) if (this.m.has(c)) return c;
    return undefined;
  }
}
const utf8 = new TextDecoder();

// ── Model-level constants ────────────────────────────────────────────────────

const MODEL_KINDS: Record<number, string> = {
  200: "pipelineClassifier", 201: "pipelineRegressor", 202: "pipeline",
  300: "glmRegressor", 301: "supportVectorRegressor", 302: "treeEnsembleRegressor",
  303: "neuralNetworkRegressor", 304: "bayesianProbitRegressor",
  400: "glmClassifier", 401: "supportVectorClassifier", 402: "treeEnsembleClassifier",
  403: "neuralNetworkClassifier", 404: "kNearestNeighborsClassifier",
  500: "neuralNetwork", 501: "itemSimilarityRecommender", 502: "mlProgram",
  555: "customModel", 556: "linkedModel", 560: "classConfidenceThresholding",
  600: "oneHotEncoder", 601: "imputer", 602: "featureVectorizer", 603: "dictVectorizer", 604: "scaler",
  606: "categoricalMapping", 607: "normalizer", 609: "arrayFeatureExtractor", 610: "nonMaximumSuppression",
  900: "identity", 2000: "textClassifier", 2001: "wordTagger", 2002: "visionFeaturePrint",
  2003: "soundAnalysisPreprocessing", 2004: "gazetteer", 2005: "wordEmbedding", 2006: "audioFeaturePrint",
  3000: "serializedModel",
};
const KIND_FIELDS = Object.keys(MODEL_KINDS).map(Number);

// NeuralNetworkLayer oneof (generated from NeuralNetwork.proto).
const NN_LAYERS: Record<number, string> = {
  100: "convolution", 120: "pooling", 130: "activation", 140: "innerProduct", 150: "embedding", 160: "batchnorm",
  165: "mvn", 170: "l2normalize", 175: "softmax", 180: "lrn", 190: "crop", 200: "padding", 210: "upsample",
  211: "resizeBilinear", 212: "cropResize", 220: "unary", 230: "add", 231: "multiply", 240: "average", 245: "scale",
  250: "bias", 260: "max", 261: "min", 270: "dot", 280: "reduce", 290: "loadConstant", 300: "reshape", 301: "flatten",
  310: "permute", 320: "concat", 330: "split", 340: "sequenceRepeat", 345: "reorganizeData", 350: "slice",
  400: "simpleRecurrent", 410: "gru", 420: "uniDirectionalLSTM", 430: "biDirectionalLSTM", 500: "custom",
  600: "copy", 605: "branch", 615: "loop", 620: "loopBreak", 625: "loopContinue", 635: "rangeStatic",
  640: "rangeDynamic", 660: "clip", 665: "ceil", 670: "floor", 680: "sign", 685: "round", 700: "exp2", 710: "sin",
  715: "cos", 720: "tan", 730: "asin", 735: "acos", 740: "atan", 750: "sinh", 755: "cosh", 760: "tanh", 770: "asinh",
  775: "acosh", 780: "atanh", 790: "erf", 795: "gelu", 815: "equal", 820: "notEqual", 825: "lessThan",
  827: "lessEqual", 830: "greaterThan", 832: "greaterEqual", 840: "logicalOr", 845: "logicalXor", 850: "logicalNot",
  855: "logicalAnd", 865: "modBroadcastable", 870: "minBroadcastable", 875: "maxBroadcastable",
  880: "addBroadcastable", 885: "powBroadcastable", 890: "divideBroadcastable", 895: "floorDivBroadcastable",
  900: "multiplyBroadcastable", 905: "subtractBroadcastable", 920: "tile", 925: "stack", 930: "gather",
  935: "scatter", 940: "gatherND", 945: "scatterND", 950: "softmaxND", 952: "gatherAlongAxis",
  954: "scatterAlongAxis", 960: "reverse", 965: "reverseSeq", 975: "splitND", 980: "concatND", 985: "transpose",
  995: "sliceStatic", 1000: "sliceDynamic", 1005: "slidingWindows", 1015: "topK", 1020: "argMin", 1025: "argMax",
  1040: "embeddingND", 1045: "batchedMatmul", 1065: "getShape", 1070: "loadConstantND", 1080: "fillLike",
  1085: "fillStatic", 1090: "fillDynamic", 1100: "broadcastToLike", 1105: "broadcastToStatic",
  1110: "broadcastToDynamic", 1120: "squeeze", 1125: "expandDims", 1130: "flattenTo2D", 1135: "reshapeLike",
  1140: "reshapeStatic", 1145: "reshapeDynamic", 1150: "rankPreservingReshape", 1155: "constantPad",
  1170: "randomNormalLike", 1175: "randomNormalStatic", 1180: "randomNormalDynamic", 1190: "randomUniformLike",
  1195: "randomUniformStatic", 1200: "randomUniformDynamic", 1210: "randomBernoulliLike",
  1215: "randomBernoulliStatic", 1220: "randomBernoulliDynamic", 1230: "categoricalDistribution",
  1250: "reduceL1", 1255: "reduceL2", 1260: "reduceMax", 1265: "reduceMin", 1270: "reduceSum", 1275: "reduceProd",
  1280: "reduceMean", 1285: "reduceLogSum", 1290: "reduceSumSquare", 1295: "reduceLogSumExp",
  1313: "whereNonZero", 1315: "matrixBandPart", 1320: "lowerTriangular", 1325: "upperTriangular",
  1330: "whereBroadcastable", 1350: "layerNormalization", 1400: "NonMaximumSuppression", 1450: "oneHot",
  1455: "cumSum", 1460: "clampedReLU", 1461: "argSort", 1465: "pooling3d", 1466: "globalPooling3d",
  1470: "sliceBySize", 1471: "convolution3d",
};
const NN_LAYER_FIELDS = Object.keys(NN_LAYERS).map(Number);

const ACTIVATIONS: Record<number, string> = {
  5: "Linear", 10: "ReLU", 15: "LeakyReLU", 20: "ThresholdedReLU", 25: "PReLU", 30: "Tanh", 31: "ScaledTanh",
  40: "Sigmoid", 41: "SigmoidHard", 50: "ELU", 60: "Softsign", 70: "Softplus", 71: "ParametricSoftplus",
};

// WeightParams-typed fields per layer kind.
const NN_WEIGHT_FIELDS: Record<string, number[]> = {
  convolution: [90, 91], innerProduct: [20, 21], batchnorm: [15, 16, 17, 18], embedding: [20, 21],
  scale: [2, 5], bias: [2], loadConstant: [2], loadConstantND: [2], batchedMatmul: [8, 9],
  embeddingND: [20, 21], layerNormalization: [3, 4], convolution3d: [60, 61],
};

const MIL_DT: Record<number, string> = {
  1: "bool", 2: "string", 10: "float16", 11: "float32", 12: "float64", 13: "bfloat16", 21: "int8", 22: "int16",
  23: "int32", 24: "int64", 25: "int4", 31: "uint8", 32: "uint16", 33: "uint32", 34: "uint64", 35: "uint4",
  36: "uint2", 37: "uint1", 38: "uint6", 39: "uint3", 40: "float8e4m3", 41: "float8e5m2",
};
const BLOB_DT: Record<number, string> = {
  1: "float16", 2: "float32", 3: "uint8", 4: "int8", 5: "bfloat16", 6: "int16", 7: "uint16", 14: "int32", 15: "uint32",
};
const PARAM_DT = /^(float|bfloat|int8|uint8|int4|uint[1-6]$)/;
const ARRAY_DT: Record<number, string> = { 65568: "float32", 65600: "float64", 131104: "int32", 131080: "int8", 65552: "float16" };

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

// ── Feature descriptions ─────────────────────────────────────────────────────

type Feature = { name: string; kind: string; shape?: Shape; dt: string; detail?: string };

function readFeature(f: Fields): Feature {
  const name = f.str(1);
  const t = f.msg(3);
  if (!t) return { name, kind: "unknown", dt: "-" };
  if (t.has(5)) {
    const a = t.msg(5)!;
    let shape = a.nums(1);
    if (!shape.length) {
      // Flexible shapes: take the first enumerated shape or range lower bounds.
      const en = a.msg(21)?.msgs(1)[0];
      if (en) shape = en.nums(1);
      else {
        const rng = a.msg(31)?.msgs(1);
        if (rng?.length) shape = rng.map((r) => { const ub = r.num(2, -1); return ub === r.num(1, 0) ? ub : DYN; });
      }
    }
    return { name, kind: "multiArray", shape: shape.length ? shape : undefined, dt: ARRAY_DT[a.num(2)] ?? "float32" };
  }
  if (t.has(4)) {
    const im = t.msg(4)!;
    const cs = im.num(3);
    const ch = cs === 10 || cs === 40 ? 1 : 3;
    const color = cs === 10 ? "GRAYSCALE" : cs === 20 ? "RGB" : cs === 30 ? "BGR" : cs === 40 ? "GRAYSCALE_FLOAT16" : "?";
    return { name, kind: "image", shape: [1, ch, im.num(2) || DYN, im.num(1) || DYN], dt: "uint8", detail: color };
  }
  if (t.has(1)) return { name, kind: "int64", shape: [1], dt: "int64" };
  if (t.has(2)) return { name, kind: "double", shape: [1], dt: "float64" };
  if (t.has(3)) return { name, kind: "string", dt: "string" };
  if (t.has(6)) return { name, kind: "dictionary", dt: "dict" };
  if (t.has(7)) return { name, kind: "sequence", dt: "sequence" };
  if (t.has(8)) {
    const a = t.msg(8)?.msg(1);
    return { name, kind: "state", shape: a?.nums(1), dt: ARRAY_DT[a?.num(2) ?? 0] ?? "float32" };
  }
  return { name, kind: "unknown", dt: "-" };
}

type Desc = { inputs: Feature[]; outputs: Feature[]; predicted: string; probs: string; meta: Record<string, string>; short: string };

function readDesc(d: Fields | undefined): Desc {
  const meta: Record<string, string> = {};
  const md = d?.msg(100);
  // map<string, string> entries: key = 1, value = 2.
  for (const e of md?.msgs(100) ?? []) meta[e.str(1)] = e.str(2);
  return {
    inputs: (d?.msgs(1) ?? []).map(readFeature),
    outputs: (d?.msgs(10) ?? []).map(readFeature),
    predicted: d?.str(11) ?? "",
    probs: d?.str(12) ?? "",
    meta,
    short: md?.str(1) ?? "",
  };
}

// ── Build context ────────────────────────────────────────────────────────────

type Ctx = {
  layers: ModelLayer[];
  edges: EdgeSet;
  producer: Map<string, number>;
  shapes: Map<string, Shape | undefined>;
  weightFiles: Map<string, Uint8Array>; // basename → bytes
  kinds: string[];
};

function addLayer(ctx: Ctx, l: Omit<Parameters<typeof makeLayer>[0], "id" | "framework">, inputs: string[], outputs: string[]): ModelLayer {
  const id = ctx.layers.length;
  for (const n of inputs) ctx.edges.add(ctx.producer.get(n), id);
  for (const n of outputs) ctx.producer.set(n, id);
  const layer = makeLayer({ ...l, id, framework: "Core ML" });
  ctx.layers.push(layer);
  return layer;
}

function addInputs(ctx: Ctx, desc: Desc): void {
  for (const f of desc.inputs) {
    if (ctx.producer.has(f.name)) continue;
    ctx.shapes.set(f.name, f.shape);
    addLayer(ctx, {
      name: f.name, type: "Input", canon: "Input", shape: f.shape, dt: f.dt, group: "input",
      outs: [{ n: f.name, s: f.shape }],
      attr: { kind: f.kind, ...(f.detail ? { colorSpace: f.detail } : {}) },
      insight: f.kind === "image"
        ? `Image input (${f.detail ?? "?"}). Core ML applies the model's image preprocessing (scale/bias) before the first layer.`
        : `Model input (${f.kind}).`,
    }, [], [f.name]);
  }
}

function markOutputs(ctx: Ctx, desc: Desc): void {
  for (const f of desc.outputs) {
    const id = ctx.producer.get(f.name);
    if (id == null) continue;
    const l = ctx.layers[id];
    if (l.group !== "input") l.group = "output";
    const o = l.outs.find((x) => x.n === f.name);
    if (o && !o.s && f.shape) o.s = f.shape;
  }
}

// ── Weights ──────────────────────────────────────────────────────────────────

type W = { count: number; dt: string; bytes?: Uint8Array };

function readWeight(f: Fields | undefined): W | null {
  if (!f) return null;
  const fl = f.m.get(1);
  if (fl?.length) {
    const b = fl[0].v;
    if (typeof b !== "number" && fl[0].wt === WT_LEN) return { count: b.byteLength >> 2, dt: "float32", bytes: b };
    return { count: fl.length, dt: "float32" };
  }
  const h = f.bytes(2);
  if (h?.byteLength) return { count: h.byteLength >> 1, dt: "float16", bytes: h };
  const i8 = f.bytes(31);
  if (i8?.byteLength) return { count: i8.byteLength, dt: "int8", bytes: i8 };
  const raw = f.bytes(30);
  if (raw?.byteLength) {
    const nbits = f.msg(40)?.num(1, 0) ?? 0;
    if (nbits > 0) return { count: Math.floor((raw.byteLength * 8) / nbits), dt: `quant${nbits}` };
    return { count: raw.byteLength >> 2, dt: "float32", bytes: raw };
  }
  return null;
}

function wStat(shape: Shape, w: W): WStat {
  const s = numel(shape) === w.count ? shape : [w.count];
  return weightStats(s, w.bytes ? sampleBytes(w.bytes, w.dt) : null);
}

// ── NeuralNetwork ────────────────────────────────────────────────────────────

const NN_CANON: Record<string, string> = {
  innerProduct: "Gemm", batchedMatmul: "MatMul", batchnorm: "BatchNormalization", softmax: "Softmax",
  softmaxND: "Softmax", concat: "Concat", concatND: "Concat", flatten: "Flatten", flattenTo2D: "Flatten",
  reshape: "Reshape", reshapeStatic: "Reshape", reshapeDynamic: "Reshape", reshapeLike: "Reshape",
  rankPreservingReshape: "Reshape", squeeze: "Reshape", expandDims: "Reshape", transpose: "Transpose",
  permute: "Transpose", add: "Add", addBroadcastable: "Add", multiply: "Mul", multiplyBroadcastable: "Mul",
  subtractBroadcastable: "Sub", divideBroadcastable: "Div", layerNormalization: "LayerNormalization",
  gelu: "Gelu", erf: "Erf", tanh: "Tanh", scale: "Mul", bias: "Add", max: "Max", min: "Min",
  maxBroadcastable: "Max", minBroadcastable: "Min", powBroadcastable: "Pow", clip: "Clip", mvn: "LayerNormalization",
};

function padAmounts(f: Fields | undefined): [number, number, number, number] {
  // ValidPadding.paddingAmounts(1) → BorderAmounts.borderAmounts(10)[h, w] → EdgeSizes start(1)/end(2)
  const edges = f?.msg(1)?.msgs(10) ?? [];
  return [edges[0]?.num(1) ?? 0, edges[0]?.num(2) ?? 0, edges[1]?.num(1) ?? 0, edges[1]?.num(2) ?? 0];
}

function spatial(i: number, k: number, s: number, p0: number, p1: number, d: number, mode: "valid" | "same" | "ceil"): number {
  if (i < 0) return DYN;
  if (mode === "same") return Math.ceil(i / s);
  const eff = d * (k - 1) + 1;
  return mode === "ceil" ? Math.ceil((i + p0 + p1 - eff) / s) + 1 : Math.floor((i + p0 + p1 - eff) / s) + 1;
}

function addNeuralNetwork(ctx: Ctx, nn: Fields, kind: number, desc: Desc): void {
  const isExact = nn.num(5, 0) === 1; // EXACT_ARRAY_MAPPING: ND shapes
  for (const lf of nn.msgs(1)) {
    const name = lf.str(1);
    const inputs = lf.strs(2);
    const outputs = lf.strs(3);
    const tf = lf.which(NN_LAYER_FIELDS);
    const kindName = tf != null ? NN_LAYERS[tf] : "unknown";
    const p = tf != null ? lf.msg(tf) ?? Fields.of(new Uint8Array(0)) : Fields.of(new Uint8Array(0));
    const ins = inputs.map((n) => ctx.shapes.get(n));
    const x = ins[0];
    let out: Shape | undefined;
    let cost: Cost = NO_COST;
    let canon = NN_CANON[kindName] ?? kindName;
    let type = cap(kindName);
    const attr: Record<string, unknown> = {};

    // Explicit output ranks/dims (newer specs).
    const ot = lf.msgs(5)[0];
    if (ot && ot.nums(2).length) out = ot.nums(2).map((d) => (d > 0 ? d : DYN));

    // Weights (params + stats on the biggest).
    let params = 0;
    let big: W | null = null;
    let bigShape: Shape = [];
    const weightShape = (fieldNo: number): Shape => {
      switch (kindName) {
        case "convolution": {
          const ks = p.nums(20);
          return fieldNo === 90
            ? (p.bool(60) ? [p.num(2), p.num(1) / Math.max(1, p.num(10, 1)), ks[0] ?? 3, ks[1] ?? 3] : [p.num(1), p.num(2), ks[0] ?? 3, ks[1] ?? 3])
            : [p.num(1)];
        }
        case "innerProduct": return fieldNo === 20 ? [p.num(2), p.num(1)] : [p.num(2)];
        case "embedding": return fieldNo === 20 ? [p.num(2), p.num(1)] : [p.num(2)];
        case "batchedMatmul": return fieldNo === 8 ? [p.num(5), p.num(6)] : [p.num(6)];
        case "embeddingND": return fieldNo === 20 ? [p.num(2), p.num(1)] : [p.num(2)];
        case "batchnorm": return [p.num(1)];
        case "scale": return fieldNo === 2 ? p.nums(1) : p.nums(4);
        case "bias": case "loadConstant": case "loadConstantND": return p.nums(1);
        default: return [];
      }
    };
    const wfields = NN_WEIGHT_FIELDS[kindName] ?? [];
    for (const wf of wfields) {
      const w = readWeight(p.msg(wf));
      if (!w) continue;
      params += w.count;
      if (!big || w.count > big.count) { big = w; bigShape = weightShape(wf); }
    }
    if (kindName === "activation") {
      const at = p.which(Object.keys(ACTIVATIONS).map(Number));
      type = at != null ? ACTIVATIONS[at] : "Activation";
      canon = { ReLU: "Relu", LeakyReLU: "LeakyRelu", PReLU: "LeakyRelu", Tanh: "Tanh", Sigmoid: "Sigmoid", ELU: "Elu" }[type] ?? type;
      if (at === 25) {
        const w = readWeight(p.msg(25)?.msg(1));
        if (w) { params += w.count; big = w; bigShape = [w.count]; }
      }
      if (at === 15) attr.alpha = p.msg(15)?.float(1);
    }

    // Shapes + cost.
    const rank = x?.length ?? 0;
    switch (kindName) {
      case "convolution": {
        const cout = p.num(1);
        const kch = p.num(2);
        const groups = Math.max(1, p.num(10, 1));
        const ks = p.nums(20); const st = p.nums(30); const dl = p.nums(40);
        const [kh, kw] = [ks[0] ?? 3, ks[1] ?? 3];
        const [sh, sw] = [st[0] ?? 1, st[1] ?? 1];
        const [dh, dw] = [dl[0] ?? 1, dl[1] ?? 1];
        const deconv = p.bool(60);
        const same = p.has(51);
        const [pt, pb, pl, pr] = padAmounts(p.msg(50));
        Object.assign(attr, { outputChannels: cout, kernelChannels: kch, groups, kernelSize: [kh, kw], stride: [sh, sw], dilation: [dh, dw], padding: same ? "same" : [pt, pb, pl, pr], hasBias: p.bool(70) });
        if (deconv) {
          canon = "ConvTranspose";
          type = "Deconvolution";
          const os = p.nums(100);
          if (!out && x && rank >= 3) {
            const H = x[rank - 2]; const Wd = x[rank - 1];
            out = os.length >= 2 ? [...x.slice(0, rank - 3), cout, os[os.length - 2], os[os.length - 1]]
              : [...x.slice(0, rank - 3), cout, H < 0 ? DYN : sh * (H - 1) + dh * (kh - 1) + 1 - pt - pb, Wd < 0 ? DYN : sw * (Wd - 1) + dw * (kw - 1) + 1 - pl - pr];
          }
          if (x) cost = linearCost(x, (cout / groups) * kh * kw);
        } else {
          canon = "Conv";
          if (!out && x && rank >= 3) {
            const mode = same ? "same" : "valid";
            out = [...x.slice(0, rank - 3), cout, spatial(x[rank - 2], kh, sh, pt, pb, dh, mode), spatial(x[rank - 1], kw, sw, pl, pr, dw, mode)];
          }
          cost = linearCost(out, kch * kh * kw);
        }
        break;
      }
      case "pooling": {
        const pt = ["MAX", "AVERAGE", "L2"][p.num(1)] ?? "MAX";
        const global = p.bool(60);
        const ks = p.nums(10); const st = p.nums(20);
        const [kh, kw] = [ks[0] ?? 1, ks[1] ?? 1];
        const [sh, sw] = [st[0] ?? 1, st[1] ?? 1];
        canon = global ? (pt === "MAX" ? "GlobalMaxPool" : "GlobalAveragePool") : pt === "MAX" ? "MaxPool" : "AveragePool";
        Object.assign(attr, { type: pt, globalPooling: global, kernelSize: [kh, kw], stride: [sh, sw] });
        if (!out && x && rank >= 3) {
          if (global) out = [...x.slice(0, rank - 2), 1, 1];
          else if (p.has(31)) out = [...x.slice(0, rank - 2), spatial(x[rank - 2], kh, sh, 0, 0, 1, "same"), spatial(x[rank - 1], kw, sw, 0, 0, 1, "same")];
          else if (p.has(32)) {
            const pa = p.msg(32)!.nums(10);
            out = [...x.slice(0, rank - 2), spatial(x[rank - 2], kh, sh, pa[0] ?? 0, pa[0] ?? 0, 1, "ceil"), spatial(x[rank - 1], kw, sw, pa[1] ?? 0, pa[1] ?? 0, 1, "ceil")];
          } else {
            const [a, b, c, d] = padAmounts(p.msg(30));
            out = [...x.slice(0, rank - 2), spatial(x[rank - 2], kh, sh, a, b, 1, "valid"), spatial(x[rank - 1], kw, sw, c, d, 1, "valid")];
          }
        }
        cost = global ? elemCost(x, 1) : elemCost(out, kh * kw);
        break;
      }
      case "innerProduct": {
        const cin = p.num(1); const cout = p.num(2);
        Object.assign(attr, { inputChannels: cin, outputChannels: cout, hasBias: p.bool(10) });
        if (!out && x) {
          // Rank-5 legacy layout puts features in C with H=W=1; ND layout uses the last axis.
          out = !isExact && rank >= 3 && x[rank - 1] === 1 && x[rank - 2] === 1
            ? [...x.slice(0, rank - 3), cout, 1, 1] : [...x.slice(0, -1), cout];
        }
        cost = linearCost(out, cin);
        break;
      }
      case "batchedMatmul": {
        const ta = p.bool(1);
        const K = p.num(5) || (x ? (ta ? x[rank - 2] : x[rank - 1]) : 0);
        if (!out && x && ins[1]) out = broadcast(x.slice(0, -1), ins[1].slice(0, -2)).concat(ins[1][ins[1].length - 1]);
        else if (!out && x && p.num(6)) out = [...x.slice(0, -1), p.num(6)];
        cost = linearCost(out, K);
        break;
      }
      case "embedding": case "embeddingND":
        if (!out && x) out = [...x.slice(0, -1), p.num(2)];
        break;
      case "reshapeStatic": case "reshape": {
        const t = p.nums(1);
        if (!out && t.length) out = t.map((d) => (d > 0 ? d : DYN));
        attr.targetShape = t;
        break;
      }
      case "flattenTo2D":
        if (!out && x) { const a = p.num(1, 1); const ax = a < 0 ? rank + a : a; out = [numel(x.slice(0, ax)), numel(x.slice(ax))]; }
        break;
      case "flatten":
        if (!out && x && rank >= 3) out = [...x.slice(0, rank - 3), numel(x.slice(rank - 3)), 1, 1];
        break;
      case "concat": case "concatND": {
        const valid = ins.filter((s): s is Shape => !!s);
        if (!out && valid.length === ins.length && valid.length) {
          const r0 = valid[0].length;
          const ax = kindName === "concat" ? r0 - 3 : (p.num(1) < 0 ? r0 + p.num(1) : p.num(1));
          out = [...valid[0]];
          out[ax] = valid.some((s) => s[ax] < 0) ? DYN : valid.reduce((a, s) => a + s[ax], 0);
        }
        break;
      }
      case "transpose": {
        const axes = p.nums(1);
        if (!out && x && axes.length === rank) out = axes.map((a) => x[a]);
        break;
      }
      case "upsample": {
        const sf = p.nums(1);
        if (!out && x && rank >= 2) out = [...x.slice(0, rank - 2), x[rank - 2] * (sf[0] ?? 1), x[rank - 1] * (sf[1] ?? 1)];
        break;
      }
      case "add": case "multiply": case "addBroadcastable": case "multiplyBroadcastable": case "subtractBroadcastable":
      case "divideBroadcastable": case "max": case "min": case "maxBroadcastable": case "minBroadcastable":
      case "average": case "powBroadcastable": {
        const valid = ins.filter((s): s is Shape => !!s);
        if (!out && valid.length) out = valid.reduce((a, b) => broadcast(a, b));
        cost = elemCost(out, Math.max(1, ins.length - 1));
        break;
      }
      default:
        if (!out && x && /^(activation|batchnorm|softmax|softmaxND|unary|scale|bias|lrn|mvn|l2normalize|clip|gelu|erf|tanh|copy|layerNormalization|sin|cos|exp2|floor|ceil|round|sign|clampedReLU)$/.test(kindName)) out = [...x];
    }
    if (cost === NO_COST && out) {
      if (/^(activation|unary|scale|bias|clip|gelu|erf|tanh|clampedReLU)$/.test(kindName)) cost = elemCost(out, 1);
      else if (/^(batchnorm|layerNormalization|mvn|l2normalize)$/.test(kindName)) cost = elemCost(out, 2);
      else if (/^softmax/.test(kindName)) cost = elemCost(out, 3);
    }
    outputs.forEach((o, i) => ctx.shapes.set(o, i === 0 ? out : undefined));

    addLayer(ctx, {
      name: name || outputs[0] || `${type}_${ctx.layers.length}`,
      type, canon, shape: out, params, cost,
      group: canon === "Gemm" || canon === "MatMul" ? "head" : undefined,
      ins: inputs.map((n, i) => ({ n, s: ins[i] })),
      outs: outputs.map((n, i) => ({ n, s: i === 0 ? out : undefined })),
      attr, w: big ? wStat(bigShape, big) : null,
      insight: nnInsight(kindName, type),
    }, inputs, outputs);
  }

  // Classifier head: Core ML maps the probability tensor to class labels.
  if (kind === 403) {
    const labels = nn.msg(100)?.strs(1) ?? [];
    const intLabels = nn.msg(101)?.nums(1) ?? [];
    const probLayer = nn.str(200);
    const lastOut = ctx.layers[ctx.layers.length - 1]?.outs[0]?.n;
    const src = probLayer || lastOut || "";
    const outs = [desc.predicted || "classLabel"];
    const nClasses = labels.length || intLabels.length;
    addLayer(ctx, {
      name: "classifier", type: "Classifier", canon: "ArgMax", group: "output", shape: [1], dt: labels.length ? "string" : "int64",
      ins: src ? [{ n: src, s: ctx.shapes.get(src) }] : [],
      outs: outs.map((n) => ({ n, s: [1] })),
      attr: { classes: nClasses, labels: (labels.length ? labels : intLabels).slice(0, 20) },
      insight: `Core ML classifier: maps ${nClasses} class probabilities to the predicted label.`,
    }, src ? [src] : [], outs);
  }
}

function nnInsight(kind: string, type: string): string | undefined {
  switch (kind) {
    case "convolution": return type === "Deconvolution" ? "Transposed convolution (upsampling)." : "Convolution. Usually the FLOP hotspot; the Neural Engine prefers channel counts that are multiples of 16.";
    case "innerProduct": return "Dense layer (InnerProduct). Largest weight tensors; the classifier head.";
    case "batchnorm": return "Batch normalization. Core ML fuses it into the preceding convolution when compiling.";
    case "custom": return "Custom layer: needs an app-side implementation; will not run on the Neural Engine.";
    case "uniDirectionalLSTM": case "biDirectionalLSTM": case "gru": case "simpleRecurrent":
      return "Recurrent layer. Runs on CPU/GPU only on most devices; parameters are not counted here.";
  }
  return undefined;
}

// ── ML Program (MIL) ─────────────────────────────────────────────────────────

type ConstInfo = { dt: string; shape: Shape; small?: unknown; bytes?: Uint8Array; blob?: { file: string; offset: number } };

function tensorType(vt: Fields | undefined): { dt: string; shape?: Shape } {
  const tt = vt?.msg(1);
  if (!tt) return { dt: vt?.has(2) ? "list" : vt?.has(3) ? "tuple" : vt?.has(4) ? "dict" : vt?.has(5) ? "state" : "-" };
  const dims = tt.msgs(3).map((d) => (d.has(1) ? d.msg(1)!.num(1) : DYN));
  return { dt: MIL_DT[tt.num(1)] ?? "float32", shape: dims };
}

function readValue(v: Fields | undefined, info: ConstInfo): void {
  if (!v) return;
  const blob = v.msg(5);
  if (blob) { info.blob = { file: blob.str(1), offset: blob.num(2) }; return; }
  const tv = v.msg(3)?.msg(1);
  if (!tv) return;
  const n = numel(info.shape);
  if (tv.has(1)) { // floats (packed)
    const b = tv.msg(1)?.bytes(1);
    if (b) { info.bytes = b; if (n <= 16) info.small = Array.from({ length: b.byteLength >> 2 }, (_, i) => new DataView(b.buffer, b.byteOffset + i * 4, 4).getFloat32(0, true)); }
  } else if (tv.has(2)) { const xs = tv.msg(2)!.nums(1); if (xs.length <= 16) info.small = info.shape.length ? xs : xs[0]; }
  else if (tv.has(3)) { const xs = tv.msg(3)!.nums(1).map((b) => b !== 0); if (xs.length <= 16) info.small = info.shape.length ? xs : xs[0]; }
  else if (tv.has(4)) { const xs = tv.msg(4)!.strs(1); if (xs.length <= 16) info.small = info.shape.length ? xs : xs[0]; }
  else if (tv.has(5)) { const xs = tv.msg(5)!.nums(1); if (xs.length <= 16) info.small = info.shape.length ? xs : xs[0]; }
  else if (tv.has(7)) info.bytes = tv.msg(7)?.bytes(1);
}

function readBlobData(ctx: Ctx, blob: { file: string; offset: number }): { dt: string; data: Uint8Array } | null {
  const base = blob.file.split(/[\\/]/).pop() ?? ""; // coremltools on Windows writes backslashes
  const file = ctx.weightFiles.get(base);
  if (!file || blob.offset + 64 > file.byteLength) return null;
  const v = new DataView(file.buffer, file.byteOffset, file.byteLength);
  if (v.getUint32(blob.offset, true) !== 0xdeadbeef) return null;
  const dt = BLOB_DT[v.getUint32(blob.offset + 4, true)];
  const size = Number(v.getBigUint64(blob.offset + 8, true));
  const off = Number(v.getBigUint64(blob.offset + 16, true));
  if (!dt || off + size > file.byteLength) return null;
  return { dt, data: file.subarray(off, off + size) };
}

const MIL_CANON: Record<string, string> = {
  conv: "Conv", conv_transpose: "ConvTranspose", linear: "Gemm", matmul: "MatMul", einsum: "MatMul",
  add: "Add", sub: "Sub", mul: "Mul", real_div: "Div", pow: "Pow", maximum: "Max", minimum: "Min",
  relu: "Relu", relu6: "Clip", clip: "Clip", leaky_relu: "LeakyRelu", prelu: "LeakyRelu", sigmoid: "Sigmoid",
  tanh: "Tanh", gelu: "Gelu", erf: "Erf", elu: "Elu", softmax: "Softmax", max_pool: "MaxPool", avg_pool: "AveragePool",
  batch_norm: "BatchNormalization", instance_norm: "InstanceNormalization", layer_norm: "LayerNormalization",
  reshape: "Reshape", squeeze: "Reshape", expand_dims: "Reshape", transpose: "Transpose", concat: "Concat",
  flatten2d: "Flatten",
};

function milCost(type: string, out: Shape | undefined, x: Shape | undefined, w: Shape | undefined, consts: Record<string, unknown>): Cost {
  const area = (s: Shape) => s.reduce((a, b) => a * Math.max(1, b), 1);
  switch (type) {
    case "conv": return w && w.length >= 3 ? linearCost(out, Math.max(1, w[1]) * area(w.slice(2))) : NO_COST;
    case "conv_transpose": return w && x && w.length >= 3 ? linearCost(x, Math.max(1, w[1]) * area(w.slice(2))) : NO_COST;
    case "linear": return w && w.length === 2 ? linearCost(out, w[1]) : x ? linearCost(out, x[x.length - 1]) : NO_COST;
    case "matmul": {
      if (!x || !x.length) return NO_COST;
      const tx = consts.transpose_x === true;
      return linearCost(out, Math.max(1, x.length === 1 ? x[0] : tx ? x[x.length - 2] : x[x.length - 1]));
    }
    case "max_pool": case "avg_pool": case "l2_pool": {
      const k = consts.kernel_sizes;
      return elemCost(out, Array.isArray(k) ? area(k as number[]) : 1);
    }
    case "reduce_mean": case "reduce_sum": case "reduce_max": return elemCost(x, 1);
    case "batch_norm": case "instance_norm": case "layer_norm": return elemCost(out, 2);
    case "softmax": return elemCost(out, 3);
  }
  if (/^(add|sub|mul|real_div|pow|maximum|minimum|relu|relu6|clip|leaky_relu|prelu|sigmoid|tanh|gelu|erf|elu|silu|exp|sqrt|rsqrt|abs|square|sigmoid_hard|softplus)$/.test(type)) return elemCost(out, 1);
  return NO_COST;
}

function addMlProgram(ctx: Ctx, prog: Fields, desc: Desc): void {
  const fns = prog.map(2);
  if (!fns.length) return;
  const [, fn] = fns.find(([k]) => k === "main") ?? fns[0];
  const opset = fn.str(2);
  const blocks = fn.map(3);
  const block = (blocks.find(([k]) => k === opset) ?? blocks[0])?.[1];
  if (!block) return;

  // Function inputs (if the description didn't already declare them).
  for (const nv of fn.msgs(1)) {
    const name = nv.str(1);
    const t = tensorType(nv.msg(2));
    if (!ctx.shapes.get(name) && t.shape) ctx.shapes.set(name, t.shape);
    if (!ctx.producer.has(name)) {
      addLayer(ctx, { name, type: "Input", canon: "Input", shape: t.shape, dt: t.dt, group: "input", outs: [{ n: name, s: t.shape }], insight: "Program input." }, [], [name]);
    }
  }

  const consts = new Map<string, ConstInfo>();
  const claimed = new Set<string>();
  for (const op of block.msgs(3)) {
    const type = op.str(1);
    const outs = op.msgs(3).map((nv) => ({ name: nv.str(1), ...tensorType(nv.msg(2)) }));
    const attrs = new Map(op.map(5));

    if (type === "const" || type.startsWith("constexpr_")) {
      for (const o of outs) {
        const info: ConstInfo = { dt: o.dt, shape: o.shape ?? [] };
        if (type === "const") readValue(attrs.get("val"), info);
        consts.set(o.name, info);
      }
      continue;
    }

    const inputs = op.map(2); // name → Argument
    const dataIns: { key: string; n: string }[] = [];
    const constArgs: Record<string, unknown> = {};
    let params = 0;
    let wInfo: ConstInfo | null = null;
    const weightShapes: Record<string, Shape> = {};
    for (const [key, arg] of inputs) {
      for (const b of arg.msgs(1)) {
        if (b.has(1)) {
          const n = b.str(1);
          const c = consts.get(n);
          if (c) {
            if (c.small !== undefined) constArgs[key] = c.small;
            if (!claimed.has(n) && PARAM_DT.test(c.dt) && c.shape.length > 0) {
              claimed.add(n);
              const cnt = numel(c.shape);
              params += cnt;
              weightShapes[key] = c.shape;
              if (!wInfo || cnt > numel(wInfo.shape)) wInfo = c;
            }
          } else dataIns.push({ key, n });
        } else if (b.has(2)) {
          const info: ConstInfo = { dt: "-", shape: [] };
          const v = b.msg(2)!;
          Object.assign(info, tensorType(v.msg(2)));
          info.shape = info.shape ?? [];
          readValue(v, info);
          if (info.small !== undefined) constArgs[key] = info.small;
        }
      }
    }

    const x = dataIns.find((d) => d.key === "x")?.n ?? dataIns[0]?.n;
    const xShape = x ? ctx.shapes.get(x) : undefined;
    const out = outs[0]?.shape;
    for (const o of outs) ctx.shapes.set(o.name, o.shape);

    let canon = MIL_CANON[type] ?? type;
    if (/^reduce_(mean|max)$/.test(type) && out && out.length === 4 && out[2] === 1 && out[3] === 1) {
      canon = type === "reduce_mean" ? "GlobalAveragePool" : "GlobalMaxPool";
    }
    if (type === "matmul" && weightShapes.y) canon = "Gemm";

    let w: WStat | null = null;
    if (wInfo) {
      let values: number[] | null = null;
      if (wInfo.blob) {
        const bd = readBlobData(ctx, wInfo.blob);
        if (bd) values = sampleBytes(bd.data, bd.dt);
      } else if (wInfo.bytes) values = sampleBytes(wInfo.bytes, wInfo.dt === "float16" && wInfo.bytes.byteLength === numel(wInfo.shape) * 4 ? "float32" : wInfo.dt);
      w = weightStats(wInfo.shape, values);
    }

    const nameInfo: ConstInfo = { dt: "string", shape: [] };
    readValue(attrs.get("name"), nameInfo);
    const name = (typeof nameInfo.small === "string" ? nameInfo.small : "") || outs[0]?.name || `${type}_${ctx.layers.length}`;
    const nested = op.msgs(4).length;

    addLayer(ctx, {
      name, type, canon, shape: out, dt: outs[0]?.dt,
      params, cost: milCost(type, out, xShape, weightShapes.weight ?? weightShapes.y, constArgs),
      group: canon === "Gemm" || canon === "MatMul" ? "head" : undefined,
      ins: dataIns.map((d) => ({ n: d.n, s: ctx.shapes.get(d.n) })),
      outs: outs.map((o) => ({ n: o.name, s: o.shape })),
      attr: { ...constArgs, ...(Object.keys(weightShapes).length ? { weights: Object.fromEntries(Object.entries(weightShapes).map(([k, s]) => [k, s.join("×")])) } : {}), ...(nested ? { blocks: nested } : {}) },
      w,
      insight: milInsight(type, nested),
    }, dataIns.map((d) => d.n), outs.map((o) => o.name));
  }

  // Block outputs are the program outputs.
  for (const n of block.strs(2)) {
    const id = ctx.producer.get(n);
    if (id != null && ctx.layers[id].group !== "input") ctx.layers[id].group = "output";
  }
  void desc;
}

function milInsight(type: string, nested: number): string | undefined {
  if (nested) return `Control flow (${type}) with ${nested} nested block(s); inner ops are not expanded.`;
  switch (type) {
    case "cast": return "Precision cast. ML Programs default to fp16 compute; casts at the boundaries convert fp32 I/O.";
    case "conv": return "Convolution. Usually the FLOP hotspot; the Neural Engine prefers channel counts that are multiples of 16.";
    case "linear": return "Dense layer (linear). Largest weight tensors; the classifier head.";
    case "classify": return "Core ML classifier: maps class probabilities to the predicted label.";
  }
  return undefined;
}

// ── Opaque (non-graph) model kinds ───────────────────────────────────────────

function addOpaque(ctx: Ctx, kind: string, desc: Desc): void {
  const ins = desc.inputs.map((f) => f.name);
  const outs = desc.outputs.map((f) => f.name);
  addLayer(ctx, {
    name: kind, type: cap(kind), canon: cap(kind), shape: desc.outputs[0]?.shape, group: "head",
    ins: desc.inputs.map((f) => ({ n: f.name, s: f.shape })),
    outs: desc.outputs.map((f) => ({ n: f.name, s: f.shape })),
    insight: `Core ML ${kind} model (not a layer graph); shown as a single node.`,
  }, ins, outs);
}

// ── Model (recursive for pipelines) ──────────────────────────────────────────

function addModel(ctx: Ctx, m: Fields, depth = 0): { spec: number; desc: Desc } {
  const spec = m.num(1);
  const desc = readDesc(m.msg(2));
  addInputs(ctx, desc);
  const kind = m.which(KIND_FIELDS);
  const body = kind != null ? m.msg(kind) : undefined;
  const kindName = kind != null ? MODEL_KINDS[kind] : "unknown";
  ctx.kinds.push(kindName);
  if (kind === 500 || kind === 403 || kind === 303) addNeuralNetwork(ctx, body!, kind, desc);
  else if (kind === 502) addMlProgram(ctx, body!, desc);
  else if ((kind === 200 || kind === 201 || kind === 202) && depth < 16) {
    const pipe = kind === 202 ? body : body?.msg(1);
    for (const sub of pipe?.msgs(1) ?? []) addModel(ctx, sub, depth + 1);
  } else addOpaque(ctx, kindName, desc);
  markOutputs(ctx, desc);
  return { spec, desc };
}

// ── Public API ───────────────────────────────────────────────────────────────

/** Heuristic: is this a Core ML Model protobuf (specificationVersion + description first)? */
export function isCoremlModel(buf: Uint8Array): boolean {
  // Field 1 varint (specificationVersion 1..20) then field 2 length-delimited.
  return buf.length > 4 && buf[0] === 0x08 && buf[1] > 0 && buf[1] < 32 && buf[2] === 0x12;
}

/** Is this a zip of an .mlpackage directory (contains Data/com.apple.CoreML/…)? */
export function isCoremlPackageZip(buf: Uint8Array): boolean {
  if (!isZip(buf)) return false;
  try {
    return readZipEntries(buf).some((e) => /(^|\/)Data\/com\.apple\.CoreML\//.test(e.name));
  } catch {
    return false;
  }
}

function unzipEntry(buf: Uint8Array, e: ZipEntry): Uint8Array | null {
  if (e.method === 0) return extractStored(buf, e);
  if (e.method === 8 && e.compressedSize < 256 * 1024 * 1024) {
    return inflateRaw(buf.subarray(e.dataOffset, e.dataOffset + e.compressedSize), e.size);
  }
  return null;
}

/**
 * Parse a Core ML model: a .mlmodel protobuf, or a zipped .mlpackage. Pass
 * `weights` (weight.bin of an unzipped .mlpackage) to get ML Program weight
 * statistics. Unparseable input falls back to detected-format metadata.
 */
export function parseCoreml(
  input: ArrayBuffer | Uint8Array,
  name = "model.mlmodel",
  weights?: ArrayBuffer | Uint8Array,
): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const weightFiles = new Map<string, Uint8Array>();
  if (weights) weightFiles.set("weight.bin", weights instanceof Uint8Array ? weights : new Uint8Array(weights));

  let spec = buf;
  let format = "Core ML";
  if (isZip(buf)) {
    let entries: ZipEntry[];
    try { entries = readZipEntries(buf); } catch { return parseContainer(buf, name); }
    const modelEntry = entries.find((e) => /(^|\/)Data\/com\.apple\.CoreML\/model\.mlmodel$/.test(e.name))
      ?? entries.find((e) => /\.mlmodel$/.test(e.name));
    const bytes = modelEntry ? unzipEntry(buf, modelEntry) : null;
    if (!bytes) return parseContainer(buf, name);
    spec = bytes;
    format = "Core ML package";
    for (const e of entries) {
      if (/\/weights\/[^/]+$/.test(e.name) && !e.name.endsWith("/")) {
        const b = unzipEntry(buf, e);
        if (b) weightFiles.set(e.name.split("/").pop()!, b);
      }
    }
  }

  const ctx: Ctx = { layers: [], edges: new EdgeSet(), producer: new Map(), shapes: new Map(), weightFiles, kinds: [] };
  let res: { spec: number; desc: Desc };
  try {
    res = addModel(ctx, Fields.of(spec));
  } catch {
    return parseContainer(buf, name);
  }
  if (!ctx.layers.length || !res.spec) return parseContainer(buf, name);

  const { desc } = res;
  const firstIn = desc.inputs[0];
  const firstOut = desc.outputs[0];
  const outShape = firstOut?.shape ?? (firstOut ? ctx.shapes.get(firstOut.name) : undefined);
  const ctVersion = desc.meta["com.github.apple.coremltools.version"];
  const source = desc.meta["com.github.apple.coremltools.source"];
  const kind = ctx.kinds[0];
  return {
    name,
    format,
    framework: kind === "mlProgram" ? "Core ML (ML Program)" : /^neuralNetwork/.test(kind) ? "Core ML (NeuralNetwork)" : `Core ML (${kind})`,
    opset: res.spec,
    sizeBytes: buf.byteLength,
    inputShape: cleanShape(firstIn?.shape),
    outputShape: cleanShape(outShape),
    layers: ctx.layers,
    edges: ctx.edges.edges,
    producer: ctVersion ? `coremltools ${ctVersion}${source ? ` (from ${source})` : ""}` : undefined,
    doc: desc.short || undefined,
  };
}
