import type { Model, ModelLayer } from "../types";
import { ProtoReader, WT_I32, WT_LEN, WT_VARINT } from "../common/protobufWire";
import { parseProtoText, ptBool, ptMsg, ptMsgs, ptNum, ptNums, ptStr, ptStrs, type PTMsg, type PTValue } from "../common/protoText";
import {
  DYN, EdgeSet, NO_COST, cleanShape, elemCost, linearCost, makeLayer, numel, sampleBytes, weightStats,
  type Cost, type Shape, type WStat,
} from "../common/graphKit";

// Caffe: .prototxt (protobuf text NetParameter, the deploy/train_val graph) and
// .caffemodel (binary NetParameter with trained blobs). Both are normalized
// into the same intermediate net: layers with bottom/top blob names, their
// *_param messages and (binary only) weight blobs. Edges follow blob names
// (in-place layers, where top == bottom, chain naturally); shapes are
// propagated from the declared input through conv / pool / inner-product /
// concat / eltwise / reshape. Field numbers come from BVLC caffe.proto.

type Blob = { shape: Shape; data?: Uint8Array; values?: number[] };
type CLayer = {
  name: string;
  type: string;
  bottom: string[];
  top: string[];
  phase?: string;
  p: Record<string, PTMsg>; // "convolution_param" → message
  blobs: Blob[];
};
type CNet = { name: string; inputs: { name: string; shape?: Shape }[]; layers: CLayer[] };

// ── V1 (legacy `layers {}`) type enum → modern type string ───────────────────
const V1_TYPES: Record<number, string> = {
  0: "None", 35: "AbsVal", 1: "Accuracy", 30: "ArgMax", 2: "BNLL", 3: "Concat", 37: "ContrastiveLoss",
  4: "Convolution", 5: "Data", 39: "Deconvolution", 6: "Dropout", 32: "DummyData", 7: "EuclideanLoss",
  25: "Eltwise", 38: "Exp", 8: "Flatten", 9: "HDF5Data", 10: "HDF5Output", 28: "HingeLoss", 11: "Im2col",
  12: "ImageData", 13: "InfogainLoss", 14: "InnerProduct", 15: "LRN", 29: "MemoryData",
  16: "MultinomialLogisticLoss", 34: "MVN", 17: "Pooling", 26: "Power", 18: "ReLU", 19: "Sigmoid",
  27: "SigmoidCrossEntropyLoss", 36: "Silence", 20: "Softmax", 21: "SoftmaxWithLoss", 22: "Split",
  33: "Slice", 23: "TanH", 24: "WindowData", 31: "Threshold",
};
const V1_BY_ID: Record<string, string> = {
  NONE: "None", ABSVAL: "AbsVal", ACCURACY: "Accuracy", ARGMAX: "ArgMax", BNLL: "BNLL", CONCAT: "Concat",
  CONTRASTIVE_LOSS: "ContrastiveLoss", CONVOLUTION: "Convolution", DATA: "Data", DECONVOLUTION: "Deconvolution",
  DROPOUT: "Dropout", DUMMY_DATA: "DummyData", EUCLIDEAN_LOSS: "EuclideanLoss", ELTWISE: "Eltwise", EXP: "Exp",
  FLATTEN: "Flatten", HDF5_DATA: "HDF5Data", HDF5_OUTPUT: "HDF5Output", HINGE_LOSS: "HingeLoss", IM2COL: "Im2col",
  IMAGE_DATA: "ImageData", INFOGAIN_LOSS: "InfogainLoss", INNER_PRODUCT: "InnerProduct", LRN: "LRN",
  MEMORY_DATA: "MemoryData", MULTINOMIAL_LOGISTIC_LOSS: "MultinomialLogisticLoss", MVN: "MVN", POOLING: "Pooling",
  POWER: "Power", RELU: "ReLU", SIGMOID: "Sigmoid", SIGMOID_CROSS_ENTROPY_LOSS: "SigmoidCrossEntropyLoss",
  SILENCE: "Silence", SOFTMAX: "Softmax", SOFTMAX_LOSS: "SoftmaxWithLoss", SPLIT: "Split", SLICE: "Slice",
  TANH: "TanH", WINDOW_DATA: "WindowData", THRESHOLD: "Threshold",
};

// ── Binary *_param schemas: field → [name, kind] ─────────────────────────────
type Kind = "u" | "i" | "f" | "b" | "shape" | string[]; // string[] = enum names by value
const POOL = ["MAX", "AVE", "STOCHASTIC"];
const SCHEMA: Record<string, Record<number, [string, Kind]>> = {
  convolution_param: {
    1: ["num_output", "u"], 2: ["bias_term", "b"], 3: ["pad", "u"], 4: ["kernel_size", "u"], 6: ["stride", "u"],
    18: ["dilation", "u"], 9: ["pad_h", "u"], 10: ["pad_w", "u"], 11: ["kernel_h", "u"], 12: ["kernel_w", "u"],
    13: ["stride_h", "u"], 14: ["stride_w", "u"], 5: ["group", "u"], 16: ["axis", "i"],
  },
  pooling_param: {
    1: ["pool", POOL], 4: ["pad", "u"], 9: ["pad_h", "u"], 10: ["pad_w", "u"], 2: ["kernel_size", "u"],
    5: ["kernel_h", "u"], 6: ["kernel_w", "u"], 3: ["stride", "u"], 7: ["stride_h", "u"], 8: ["stride_w", "u"],
    12: ["global_pooling", "b"], 13: ["round_mode", ["CEIL", "FLOOR"]],
  },
  inner_product_param: { 1: ["num_output", "u"], 2: ["bias_term", "b"], 5: ["axis", "i"], 6: ["transpose", "b"] },
  concat_param: { 2: ["axis", "i"], 1: ["concat_dim", "u"] },
  eltwise_param: { 1: ["operation", ["PROD", "SUM", "MAX"]], 2: ["coeff", "f"] },
  reshape_param: { 1: ["shape", "shape"], 2: ["axis", "i"], 3: ["num_axes", "i"] },
  flatten_param: { 1: ["axis", "i"], 2: ["end_axis", "i"] },
  slice_param: { 3: ["axis", "i"], 2: ["slice_point", "u"], 1: ["slice_dim", "u"] },
  scale_param: { 1: ["axis", "i"], 2: ["num_axes", "i"], 4: ["bias_term", "b"] },
  bias_param: { 1: ["axis", "i"], 2: ["num_axes", "i"] },
  softmax_param: { 2: ["axis", "i"] },
  embed_param: { 1: ["num_output", "u"], 2: ["input_dim", "u"], 3: ["bias_term", "b"] },
  prelu_param: { 2: ["channel_shared", "b"] },
  dropout_param: { 1: ["dropout_ratio", "f"] },
  lrn_param: { 1: ["local_size", "u"], 2: ["alpha", "f"], 3: ["beta", "f"], 4: ["norm_region", ["ACROSS_CHANNELS", "WITHIN_CHANNEL"]], 5: ["k", "f"] },
  input_param: { 1: ["shape", "shape"] },
  batch_norm_param: { 1: ["use_global_stats", "b"], 3: ["eps", "f"] },
  relu_param: { 1: ["negative_slope", "f"] },
  crop_param: { 1: ["axis", "i"], 2: ["offset", "u"] },
  argmax_param: { 1: ["out_max_val", "b"], 2: ["top_k", "u"], 3: ["axis", "i"] },
};
const LAYER_PARAM_FIELDS: Record<number, string> = {
  106: "convolution_param", 121: "pooling_param", 117: "inner_product_param", 104: "concat_param",
  110: "eltwise_param", 133: "reshape_param", 135: "flatten_param", 126: "slice_param", 142: "scale_param",
  141: "bias_param", 125: "softmax_param", 137: "embed_param", 131: "prelu_param", 108: "dropout_param",
  118: "lrn_param", 143: "input_param", 139: "batch_norm_param", 123: "relu_param", 144: "crop_param",
  103: "argmax_param",
};
const V1_PARAM_FIELDS: Record<number, string> = {
  10: "convolution_param", 19: "pooling_param", 17: "inner_product_param", 9: "concat_param",
  24: "eltwise_param", 12: "dropout_param", 18: "lrn_param", 30: "relu_param", 39: "softmax_param",
  31: "slice_param", 23: "argmax_param",
};

function readShapeMsg(r: ProtoReader): PTMsg {
  const dim: PTValue[] = [];
  for (let t = r.next(); t; t = r.next()) {
    if (t.field === 1 && (t.wireType === WT_LEN || t.wireType === WT_VARINT)) dim.push(...r.packedVarints(t.wireType));
    else r.skip(t.wireType);
  }
  return { dim };
}

function decodeParam(r: ProtoReader, schema: Record<number, [string, Kind]>): PTMsg {
  const m: PTMsg = {};
  for (let t = r.next(); t; t = r.next()) {
    const f = schema[t.field];
    if (!f) { r.skip(t.wireType); continue; }
    const [key, kind] = f;
    const arr = (m[key] ??= []);
    if (kind === "shape" && t.wireType === WT_LEN) arr.push(readShapeMsg(r.sub()));
    else if (kind === "f" && (t.wireType === WT_I32 || t.wireType === WT_LEN)) arr.push(...r.packedFloats(t.wireType));
    else if (t.wireType === WT_VARINT || t.wireType === WT_LEN) {
      const vals = r.packedVarints(t.wireType);
      for (const v of vals) {
        if (kind === "b") arr.push(v !== 0);
        else if (Array.isArray(kind)) arr.push(kind[v] ?? v);
        else arr.push(v);
      }
    } else r.skip(t.wireType);
  }
  return m;
}

function readBlob(r: ProtoReader): Blob {
  let shape: Shape | null = null;
  const legacy = [0, 0, 0, 0];
  let hasLegacy = false;
  let data: Uint8Array | undefined;
  let values: number[] | undefined;
  for (let t = r.next(); t; t = r.next()) {
    switch (t.field) {
      case 7: shape = ptNums(readShapeMsg(r.sub()), "dim"); break;
      case 5:
        if (t.wireType === WT_LEN) data = r.bytes();
        else (values ??= []).push(r.float());
        break;
      case 8:
        if (t.wireType === WT_LEN) { const b = r.bytes(); const v = new DataView(b.buffer, b.byteOffset, b.byteLength); values = []; for (let i = 0; i + 8 <= b.byteLength; i += 8) values.push(v.getFloat64(i, true)); }
        else r.skip(t.wireType);
        break;
      case 1: case 2: case 3: case 4:
        if (t.wireType === WT_VARINT) { legacy[t.field - 1] = r.int64(); hasLegacy = true; } else r.skip(t.wireType);
        break;
      default: r.skip(t.wireType);
    }
  }
  if (!shape) {
    if (hasLegacy) {
      // Legacy 4-D num/channels/height/width; drop leading 1s (IP weights are 1×1×out×in).
      const s = legacy.map((d) => d || 1);
      while (s.length > 1 && s[0] === 1) s.shift();
      shape = s;
    } else {
      const n = data ? data.byteLength >> 2 : values?.length ?? 0;
      shape = [n];
    }
  }
  return { shape, data, values };
}

function readRule(r: ProtoReader): string | undefined {
  let phase: string | undefined;
  for (let t = r.next(); t; t = r.next()) {
    if (t.field === 1 && t.wireType === WT_VARINT) phase = r.varint() === 0 ? "TRAIN" : "TEST";
    else r.skip(t.wireType);
  }
  return phase;
}

function readLayer(r: ProtoReader, v1: boolean): CLayer {
  const L: CLayer = { name: "", type: "", bottom: [], top: [], p: {}, blobs: [] };
  const F = v1
    ? { name: 4, type: 5, bottom: 2, top: 3, blobs: 6, include: 32 }
    : { name: 1, type: 2, bottom: 3, top: 4, blobs: 7, include: 8 };
  const params = v1 ? V1_PARAM_FIELDS : LAYER_PARAM_FIELDS;
  for (let t = r.next(); t; t = r.next()) {
    const f = t.field;
    if (f === F.name && t.wireType === WT_LEN) L.name = r.string();
    else if (f === F.type && !v1 && t.wireType === WT_LEN) L.type = r.string();
    else if (f === F.type && v1 && t.wireType === WT_VARINT) { const e = r.varint(); L.type = V1_TYPES[e] ?? `V1_${e}`; }
    else if (f === F.bottom && t.wireType === WT_LEN) L.bottom.push(r.string());
    else if (f === F.top && t.wireType === WT_LEN) L.top.push(r.string());
    else if (f === F.blobs && t.wireType === WT_LEN) L.blobs.push(readBlob(r.sub()));
    else if (f === F.include && t.wireType === WT_LEN) L.phase = readRule(r.sub()) ?? L.phase;
    else if (params[f] && t.wireType === WT_LEN) L.p[params[f]] = decodeParam(r.sub(), SCHEMA[params[f]]);
    else r.skip(t.wireType);
  }
  return L;
}

function readNetBinary(buf: Uint8Array): CNet {
  const net: CNet = { name: "", inputs: [], layers: [] };
  const r = new ProtoReader(buf);
  const inputNames: string[] = [];
  const inputDims: number[] = [];
  const inputShapes: Shape[] = [];
  for (let t = r.next(); t; t = r.next()) {
    switch (t.field) {
      case 1: if (t.wireType === WT_LEN) net.name = r.string(); else r.skip(t.wireType); break;
      case 3: if (t.wireType === WT_LEN) inputNames.push(r.string()); else r.skip(t.wireType); break;
      case 4: if (t.wireType === WT_LEN || t.wireType === WT_VARINT) inputDims.push(...r.packedVarints(t.wireType)); else r.skip(t.wireType); break;
      case 8: if (t.wireType === WT_LEN) inputShapes.push(ptNums(readShapeMsg(r.sub()), "dim")); else r.skip(t.wireType); break;
      case 100: if (t.wireType === WT_LEN) net.layers.push(readLayer(r.sub(), false)); else r.skip(t.wireType); break;
      case 2: if (t.wireType === WT_LEN) net.layers.push(readLayer(r.sub(), true)); else r.skip(t.wireType); break;
      default: r.skip(t.wireType);
    }
  }
  net.inputs = netInputs(inputNames, inputShapes, inputDims);
  return net;
}

function netInputs(names: string[], shapes: Shape[], flatDims: number[]): CNet["inputs"] {
  return names.map((name, i) => ({
    name,
    shape: shapes[i] ?? (flatDims.length >= (i + 1) * 4 ? flatDims.slice(i * 4, i * 4 + 4) : undefined),
  }));
}

function readNetText(text: string): CNet {
  const m = parseProtoText(text);
  const shapes = ptMsgs(m, "input_shape").map((s) => ptNums(s, "dim"));
  const net: CNet = {
    name: ptStr(m, "name") ?? "",
    inputs: netInputs(ptStrs(m, "input"), shapes, ptNums(m, "input_dim")),
    layers: [],
  };
  const toLayer = (lm: PTMsg, v1: boolean): CLayer => {
    const rawType = ptStr(lm, "type") ?? "";
    const p: Record<string, PTMsg> = {};
    for (const [k, vals] of Object.entries(lm)) {
      if (k.endsWith("_param") && vals[0] && typeof vals[0] === "object") p[k] = vals[0] as PTMsg;
    }
    const phases = ptMsgs(lm, "include").map((r) => ptStr(r, "phase")).filter(Boolean) as string[];
    const blobs: Blob[] = ptMsgs(lm, "blobs").map((b) => {
      const sm = ptMsg(b, "shape");
      const shape = sm ? ptNums(sm, "dim") : [ptNum(b, "num", 1), ptNum(b, "channels", 1), ptNum(b, "height", 1), ptNum(b, "width", 1)];
      const values = ptNums(b, "data");
      return { shape, values: values.length ? values : undefined };
    });
    return {
      name: ptStr(lm, "name") ?? "",
      type: v1 ? V1_BY_ID[rawType] ?? rawType : rawType,
      bottom: ptStrs(lm, "bottom"),
      top: ptStrs(lm, "top"),
      phase: phases[0] === "0" ? "TRAIN" : phases[0] === "1" ? "TEST" : phases[0],
      p,
      blobs,
    };
  };
  for (const lm of ptMsgs(m, "layer")) net.layers.push(toLayer(lm, false));
  for (const lm of ptMsgs(m, "layers")) net.layers.push(toLayer(lm, true));
  return net;
}

// ── Shape inference ──────────────────────────────────────────────────────────

const pair = (pm: PTMsg | undefined, base: string, h: string, w: string, d: number): [number, number] => {
  const list = ptNums(pm, base);
  const hv = ptNum(pm, h, NaN);
  const wv = ptNum(pm, w, NaN);
  const a = Number.isFinite(hv) ? hv : list[0] ?? d;
  const b = Number.isFinite(wv) ? wv : list[1] ?? list[0] ?? d;
  return [a, b];
};

function convOut(inS: number, k: number, s: number, p: number, d: number): number {
  if (inS < 0) return DYN;
  return Math.floor((inS + 2 * p - (d * (k - 1) + 1)) / s) + 1;
}

function poolOut(inS: number, k: number, s: number, p: number, ceil: boolean): number {
  if (inS < 0) return DYN;
  let o = (ceil ? Math.ceil((inS + 2 * p - k) / s) : Math.floor((inS + 2 * p - k) / s)) + 1;
  if (p > 0 && (o - 1) * s >= inS + p) o--; // last window must start inside the image
  return o;
}

const axisOf = (a: number, rank: number) => (a < 0 ? rank + a : a);

type Inferred = { outs: (Shape | undefined)[]; params: number; cost: Cost; attr: Record<string, unknown> };

function infer(L: CLayer, ins: (Shape | undefined)[]): Inferred {
  const x = ins[0];
  const res: Inferred = { outs: [], params: 0, cost: NO_COST, attr: {} };
  const same = () => { res.outs = L.top.map((_, i) => (ins[i] ?? x) ? [...(ins[i] ?? x)!] : undefined); };
  const cp = L.p.convolution_param;

  switch (L.type) {
    case "Input": {
      const shapes = ptMsgs(L.p.input_param, "shape").map((s) => ptNums(s, "dim"));
      res.outs = L.top.map((_, i) => shapes[i] ?? shapes[0]);
      break;
    }
    case "Convolution":
    case "Deconvolution":
    case "DepthwiseConvolution": {
      const [kh, kw] = pair(cp, "kernel_size", "kernel_h", "kernel_w", 1);
      const [sh, sw] = pair(cp, "stride", "stride_h", "stride_w", 1);
      const [ph, pw] = pair(cp, "pad", "pad_h", "pad_w", 0);
      const dil = ptNums(cp, "dilation");
      const [dh, dw] = [dil[0] ?? 1, dil[1] ?? dil[0] ?? 1];
      const cout = ptNum(cp, "num_output", 0);
      const group = ptNum(cp, "group", 1) || 1;
      const bias = ptBool(cp, "bias_term", true);
      res.attr = { kernel: [kh, kw], stride: [sh, sw], pad: [ph, pw], dilation: [dh, dw], group, num_output: cout, bias_term: bias };
      const cin = x && x.length >= 2 ? x[1] : DYN;
      if (L.type === "Deconvolution") {
        if (x && x.length === 4) {
          const o = (i: number, k: number, s: number, p: number, d: number) => (i < 0 ? DYN : s * (i - 1) + d * (k - 1) + 1 - 2 * p);
          res.outs = [[x[0], cout, o(x[2], kh, sh, ph, dh), o(x[3], kw, sw, pw, dw)]];
          res.cost = linearCost(x, (cout / group) * kh * kw);
        }
        if (cin > 0) res.params = cin * (cout / group) * kh * kw + (bias ? cout : 0);
      } else {
        if (x && x.length === 4) {
          res.outs = [[x[0], cout, convOut(x[2], kh, sh, ph, dh), convOut(x[3], kw, sw, pw, dw)]];
          if (cin > 0) res.cost = linearCost(res.outs[0], (cin / group) * kh * kw);
        }
        if (cin > 0) res.params = cout * (cin / group) * kh * kw + (bias ? cout : 0);
      }
      break;
    }
    case "Pooling": {
      const pp = L.p.pooling_param;
      const global = ptBool(pp, "global_pooling", false);
      const ceil = (ptStr(pp, "round_mode") ?? "CEIL") !== "FLOOR";
      const pool = ptStr(pp, "pool") ?? "MAX";
      res.attr = { pool, global_pooling: global };
      if (!x || x.length !== 4) break;
      if (global) {
        res.outs = [[x[0], x[1], 1, 1]];
        res.cost = elemCost(x, 1);
      } else {
        const [kh, kw] = pair(pp, "kernel_size", "kernel_h", "kernel_w", 1);
        const [sh, sw] = pair(pp, "stride", "stride_h", "stride_w", 1);
        const [ph, pw] = pair(pp, "pad", "pad_h", "pad_w", 0);
        res.attr = { ...res.attr, kernel: [kh, kw], stride: [sh, sw], pad: [ph, pw], round_mode: ceil ? "CEIL" : "FLOOR" };
        res.outs = [[x[0], x[1], poolOut(x[2], kh, sh, ph, ceil), poolOut(x[3], kw, sw, pw, ceil)]];
        res.cost = elemCost(res.outs[0], kh * kw);
      }
      break;
    }
    case "InnerProduct": {
      const ip = L.p.inner_product_param;
      const nout = ptNum(ip, "num_output", 0);
      const bias = ptBool(ip, "bias_term", true);
      res.attr = { num_output: nout, bias_term: bias };
      if (!x) break;
      const a = axisOf(ptNum(ip, "axis", 1), x.length);
      const K = x.slice(a).reduce((p, d) => (d < 0 ? p : p * d), 1);
      res.outs = [[...x.slice(0, a), nout]];
      res.params = nout * K + (bias ? nout : 0);
      res.cost = linearCost(res.outs[0], K);
      break;
    }
    case "Embed": {
      const ep = L.p.embed_param;
      const nout = ptNum(ep, "num_output", 0);
      const nin = ptNum(ep, "input_dim", 0);
      res.params = nout * nin + (ptBool(ep, "bias_term", true) ? nout : 0);
      if (x) res.outs = [[...x, nout]];
      break;
    }
    case "Concat": {
      const cpm = L.p.concat_param;
      const valid = ins.filter((s): s is Shape => !!s);
      if (valid.length !== ins.length || !valid.length) break;
      const a = axisOf(cpm?.axis ? ptNum(cpm, "axis", 1) : ptNum(cpm, "concat_dim", 1), valid[0].length);
      const out = [...valid[0]];
      out[a] = valid.some((s) => s[a] < 0) ? DYN : valid.reduce((p, s) => p + s[a], 0);
      res.outs = [out];
      res.attr = { axis: a };
      break;
    }
    case "Eltwise": {
      const op = ptStr(L.p.eltwise_param, "operation") ?? "SUM";
      res.attr = { operation: op };
      if (x) { res.outs = [[...x]]; res.cost = elemCost(x, Math.max(1, ins.length - 1)); }
      break;
    }
    case "Flatten": {
      if (!x) break;
      const fp = L.p.flatten_param;
      const a = axisOf(ptNum(fp, "axis", 1), x.length);
      const e = axisOf(ptNum(fp, "end_axis", -1), x.length);
      const mid = x.slice(a, e + 1);
      res.outs = [[...x.slice(0, a), mid.some((d) => d < 0) ? DYN : mid.reduce((p, d) => p * d, 1), ...x.slice(e + 1)]];
      break;
    }
    case "Reshape": {
      const rp = L.p.reshape_param;
      const dimsSpec = ptNums(ptMsg(rp, "shape"), "dim");
      if (!x || !dimsSpec.length) break;
      const a = axisOf(ptNum(rp, "axis", 0), x.length + 1);
      const na = ptNum(rp, "num_axes", -1);
      const end = na < 0 ? x.length : a + na;
      const mid = dimsSpec.map((d, i) => (d === 0 ? x[a + i] ?? DYN : d));
      const out = [...x.slice(0, a), ...mid, ...x.slice(end)];
      const inf = out.indexOf(-1);
      if (inf >= 0 && x.every((d) => d >= 0)) {
        const rest = out.reduce((p, d, i) => (i === inf ? p : p * d), 1);
        out[inf] = rest > 0 ? numel(x) / rest : DYN;
      }
      res.outs = [out];
      break;
    }
    case "Slice": {
      if (!x) break;
      const sp = L.p.slice_param;
      const a = axisOf(sp?.axis ? ptNum(sp, "axis", 1) : ptNum(sp, "slice_dim", 1), x.length);
      const points = ptNums(sp, "slice_point");
      const bounds = points.length ? [0, ...points, x[a]] : L.top.map((_, i) => Math.floor((x[a] * i) / L.top.length)).concat(x[a]);
      res.outs = L.top.map((_, i) => { const s = [...x]; s[a] = x[a] < 0 ? DYN : bounds[i + 1] - bounds[i]; return s; });
      break;
    }
    case "Crop": {
      const ref = ins[1];
      if (!x || !ref) break;
      const a = axisOf(ptNum(L.p.crop_param, "axis", 2), x.length);
      res.outs = [[...x.slice(0, a), ...ref.slice(a)]];
      break;
    }
    case "BatchNorm": {
      same();
      const c = x && x.length > 1 ? x[1] : 0;
      if (c > 0) res.params = 2 * c + 1; // mean, variance, moving-average factor
      res.cost = elemCost(x, 2);
      break;
    }
    case "Scale":
    case "Bias": {
      same();
      const c = x && x.length > 1 ? x[1] : 0;
      const withBias = L.type === "Scale" && ptBool(L.p.scale_param, "bias_term", false);
      if (c > 0 && ins.length < 2) res.params = c + (withBias ? c : 0);
      res.cost = elemCost(x, withBias ? 2 : 1);
      break;
    }
    case "PReLU": {
      same();
      const c = x && x.length > 1 ? x[1] : 0;
      res.params = ptBool(L.p.prelu_param, "channel_shared", false) ? 1 : c;
      res.cost = elemCost(x, 1);
      break;
    }
    case "Softmax":
      same();
      res.cost = elemCost(x, 3);
      break;
    case "LRN":
      same();
      res.cost = elemCost(x, ptNum(L.p.lrn_param, "local_size", 5) + 2);
      break;
    case "Split":
      res.outs = L.top.map(() => (x ? [...x] : undefined));
      break;
    case "ReLU": case "Sigmoid": case "TanH": case "ELU": case "AbsVal": case "BNLL": case "Power": case "Exp":
    case "Log": case "Threshold": case "Swish": case "Clip": case "ReLU6": case "MVN":
      same();
      res.cost = elemCost(x, L.type === "MVN" ? 2 : 1);
      break;
    case "Dropout": case "Silence":
      same();
      break;
    default:
      // Unknown / data / loss layers: shapes unknown unless pass-through in-place.
      if (L.top.length === 1 && L.bottom.length === 1 && L.top[0] === L.bottom[0]) same();
  }
  return res;
}

// ── Display metadata ─────────────────────────────────────────────────────────

function canonOf(L: CLayer): string {
  switch (L.type) {
    case "Convolution": case "DepthwiseConvolution": return "Conv";
    case "Deconvolution": return "ConvTranspose";
    case "InnerProduct": return "Gemm";
    case "Pooling": {
      const pp = L.p.pooling_param;
      const ave = (ptStr(pp, "pool") ?? "MAX") === "AVE";
      if (ptBool(pp, "global_pooling", false)) return ave ? "GlobalAveragePool" : "GlobalMaxPool";
      return ave ? "AveragePool" : "MaxPool";
    }
    case "ReLU": return ptNum(L.p.relu_param, "negative_slope", 0) ? "LeakyRelu" : "Relu";
    case "PReLU": return "LeakyRelu";
    case "Sigmoid": return "Sigmoid";
    case "TanH": return "Tanh";
    case "Softmax": return "Softmax";
    case "BatchNorm": return "BatchNormalization";
    case "Scale": return "Mul";
    case "Bias": return "Add";
    case "Eltwise": {
      const op = ptStr(L.p.eltwise_param, "operation") ?? "SUM";
      return op === "PROD" ? "Mul" : op === "MAX" ? "Max" : "Add";
    }
    case "Concat": return "Concat";
    case "Flatten": return "Flatten";
    case "Reshape": return "Reshape";
    default: return L.type;
  }
}

const isDataLayer = (t: string) => /^(Input|Data|ImageData|MemoryData|HDF5Data|WindowData|DummyData|Python)$/.test(t);
const isLossLayer = (t: string) => /Loss$|^(Accuracy|ArgMax)$/.test(t);

function insightFor(L: CLayer, inPlace: boolean): string | undefined {
  const tail = inPlace ? " Runs in place (top == bottom), so it adds no activation memory in Caffe." : "";
  switch (L.type) {
    case "BatchNorm": return "Caffe BatchNorm only normalizes (mean/variance); γ/β live in the following Scale layer. Fold both into the preceding convolution for deployment." + tail;
    case "Scale": return "Per-channel scale (+bias): usually the γ/β half of a Caffe BatchNorm pair; foldable into the previous convolution." + tail;
    case "Pooling": return "Caffe pooling rounds the output size UP (ceil) by default, unlike ONNX/TF floor. A common source of off-by-one shape mismatches after conversion.";
    case "Eltwise": return "Elementwise combine of several blobs (residual add / product / max).";
    case "Split": return "Fan-out of one blob to several consumers (inserted by Caffe for blobs used more than once).";
    case "Dropout": return "Training-only regularizer; identity at inference and removed by converters.";
    case "LRN": return "Local response normalization (AlexNet/GoogLeNet era). Poorly supported on edge NPUs; consider removing or replacing.";
    case "Input": case "Data": return "Network input blob.";
  }
  if (isLossLayer(L.type)) return "Training/evaluation layer; not part of the deploy graph.";
  return tail ? `Caffe ${L.type} layer.${tail}` : undefined;
}

// ── Public API ───────────────────────────────────────────────────────────────

/** Does this text look like a Caffe prototxt (layer {…} / layers {…} / input:)? */
export function isCaffePrototxt(text: string): boolean {
  const head = text.slice(0, 65536);
  return /(^|\n)\s*(layers?)\s*\{/.test(head) && /\b(bottom|top)\s*:/.test(head);
}

function looksLikeText(buf: Uint8Array): boolean {
  const n = Math.min(buf.length, 512);
  if (n === 0) return false;
  for (let i = 0; i < n; i++) {
    const c = buf[i];
    if (c < 9 || (c > 13 && c < 32 && c !== 27)) return false;
  }
  return true;
}

/** Parse a Caffe .prototxt (text) or .caffemodel (binary). */
export function parseCaffe(input: ArrayBuffer | Uint8Array | string, name = "model.caffemodel"): Model {
  if (typeof input === "string") return parseCaffePrototxt(input, name);
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (/\.(prototxt|pbtxt|txt)$/i.test(name) || looksLikeText(buf)) {
    return parseCaffePrototxt(new TextDecoder().decode(buf), name, buf.byteLength);
  }
  return parseCaffemodel(buf, name);
}

export function parseCaffePrototxt(text: string, name = "deploy.prototxt", sizeBytes = text.length): Model {
  const net = readNetText(text);
  if (!net.layers.length) throw new Error("Caffe prototxt has no layers.");
  return buildModel(net, name, sizeBytes, "Caffe prototxt");
}

export function parseCaffemodel(input: ArrayBuffer | Uint8Array, name = "model.caffemodel"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const net = readNetBinary(buf);
  if (!net.layers.length) throw new Error("Caffe model has no layers (not a NetParameter?).");
  return buildModel(net, name, buf.byteLength, "Caffe");
}

function blobStat(b: Blob): WStat {
  if (b.data) return weightStats(b.shape, sampleBytes(b.data, "float32"));
  return weightStats(b.shape, b.values ?? null);
}

function buildModel(net: CNet, name: string, sizeBytes: number, format: string): Model {
  // Drop TRAIN-only layers when the net also has a TEST phase (train_val nets).
  const hasTest = net.layers.some((l) => l.phase === "TEST");
  const cl = net.layers.filter((l) => !(hasTest && l.phase === "TRAIN"));

  const layers: ModelLayer[] = [];
  const edges = new EdgeSet();
  const producer = new Map<string, number>(); // blob → layer id
  const shapeOf = new Map<string, Shape | undefined>();
  // A layer is a sink when no LATER layer reads any of its tops.
  const lastRead = new Map<string, number>();
  cl.forEach((l, i) => l.bottom.forEach((b) => lastRead.set(b, i)));
  const sink = cl.map((l, i) => !l.top.some((t) => (lastRead.get(t) ?? -1) > i));

  // Net-level inputs (`input: "data"` + input_shape / input_dim).
  for (const inp of net.inputs) {
    const id = layers.length;
    producer.set(inp.name, id);
    shapeOf.set(inp.name, inp.shape);
    layers.push(makeLayer({
      id, name: inp.name, type: "Input", canon: "Input", shape: inp.shape, group: "input",
      outs: [{ n: inp.name, s: inp.shape }], insight: "Network input blob (declared at net level).", framework: "Caffe",
    }));
  }

  cl.forEach((L, li) => {
    const id = layers.length;
    const inShapes = L.bottom.map((b) => shapeOf.get(b));
    const inf = infer(L, inShapes);
    const blobParams = L.blobs.reduce((a, b) => a + numel(b.shape), 0);
    const params = L.blobs.length ? blobParams : inf.params;
    const biggest = L.blobs.length ? L.blobs.reduce((a, b) => (numel(b.shape) > numel(a.shape) ? b : a)) : null;

    for (const b of L.bottom) edges.add(producer.get(b), id);
    L.top.forEach((t, i) => { producer.set(t, id); shapeOf.set(t, inf.outs[i]); });

    const canon = canonOf(L);
    const inPlace = L.top.length > 0 && L.top.every((t) => L.bottom.includes(t));
    const isSink = sink[li];
    let group: ModelLayer["group"] | undefined;
    if (isDataLayer(L.type)) group = "input";
    else if (isLossLayer(L.type) || L.type === "Softmax") group = "output";
    else if (canon === "Gemm") group = "head";
    else if (isSink) group = "output";

    const outShape = inf.outs[0];
    layers.push(makeLayer({
      id,
      name: L.name || `${L.type}_${id}`,
      type: L.type,
      canon,
      shape: outShape,
      params,
      cost: inf.cost,
      group,
      ins: L.bottom.map((b, i) => ({ n: b, s: inShapes[i] })),
      outs: L.top.map((t, i) => ({ n: t, s: inf.outs[i] })),
      attr: { ...flattenParams(L.p), ...inf.attr, ...(L.phase ? { phase: L.phase } : {}), ...(L.blobs.length ? { blobs: L.blobs.map((b) => b.shape.join("×")) } : {}) },
      w: biggest ? blobStat(biggest) : params > 0 ? { shape: "[multi]", size: params } : null,
      insight: insightFor(L, inPlace),
      framework: "Caffe",
    }));
    // In-place layers write no new activation buffer.
    if (inPlace) layers[layers.length - 1].mem = 0;
  });

  const firstIn = layers.find((l) => l.group === "input");
  const lastShaped = [...layers].reverse().find((l) => l.outs[0]?.s && l.outs[0].s.length);
  return {
    name,
    format,
    framework: "Caffe",
    opset: 0,
    sizeBytes,
    inputShape: cleanShape(firstIn?.outs[0]?.s),
    outputShape: cleanShape(lastShaped?.outs[0]?.s),
    layers,
    edges: edges.edges,
    doc: net.name || undefined,
  };
}

function flattenParams(p: Record<string, PTMsg>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, m] of Object.entries(p)) {
    const prefix = k.replace(/_param$/, "");
    for (const [f, vals] of Object.entries(m)) {
      if (vals.some((v) => typeof v === "object")) continue; // fillers etc.
      out[`${prefix}.${f}`] = vals.length === 1 ? vals[0] : vals;
    }
  }
  return out;
}
