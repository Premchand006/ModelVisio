import type { Model, ModelLayer } from "../types";
import {
  EdgeSet, NO_COST, cleanShape, elemCost, linearCost, makeLayer, numel, weightStats,
  type Cost, type Shape,
} from "../common/graphKit";
import { fromHalf } from "../common/float16";

// Tencent ncnn `.param` (text, magic 7767517) and `.param.bin` (binary, magic
// 0x007685dd) graph parser. Each layer line is
//   Type Name in_count out_count in_blobs… out_blobs… id=value…
// where integer ids carry per-type meaning (from ncnn's src/layer/*.cpp
// load_param) and ids <= -23300 are arrays ("-23300=n,v1,…,vn").
// Shapes follow ncnn's Mat model (no batch dim): we store them outermost-first,
// i.e. [c, h, w] for 3-D blobs, [h, w] for 2-D and [w] for 1-D. An optional
// `.bin` weight file adds real weight statistics.

const MAGIC_TEXT = "7767517";
const MAGIC_BIN = 0x007685dd;
const FRAMEWORK = "ncnn";

type ParamVal = number | number[];
type NcnnLayer = { type: string; name: string; ins: string[]; outs: string[]; p: Map<number, ParamVal> };

export type NcnnOptions = {
  /** Contents of the matching `.bin` weight file (enables weight statistics). */
  bin?: Uint8Array;
  /** Input blob shape [c, h, w] when the Input layer carries no dims. */
  inputShape?: number[];
};

// Layer type index → name (ncnn layer_type_enum order), used by `.param.bin`.
const TYPE_INDEX: string[] = [];
(
  "AbsVal ArgMax BatchNorm Bias BNLL Concat Convolution Crop Deconvolution Dropout Eltwise ELU Embed Exp " +
  "Flatten InnerProduct Input Log LRN MemoryData MVN Pooling Power PReLU Proposal Reduction ReLU Reshape " +
  "ROIPooling Scale Sigmoid Slice Softmax Split SPP TanH Threshold Tile RNN LSTM BinaryOp UnaryOp " +
  "ConvolutionDepthWise Padding Squeeze ExpandDims Normalize Permute PriorBox DetectionOutput Interp " +
  "DeconvolutionDepthWise ShuffleChannel InstanceNorm Clip Reorg YoloDetectionOutput Quantize Dequantize " +
  "Yolov3DetectionOutput PSROIPooling ROIAlign Packing Requantize Cast HardSigmoid SELU HardSwish Noop " +
  "PixelShuffle DeepCopy Mish StatisticsPooling Swish Gemm GroupNorm LayerNorm Softplus GRU MultiHeadAttention " +
  "GELU Convolution1D Pooling1D ConvolutionDepthWise1D Convolution3D ConvolutionDepthWise3D Pooling3D MatMul " +
  "Deconvolution1D DeconvolutionDepthWise1D Deconvolution3D DeconvolutionDepthWise3D Einsum DeformableConv2D " +
  "GLU Fold Unfold GridSample CumulativeSum CopyTo Erf Diag CELU Shrink RMSNorm Spectrogram InverseSpectrogram " +
  "Flip SDPA RotaryEmbed"
).split(" ").forEach((n, i) => { TYPE_INDEX[i] = n; });

// Param id → attribute name per layer type (ids from ncnn src/layer/*.cpp).
const CONV_NAMES: Record<number, string> = {
  0: "num_output", 1: "kernel_w", 11: "kernel_h", 2: "dilation_w", 12: "dilation_h", 3: "stride_w", 13: "stride_h",
  4: "pad_left", 15: "pad_right", 14: "pad_top", 16: "pad_bottom", 18: "pad_value", 5: "bias_term",
  6: "weight_data_size", 7: "group", 8: "int8_scale_term", 9: "activation_type", 10: "activation_params", 19: "dynamic_weight",
};
const DECONV_NAMES: Record<number, string> = {
  ...CONV_NAMES, 18: "output_pad_right", 19: "output_pad_bottom", 20: "output_w", 21: "output_h", 28: "dynamic_weight",
};
const POOL_NAMES: Record<number, string> = {
  0: "pooling_type", 1: "kernel_w", 11: "kernel_h", 2: "stride_w", 12: "stride_h", 3: "pad_left", 14: "pad_right",
  13: "pad_top", 15: "pad_bottom", 4: "global_pooling", 5: "pad_mode", 6: "avgpool_count_include_pad",
  7: "adaptive_pooling", 8: "out_w", 18: "out_h",
};
const NAMES: Record<string, Record<number, string>> = {
  Input: { 0: "w", 1: "h", 11: "d", 2: "c" },
  MemoryData: { 0: "w", 1: "h", 11: "d", 2: "c", 21: "load_type" },
  Convolution: CONV_NAMES, ConvolutionDepthWise: CONV_NAMES, Convolution1D: CONV_NAMES, Convolution3D: CONV_NAMES,
  Deconvolution: DECONV_NAMES, DeconvolutionDepthWise: DECONV_NAMES,
  InnerProduct: { 0: "num_output", 1: "bias_term", 2: "weight_data_size", 8: "int8_scale_term", 9: "activation_type", 10: "activation_params" },
  Pooling: POOL_NAMES, Pooling1D: POOL_NAMES, Pooling3D: POOL_NAMES,
  BatchNorm: { 0: "channels", 1: "eps" },
  Scale: { 0: "scale_data_size", 1: "bias_term" },
  BinaryOp: { 0: "op_type", 1: "with_scalar", 2: "b" },
  UnaryOp: { 0: "op_type" },
  Eltwise: { 0: "op_type", 1: "coeffs" },
  Concat: { 0: "axis" },
  Slice: { 0: "slices", 1: "axis", 2: "indices" },
  Softmax: { 0: "axis", 1: "fixbug0" },
  ReLU: { 0: "slope" },
  Clip: { 0: "min", 1: "max" },
  Reshape: { 0: "w", 1: "h", 11: "d", 2: "c", 6: "shape_expr" },
  Interp: { 0: "resize_type", 1: "height_scale", 2: "width_scale", 3: "output_height", 4: "output_width", 5: "dynamic_target_size", 6: "align_corner" },
  Padding: { 0: "top", 1: "bottom", 2: "left", 3: "right", 4: "type", 5: "value", 6: "per_channel_pad_data_size", 7: "front", 8: "behind" },
  Permute: { 0: "order_type" },
  LayerNorm: { 0: "affine_size", 1: "eps", 2: "affine" },
  InstanceNorm: { 0: "channels", 1: "eps", 2: "affine" },
  GroupNorm: { 0: "group", 1: "channels", 2: "eps", 3: "affine" },
  PReLU: { 0: "num_slope" },
  HardSigmoid: { 0: "alpha", 1: "beta" },
  HardSwish: { 0: "alpha", 1: "beta" },
  Embed: { 0: "num_output", 1: "input_dim", 2: "bias_term", 3: "weight_data_size" },
  LSTM: { 0: "num_output", 1: "weight_data_size", 2: "direction", 3: "hidden_size" },
  GRU: { 0: "num_output", 1: "weight_data_size", 2: "direction" },
  RNN: { 0: "num_output", 1: "weight_data_size", 2: "direction" },
  MultiHeadAttention: { 0: "embed_dim", 1: "num_heads", 2: "weight_data_size", 3: "kdim", 4: "vdim", 5: "attn_mask", 6: "scale" },
  Gemm: { 0: "alpha", 1: "beta", 2: "transA", 3: "transB" },
  Dropout: { 0: "scale" },
  ELU: { 0: "alpha" },
  GELU: { 0: "fast_gelu" },
};

// Param ids holding floats (binary `.param.bin` stores raw 32-bit words).
const FLOAT_IDS: Record<string, number[]> = {
  BatchNorm: [1], ReLU: [0], Clip: [0, 1], Interp: [1, 2], Dropout: [0], ELU: [0], HardSigmoid: [0, 1],
  HardSwish: [0, 1], LayerNorm: [1], InstanceNorm: [1], GroupNorm: [2], Padding: [5], BinaryOp: [2],
  Power: [0, 1, 2], Convolution: [18, 10], ConvolutionDepthWise: [18, 10], InnerProduct: [10], Gemm: [0, 1],
  MultiHeadAttention: [6], Eltwise: [1], Threshold: [0],
};

const BINARY_OPS = ["Add", "Sub", "Mul", "Div", "Max", "Min", "Pow", "RSub", "RDiv", "RPow", "Atan2", "RAtan2"];
const UNARY_OPS = ["Abs", "Neg", "Floor", "Ceil", "Square", "Sqrt", "Rsqrt", "Exp", "Log", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Reciprocal", "Tanh", "Log10", "Round", "Trunc"];
const ACTIVATIONS = ["", "ReLU", "LeakyReLU", "Clip", "Sigmoid", "Mish", "HardSwish"];

/** True when the buffer starts with an ncnn param magic (text or binary). */
export function isNcnnParam(buf: Uint8Array): boolean {
  if (buf.length >= 4 && new DataView(buf.buffer, buf.byteOffset, 4).getInt32(0, true) === MAGIC_BIN) return true;
  const head = new TextDecoder().decode(buf.subarray(0, 16)).replace(/^﻿/, "");
  return head.trimStart().startsWith(MAGIC_TEXT);
}

function parseText(text: string): NcnnLayer[] {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length);
  if (lines[0] !== MAGIC_TEXT) throw new Error("Not an ncnn .param file (missing 7767517 magic).");
  const header = lines[1]?.split(/\s+/).map(Number) ?? [];
  if (header.length !== 2 || header.some((n) => !Number.isFinite(n))) throw new Error("Invalid ncnn header.");
  const layers: NcnnLayer[] = [];
  for (const line of lines.slice(2)) {
    const cols = line.split(/\s+/);
    if (cols.length < 4) continue;
    const [type, name] = cols;
    const nin = parseInt(cols[2], 10);
    const nout = parseInt(cols[3], 10);
    if (!Number.isFinite(nin) || !Number.isFinite(nout)) continue;
    const ins = cols.slice(4, 4 + nin);
    const outs = cols.slice(4 + nin, 4 + nin + nout);
    const p = new Map<number, ParamVal>();
    for (const kv of cols.slice(4 + nin + nout)) {
      const eq = kv.indexOf("=");
      if (eq < 0) continue;
      let id = parseInt(kv.slice(0, eq), 10);
      const raw = kv.slice(eq + 1);
      if (id <= -23300) {
        id = -id - 23300;
        const vals = raw.split(",").map(Number);
        p.set(id, vals.slice(1));
      } else {
        const v = Number(raw);
        p.set(id, Number.isFinite(v) ? v : 0);
      }
    }
    layers.push({ type, name, ins, outs, p });
  }
  return layers;
}

function parseBinary(buf: Uint8Array): NcnnLayer[] {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 0;
  const i32 = () => { const v = dv.getInt32(o, true); o += 4; return v; };
  const f32At = (bits: number) => { const t = new DataView(new ArrayBuffer(4)); t.setInt32(0, bits, true); return t.getFloat32(0, true); };
  if (i32() !== MAGIC_BIN) throw new Error("Not an ncnn .param.bin file.");
  const layerCount = i32();
  i32(); // blob count
  const layers: NcnnLayer[] = [];
  for (let i = 0; i < layerCount; i++) {
    const ti = i32();
    const type = TYPE_INDEX[ti & 0xff] && !(ti & 0x100) ? TYPE_INDEX[ti] : `Custom_${ti}`;
    const nin = i32();
    const nout = i32();
    const ins: string[] = []; const outs: string[] = [];
    for (let k = 0; k < nin; k++) ins.push(String(i32()));
    for (let k = 0; k < nout; k++) outs.push(String(i32()));
    const p = new Map<number, ParamVal>();
    const floats = new Set(FLOAT_IDS[type] ?? []);
    for (let id = i32(); id !== -233; id = i32()) {
      if (id <= -23300) {
        const key = -id - 23300;
        const n = i32();
        const vals: number[] = [];
        for (let k = 0; k < n; k++) { const bits = i32(); vals.push(floats.has(key) ? f32At(bits) : bits); }
        p.set(key, vals);
      } else {
        const bits = i32();
        p.set(id, floats.has(id) ? f32At(bits) : bits);
      }
      if (o > buf.length) throw new Error("Truncated ncnn .param.bin.");
    }
    layers.push({ type, name: `${type}_${i}`, ins, outs, p });
  }
  return layers;
}

const num = (p: Map<number, ParamVal>, id: number, d: number): number => {
  const v = p.get(id);
  return typeof v === "number" ? v : d;
};
const arr = (p: Map<number, ParamVal>, id: number): number[] | undefined => {
  const v = p.get(id);
  return Array.isArray(v) ? v : undefined;
};

/** ncnn dims (w, h, d, c with 0 = absent) → outermost-first shape. */
function dimsShape(w: number, h: number, d: number, c: number): Shape | undefined {
  if (c > 0) return d > 0 ? [c, d, h, w] : [c, h, w];
  if (h > 0) return [h, w];
  if (w > 0) return [w];
  return undefined;
}

function convOut(inSize: number, k: number, s: number, dil: number, pa: number, pb: number): number {
  if (inSize < 0) return -1;
  if (pa === -233 || pa === -234) return Math.ceil(inSize / s); // SAME_UPPER / SAME_LOWER
  return Math.floor((inSize + pa + pb - (dil * (k - 1) + 1)) / s) + 1;
}

type Geo = { kw: number; kh: number; sw: number; sh: number; dw: number; dh: number; pl: number; pr: number; pt: number; pb: number };
function convGeo(p: Map<number, ParamVal>): Geo {
  const kw = num(p, 1, 0), sw = num(p, 3, 1), dw = num(p, 2, 1), pl = num(p, 4, 0);
  const pt = num(p, 14, pl);
  return { kw, kh: num(p, 11, kw), sw, sh: num(p, 13, sw), dw, dh: num(p, 12, dw), pl, pr: num(p, 15, pl), pt, pb: num(p, 16, pt) };
}

type Inferred = { outs: (Shape | undefined)[]; params: number; cost: Cost; canon: string; wShape?: Shape; attrs?: Record<string, unknown> };

function infer(l: NcnnLayer, inS: (Shape | undefined)[], opts: NcnnOptions): Inferred {
  const p = l.p;
  const x = inS[0];
  const same = (canon: string, perElem = 1): Inferred => ({ outs: l.outs.map(() => x), params: 0, cost: elemCost(x, perElem), canon });
  const rank = x?.length ?? 0;
  switch (l.type) {
    case "Input": {
      const s = dimsShape(num(p, 0, 0), num(p, 1, 0), num(p, 11, 0), num(p, 2, 0)) ?? (opts.inputShape?.length ? opts.inputShape : undefined);
      return { outs: [s], params: 0, cost: NO_COST, canon: "Input" };
    }
    case "MemoryData": {
      const s = dimsShape(num(p, 0, 0), num(p, 1, 0), num(p, 11, 0), num(p, 2, 0));
      return { outs: [s], params: numel(s), cost: NO_COST, canon: "Constant", wShape: s };
    }
    case "Convolution": case "ConvolutionDepthWise": case "Convolution1D": {
      const g = convGeo(p);
      const nout = num(p, 0, 0);
      const wds = num(p, 6, 0);
      const group = l.type === "ConvolutionDepthWise" ? num(p, 7, 1) : 1;
      const params = wds + (num(p, 5, 0) ? nout : 0);
      let out: Shape | undefined;
      if (x && l.type === "Convolution1D" && rank === 2) out = [nout, convOut(x[1], g.kw, g.sw, g.dw, g.pl, g.pr)];
      else if (x && rank === 3) out = [nout, convOut(x[1], g.kh, g.sh, g.dh, g.pt, g.pb), convOut(x[2], g.kw, g.sw, g.dw, g.pl, g.pr)];
      const kArea = l.type === "Convolution1D" ? g.kw : g.kw * g.kh;
      const perOut = nout > 0 && kArea > 0 ? wds / nout : 0; // = (cin/group)·kArea
      const act = num(p, 9, 0);
      const cost = linearCost(out, perOut);
      const wShape = nout && kArea ? (l.type === "Convolution1D" ? [nout, wds / nout / kArea, g.kw] : [nout, wds / nout / kArea, g.kh, g.kw]) : [wds];
      return {
        outs: [out], params, cost: act && out ? { ...cost, flops: cost.flops + numel(out) } : cost, canon: "Conv", wShape,
        attrs: { group, ...(act ? { activation: ACTIVATIONS[act] ?? act } : {}) },
      };
    }
    case "Deconvolution": case "DeconvolutionDepthWise": {
      const g = convGeo(p);
      const nout = num(p, 0, 0);
      const wds = num(p, 6, 0);
      const params = wds + (num(p, 5, 0) ? nout : 0);
      let out: Shape | undefined;
      if (x && rank === 3) {
        const ow = num(p, 20, 0), oh = num(p, 21, ow);
        const opr = num(p, 18, 0), opb = num(p, 19, opr);
        const full = (i: number, k: number, s: number, d: number, pa: number, pb: number, op: number) =>
          i < 0 ? -1 : (i - 1) * s + d * (k - 1) + 1 - (pa > 0 ? pa : 0) - (pb > 0 ? pb : 0) + op;
        out = [nout, oh > 0 ? oh : full(x[1], g.kh, g.sh, g.dh, g.pt, g.pb, opb), ow > 0 ? ow : full(x[2], g.kw, g.sw, g.dw, g.pl, g.pr, opr)];
      }
      // Every input pixel scatters into kArea·cout/group outputs: MACs = numel(in)·wds/cin.
      const cin = x?.[0] ?? 0;
      const macs = x && cin > 0 ? numel(x) * (wds / cin) : 0;
      return { outs: [out], params, cost: { macs, flops: 2 * macs }, canon: "ConvTranspose", wShape: [wds] };
    }
    case "InnerProduct": {
      const nout = num(p, 0, 0);
      const wds = num(p, 2, 0);
      const params = wds + (num(p, 1, 0) ? nout : 0);
      const out: Shape = x && rank === 2 && x[1] * nout === wds ? [x[0], nout] : [nout];
      const k = nout > 0 ? wds / nout : 0;
      return { outs: [out], params, cost: linearCost(out, k), canon: "Gemm", wShape: nout ? [nout, k] : [wds] };
    }
    case "Pooling": {
      const type = num(p, 0, 0);
      const canonP = type === 0 ? "MaxPool" : "AveragePool";
      if (num(p, 4, 0)) {
        return { outs: [x ? [x[0]] : undefined], params: 0, cost: elemCost(x), canon: type === 0 ? "GlobalMaxPool" : "GlobalAveragePool" };
      }
      if (!x || rank !== 3) return { outs: [undefined], params: 0, cost: NO_COST, canon: canonP };
      if (num(p, 7, 0)) {
        const ow = num(p, 8, 0), oh = num(p, 18, ow);
        const out: Shape = [x[0], oh, ow];
        return { outs: [out], params: 0, cost: elemCost(x), canon: canonP };
      }
      const kw = num(p, 1, 0), kh = num(p, 11, kw), sw = num(p, 2, 1), sh = num(p, 12, sw);
      const pl = num(p, 3, 0), pr = num(p, 14, pl), pt = num(p, 13, pl), pb = num(p, 15, pt);
      const mode = num(p, 5, 0);
      const dim = (i: number, k: number, s: number, a: number, b: number) => {
        if (i < 0) return -1;
        if (mode === 2 || mode === 3) return Math.ceil(i / s);
        const span = i + a + b - k;
        return (mode === 0 ? Math.ceil(span / s) : Math.floor(span / s)) + 1;
      };
      const out: Shape = [x[0], dim(x[1], kh, sh, pt, pb), dim(x[2], kw, sw, pl, pr)];
      return { outs: [out], params: 0, cost: elemCost(out, kw * kh), canon: canonP };
    }
    case "BatchNorm": return { ...same("BatchNormalization", 2), params: 4 * num(p, 0, 0), wShape: [num(p, 0, 0)] };
    case "Scale": {
      const n = num(p, 0, 0);
      return { ...same("Mul", num(p, 1, 0) ? 2 : 1), params: n > 0 ? n * (num(p, 1, 0) ? 2 : 1) : 0, wShape: n > 0 ? [n] : undefined };
    }
    case "PReLU": return { ...same("PRelu"), params: num(p, 0, 0), wShape: [num(p, 0, 0)] };
    case "LayerNorm": return { ...same("LayerNormalization", 5), params: num(p, 2, 1) ? 2 * num(p, 0, 0) : 0 };
    case "InstanceNorm": return { ...same("InstanceNormalization", 5), params: num(p, 2, 1) ? 2 * num(p, 0, 0) : 0 };
    case "GroupNorm": return { ...same("GroupNormalization", 5), params: num(p, 3, 1) ? 2 * num(p, 1, 0) : 0 };
    case "ReLU": return same(num(p, 0, 0) ? "LeakyRelu" : "Relu");
    case "Clip": return same("Clip");
    case "Sigmoid": return same("Sigmoid");
    case "TanH": return same("Tanh");
    case "ELU": return same("Elu");
    case "GELU": return same("Gelu", 8);
    case "Swish": case "Mish": case "HardSwish": case "HardSigmoid": case "SELU": case "Softplus": case "AbsVal": case "BNLL":
    case "Exp": case "Log": case "Power": case "Threshold": case "Erf":
      return same(l.type, 2);
    case "Dropout": case "Noop": case "Cast": case "DeepCopy": case "Packing": case "Quantize": case "Dequantize": case "Requantize":
      return same(l.type === "Dropout" ? "Identity" : l.type, l.type === "Dropout" ? 0 : 1);
    case "Softmax": return same("Softmax", 3);
    case "UnaryOp": return { ...same(UNARY_OPS[num(p, 0, 0)] ?? "UnaryOp"), attrs: { op: UNARY_OPS[num(p, 0, 0)] } };
    case "BinaryOp": {
      const op = BINARY_OPS[num(p, 0, 0)] ?? "BinaryOp";
      let out = x;
      const y = inS[1];
      if (!num(p, 1, 0) && x && y) out = x.length >= y.length ? x : y;
      return { outs: [out], params: 0, cost: elemCost(out), canon: op, attrs: { op } };
    }
    case "Eltwise": {
      const op = ["Prod", "Sum", "Max"][num(p, 0, 1)] ?? "Eltwise";
      return { outs: [x], params: 0, cost: elemCost(x, Math.max(1, l.ins.length - 1)), canon: op === "Sum" ? "Add" : op === "Prod" ? "Mul" : "Max", attrs: { op } };
    }
    case "Split": return { outs: l.outs.map(() => x), params: 0, cost: NO_COST, canon: "Split" };
    case "Flatten": return { outs: [x ? [numel(x)] : undefined], params: 0, cost: NO_COST, canon: "Flatten" };
    case "Reshape": {
      if (!x) return { outs: [undefined], params: 0, cost: NO_COST, canon: "Reshape" };
      const w = num(p, 0, -233), h = num(p, 1, -233), d = num(p, 11, -233), c = num(p, 2, -233);
      const fromIn = (k: number) => x[x.length - 1 - k] ?? 1; // ncnn axis k: 0=w 1=h 2=d/c
      const spec: number[] = c !== -233 ? (d !== -233 ? [c, d, h, w] : [c, h, w]) : h !== -233 ? [h, w] : [w];
      const out = spec.map((v, i) => (v === 0 ? fromIn(spec.length - 1 - i) : v));
      const known = out.filter((v) => v > 0).reduce((a, b) => a * b, 1);
      const inferIdx = out.indexOf(-1);
      if (inferIdx >= 0) out[inferIdx] = known > 0 ? numel(x) / known : -1;
      return { outs: [out], params: 0, cost: NO_COST, canon: "Reshape" };
    }
    case "Concat": {
      const shapes = inS.filter((s): s is Shape => !!s);
      if (!x || shapes.length !== inS.length) return { outs: [undefined], params: 0, cost: NO_COST, canon: "Concat" };
      let axis = num(p, 0, 0);
      if (axis < 0) axis += rank;
      const out = [...x];
      out[axis] = shapes.reduce((s, sh) => s + (sh[axis] ?? 0), 0);
      return { outs: [out], params: 0, cost: NO_COST, canon: "Concat" };
    }
    case "Slice": {
      const slices = arr(p, 0);
      let axis = num(p, 1, 0);
      if (!x || !slices) return { outs: l.outs.map(() => undefined), params: 0, cost: NO_COST, canon: "Split" };
      if (axis < 0) axis += rank;
      const total = x[axis];
      const fixed = slices.filter((v) => v !== -233).reduce((a, b) => a + b, 0);
      const rest = slices.filter((v) => v === -233).length;
      const outs = slices.map((v) => {
        const s = [...x];
        s[axis] = v === -233 ? Math.floor((total - fixed) / Math.max(1, rest)) : v;
        return s;
      });
      return { outs, params: 0, cost: NO_COST, canon: "Split" };
    }
    case "Interp": {
      if (!x || rank !== 3) return { outs: [undefined], params: 0, cost: NO_COST, canon: "Resize" };
      const hs = num(p, 1, 1), ws = num(p, 2, 1), oh = num(p, 3, 0), ow = num(p, 4, 0);
      const out: Shape = [x[0], oh > 0 ? oh : Math.floor(x[1] * hs), ow > 0 ? ow : Math.floor(x[2] * ws)];
      return { outs: [out], params: 0, cost: elemCost(out, 4), canon: "Resize" };
    }
    case "Padding": {
      if (!x || rank !== 3) return { outs: [undefined], params: 0, cost: NO_COST, canon: "Pad" };
      const out: Shape = [x[0] + num(p, 7, 0) + num(p, 8, 0), x[1] + num(p, 0, 0) + num(p, 1, 0), x[2] + num(p, 2, 0) + num(p, 3, 0)];
      return { outs: [out], params: 0, cost: NO_COST, canon: "Pad" };
    }
    case "Embed": {
      const nout = num(p, 0, 0), wds = num(p, 3, 0);
      return { outs: [x ? [...x, nout] : undefined], params: wds + (num(p, 2, 0) ? nout : 0), cost: NO_COST, canon: "Gather", wShape: [num(p, 1, 0), nout] };
    }
    case "LSTM": case "GRU": case "RNN": {
      const H = num(p, 0, 0), wds = num(p, 1, 0), dirs = num(p, 2, 0) === 2 ? 2 : 1;
      const gates = l.type === "LSTM" ? 4 : l.type === "GRU" ? 3 : 1;
      const params = wds + dirs * (gates * H * H + (l.type === "GRU" ? 4 : gates) * H);
      const out: Shape | undefined = x && rank === 2 ? [x[0], H * dirs] : undefined;
      const T = x?.[0] ?? 1;
      const macs = T * (wds + dirs * gates * H * H);
      return { outs: [out, ...l.outs.slice(1).map(() => undefined)], params, cost: { macs, flops: 2 * macs }, canon: l.type };
    }
    case "MultiHeadAttention": {
      const e = num(p, 0, 0), wds = num(p, 2, 0);
      const params = 4 * wds + 4 * e;
      const T = x?.[0] ?? 0;
      const macs = T * 4 * wds + 2 * T * T * e;
      return { outs: [x], params, cost: { macs, flops: 2 * macs }, canon: "Attention" };
    }
    case "Gemm": case "MatMul": {
      const y = inS[1];
      if (x && y && x.length >= 2 && y.length >= 2) {
        const out = [...x.slice(0, -1), y[y.length - 1]];
        return { outs: [out], params: 0, cost: linearCost(out, x[x.length - 1]), canon: l.type === "Gemm" ? "Gemm" : "MatMul" };
      }
      return { outs: [undefined], params: 0, cost: NO_COST, canon: l.type === "Gemm" ? "Gemm" : "MatMul" };
    }
    default:
      return { outs: l.outs.map(() => undefined), params: 0, cost: NO_COST, canon: l.type };
  }
}

// ── .bin weights ─────────────────────────────────────────────────────────────

class WeightCursor {
  private o = 0;
  ok = true;
  constructor(private bin: Uint8Array) {}
  private dv = () => new DataView(this.bin.buffer, this.bin.byteOffset, this.bin.byteLength);
  private take(n: number): Uint8Array | null {
    if (!this.ok || this.o + n > this.bin.length) { this.ok = false; return null; }
    const b = this.bin.subarray(this.o, this.o + n);
    this.o += n;
    return b;
  }
  /** mb.load(n, 1): raw float32, no header. */
  raw(n: number): Float32Array | null {
    const b = this.take(n * 4);
    if (!b) return null;
    const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = v.getFloat32(i * 4, true);
    return out;
  }
  /** mb.load(n, 0): 4-byte storage tag, then fp32 / fp16 / int8 / quantized-table data. */
  flagged(n: number): { values: ArrayLike<number> | null; dt: string } {
    const tagB = this.take(4);
    if (!tagB) return { values: null, dt: "float32" };
    const tag = this.dv().getUint32(this.o - 4, true);
    const align4 = (b: number) => (b + 3) & ~3;
    if (tag === 0x01306b47) { // float16
      const b = this.take(align4(n * 2));
      if (!b) return { values: null, dt: "float16" };
      const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = fromHalf(v.getUint16(i * 2, true));
      return { values: out, dt: "float16" };
    }
    if (tag === 0x01348b83) { // bfloat16
      const b = this.take(align4(n * 2));
      if (!b) return { values: null, dt: "bfloat16" };
      const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
      const out = new Float32Array(n);
      const t = new DataView(new ArrayBuffer(4));
      for (let i = 0; i < n; i++) { t.setUint32(0, v.getUint16(i * 2, true) << 16, true); out[i] = t.getFloat32(0, true); }
      return { values: out, dt: "bfloat16" };
    }
    if (tag === 0x000d4b38) { // int8
      const b = this.take(align4(n));
      return { values: b ? Int8Array.from(b.subarray(0, n), (u) => (u << 24) >> 24) : null, dt: "int8" };
    }
    if (tag === 0x0002c056 || tag === 0) return { values: this.raw(n), dt: "float32" };
    // Non-zero tag bytes: 256-entry float table + uint8 indices.
    const table = this.raw(256);
    const idx = this.take(align4(n));
    if (!table || !idx) return { values: null, dt: "uint8" };
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = table[idx[i]];
    return { values: out, dt: "uint8" };
  }
}

/** Read this layer's weights from the cursor; returns the main weight values or null. */
function readWeights(l: NcnnLayer, c: WeightCursor): { values: ArrayLike<number>; dt: string } | null {
  if (!c.ok) return null;
  const p = l.p;
  switch (l.type) {
    case "Convolution": case "ConvolutionDepthWise": case "Convolution1D": case "Deconvolution": case "DeconvolutionDepthWise": {
      if (num(p, 19, 0) || (l.type.startsWith("Deconv") && num(p, 28, 0))) return null; // dynamic weight
      const nout = num(p, 0, 0);
      const w = c.flagged(num(p, 6, 0));
      if (num(p, 5, 0)) c.raw(nout);
      if (num(p, 8, 0)) c.ok = false; // int8 scale layout varies; stop reading further layers
      return w.values ? { values: w.values, dt: w.dt } : null;
    }
    case "InnerProduct": {
      const nout = num(p, 0, 0);
      const w = c.flagged(num(p, 2, 0));
      if (num(p, 1, 0)) c.raw(nout);
      if (num(p, 8, 0)) c.ok = false;
      return w.values ? { values: w.values, dt: w.dt } : null;
    }
    case "Embed": {
      const w = c.flagged(num(p, 3, 0));
      if (num(p, 2, 0)) c.raw(num(p, 0, 0));
      return w.values ? { values: w.values, dt: w.dt } : null;
    }
    case "BatchNorm": {
      const n = num(p, 0, 0);
      const slope = c.raw(n); c.raw(n); c.raw(n); c.raw(n);
      return slope ? { values: slope, dt: "float32" } : null;
    }
    case "Scale": {
      const n = num(p, 0, 0);
      if (n === -233) return null;
      const s = c.raw(n);
      if (num(p, 1, 0)) c.raw(n);
      return s ? { values: s, dt: "float32" } : null;
    }
    case "PReLU": { const s = c.raw(num(p, 0, 0)); return s ? { values: s, dt: "float32" } : null; }
    case "LayerNorm": case "InstanceNorm": case "GroupNorm": {
      if (!num(p, l.type === "GroupNorm" ? 3 : 2, 1)) return null;
      const n = num(p, l.type === "GroupNorm" ? 1 : 0, 0);
      const g = c.raw(n); c.raw(n);
      return g ? { values: g, dt: "float32" } : null;
    }
    case "MemoryData": {
      const n = Math.max(1, num(p, 0, 0)) * Math.max(1, num(p, 1, 0)) * Math.max(1, num(p, 11, 0)) * Math.max(1, num(p, 2, 0));
      const s = c.raw(n);
      return s ? { values: s, dt: "float32" } : null;
    }
    default:
      // Unknown weight-bearing layers would desynchronize the stream: stop.
      if (["LSTM", "GRU", "RNN", "MultiHeadAttention", "Gemm", "Padding", "Bias", "Normalize", "Convolution3D", "ConvolutionDepthWise1D", "ConvolutionDepthWise3D", "Deconvolution1D", "Deconvolution3D", "DeformableConv2D", "RMSNorm"].includes(l.type)) c.ok = false;
      return null;
  }
}

function groupFor(type: string, canon: string): ModelLayer["group"] | undefined {
  if (type === "Input") return "input";
  if (/Softmax|InnerProduct|Gemm|DetectionOutput|Yolo/.test(type)) return "head";
  if (/Pooling|Concat|Reshape|Flatten|Interp|Permute|Slice|Split|Crop|Padding/.test(type)) return "neck";
  return canon === "Conv" || canon === "ConvTranspose" ? "backbone" : undefined;
}

/** Parse an ncnn `.param` (text) or `.param.bin` (binary) graph. */
export function parseNcnn(input: ArrayBuffer | Uint8Array | string, name = "model.param", opts: NcnnOptions = {}): Model {
  let buf: Uint8Array;
  let raw: NcnnLayer[];
  let binary = false;
  if (typeof input === "string") {
    buf = new TextEncoder().encode(input);
    raw = parseText(input);
  } else {
    buf = input instanceof Uint8Array ? input : new Uint8Array(input);
    binary = buf.length >= 4 && new DataView(buf.buffer, buf.byteOffset, 4).getInt32(0, true) === MAGIC_BIN;
    raw = binary ? parseBinary(buf) : parseText(new TextDecoder().decode(buf));
  }

  const shapes = new Map<string, Shape | undefined>();
  const producer = new Map<string, number>();
  const consumed = new Set<string>();
  for (const l of raw) for (const b of l.ins) consumed.add(b);
  const cursor = opts.bin ? new WeightCursor(opts.bin) : null;
  const edges = new EdgeSet();
  const layers: ModelLayer[] = [];

  raw.forEach((l, id) => {
    const inS = l.ins.map((b) => shapes.get(b));
    const r = infer(l, inS, opts);
    l.outs.forEach((b, k) => { shapes.set(b, r.outs[k]); producer.set(b, id); });
    for (const b of l.ins) edges.add(producer.get(b), id);

    const wv = cursor ? readWeights(l, cursor) : null;
    const attr: Record<string, unknown> = {};
    const names = NAMES[l.type] ?? {};
    for (const [k, v] of l.p) attr[names[k] ?? String(k)] = v;
    Object.assign(attr, r.attrs ?? {});
    const out0 = r.outs[0];
    const isOutput = l.outs.length > 0 && l.outs.every((b) => !consumed.has(b));
    const w = r.params > 0 || wv ? weightStats(r.wShape ?? [r.params], wv?.values ?? null) : null;
    if (w && r.params > 0 && !wv) w.size = r.params;

    layers.push(makeLayer({
      id, name: l.name, type: l.type, canon: r.canon, shape: out0,
      dt: wv?.dt && wv.dt !== "float32" ? wv.dt : "float32",
      params: r.params, cost: r.cost,
      group: isOutput && l.type !== "Input" ? "output" : groupFor(l.type, r.canon),
      ins: l.ins.map((b) => ({ n: b, s: shapes.get(b) })),
      outs: l.outs.map((b, k) => ({ n: b, s: r.outs[k] })),
      attr, w, framework: FRAMEWORK,
    }));
  });

  const inputLayer = layers.find((l) => l.type === "Input");
  const outLayer = [...layers].reverse().find((l) => l.group === "output") ?? layers[layers.length - 1];
  const inShape = inputLayer?.outs[0]?.s;
  const outShape = outLayer?.outs[0]?.s;
  return {
    name, format: binary ? "ncnn (.param.bin)" : "ncnn", framework: FRAMEWORK, opset: 0,
    sizeBytes: buf.byteLength + (opts.bin?.byteLength ?? 0),
    inputShape: cleanShape(inShape), outputShape: cleanShape(outShape),
    layers, edges: edges.edges,
    producer: `${layers.length} layers, ${shapes.size} blobs`,
    doc: inShape ? undefined : "Input layer has no dims; pass opts.inputShape to propagate shapes.",
  };
}
