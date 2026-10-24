import type { Model, ModelLayer } from "../types";
import { FlatBuf } from "../common/flatbuffers";
import {
  DYN, EdgeSet, NO_COST, broadcast, cleanShape, elemCost, linearCost, makeLayer, numel, weightStats,
  type Cost, type Shape,
} from "../common/graphKit";

// Alibaba MNN `.mnn` models are FlatBuffers with root table `Net`
// (schema/default/MNN.fbs). Field indices below are declaration order; a union
// takes two slots (`main_type`, then `main`).
//   Net: bizCode 0, extraTensorDescribe 1, extraInfo 2, oplists 3, outputName 4,
//        preferForwardType 5, sourceType 6, tensorName 7, tensorNumber 8, usage 9,
//        subgraphs 10, mnn_uuid 11
//   Op:  inputIndexes 0, main_type 1, main 2, name 3, outputIndexes 4, type 5,
//        defaultDimentionFormat 6
// Edges come from tensor indexes; shapes propagate from the Input dims, with a
// small integer constant-folder for the Shape/Slice/Concat subgraphs ONNX
// converters emit to feed Reshape.

const FRAMEWORK = "MNN";

const OP_SEQ = (
  "AbsVal QuantizedAdd ArgMax AsString InstanceNorm BatchToSpaceND Copy BinaryOp Bnll Cast Concat Const Convolution " +
  "ConvolutionDepthwise Crop CropAndResize ImageProcess Deconvolution DeconvolutionDepthwise Dequantize DetectionOutput " +
  "Dropout Eltwise ELU Unique Exp ExpandDims Fill Flatten Im2Col Gather GatherV2 Im2Seq InnerProduct Input Interp Log " +
  "LRN LSTM MatMul MoE NonMaxSuppression NonMaxSuppressionV2 Normalize Pack Padding Permute Pooling Power PReLU PriorBox " +
  "Proposal QuantizedAvgPool QuantizedBiasAdd QuantizedConcat QuantizedDepthwiseConv2D QuantizedLogistic " +
  "RasterAndInterpolate QuantizedMaxPool Texture RasterDiff QuantizedReshape QuantizedSoftmax QuantizeMaxMin QuantizeV2 " +
  "Range Rank ReduceJoin Reduction ReLU ReLU6 RequantizationRange Requantize Reshape Resize RNN ROIPooling Scale Selu " +
  "Seq2Out Shape Sigmoid Size Slice SliceTf Softmax SpaceToBatchND SpatialProduct Col2Im Segment Squeeze StridedSlice " +
  "CastLike StringSplit StringToNumber TanH TfQuantizedConv2D Threshold Tile TopKV2 Transpose UnaryOp Unpack Where " +
  "Moments RNNSequenceGRU BatchMatMul Unsqueeze CosineSimilarity DepthToSpace SpaceToDepth ReverseSequence Pooling3D " +
  "Convolution3D MatrixBandPart GatherND DetectionPostProcess UnravelIndex ScatterNd OneHot BroadcastTo Dilation2D Interp3D"
).split(" ");
const OP_EXTRA: Record<number, string> = {
  128: "Raster", 129: "ConvertTensor", 130: "ArgMin", 131: "LinSpace", 132: "RandomUniform", 133: "TensorArray",
  141: "LSTMBlockCell", 142: "Reverse", 143: "ROIAlign", 144: "RandomNormal", 147: "EyeLike", 148: "CumSum", 149: "Det",
  150: "CumProd", 151: "ScatterElements", 152: "GatherElements", 153: "Svd", 154: "Histogram", 155: "DynamicQuant", 156: "Stft",
  256: "Plugin", 257: "Select", 258: "ZerosLike", 259: "Broastcast", 260: "SetDiff1D", 261: "ReluGrad", 262: "Identity",
  263: "PoolGrad", 264: "SoftmaxGrad", 265: "Conv2DBackPropFilter", 266: "TrainableParam", 267: "BatchNorm",
  268: "ConvTranspose3D", 269: "ZeroGrad", 299: "Attention", 300: "FmhaV2", 301: "Fmhca", 302: "SeqLen2Spatial",
  303: "SplitGeLU", 304: "GroupNorm", 305: "LinearAttention", 306: "RoPE", 307: "FusedLinear", 308: "GatedRMSNorm",
  512: "Extra", 513: "ConvInt8", 514: "Int8ToFloat", 515: "DepthwiseConvInt8", 517: "FloatToInt8", 518: "EltwiseInt8",
  600: "While", 601: "If", 603: "LayerNorm", 604: "GridSample",
};
const opName = (t: number) => OP_SEQ[t] ?? OP_EXTRA[t] ?? `OpType_${t}`;

// OpParameter union members (1-based; 0 = NONE).
const PARAM_UNION = (
  "QuantizedAdd ArgMax AsString Axis BatchNorm BinaryOp Blob CastParam Convolution2D Crop CropAndResize Dequantize " +
  "DetectionOutput Eltwise ExpandDims Fill Flatten Gather GatherV2 InnerProduct Input Interp LRN LSTM MatMul " +
  "NonMaxSuppressionV2 Normalize PackParam Permute Plugin Pool PRelu PriorBox Proposal QuantizedAvgPool QuantizedBiasAdd " +
  "QuantizedConcat QuantizedLogistic QuantizedMatMul QuantizedMaxPool QuantizedRelu QuantizedRelu6 QuantizedReshape " +
  "QuantizedSoftmax QuantizeMaxMin QuantizeV2 Range Rank ReduceJoin ReductionParam Relu Relu6 RequantizationRange " +
  "Requantize Reshape Resize RoiParameters Scale Selu Size Slice SliceTf SpaceBatch SqueezeParam StridedSliceParam " +
  "TensorConvertInfo TfQuantizedConv2D TopKV2 Transpose UnaryOp MomentsParam RNNParam BatchMatMulParam " +
  "QuantizedFloatParam DepthSpaceParam EltwiseInt8 ReverseSequenceParam Extra Pool3D Convolution3D ELU " +
  "DetectionPostProcessParam OneHotParam PadParam WhileParam IfParam RandomUniform LayerNorm TensorArray LSTMBlockCell " +
  "GridSample LoopParam ImageProcessParam CumSum GroupNorm FmhaV2Param FmhcaParam AttentionParam StftParam " +
  "LinearAttentionParam ShapeParam RoPEParam FusedLinearParam"
).split(" ");

const SOURCES = ["Caffe", "TensorFlow", "TFLite", "ONNX", "Torch"];
const DATA_TYPES: Record<number, string> = { 1: "float32", 2: "float64", 3: "int32", 4: "uint8", 5: "int16", 6: "int8", 9: "int64", 10: "bool", 14: "bfloat16", 17: "uint16", 19: "float16" };
const FORMATS = ["NCHW", "NHWC", "NC4HW4", "NHWC4", "UNKNOWN"];
const BINARY = ["Add", "Sub", "Mul", "Div", "MaxTemp", "MinTemp", "Pow", "RealDiv", "Min", "Max", "Greater", "GreaterEqual", "Less", "FloorDiv", "SquaredDifference", "Equal", "LessEqual", "FloorMod", "", "Mod", "Atan2", "LogicalOr", "NotEqual", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "LogicalXor", "LeftShift", "RightShift", "MulSilu"];
const POOL_PAD = ["CAFFE", "VALID", "SAME"];

const CANON: Record<string, string> = {
  Convolution: "Conv", ConvolutionDepthwise: "Conv", Convolution3D: "Conv", ConvInt8: "Conv", DepthwiseConvInt8: "Conv",
  Deconvolution: "ConvTranspose", DeconvolutionDepthwise: "ConvTranspose", InnerProduct: "Gemm", MatMul: "MatMul",
  BatchMatMul: "MatMul", ReLU: "Relu", ReLU6: "Clip", Sigmoid: "Sigmoid", TanH: "Tanh", Softmax: "Softmax",
  Concat: "Concat", Reshape: "Reshape", Flatten: "Flatten", Permute: "Transpose", Transpose: "Transpose",
  BatchNorm: "BatchNormalization", Scale: "Mul", Eltwise: "Add", Interp: "Resize", Resize: "Resize", Padding: "Pad",
  PReLU: "PRelu", ELU: "Elu", Selu: "Selu", LayerNorm: "LayerNormalization", GroupNorm: "GroupNormalization",
  InstanceNorm: "InstanceNormalization", Dropout: "Identity", Input: "Input",
};

type MOp = { name: string; type: string; ins: number[]; outs: number[]; param: string; main: number | null; fmt: number };

/** Structural check: a FlatBuffer whose root looks like an MNN Net. */
export function isMnn(buf: Uint8Array): boolean {
  try {
    if (buf.length < 16) return false;
    const fb = new FlatBuf(buf);
    const root = fb.root();
    if (root < 4 || root >= buf.length) return false;
    const ops = fb.tables(root, 3);
    if (ops.length === 0) return false;
    const names = fb.vecLen(root, 7);
    const t = fb.int32(ops[0], 5, -1);
    return names > 0 && t >= 0 && t < 1024;
  } catch {
    return false;
  }
}

function readOp(fb: FlatBuf, t: number): MOp {
  const mainType = fb.uint8(t, 1, 0);
  return {
    name: fb.str(t, 3),
    type: opName(fb.int32(t, 5, 0)),
    ins: fb.ints(t, 0),
    outs: fb.ints(t, 4),
    param: mainType > 0 ? PARAM_UNION[mainType - 1] ?? `Param_${mainType}` : "",
    main: mainType > 0 ? fb.table(t, 2) : null,
    fmt: fb.int8(t, 6, 1),
  };
}

const convDim = (i: number, k: number, s: number, d: number, pa: number, pb: number, mode: number): number => {
  if (i < 0) return DYN;
  if (mode === 2) return Math.ceil(i / s); // SAME
  const eff = d * (k - 1) + 1;
  if (mode === 1) return Math.floor((i - eff) / s) + 1; // VALID
  return Math.floor((i + pa + pb - eff) / s) + 1;
};

function permuteFormat(s: Shape | undefined, from: number, to: number): Shape | undefined {
  if (!s || s.length !== 4) return s;
  const nhwc = (f: number) => f === 1 || f === 3;
  if (nhwc(from) && !nhwc(to)) return [s[0], s[3], s[1], s[2]];
  if (!nhwc(from) && nhwc(to)) return [s[0], s[2], s[3], s[1]];
  return s;
}

/** Parse an MNN `.mnn` FlatBuffer into the normalized graph. */
export function parseMnn(input: ArrayBuffer | Uint8Array, name = "model.mnn"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const fb = new FlatBuf(buf);
  const net = fb.root();
  const opTables = fb.tables(net, 3);
  if (!opTables.length) throw new Error("MNN: no operators (not an MNN Net FlatBuffer).");
  const tensorNames = fb.strings(net, 7);
  const outputNames = new Set(fb.strings(net, 4));
  const source = SOURCES[fb.int8(net, 6, 0)] ?? "unknown";
  const biz = fb.str(net, 0);
  const extra = fb.table(net, 2);
  const version = extra != null ? fb.str(extra, 2) : "";

  const ops = opTables.map((t) => readOp(fb, t));
  const tname = (i: number) => tensorNames[i] || `t${i}`;
  const shapes = new Map<number, Shape | undefined>();
  const values = new Map<number, number[]>(); // folded integer tensors
  const fmts = new Map<number, number>();
  const producer = new Map<number, number>();
  const consumed = new Set<number>();
  for (const o of ops) for (const i of o.ins) consumed.add(i);

  // Const ops are folded into their consumers (weights / shape operands).
  const constOf = new Map<number, { shape: Shape; dt: string; vals: ArrayLike<number> | null }>();
  const layers: ModelLayer[] = [];
  const edges = new EdgeSet();
  const layerOfOp: number[] = [];

  for (const op of ops) {
    const m = op.main;
    const inS = op.ins.map((i) => shapes.get(i));
    const x = inS[0];
    const attr: Record<string, unknown> = {};
    let out: Shape | undefined;
    let outVals: number[] | undefined;
    let params = 0;
    let cost: Cost = NO_COST;
    let wShape: Shape | undefined;
    let wVals: ArrayLike<number> | null = null;
    let dt = "float32";
    let canon = CANON[op.type] ?? op.type;
    let outFmt = x ? fmts.get(op.ins[0]) ?? 0 : op.fmt;

    if (op.type === "Const" || (op.type === "TrainableParam" && op.param === "Blob")) {
      if (m != null && op.param === "Blob") {
        const dims = fb.ints(m, 0);
        const bdt = fb.int32(m, 2, 1);
        const ints = fb.ints(m, 5);
        const floats = fb.floats(m, 7);
        const vals: ArrayLike<number> | null = ints.length ? ints : floats ?? (fb.has(m, 6) ? fb.longs(m, 6) : null);
        const shape = dims.length ? dims : vals && vals.length > 1 ? [vals.length] : [];
        for (const o of op.outs) {
          shapes.set(o, shape);
          constOf.set(o, { shape, dt: DATA_TYPES[bdt] ?? "float32", vals });
          if (ints.length || (bdt === 3 || bdt === 9)) values.set(o, Array.from(vals ?? []));
          fmts.set(o, fb.int8(m, 1, 0));
        }
      }
      continue;
    }

    const constIns = op.ins.filter((i) => constOf.has(i));
    const constParams = constIns.reduce((s, i) => {
      const c = constOf.get(i)!;
      return s + (values.has(i) && !/float/.test(c.dt) ? 0 : numel(c.shape));
    }, 0);

    switch (op.type) {
      case "Input": {
        if (m != null) {
          const dims = fb.ints(m, 0);
          out = dims.map((d) => (d > 0 ? d : DYN));
          dt = DATA_TYPES[fb.int32(m, 1, 1)] ?? "float32";
          outFmt = fb.int8(m, 2, 2);
          attr.dformat = FORMATS[outFmt];
        }
        break;
      }
      case "ConvertTensor": {
        const from = m != null ? fb.int8(m, 0, 0) : 0;
        const to = m != null ? fb.int8(m, 1, 0) : 0;
        out = permuteFormat(x, from, to);
        outFmt = to;
        attr.source = FORMATS[from]; attr.dest = FORMATS[to];
        canon = "Identity";
        break;
      }
      case "Convolution": case "ConvolutionDepthwise": case "Deconvolution": case "DeconvolutionDepthwise":
      case "ConvInt8": case "DepthwiseConvInt8": {
        if (m == null || op.param !== "Convolution2D") break;
        const c = fb.table(m, 0);
        if (c == null) break;
        const kx = fb.int32(c, 2, 1), ky = fb.int32(c, 3, 1), sx = fb.int32(c, 4, 1), sy = fb.int32(c, 5, 1);
        const dx = fb.int32(c, 6, 1), dy = fb.int32(c, 7, 1), mode = fb.int8(c, 8, 0);
        const group = Math.max(1, fb.int32(c, 9, 1)), cout = fb.int32(c, 10, 0);
        let cin = fb.int32(c, 11, 0);
        if (!cin && x && x.length === 4 && x[1] > 0) cin = x[1];
        const pads = fb.ints(c, 14);
        const px = fb.int32(c, 0, 0), py = fb.int32(c, 1, 0);
        const [pt, pl, pb, pr] = pads.length >= 4 ? pads : [py, px, py, px];
        Object.assign(attr, { kernel: [ky, kx], stride: [sy, sx], dilation: [dy, dx], pads: [pt, pl, pb, pr], padMode: ["CAFFE", "VALID", "SAME"][mode], group, outputCount: cout, inputCount: cin });
        if (fb.bool(c, 12)) attr.relu = true;
        if (fb.bool(c, 13)) attr.relu6 = true;
        const deconv = op.type.startsWith("Deconv");
        if (x && x.length === 4) {
          if (deconv) {
            const full = (i: number, k: number, s: number, d: number, a: number, b: number) => (i < 0 ? DYN : (i - 1) * s + d * (k - 1) + 1 - a - b);
            out = [x[0], cout, full(x[2], ky, sy, dy, pt, pb), full(x[3], kx, sx, dx, pl, pr)];
          } else {
            out = [x[0], cout, convDim(x[2], ky, sy, dy, pt, pb, mode), convDim(x[3], kx, sx, dx, pl, pr, mode)];
          }
        }
        const weight = fb.floats(m, 1);
        const bias = fb.floats(m, 2);
        const quant = fb.table(m, 3);
        const perOut = (cin / group) * kx * ky;
        const expectedW = cout * perOut;
        params = (weight?.length || expectedW) + (bias?.length ?? cout);
        params += constParams;
        if (quant != null) { dt = "int8"; attr.quantized = true; }
        wShape = deconv ? [cin, cout / group, ky, kx] : [cout, cin / group, ky, kx];
        wVals = weight && weight.length ? weight : null;
        cost = deconv && x ? { macs: numel(x) * (cout / group) * kx * ky, flops: 2 * numel(x) * (cout / group) * kx * ky } : linearCost(out, perOut);
        break;
      }
      case "Convolution3D": {
        if (m == null) break;
        const c = fb.table(m, 0);
        const weight = fb.floats(m, 1), bias = fb.floats(m, 2);
        params = (weight?.length ?? 0) + (bias?.length ?? 0);
        if (c != null) {
          const k = fb.ints(c, 2), cin = fb.int32(c, 5, 0), cout = fb.int32(c, 6, 0), g = Math.max(1, fb.int32(c, 9, 1));
          attr.kernels = k; attr.strides = fb.ints(c, 1); attr.outputCount = cout;
          if (!params) params = cout * (cin / g) * k.reduce((a, b) => a * b, 1) + cout;
        }
        wVals = weight;
        break;
      }
      case "InnerProduct": {
        if (m == null) break;
        const nout = fb.int32(m, 0, 0);
        const weight = fb.floats(m, 3), bias = fb.floats(m, 4);
        const ws = fb.int32(m, 2, 0) || weight?.length || 0;
        params = ws + (bias?.length ?? (fb.int32(m, 1, 0) ? nout : 0));
        out = x ? [x[0], nout] : [1, nout];
        cost = linearCost(out, nout ? ws / nout : 0);
        wShape = [nout, nout ? ws / nout : 0];
        wVals = weight;
        break;
      }
      case "MatMul": case "BatchMatMul": {
        const y = inS[1];
        const ta = m != null && op.param === "MatMul" && fb.bool(m, 1);
        const tb = m != null && op.param === "MatMul" && fb.bool(m, 2);
        if (x && y && x.length >= 2 && y.length >= 2) {
          const M = ta ? x[x.length - 1] : x[x.length - 2];
          const K = ta ? x[x.length - 2] : x[x.length - 1];
          const N = tb ? y[y.length - 2] : y[y.length - 1];
          out = [...(x.length >= y.length ? x : y).slice(0, -2), M, N];
          cost = linearCost(out, K);
        }
        params = constParams;
        const ci = constIns.find((i) => constOf.get(i)!.vals);
        if (ci != null) { wShape = constOf.get(ci)!.shape; wVals = constOf.get(ci)!.vals; }
        break;
      }
      case "Pooling": {
        if (m == null || !x || x.length !== 4) break;
        const global = fb.bool(m, 2);
        const kx = fb.int32(m, 3, 1), ky = fb.int32(m, 4, 1), sx = fb.int32(m, 5, 1), sy = fb.int32(m, 6, 1);
        const type = fb.int8(m, 7, 0), padType = fb.int8(m, 8, 0), ceil = fb.bool(m, 10, true);
        const pads = fb.ints(m, 11);
        const px = fb.int32(m, 0, 0), py = fb.int32(m, 1, 0);
        const [pt, pl, pb, pr] = pads.length >= 4 ? pads : [py, px, py, px];
        canon = global ? (type === 0 ? "GlobalMaxPool" : "GlobalAveragePool") : type === 0 ? "MaxPool" : "AveragePool";
        Object.assign(attr, { pool: type === 0 ? "max" : "avg", global, kernel: [ky, kx], stride: [sy, sx], padType: POOL_PAD[padType], ceilModel: ceil });
        if (global) { out = [x[0], x[1], 1, 1]; cost = elemCost(x); break; }
        const dim = (i: number, k: number, s: number, a: number, b: number) => {
          if (i < 0) return DYN;
          if (padType === 2) return Math.ceil(i / s);
          if (padType === 1) return Math.ceil((i - k + 1) / s);
          const span = (i + a + b - k) / s;
          return (ceil ? Math.ceil(span) : Math.floor(span)) + 1;
        };
        out = [x[0], x[1], dim(x[2], ky, sy, pt, pb), dim(x[3], kx, sx, pl, pr)];
        cost = elemCost(out, kx * ky);
        break;
      }
      case "Pooling3D": {
        if (m == null || !x) break;
        const global = fb.bool(m, 5);
        const type = fb.int8(m, 3, 0);
        canon = global ? (type === 0 ? "GlobalMaxPool" : "GlobalAveragePool") : type === 0 ? "MaxPool" : "AveragePool";
        attr.global = global; attr.pool = type === 0 ? "max" : "avg";
        if (global) { out = [x[0], x[1], ...x.slice(2).map(() => 1)]; cost = elemCost(x); }
        break;
      }
      case "BinaryOp": {
        const t = m != null ? fb.int32(m, 0, 0) : 0;
        const opn = BINARY[t] || `Binary_${t}`;
        attr.op = opn;
        canon = ["Add", "Sub", "Mul", "Div"].includes(opn) ? opn : opn === "Max" ? "Max" : opn;
        const y = inS[1];
        out = x && y ? broadcast(x, y) : x ?? y;
        cost = elemCost(out);
        params = constParams;
        const a = values.get(op.ins[0]), b = values.get(op.ins[1]);
        if (a && b) {
          const f = (p: number, q: number) => [p + q, p - q, p * q, Math.trunc(p / q)][t] ?? NaN;
          const n = Math.max(a.length, b.length);
          const r = Array.from({ length: n }, (_, i) => f(a[a.length === 1 ? 0 : i], b[b.length === 1 ? 0 : i]));
          if (r.every(Number.isFinite)) outVals = r;
        }
        break;
      }
      case "Eltwise": {
        const t = m != null ? fb.int8(m, 0, 0) : 1;
        attr.op = ["Prod", "Sum", "Max", "Sub"][t];
        canon = ["Mul", "Add", "Max", "Sub"][t] ?? "Add";
        out = x; cost = elemCost(x, Math.max(1, op.ins.length - 1));
        break;
      }
      case "ReLU": case "ReLU6": case "Sigmoid": case "TanH": case "ELU": case "Selu": case "Dropout": case "Cast": case "UnaryOp":
      case "Identity": case "Exp": case "Log": case "Power": case "AbsVal": case "Threshold": case "Int8ToFloat": case "FloatToInt8":
        out = x;
        if (op.type === "ReLU" && m != null && fb.float32(m, 0, 0) !== 0) { canon = "LeakyRelu"; attr.slope = fb.float32(m, 0, 0); }
        cost = op.type === "Dropout" || op.type === "Cast" || op.type === "Identity" ? NO_COST : elemCost(x);
        if (values.has(op.ins[0]) && (op.type === "Cast" || op.type === "Identity")) outVals = values.get(op.ins[0]);
        break;
      case "PReLU": {
        out = x; cost = elemCost(x);
        if (m != null) { const s = fb.floats(m, 1); params = s?.length ?? fb.int32(m, 0, 0); wVals = s; }
        break;
      }
      case "Softmax": out = x; cost = elemCost(x, 3); if (m != null) attr.axis = fb.int32(m, 0, 0); break;
      case "Scale": {
        out = x;
        if (m != null) {
          const s = fb.floats(m, 1), b = fb.floats(m, 2);
          params = (s?.length ?? 0) + (b?.length ?? 0);
          wVals = s; attr.channels = fb.int32(m, 0, 0);
        }
        cost = elemCost(x, 2);
        break;
      }
      case "BatchNorm": {
        out = x;
        if (m != null) {
          const ch = fb.int32(m, 0, 0);
          const s = fb.floats(m, 1);
          params = [1, 2, 3, 4].reduce((acc, f) => acc + (fb.vecLen(m, f)), 0) || 4 * ch;
          wVals = s; attr.channels = ch; attr.epsilon = fb.float32(m, 7, 0.001);
        }
        cost = elemCost(x, 2);
        break;
      }
      case "LayerNorm": case "GroupNorm": {
        out = x;
        if (m != null) {
          const g = fb.floats(m, op.type === "LayerNorm" ? 2 : 2), b = fb.floats(m, 3);
          params = (g?.length ?? 0) + (b?.length ?? 0);
          wVals = g;
        }
        cost = elemCost(x, 5);
        break;
      }
      case "Flatten": {
        if (!x) break;
        let axis = m != null ? fb.int32(m, 0, 1) : 1;
        if (axis < 0) axis += x.length;
        let end = m != null ? fb.int32(m, 1, 0) : 0;
        if (end <= 0) end += x.length;
        const pre = x.slice(0, axis).reduce((a, b) => a * b, 1);
        const mid = x.slice(axis, Math.max(end, axis + 1)).reduce((a, b) => a * b, 1);
        out = [pre, mid, ...x.slice(Math.max(end, axis + 1))];
        break;
      }
      case "Reshape": {
        const target = (m != null && op.param === "Reshape" ? fb.ints(m, 0) : []).length
          ? fb.ints(m!, 0)
          : values.get(op.ins[1]);
        if (target && x) {
          const o = target.map((d, i) => (d === 0 ? x[i] ?? 1 : d));
          const known = o.filter((d) => d > 0).reduce((a, b) => a * b, 1);
          const k = o.indexOf(-1);
          if (k >= 0) o[k] = x.every((d) => d > 0) ? numel(x) / known : DYN;
          out = o;
        }
        if (values.has(op.ins[0])) outVals = values.get(op.ins[0]);
        break;
      }
      case "Concat": {
        const axisRaw = m != null ? fb.int32(m, 0, 0) : 0;
        const vs = op.ins.map((i) => values.get(i));
        if (vs.every((v) => v)) outVals = vs.flatMap((v) => v!);
        const ss = inS.filter((s): s is Shape => !!s);
        if (x && ss.length === inS.length) {
          const axis = axisRaw < 0 ? axisRaw + x.length : axisRaw;
          out = [...x];
          out[axis] = ss.reduce((a, s) => a + (s[axis] ?? 0), 0);
        }
        break;
      }
      case "Permute": case "Transpose": {
        const perm = op.type === "Permute" && m != null ? fb.ints(m, 0) : values.get(op.ins[1]);
        if (x && perm && perm.length === x.length) out = perm.map((p) => x[p]);
        break;
      }
      case "Interp": {
        if (x && x.length === 4 && m != null) {
          const ws = fb.float32(m, 0, 0), hs = fb.float32(m, 1, 0), ow = fb.int32(m, 2, 0), oh = fb.int32(m, 3, 0);
          out = [x[0], x[1], oh || Math.floor(x[2] * hs) || DYN, ow || Math.floor(x[3] * ws) || DYN];
          cost = elemCost(out, 4);
        }
        break;
      }
      // ── Integer shape-computation ops (folded so Reshape targets resolve) ──
      case "Shape": {
        if (x) { outVals = permuteFormat(x, fmts.get(op.ins[0]) ?? 0, 0)!; out = [x.length]; }
        break;
      }
      case "Rank": if (x) { outVals = [x.length]; out = []; } break;
      case "Size": if (x) { outVals = [numel(x)]; out = []; } break;
      case "Unsqueeze": case "Squeeze": case "ExpandDims": {
        const v = values.get(op.ins[0]);
        if (v) outVals = v;
        if (x && m != null && op.param === "SqueezeParam") {
          const dims = fb.ints(m, 0);
          if (op.type === "Unsqueeze") {
            const o = [...x];
            for (const d of [...dims].sort((a, b) => a - b)) o.splice(d < 0 ? d + o.length + 1 : d, 0, 1);
            out = o;
          } else {
            out = x.filter((d, i) => !(dims.length ? dims.map((q) => (q < 0 ? q + x.length : q)).includes(i) : d === 1));
          }
        }
        break;
      }
      case "SliceTf": {
        const v = values.get(op.ins[0]), b = values.get(op.ins[1]), s = values.get(op.ins[2]);
        if (v && b && s && b.length === 1 && s.length === 1) {
          outVals = v.slice(b[0], s[0] < 0 ? undefined : b[0] + s[0]);
          out = [outVals.length];
        }
        break;
      }
      case "Gather": case "GatherV2": {
        const v = values.get(op.ins[0]), idx = values.get(op.ins[1]);
        if (v && idx) { outVals = idx.map((i) => v[i < 0 ? i + v.length : i]); out = [outVals.length]; }
        break;
      }
      default:
        params = constParams;
        break;
    }

    if (!wVals && constIns.length && op.type !== "Reshape") {
      const ci = constIns.map((i) => constOf.get(i)!).filter((c) => /float/.test(c.dt)).sort((a, b) => numel(b.shape) - numel(a.shape))[0];
      if (ci && ci.vals) { wVals = ci.vals; wShape = wShape ?? ci.shape; }
    }

    if (outVals && !out) out = [outVals.length];
    for (const o of op.outs) {
      shapes.set(o, out);
      fmts.set(o, outFmt);
      if (outVals) values.set(o, outVals);
    }

    const id = layers.length;
    layerOfOp.push(id);
    for (const i of op.ins) edges.add(producer.get(i), id);
    for (const o of op.outs) producer.set(o, id);
    const isOutput = op.outs.some((o) => outputNames.has(tname(o)) || !consumed.has(o));
    const w = params > 0 ? weightStats(wShape && numel(wShape) === (wVals?.length ?? -1) ? wShape : wVals ? [wVals.length] : (wShape ?? [params]), wVals) : null;
    if (w && !wVals) w.size = params;
    if (constIns.length) attr.constInputs = constIns.map((i) => `${tname(i)} ${constOf.get(i)!.shape.join("×") || "scalar"}`);
    if (outVals && outVals.length <= 16) attr.value = outVals;

    layers.push(makeLayer({
      id, name: op.name || `${op.type}_${id}`, type: op.type, canon, shape: out, dt,
      params, cost,
      group: op.type === "Input" ? "input" : isOutput ? "output" : groupFor(op.type),
      ins: op.ins.map((i) => ({ n: tname(i), s: shapes.get(i) })),
      outs: op.outs.map((o) => ({ n: tname(o), s: shapes.get(o) })),
      attr, w, framework: FRAMEWORK,
    }));
  }

  const inLayer = layers.find((l) => l.type === "Input");
  const outLayer = layers.filter((l) => l.group === "output").pop() ?? layers[layers.length - 1];
  return {
    name, format: "MNN", framework: FRAMEWORK, opset: 0, sizeBytes: buf.byteLength,
    inputShape: cleanShape(inLayer?.outs[0]?.s), outputShape: cleanShape(outLayer?.outs[0]?.s),
    layers, edges: edges.edges,
    producer: [`source ${source}`, version && `MNN ${version}`, biz && `bizCode ${biz}`].filter(Boolean).join(", "),
  };
}

function groupFor(type: string): ModelLayer["group"] | undefined {
  if (/Conv/.test(type)) return "backbone";
  if (/InnerProduct|MatMul|Softmax|Detection/.test(type)) return "head";
  if (/Pool|Concat|Reshape|Flatten|Interp|Resize|Permute|Transpose|Slice|Convert|Squeeze|Shape|Rank|Gather/.test(type)) return "neck";
  return undefined;
}
