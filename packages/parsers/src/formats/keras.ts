import type { Model, ModelLayer } from "../types";
import { extractStored, isZip, readZipEntries } from "../common/zip";
import { inflateRaw } from "../common/inflate";
import { findJsonObject } from "../common/jsonScan";
import {
  broadcast, cleanShape, DT_SIZE, DYN, EdgeSet, elemCost, linearCost, makeLayer, NO_COST, numel, padOut, prod,
  sampleBytes, weightStats, type Cost, type Shape,
} from "../common/graphKit";

// Keras models: the architecture is a JSON config.
//   .keras (Keras 3): ZIP with config.json (+ metadata.json, model.weights.h5)
//   .h5 / .hdf5 (legacy): HDF5 file whose root has a `model_config` attribute
//     holding the same JSON. We locate it by scanning (no full HDF5 reader).
// Both Keras 2 and Keras 3 config dialects are handled, including Sequential,
// Functional, and nested sub-models (flattened with a "outer/inner" prefix).
// Params come from per-layer formulas (Keras' count_params, incl. non-trainable
// BN statistics), shapes are propagated from the InputLayer batch shape.

type Json = Record<string, unknown>;
type KTensor = { id: number; shape?: Shape };
type LayerOut = KTensor[];

const FW = "Keras";

const obj = (v: unknown): Json => (v && typeof v === "object" && !Array.isArray(v) ? (v as Json) : {});
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const toShape = (v: unknown): Shape | undefined =>
  Array.isArray(v) ? v.map((d) => (typeof d === "number" ? d : DYN)) : undefined;
const ints = (v: unknown, n: number, dflt: number): number[] => {
  if (typeof v === "number") return Array(n).fill(v);
  if (Array.isArray(v) && v.length) return v.map((x) => (typeof x === "number" ? x : dflt));
  return Array(n).fill(dflt);
};
const known = (d: number | undefined) => (d == null || d < 0 ? 1 : d);

/** Layer config dtype: a string (Keras 2) or a DTypePolicy object (Keras 3). */
function dtypeOf(cfg: Json): string {
  const d = cfg.dtype;
  if (typeof d === "string") return /float32|float16|bfloat16|float64|int/.exec(d)?.[0] ?? "float32";
  const name = obj(obj(d).config).name;
  if (typeof name === "string") {
    if (name.startsWith("mixed_")) return name.endsWith("bfloat16") ? "bfloat16" : "float16";
    return name;
  }
  return "float32";
}

// ── Per-layer semantics ──────────────────────────────────────────────────────

type Sem = { out?: Shape; outs?: Shape[]; params: number; cost: Cost; canon: string };

const CANON: Record<string, string> = {
  Dense: "Gemm", EinsumDense: "MatMul",
  Conv1D: "Conv", Conv2D: "Conv", Conv3D: "Conv", Convolution2D: "Conv",
  DepthwiseConv1D: "Conv", DepthwiseConv2D: "Conv", SeparableConv1D: "Conv", SeparableConv2D: "Conv",
  Conv1DTranspose: "ConvTranspose", Conv2DTranspose: "ConvTranspose", Conv3DTranspose: "ConvTranspose",
  BatchNormalization: "BatchNormalization", LayerNormalization: "LayerNormalization", GroupNormalization: "GroupNormalization",
  MaxPooling1D: "MaxPool", MaxPooling2D: "MaxPool", MaxPooling3D: "MaxPool",
  AveragePooling1D: "AveragePool", AveragePooling2D: "AveragePool", AveragePooling3D: "AveragePool",
  GlobalAveragePooling1D: "GlobalAveragePool", GlobalAveragePooling2D: "GlobalAveragePool", GlobalAveragePooling3D: "GlobalAveragePool",
  GlobalMaxPooling1D: "GlobalMaxPool", GlobalMaxPooling2D: "GlobalMaxPool", GlobalMaxPooling3D: "GlobalMaxPool",
  Flatten: "Flatten", Reshape: "Reshape", Permute: "Transpose",
  Add: "Add", Subtract: "Sub", Multiply: "Mul", Average: "Mean", Maximum: "Max", Minimum: "Min",
  Concatenate: "Concat", ReLU: "Relu", LeakyReLU: "LeakyRelu", Softmax: "Softmax", ELU: "Elu", PReLU: "PRelu",
  Dropout: "Identity", SpatialDropout2D: "Identity", GaussianNoise: "Identity", Rescaling: "Mul",
  ZeroPadding1D: "Pad", ZeroPadding2D: "Pad", ZeroPadding3D: "Pad", Cropping2D: "Slice",
  UpSampling2D: "Resize", Resizing: "Resize", Embedding: "Gather",
  LSTM: "LSTM", GRU: "GRU", SimpleRNN: "RNN", Bidirectional: "RNN",
  MultiHeadAttention: "Attention", Attention: "Attention",
};

const ACT_CANON: Record<string, string> = {
  relu: "Relu", relu6: "Relu", sigmoid: "Sigmoid", tanh: "Tanh", softmax: "Softmax", leaky_relu: "LeakyRelu",
  elu: "Elu", gelu: "Gelu", swish: "Swish", silu: "Swish", hard_sigmoid: "HardSigmoid", hard_swish: "HardSwish",
  linear: "Identity",
};

const activationName = (cfg: Json): string | undefined => {
  const a = cfg.activation;
  if (typeof a === "string") return a;
  const n = obj(a).config ? obj(obj(a).config).name : undefined;
  return typeof n === "string" ? n : undefined;
};

function semantics(cls: string, cfg: Json, ins: (Shape | undefined)[]): Sem {
  const x = ins[0];
  const canon = CANON[cls] ?? cls;
  const same = (params = 0, cost: Cost = elemCost(x)): Sem => ({ out: x ? [...x] : undefined, params, cost, canon });
  const chFirst = cfg.data_format === "channels_first";
  const useBias = cfg.use_bias !== false;
  const act = activationName(cfg);
  const actCost = (out: Shape | undefined) => (act && act !== "linear" ? elemCost(out).flops : 0);

  // Split a feature-map shape into [batch, spatial..., channels] (channels_last view).
  const split = (s: Shape, nsp: number) =>
    chFirst
      ? { b: s[0], c: s[1], sp: s.slice(2, 2 + nsp) }
      : { b: s[0], c: s[s.length - 1], sp: s.slice(1, 1 + nsp) };
  const join = (b: number, sp: number[], c: number): Shape => (chFirst ? [b, c, ...sp] : [b, ...sp, c]);

  const nspOf = (c: string) => (/3D/.test(c) ? 3 : /1D/.test(c) ? 1 : 2);

  switch (cls) {
    case "InputLayer":
      return { out: x, params: 0, cost: NO_COST, canon: "Input" };

    case "Dense": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const units = Number(cfg.units);
      const d = known(x[x.length - 1]);
      const out = [...x.slice(0, -1), units];
      const c = linearCost(out, d);
      return { out, params: d * units + (useBias ? units : 0), cost: { macs: c.macs, flops: c.flops + actCost(out) }, canon };
    }

    case "Conv1D": case "Conv2D": case "Conv3D": case "Convolution2D":
    case "DepthwiseConv1D": case "DepthwiseConv2D":
    case "SeparableConv1D": case "SeparableConv2D": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const nsp = nspOf(cls);
      const { b, c: cin0, sp } = split(x, nsp);
      const cin = known(cin0);
      const k = ints(cfg.kernel_size, nsp, 1);
      const st = ints(cfg.strides, nsp, 1);
      const dil = ints(cfg.dilation_rate, nsp, 1);
      const pad = String(cfg.padding ?? "valid");
      const osp = sp.map((d, i) => padOut(d, k[i], st[i], pad, dil[i]));
      const kArea = prod(k);
      let cout: number, params: number, macs: number;
      if (cls.startsWith("Depthwise")) {
        const dm = Number(cfg.depth_multiplier ?? 1);
        cout = cin * dm;
        params = kArea * cin * dm + (useBias ? cout : 0);
        macs = prod([...osp.map(known), cout]) * kArea;
      } else if (cls.startsWith("Separable")) {
        const dm = Number(cfg.depth_multiplier ?? 1);
        cout = Number(cfg.filters);
        params = kArea * cin * dm + cin * dm * cout + (useBias ? cout : 0);
        const spN = prod(osp.map(known));
        macs = spN * cin * dm * kArea + spN * cin * dm * cout;
      } else {
        const groups = Number(cfg.groups ?? 1) || 1;
        cout = Number(cfg.filters);
        params = kArea * (cin / groups) * cout + (useBias ? cout : 0);
        macs = prod([...osp.map(known), cout]) * kArea * (cin / groups);
      }
      const out = join(b, osp, cout);
      return { out, params, cost: { macs, flops: 2 * macs + actCost(out) }, canon };
    }

    case "Conv1DTranspose": case "Conv2DTranspose": case "Conv3DTranspose": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const nsp = nspOf(cls);
      const { b, c: cin0, sp } = split(x, nsp);
      const cin = known(cin0);
      const k = ints(cfg.kernel_size, nsp, 1);
      const st = ints(cfg.strides, nsp, 1);
      const pad = String(cfg.padding ?? "valid");
      const cout = Number(cfg.filters);
      const osp = sp.map((d, i) => (d < 0 ? DYN : /same/.test(pad) ? d * st[i] : (d - 1) * st[i] + k[i]));
      const out = join(b, osp, cout);
      const macs = prod([...sp.map(known), cin]) * prod(k) * cout;
      return { out, params: prod(k) * cin * cout + (useBias ? cout : 0), cost: { macs, flops: 2 * macs + actCost(out) }, canon };
    }

    case "BatchNormalization": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const axis = Array.isArray(cfg.axis) ? Number(cfg.axis[0]) : Number(cfg.axis ?? -1);
      const c = known(x[axis < 0 ? x.length + axis : axis]);
      const n = (cfg.scale !== false ? 1 : 0) + (cfg.center !== false ? 1 : 0) + 2; // + moving mean/var
      return same(n * c, elemCost(x, 2));
    }
    case "LayerNormalization": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const c = known(x[x.length - 1]);
      const n = (cfg.scale !== false ? 1 : 0) + (cfg.center !== false ? 1 : 0);
      return same(n * c, elemCost(x, 5));
    }
    case "GroupNormalization": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const c = known(x[x.length - 1]);
      return same(((cfg.scale !== false ? 1 : 0) + (cfg.center !== false ? 1 : 0)) * c, elemCost(x, 5));
    }

    case "MaxPooling1D": case "MaxPooling2D": case "MaxPooling3D":
    case "AveragePooling1D": case "AveragePooling2D": case "AveragePooling3D": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const nsp = nspOf(cls);
      const { b, c, sp } = split(x, nsp);
      const k = ints(cfg.pool_size, nsp, 2);
      const st = cfg.strides == null ? k : ints(cfg.strides, nsp, 1);
      const pad = String(cfg.padding ?? "valid");
      const out = join(b, sp.map((d, i) => padOut(d, k[i], st[i], pad)), c);
      return { out, params: 0, cost: elemCost(out, prod(k)), canon };
    }

    case "GlobalAveragePooling1D": case "GlobalAveragePooling2D": case "GlobalAveragePooling3D":
    case "GlobalMaxPooling1D": case "GlobalMaxPooling2D": case "GlobalMaxPooling3D": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const nsp = nspOf(cls);
      const { b, c } = split(x, nsp);
      const out = cfg.keepdims === true ? join(b, Array(nsp).fill(1), c) : [b, c];
      return { out, params: 0, cost: elemCost(x), canon };
    }

    case "Flatten": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const rest = x.slice(1);
      return { out: [x[0], rest.some((d) => d < 0) ? DYN : prod(rest)], params: 0, cost: NO_COST, canon };
    }
    case "Reshape": {
      const t = toShape(cfg.target_shape) ?? [];
      if (x && t.includes(DYN)) {
        const total = prod(x.slice(1));
        const knownPart = prod(t.filter((d) => d >= 0));
        return { out: [x[0], ...t.map((d) => (d < 0 ? total / knownPart : d))], params: 0, cost: NO_COST, canon };
      }
      return { out: [x ? x[0] : DYN, ...t], params: 0, cost: NO_COST, canon };
    }
    case "Permute": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const dims = arr(cfg.dims).map(Number);
      return { out: [x[0], ...dims.map((d) => x[d])], params: 0, cost: NO_COST, canon };
    }
    case "RepeatVector":
      return { out: x ? [x[0], Number(cfg.n), ...x.slice(1)] : undefined, params: 0, cost: NO_COST, canon };

    case "Add": case "Subtract": case "Multiply": case "Average": case "Maximum": case "Minimum": {
      const valid = ins.filter((s): s is Shape => !!s);
      if (!valid.length) return { params: 0, cost: NO_COST, canon };
      const out = valid.reduce((a, b2) => broadcast(a, b2));
      return { out, params: 0, cost: elemCost(out, Math.max(1, valid.length - 1)), canon };
    }
    case "Concatenate": {
      const valid = ins.filter((s): s is Shape => !!s);
      if (!valid.length) return { params: 0, cost: NO_COST, canon };
      const out = [...valid[0]];
      const ax0 = Number(cfg.axis ?? -1);
      const ax = ax0 < 0 ? out.length + ax0 : ax0;
      out[ax] = valid.some((s) => s[ax] < 0) ? DYN : valid.reduce((a, s) => a + s[ax], 0);
      return { out, params: 0, cost: NO_COST, canon };
    }

    case "Activation": {
      const a = activationName(cfg) ?? "linear";
      return { out: x, params: 0, cost: elemCost(x, a === "softmax" ? 3 : 1), canon: ACT_CANON[a] ?? a };
    }
    case "ReLU": case "LeakyReLU": case "ELU": case "ThresholdedReLU":
      return same();
    case "Softmax":
      return same(0, elemCost(x, 3));
    case "PReLU":
      return same(x ? numel(x.slice(1).map(known)) : 0);

    case "ZeroPadding1D": case "ZeroPadding2D": case "ZeroPadding3D": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const nsp = nspOf(cls);
      const { b, c, sp } = split(x, nsp);
      const p = cfg.padding;
      const pairs: [number, number][] = sp.map((_, i) => {
        if (typeof p === "number") return [p, p];
        const e = arr(p)[i];
        if (typeof e === "number") return [e, e];
        const pe = arr(e).map(Number);
        return [pe[0] ?? 0, pe[1] ?? 0];
      });
      return { out: join(b, sp.map((d, i) => (d < 0 ? d : d + pairs[i][0] + pairs[i][1])), c), params: 0, cost: NO_COST, canon };
    }
    case "Cropping2D": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const { b, c, sp } = split(x, 2);
      const p = arr(cfg.cropping).map((e) => (typeof e === "number" ? [e, e] : arr(e).map(Number)));
      return { out: join(b, sp.map((d, i) => (d < 0 ? d : d - (p[i]?.[0] ?? 0) - (p[i]?.[1] ?? 0))), c), params: 0, cost: NO_COST, canon };
    }
    case "UpSampling2D": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const { b, c, sp } = split(x, 2);
      const sz = ints(cfg.size, 2, 2);
      const out = join(b, sp.map((d, i) => (d < 0 ? d : d * sz[i])), c);
      return { out, params: 0, cost: elemCost(out), canon };
    }
    case "Resizing": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const { b, c } = split(x, 2);
      const out = join(b, [Number(cfg.height), Number(cfg.width)], c);
      return { out, params: 0, cost: elemCost(out), canon };
    }

    case "Embedding": {
      const od = Number(cfg.output_dim);
      return { out: x ? [...x, od] : undefined, params: Number(cfg.input_dim) * od, cost: NO_COST, canon };
    }

    case "SimpleRNN": case "LSTM": case "GRU": {
      if (!x) return { params: 0, cost: NO_COST, canon };
      const u = Number(cfg.units);
      const d = known(x[x.length - 1]);
      const gates = cls === "LSTM" ? 4 : cls === "GRU" ? 3 : 1;
      const biasPer = cls === "GRU" && cfg.reset_after !== false ? 2 * u : u;
      const params = gates * (u * (d + u) + (useBias ? biasPer : 0));
      const T = known(x[1]);
      const macs = T * gates * u * (d + u);
      const out = cfg.return_sequences === true ? [x[0], x[1], u] : [x[0], u];
      return { out, params, cost: { macs, flops: 2 * macs }, canon };
    }
    case "Bidirectional": {
      const inner = obj(cfg.layer);
      const sem = semantics(String(inner.class_name), obj(inner.config), ins);
      const mode = cfg.merge_mode ?? "concat";
      let out = sem.out;
      if (out && mode === "concat") out = [...out.slice(0, -1), out[out.length - 1] * 2];
      return { out, params: sem.params * 2, cost: { macs: sem.cost.macs * 2, flops: sem.cost.flops * 2 }, canon: "RNN" };
    }
    case "TimeDistributed": {
      const inner = obj(cfg.layer);
      if (!x) return { params: 0, cost: NO_COST, canon };
      const sem = semantics(String(inner.class_name), obj(inner.config), [[x[0], ...x.slice(2)]]);
      const T = known(x[1]);
      return {
        out: sem.out ? [x[0], x[1], ...sem.out.slice(1)] : undefined, params: sem.params,
        cost: { macs: sem.cost.macs * T, flops: sem.cost.flops * T }, canon: sem.canon,
      };
    }

    case "MultiHeadAttention": {
      const q = ins[0];
      const v = ins[1] ?? q;
      if (!q) return { params: 0, cost: NO_COST, canon };
      const h = Number(cfg.num_heads);
      const kd = Number(cfg.key_dim);
      const vd = Number(cfg.value_dim ?? kd);
      const dq = known(q[q.length - 1]);
      const dv = known(v?.[v.length - 1]);
      const b = useBias ? 1 : 0;
      const params = dq * h * kd + b * h * kd + dv * h * kd + b * h * kd + dv * h * vd + b * h * vd + h * vd * dq + b * dq;
      const Tq = known(q[1]);
      const Tv = known(v?.[1]);
      const macs = Tq * dq * h * kd + Tv * dv * h * kd + Tv * dv * h * vd + Tq * Tv * h * kd + Tq * Tv * h * vd + Tq * h * vd * dq;
      return { out: [...q], params, cost: { macs, flops: 2 * macs }, canon };
    }

    default:
      return same(0, NO_COST);
  }
}

// ── Inbound-node decoding (Keras 2 and Keras 3 dialects) ─────────────────────

type Ref = { layer: string; node: number; tensor: number };

function inboundRefs(node: unknown): Ref[] {
  const refs: Ref[] = [];
  const walk = (v: unknown) => {
    if (Array.isArray(v)) {
      // Keras 2: [layer_name, node_index, tensor_index, kwargs]
      if (typeof v[0] === "string" && typeof v[1] === "number" && typeof v[2] === "number") {
        refs.push({ layer: v[0], node: v[1], tensor: v[2] });
        walk(v[3]); // kwargs may carry extra tensors
        return;
      }
      v.forEach(walk);
      return;
    }
    if (v && typeof v === "object") {
      const o = v as Json;
      // Keras 3: {"class_name": "__keras_tensor__", "config": {"keras_history": [name, node, tensor], "shape": [...]}}
      if (o.class_name === "__keras_tensor__") {
        const h = arr(obj(o.config).keras_history);
        if (typeof h[0] === "string") refs.push({ layer: h[0], node: Number(h[1] ?? 0), tensor: Number(h[2] ?? 0) });
        return;
      }
      for (const val of Object.values(o)) walk(val);
    }
  };
  walk(node);
  return refs;
}

// ── Graph builder ────────────────────────────────────────────────────────────

type Builder = {
  layers: ModelLayer[];
  edges: EdgeSet;
  inputs: KTensor[];
  outputs: KTensor[];
};

const isModelClass = (c: string) => c === "Functional" || c === "Sequential" || c === "Model";

function addLayer(
  b: Builder, cls: string, cfg: Json, name: string, inT: KTensor[], prefix: string,
): LayerOut {
  const sem = semantics(cls, cfg, inT.map((t) => t.shape));
  const id = b.layers.length;
  const act = activationName(cfg);
  const attr: Json = {};
  for (const [k, v] of Object.entries(cfg)) {
    if (k === "name" || k === "trainable" || k === "dtype" || /initializer|regularizer|constraint/.test(k)) continue;
    if (k === "activation" && typeof v !== "string") { attr[k] = act; continue; }
    if (v !== null && typeof v === "object" && !Array.isArray(v)) continue;
    attr[k] = v;
  }
  const isInput = cls === "InputLayer";
  const layer = makeLayer({
    id, name: prefix + name, type: cls, canon: sem.canon, shape: sem.out, dt: dtypeOf(cfg),
    params: sem.params, cost: sem.cost, group: isInput ? "input" : undefined,
    ins: inT.map((t) => ({ n: b.layers[t.id]?.name ?? String(t.id), s: t.shape })),
    outs: [{ n: prefix + name, s: sem.out }],
    attr, w: sem.params > 0 ? { shape: weightShape(cls, cfg, inT[0]?.shape, sem), size: sem.params } : null,
    framework: FW,
    insight: isInput ? "Model input. Apply preprocessing (resize/normalize) upstream." : undefined,
  });
  if (act && act !== "linear" && cls !== "Activation") layer.math += ` → ${act}`;
  if (isInput) layer.math = "model input";
  b.layers.push(layer);
  for (const t of inT) b.edges.add(t.id, id);
  return [{ id, shape: sem.out }];
}

/** Main kernel shape for the inspector (falls back to a "N params" string). */
function weightShape(cls: string, cfg: Json, x: Shape | undefined, sem: Sem): number[] | string {
  const cin = x ? known(cfg.data_format === "channels_first" ? x[1] : x[x.length - 1]) : 0;
  if (cls === "Dense") return [cin, Number(cfg.units)];
  if (/^Conv\dD$|^Convolution2D$/.test(cls)) return [...ints(cfg.kernel_size, /3D/.test(cls) ? 3 : /1D/.test(cls) ? 1 : 2, 1), cin, Number(cfg.filters)];
  if (cls === "Embedding") return [Number(cfg.input_dim), Number(cfg.output_dim)];
  return `${sem.params} params`;
}

/**
 * Build a (possibly nested) Functional or Sequential model into `b`.
 * `outerInputs` binds a nested model's InputLayers to the caller's tensors.
 */
function buildModel(b: Builder, model: Json, prefix: string, outerInputs?: KTensor[]): KTensor[] {
  const cls = String(model.class_name ?? "");
  const cfg = obj(model.config);
  const specs = arr(cfg.layers).map(obj);

  if (cls === "Sequential" || (!cfg.input_layers && !specs.some((s) => arr(s.inbound_nodes).length))) {
    let cur: KTensor[] = [];
    let first = true;
    for (const s of specs) {
      const lc = String(s.class_name);
      const lcfg = obj(s.config);
      const lname = String(lcfg.name ?? s.name ?? `${lc}_${b.layers.length}`);
      if (lc === "InputLayer") {
        if (outerInputs) { cur = outerInputs; first = false; continue; }
        cur = addLayer(b, lc, { ...lcfg, batch_shape: lcfg.batch_shape ?? lcfg.batch_input_shape }, lname, [], prefix);
        setInputShape(b, cur[0], lcfg);
        b.inputs.push(cur[0]);
        first = false;
        continue;
      }
      if (first) {
        first = false;
        if (outerInputs) cur = outerInputs;
        else {
          // Keras 2 Sequential without InputLayer: shape lives on the first layer.
          const bs = lcfg.batch_input_shape ?? lcfg.batch_shape ?? obj(s.build_config).input_shape;
          if (bs) {
            cur = addLayer(b, "InputLayer", { batch_shape: bs, dtype: lcfg.dtype }, `${lname}_input`, [], prefix);
            setInputShape(b, cur[0], { batch_shape: bs });
            b.inputs.push(cur[0]);
          }
        }
      }
      if (isModelClass(lc)) cur = buildModel(b, s, `${prefix}${lname}/`, cur);
      else cur = addLayer(b, lc, lcfg, lname, cur, prefix);
    }
    return cur;
  }

  // Functional: resolve inbound refs by (layer, node_index, tensor_index).
  const produced = new Map<string, KTensor[][]>(); // layer name → per call node → outputs
  const inputLayerNames = arr(cfg.input_layers);
  const inputNameList: string[] = (Array.isArray(inputLayerNames[0]) ? inputLayerNames : [inputLayerNames])
    .map((r) => String(arr(r)[0]));
  const lookup = (r: Ref): KTensor | undefined => produced.get(r.layer)?.[r.node]?.[r.tensor] ?? produced.get(r.layer)?.[r.node]?.[0];

  // Process layers in dependency order (configs are usually topological already).
  const pending = specs.map((s, i) => ({ s, i }));
  let guard = pending.length * pending.length + 10;
  while (pending.length && guard-- > 0) {
    const { s } = pending.shift()!;
    const lc = String(s.class_name);
    const lcfg = obj(s.config);
    const lname = String(s.name ?? lcfg.name);
    const nodes = arr(s.inbound_nodes);

    if (lc === "InputLayer") {
      let t: KTensor[];
      const idx = inputNameList.indexOf(lname);
      if (outerInputs) t = [outerInputs[Math.max(0, idx)] ?? outerInputs[0]];
      else {
        t = addLayer(b, lc, lcfg, lname, [], prefix);
        setInputShape(b, t[0], lcfg);
        b.inputs.push(t[0]);
      }
      produced.set(lname, [t]);
      continue;
    }

    const refsPerNode = nodes.map(inboundRefs);
    if (refsPerNode.some((rs) => rs.some((r) => !lookup(r)))) {
      // A dependency is defined later in the list; retry after the others.
      pending.push({ s, i: 0 });
      continue;
    }
    const outsPerNode: KTensor[][] = [];
    for (const rs of refsPerNode) {
      const inT = rs.map(lookup).filter((t): t is KTensor => !!t);
      // Shared layers called more than once render one node per call.
      const suffix = refsPerNode.length > 1 && outsPerNode.length ? `_${outsPerNode.length}` : "";
      outsPerNode.push(
        isModelClass(lc)
          ? buildModel(b, s, `${prefix}${lname}${suffix}/`, inT)
          : addLayer(b, lc, lcfg, lname + suffix, inT, prefix),
      );
    }
    produced.set(lname, outsPerNode);
  }

  const outRefs = arr(cfg.output_layers);
  const outList = (Array.isArray(outRefs[0]) ? outRefs : [outRefs]).map((r) => {
    const a = arr(r);
    return { layer: String(a[0]), node: Number(a[1] ?? 0), tensor: Number(a[2] ?? 0) };
  });
  return outList.map(lookup).filter((t): t is KTensor => !!t);
}

function setInputShape(b: Builder, t: KTensor, cfg: Json): void {
  const s = toShape(cfg.batch_shape ?? cfg.batch_input_shape);
  if (!s) return;
  t.shape = s;
  const l = b.layers[t.id];
  const rebuilt = makeLayer({ id: l.id, name: l.name, type: l.type, canon: "Input", shape: s, dt: l.dt, group: "input", outs: [{ n: l.name, s }], attr: l.attr, framework: FW, insight: l.insight });
  rebuilt.math = "model input";
  b.layers[t.id] = rebuilt;
}

/** Build the normalized graph from a parsed Keras model config object. */
export function kerasConfigToModel(
  config: Json, meta: { name: string; format: string; sizeBytes: number; kerasVersion?: string },
): Model {
  // The h5 attribute and config.json are the model object itself; some tools wrap it.
  const root = config.class_name ? config : obj(config.model_config ?? config.config);
  if (!root.class_name || !obj(root.config).layers) throw new Error("Keras config has no layers.");
  const b: Builder = { layers: [], edges: new EdgeSet(), inputs: [], outputs: [] };
  const outs = buildModel(b, root, "");
  for (const t of outs) if (b.layers[t.id] && b.layers[t.id].group !== "input") b.layers[t.id].group = "output";
  return {
    name: meta.name,
    format: meta.format,
    framework: FW,
    opset: 0,
    sizeBytes: meta.sizeBytes,
    inputShape: cleanShape(b.inputs[0]?.shape),
    outputShape: cleanShape(outs[0]?.shape),
    layers: b.layers,
    edges: b.edges.edges,
    producer: meta.kerasVersion ? `Keras ${meta.kerasVersion}` : undefined,
    doc: typeof obj(root.config).name === "string" ? `${root.class_name} "${obj(root.config).name}"` : undefined,
  };
}

// ── File entry points ────────────────────────────────────────────────────────

const td = new TextDecoder();

function zipMember(buf: Uint8Array, name: string): Uint8Array | null {
  const e = readZipEntries(buf).find((x) => x.name === name || x.name.endsWith(`/${name}`));
  if (!e) return null;
  if (e.method === 0) return extractStored(buf, e);
  if (e.method === 8) return inflateRaw(buf.subarray(e.dataOffset, e.dataOffset + e.compressedSize), e.size);
  throw new Error(`Unsupported ZIP compression method ${e.method} for ${e.name}.`);
}

/** Parse a Keras 3 `.keras` archive (ZIP with config.json). */
export function parseKerasArchive(input: ArrayBuffer | Uint8Array, name = "model.keras"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (!isZip(buf)) throw new Error("Not a .keras archive (expected a ZIP).");
  const cfgBytes = zipMember(buf, "config.json");
  if (!cfgBytes) throw new Error(".keras archive has no config.json.");
  const metaBytes = zipMember(buf, "metadata.json");
  const meta = metaBytes ? obj(JSON.parse(td.decode(metaBytes))) : {};
  return kerasConfigToModel(JSON.parse(td.decode(cfgBytes)), {
    name, format: "Keras (.keras)", sizeBytes: buf.byteLength,
    kerasVersion: typeof meta.keras_version === "string" ? meta.keras_version : undefined,
  });
}

/**
 * Parse a legacy Keras HDF5 model (`.h5`/`.hdf5`). The architecture JSON is
 * the root `model_config` attribute; we find it by scanning for a balanced
 * JSON object that looks like a Keras model config, without a full HDF5 reader.
 */
export function parseKerasH5(input: ArrayBuffer | Uint8Array, name = "model.h5"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const cfg = findJsonObject(buf, ['{"class_name"', '{"module"'], (o) => {
    const r = o as Json;
    return typeof r.class_name === "string" && Array.isArray(obj(r.config).layers);
  });
  if (!cfg) {
    throw new Error("No Keras model_config found in this HDF5 file (weights-only checkpoint?).");
  }
  return kerasConfigToModel(cfg as Json, { name, format: "Keras (HDF5)", sizeBytes: buf.byteLength });
}

/** Parse either Keras container, dispatching on the ZIP magic. */
export function parseKeras(input: ArrayBuffer | Uint8Array, name = "model.keras"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  return isZip(buf) ? parseKerasArchive(buf, name) : parseKerasH5(buf, name);
}

/** HDF5 signature check (superblock may sit at 0, 512, 1024, … ). */
export function isHdf5(buf: Uint8Array): boolean {
  const sig = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let off = 0; off + 8 <= buf.length && off <= 1 << 20; off = off ? off * 2 : 512) {
    if (sig.every((b, i) => buf[off + i] === b)) return true;
  }
  return false;
}

/**
 * Parse a Keras architecture JSON (`model.to_json()`) or a TensorFlow.js
 * layers-model `model.json` (its `modelTopology` is a Keras config). TF.js
 * graph-models (`format: "graph-model"`) are GraphDefs, not Keras; this throws.
 */
export function parseKerasJson(
  input: ArrayBuffer | Uint8Array | string, name = "model.json", shards?: Record<string, Uint8Array>,
): Model {
  const text = typeof input === "string" ? input : td.decode(input instanceof Uint8Array ? input : new Uint8Array(input));
  const j = obj(JSON.parse(text));
  const isTfjs = j.modelTopology != null;
  if (isTfjs && j.format === "graph-model") throw new Error("TF.js graph-model is not a Keras config.");
  const topo = isTfjs ? obj(j.modelTopology) : j;
  const kv = topo.keras_version;
  const model = kerasConfigToModel(topo, {
    name, format: isTfjs ? "TensorFlow.js (layers)" : "Keras (JSON)", sizeBytes: text.length,
    kerasVersion: typeof kv === "string" ? kv.replace(/^tfjs-layers\s*/, "") : undefined,
  });
  if (isTfjs && shards) attachTfjsWeights(model, j.weightsManifest, shards);
  return model;
}

type TfjsWeight = { name: string; shape: number[]; dtype: string; quantization?: { dtype: string; scale?: number; min?: number } };

/**
 * Decode a TF.js weightsManifest against its shard files and put value
 * statistics on the layers. Each manifest group's shards concatenate into one
 * buffer holding its weights back to back, optionally quantized
 * (uint8/uint16: value = q · scale + min; float16). Weights are named
 * "<layer>/<variable>"; a layer takes the one matching its weight shape, else
 * its largest. A group with a missing shard or an unknown dtype is skipped.
 */
function attachTfjsWeights(model: Model, manifest: unknown, shards: Record<string, Uint8Array>): void {
  const values = new Map<string, { shape: number[]; vals: number[] }>();
  for (const g of Array.isArray(manifest) ? (manifest as { paths?: string[]; weights?: TfjsWeight[] }[]) : []) {
    const parts = (g.paths ?? []).map((p) => shards[p] ?? shards[p.split("/").pop() ?? p]);
    if (!parts.length || parts.some((p) => !p)) continue;
    const buf = new Uint8Array(parts.reduce((n, p) => n + p!.length, 0));
    let o = 0;
    for (const p of parts) { buf.set(p!, o); o += p!.length; }
    o = 0;
    for (const w of g.weights ?? []) {
      const q = w.quantization;
      const dt = q?.dtype ?? w.dtype;
      const bpe = dt === "bool" ? 1 : DT_SIZE[dt];
      if (!bpe) break; // e.g. string weights: sizes unknowable, the rest of the group can't be located
      const n = w.shape.reduce((a, b) => a * b, 1) * bpe;
      if (o + n > buf.length) break;
      let vals = sampleBytes(buf.subarray(o, o + n), dt === "bool" ? "uint8" : dt);
      if (vals && q && (dt === "uint8" || dt === "uint16")) vals = vals.map((v) => v * (q.scale ?? 1) + (q.min ?? 0));
      if (vals) values.set(w.name, { shape: w.shape, vals });
      o += n;
    }
  }
  for (const l of model.layers) {
    if (!l.w) continue;
    const mine = [...values.entries()].filter(([k]) => k.startsWith(`${l.name}/`)).map(([, v]) => v);
    const want = Array.isArray(l.w.shape) ? l.w.shape.join("×") : String(l.w.shape);
    const pick = mine.find((v) => v.shape.join("×") === want) ?? mine.sort((a, b) => b.vals.length - a.vals.length)[0];
    if (pick) l.w = { ...weightStats(pick.shape, pick.vals), size: l.w.size };
  }
}

/** True for a Keras config / TF.js layers-model JSON (cheap textual sniff). */
export function isKerasJson(text: string): boolean {
  const head = text.slice(0, 4096);
  return /"class_name"\s*:/.test(head) && /"(layers|model_config|modelTopology)"\s*:/.test(text.slice(0, 1 << 16))
    && !/"format"\s*:\s*"graph-model"/.test(head);
}
