import type { Model, ModelLayer } from "../types";
import { isGlobal, PyObj, unpickle, type GlobalRef, type Handlers } from "../common/pickle";
import { inflateRaw } from "../common/inflate";
import { EdgeSet, NO_COST, makeLayer, sampleBytes, weightStats, type Cost, type WStat } from "../common/graphKit";

// scikit-learn models saved with pickle (`.pkl`/`.pickle`) or joblib
// (`.joblib`, also often `.pkl`). The pickle stream is decoded with the shared
// safe unpickler (common/pickle.ts): nothing is executed; classes become inert
// PyObj records whose `__dict__` we walk. NumPy arrays are reconstructed from
//   numpy.core.multiarray._reconstruct + ndarray.__setstate__ (protocol ≤ 4),
//   numpy.core.numeric._frombuffer (protocol 5 in-band buffers),
//   joblib.numpy_pickle.NumpyArrayWrapper (raw bytes stored inline after the
//   wrapper's BUILD opcode, optionally 1-byte-length padded for alignment).
// joblib zlib/gzip compression is inflated; lzma/bz2/lz4 fall back to metadata.
// The graph: Pipeline steps in sequence, ColumnTransformer / FeatureUnion /
// voting/stacking members as parallel branches, tree ensembles as parallel
// trees, MLPs expanded into their dense layers.

const FRAMEWORK = "scikit-learn";

type DType = { __dtype__: true; kind: string; itemsize: number; descr: string };
type NdArray = { __ndarray__: true; shape: number[]; dtype: DType | null; data: Uint8Array | null; items: unknown[] | null };

const isDType = (x: unknown): x is DType => typeof x === "object" && x !== null && (x as DType).__dtype__ === true;
const isArr = (x: unknown): x is NdArray => typeof x === "object" && x !== null && (x as NdArray).__ndarray__ === true;

function makeDType(descr: string): DType {
  const m = descr.match(/^[<>|=]?([a-zA-Z]+)(\d*)$/);
  const kind = m?.[1] ?? "V";
  let size = m?.[2] ? Number(m[2]) : 0;
  if (kind === "U") size *= 4;
  if (!size) size = { b: 1, O: 8, f: 8, d: 8, i: 8, l: 8, q: 8 }[kind] ?? 0;
  return { __dtype__: true, kind, itemsize: size, descr };
}

const DT_NAME: Record<string, string> = { f2: "float16", f4: "float32", f8: "float64", i1: "int8", i2: "int16", i4: "int32", i8: "int64", u1: "uint8", u2: "uint16", u4: "uint32", u8: "uint64", b1: "bool" };
const dtName = (d: DType | null) => (d ? DT_NAME[`${d.kind}${d.itemsize}`] ?? (d.kind === "O" ? "object" : d.kind === "V" ? "struct" : d.descr) : "?");
const numelOf = (s: number[]) => s.reduce((a, b) => a * b, 1);

function newArray(shape: unknown, dtype: unknown, data: unknown): NdArray {
  const sh = Array.isArray(shape) ? shape.map(Number) : typeof shape === "number" ? [shape] : [];
  return {
    __ndarray__: true, shape: sh, dtype: isDType(dtype) ? dtype : null,
    data: data instanceof Uint8Array ? data : null,
    items: Array.isArray(data) ? data : null,
  };
}

/** Byte length of a pickle stream starting at `pos` (opcode walk, nothing decoded). */
function pickleEnd(buf: Uint8Array, pos: number): number {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const line = () => { while (pos < buf.length && buf[pos] !== 0x0a) pos++; pos++; };
  while (pos < buf.length) {
    const op = buf[pos++];
    switch (op) {
      case 0x2e: return pos; // STOP
      case 0x80: case 0x4b: case 0x68: case 0x71: case 0x82: pos += 1; break;
      case 0x4d: case 0x83: pos += 2; break;
      case 0x4a: case 0x6a: case 0x72: case 0x84: pos += 4; break;
      case 0x95: case 0x47: pos += 8; break;
      case 0x8a: pos += 1 + buf[pos]; break;
      case 0x8c: case 0x55: case 0x43: pos += 1 + buf[pos]; break;
      case 0x8b: case 0x58: case 0x54: case 0x42: pos += 4 + dv.getUint32(pos, true); break;
      case 0x8d: case 0x8e: case 0x96: pos += 8 + Number(dv.getBigUint64(pos, true)); break;
      case 0x63: case 0x69: line(); line(); break;
      case 0x49: case 0x4c: case 0x46: case 0x53: case 0x56: case 0x50: case 0x67: case 0x70: line(); break;
      default: break; // no-argument opcode
    }
  }
  return pos;
}

function handlers(seen: { joblib: boolean } = { joblib: false }): Handlers {
  const h: Handlers = {
    persistentLoad: () => null,
    findClass: (module, name): GlobalRef => ({ __global__: true, module, name }),
    reduce(func, args) {
      if (!isGlobal(func)) return new PyObj(null);
      const { module, name } = func;
      if (/^numpy(\._core|\.core)?\.multiarray$/.test(module) && name === "_reconstruct") return newArray([], null, null);
      if (/^numpy(\._core|\.core)?\.numeric$/.test(module) && name === "_frombuffer") return newArray(args[2], args[1], args[0]);
      if (module === "numpy" && name === "dtype") return makeDType(String(args[0]));
      if (/^numpy(\._core|\.core)?\.multiarray$/.test(module) && name === "scalar") {
        const d = args[0], b = args[1];
        if (isDType(d) && b instanceof Uint8Array) return sampleBytes(b, DT_NAME[`${d.kind}${d.itemsize}`] ?? "")?.[0] ?? null;
        return null;
      }
      if (module === "copyreg" && name === "_reconstructor") return new PyObj(isGlobal(args[0]) ? args[0] : null);
      if (module === "collections" && name === "OrderedDict") return new Map();
      if (module === "_codecs" && name === "encode") return new TextEncoder().encode(String(args[0]));
      if (module === "builtins" && (name === "set" || name === "frozenset" || name === "list" || name === "tuple")) return Array.isArray(args[0]) ? [...args[0]] : [];
      const o = new PyObj(func);
      if (args.length) o.attrs.set("__args__", args);
      return o;
    },
    build(obj, state) {
      if (isArr(obj) && Array.isArray(state)) {
        // ndarray.__setstate__((version, shape, dtype, is_fortran, data))
        const [, shape, dtype, , data] = state.length === 5 ? state : [0, ...state];
        Object.assign(obj, newArray(shape, dtype, data));
        return obj;
      }
      if (isDType(obj)) {
        if (Array.isArray(state) && typeof state[5] === "number" && state[5] > 0) obj.itemsize = state[5];
        return obj;
      }
      return undefined;
    },
    afterBuild(obj, buf, pos) {
      if (!(obj instanceof PyObj) || obj.cls?.name !== "NumpyArrayWrapper") return undefined;
      seen.joblib = true;
      const dt = obj.attrs.get("dtype");
      const shape = obj.attrs.get("shape");
      const sh = Array.isArray(shape) ? shape.map(Number) : [];
      const dtype = isDType(dt) ? dt : null;
      if (dtype?.kind === "O") {
        const end = pickleEnd(buf, pos);
        let items: unknown = null;
        try { items = unpickle(buf.subarray(pos, end), h); } catch { items = null; }
        const arr = newArray(sh, dtype, null);
        const inner = isArr(items) ? items.items : Array.isArray(items) ? items : null;
        arr.items = inner;
        obj.attrs.set("__array__", arr);
        return end;
      }
      let p = pos;
      if (obj.attrs.get("numpy_array_alignment_bytes") != null) p += 1 + buf[p];
      const bytes = numelOf(sh) * (dtype?.itemsize ?? 0);
      obj.attrs.set("__array__", newArray(sh, dtype, buf.subarray(p, p + bytes)));
      return p + bytes;
    },
  };
  return h;
}

// ── Object-tree helpers ──────────────────────────────────────────────────────

const clsName = (o: unknown) => (o instanceof PyObj ? o.cls?.name ?? "object" : "");
const clsModule = (o: unknown) => (o instanceof PyObj ? o.cls?.module ?? "" : "");
const isEstimator = (o: unknown): o is PyObj => o instanceof PyObj && /^sklearn\./.test(clsModule(o));
const attr = (o: PyObj, k: string) => o.attrs.get(k);
function arr(o: unknown): NdArray | null {
  if (isArr(o)) return o;
  if (o instanceof PyObj && o.cls?.name === "NumpyArrayWrapper") { const a = o.attrs.get("__array__"); return isArr(a) ? a : null; }
  return null;
}
const arrItems = (o: unknown): unknown[] => (Array.isArray(o) ? o : arr(o)?.items ?? []);
const numel = (a: NdArray | null) => (a ? numelOf(a.shape) : 0);

function arrStats(a: NdArray | null): WStat | null {
  if (!a) return null;
  const n = DT_NAME[a.dtype ? `${a.dtype.kind}${a.dtype.itemsize}` : ""];
  const vals = a.data && n ? sampleBytes(a.data, n) : null;
  return weightStats(a.shape, vals);
}

/** Hyperparameters (get_params-style scalars) + fitted-array shapes. */
function describe(o: PyObj): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of o.attrs) {
    const key = String(k);
    if (key.startsWith("__") || key === "_sklearn_version") continue;
    const a = arr(v);
    if (a) { out[key] = `${dtName(a.dtype)}[${a.shape.join("×")}]`; continue; }
    if (v === null || ["string", "number", "boolean"].includes(typeof v)) { out[key] = v; continue; }
    if (Array.isArray(v) && v.length <= 16 && v.every((x) => x === null || ["string", "number", "boolean"].includes(typeof x))) out[key] = v;
    else if (v instanceof PyObj) out[key] = clsName(v);
  }
  return out;
}

type Est = { params: number; cost: Cost; out: number | null; canon: string; w: WStat | null; insight?: string };

const LINEAR = (o: PyObj, inW: number | null): Est | null => {
  const coef = arr(attr(o, "coef_"));
  if (!coef) return null;
  const icpt = arr(attr(o, "intercept_"));
  const params = numel(coef) + numel(icpt);
  const macs = numel(coef);
  const classes = arr(attr(o, "classes_"));
  const out = classes ? classes.shape[0] : coef.shape.length === 2 ? coef.shape[0] : 1;
  return { params, cost: { macs, flops: 2 * macs }, out: out ?? inW, canon: "Gemm", w: arrStats(coef) };
};

function analyze(o: PyObj, inW: number | null): Est {
  const name = clsName(o);
  const lin = LINEAR(o, inW);
  if (lin) return lin;
  const sumArr = (...keys: string[]) => keys.reduce((s, k) => s + numel(arr(attr(o, k))), 0);
  if (/Scaler$/.test(name) || name === "Normalizer") {
    const keys = name === "MinMaxScaler" ? ["min_", "scale_"] : name === "RobustScaler" ? ["center_", "scale_"] : name === "MaxAbsScaler" ? ["scale_"] : ["mean_", "scale_"];
    const p = sumArr(...keys);
    const width = arr(attr(o, keys[keys.length - 1]))?.shape[0] ?? inW;
    return { params: p, cost: { macs: 0, flops: 2 * (width ?? 0) }, out: width, canon: "Mul", w: arrStats(arr(attr(o, keys[0]))) };
  }
  const comps = arr(attr(o, "components_"));
  if (comps) {
    const p = numel(comps) + sumArr("mean_");
    return { params: p, cost: { macs: numel(comps), flops: 2 * numel(comps) }, out: comps.shape[0], canon: "MatMul", w: arrStats(comps) };
  }
  const sv = arr(attr(o, "support_vectors_"));
  if (sv) {
    const p = numel(sv) + sumArr("dual_coef_", "intercept_", "_dual_coef", "_intercept_");
    const classes = arr(attr(o, "classes_"));
    return { params: p, cost: { macs: numel(sv), flops: 3 * numel(sv) }, out: classes?.shape[0] ?? 1, canon: "SVM", w: arrStats(sv) };
  }
  const tree = attr(o, "tree_");
  if (tree instanceof PyObj) {
    const nodes = Number(attr(tree, "node_count") ?? 0);
    const depth = Number(attr(tree, "max_depth") ?? 0);
    const classes = arr(attr(o, "classes_"));
    return { params: nodes, cost: { macs: 0, flops: depth }, out: classes?.shape[0] ?? 1, canon: "Tree", w: null, insight: `Decision tree: ${nodes} nodes, depth ${depth}. Params count tree nodes; inference is ${depth} comparisons.` };
  }
  const fitX = arr(attr(o, "_fit_X"));
  if (fitX) {
    const classes = arr(attr(o, "classes_"));
    return { params: numel(fitX), cost: { macs: numel(fitX), flops: 3 * numel(fitX) }, out: classes?.shape[0] ?? 1, canon: "KNN", w: null, insight: "Nearest-neighbours model: stores the training set; predict cost scales with it." };
  }
  if (attr(o, "theta_") != null) {
    const p = sumArr("theta_", "var_", "sigma_", "class_prior_");
    const classes = arr(attr(o, "classes_"));
    return { params: p, cost: { macs: p, flops: 3 * p }, out: classes?.shape[0] ?? 1, canon: "NaiveBayes", w: arrStats(arr(attr(o, "theta_"))) };
  }
  // Generic: count numeric fitted arrays.
  let p = 0;
  for (const [k, v] of o.attrs) {
    const a = arr(v);
    if (a && String(k).endsWith("_") && a.dtype && /[fiu]/.test(a.dtype.kind) && !/^(classes_|n_iter_|feature_names_in_|n_features_in_|loss_curve_)$/.test(String(k))) p += numel(a);
  }
  return { params: p, cost: NO_COST, out: inW, canon: name, w: null };
}

// ── Graph construction ───────────────────────────────────────────────────────

class Builder {
  layers: ModelLayer[] = [];
  edges = new EdgeSet();
  add(name: string, type: string, canon: string, from: number[], width: number | null, extra: Partial<Est> & { attr?: Record<string, unknown>; group?: ModelLayer["group"] } = {}): number {
    const id = this.layers.length;
    for (const f of from) this.edges.add(f, id);
    this.layers.push(makeLayer({
      id, name, type, canon, shape: width != null ? [width] : undefined, dt: "float64",
      params: extra.params ?? 0, cost: extra.cost ?? NO_COST, group: extra.group,
      ins: from.map((f) => ({ n: this.layers[f]?.name ?? `#${f}`, s: this.layers[f]?.outs[0]?.s })),
      outs: [{ n: name, s: width != null ? [width] : undefined }],
      attr: extra.attr ?? {}, w: extra.w ?? null, insight: extra.insight, framework: FRAMEWORK,
    }));
    return id;
  }
}

type Flow = { ids: number[]; width: number | null };
const MAX_BRANCHES = 32;

function emit(b: Builder, o: unknown, label: string, flow: Flow): Flow {
  if (o === "passthrough" || o === null) return flow;
  if (o === "drop") return { ids: [], width: 0 };
  if (!(o instanceof PyObj)) return flow;
  const name = clsName(o);
  const desc = describe(o);
  const title = label ? `${label} (${name})` : name;

  // Pipeline: sequential steps.
  const steps = attr(o, "steps");
  if (name === "Pipeline" && Array.isArray(steps)) {
    let cur = flow;
    for (const s of steps) if (Array.isArray(s)) cur = emit(b, s[1], `${label ? label + "." : ""}${String(s[0])}`, cur);
    return cur;
  }

  // ColumnTransformer / FeatureUnion: parallel branches + horizontal concat.
  const transformers = attr(o, "transformers_") ?? attr(o, "transformers") ?? attr(o, "transformer_list");
  if ((name === "ColumnTransformer" || name === "FeatureUnion") && Array.isArray(transformers)) {
    const split = b.add(title, name, "Split", flow.ids, flow.width, { attr: desc, group: "neck" });
    const outs: number[] = [];
    let width = 0;
    for (const t of transformers) {
      if (!Array.isArray(t)) continue;
      const [tn, est, cols] = t as [unknown, unknown, unknown];
      if (est === "drop") continue;
      const ncols = Array.isArray(cols) ? cols.length : arr(cols)?.shape[0] ?? flow.width;
      if (est === "passthrough") {
        const p = b.add(`${String(tn)} (passthrough)`, "passthrough", "Identity", [split], ncols ?? null, { attr: { columns: cols as unknown } });
        outs.push(p); width += ncols ?? 0;
        continue;
      }
      const r = emit(b, est, String(tn), { ids: [split], width: ncols ?? null });
      if (Array.isArray(cols)) b.layers[r.ids[0]] && ((b.layers[r.ids[0]].attr.columns as unknown) ??= cols);
      outs.push(...r.ids); width += r.width ?? 0;
    }
    const cat = b.add(`${label || name}.hstack`, "hstack", "Concat", outs, width || null, { group: "neck" });
    return { ids: [cat], width: width || null };
  }

  // Tree ensembles / bagging: parallel members + aggregation.
  const members = attr(o, "estimators_");
  if (members != null && !Array.isArray(attr(o, "estimators"))) {
    const list = arrItems(members).flatMap((m) => (Array.isArray(m) ? m : arrItems(m).length ? arrItems(m) : [m])).filter((m) => m instanceof PyObj) as PyObj[];
    if (list.length) {
      const ids: number[] = [];
      let total = 0, flops = 0, width: number | null = null;
      list.forEach((m, i) => {
        const e = analyze(m, flow.width);
        total += e.params; flops += e.cost.flops; width = e.out;
        if (i < MAX_BRANCHES) ids.push(b.add(`${label || name}.estimators_[${i}]`, clsName(m), e.canon, flow.ids, e.out, { ...e, attr: describe(m) }));
      });
      if (list.length > MAX_BRANCHES) {
        const rest = list.slice(MAX_BRANCHES).reduce((s, m) => s + analyze(m, flow.width).params, 0);
        ids.push(b.add(`${label || name}: ${list.length - MAX_BRANCHES} more`, "Estimators", "Tree", flow.ids, width, { params: rest, attr: { count: list.length - MAX_BRANCHES } }));
      }
      const classes = arr(attr(o, "classes_"));
      const agg = b.add(title, name, "Mean", ids, classes?.shape[0] ?? width, {
        attr: { ...desc, n_members: list.length, total_member_params: total },
        cost: { macs: 0, flops: flops + list.length },
        insight: `${name}: ${list.length} members, ${total} total ${list[0] && clsName(list[0]).includes("Tree") ? "tree nodes" : "parameters"} (counted on the member nodes).`,
      });
      return { ids: [agg], width: b.layers[agg].outs[0].s?.[0] ?? width };
    }
  }

  // Voting / stacking: named member estimators in parallel, then the meta node.
  const named = attr(o, "estimators");
  if (Array.isArray(named) && /Voting|Stacking/.test(name)) {
    const fitted = arrItems(attr(o, "estimators_"));
    const ids: number[] = [];
    named.forEach((t, i) => {
      if (!Array.isArray(t)) return;
      const est = fitted[i] ?? t[1];
      ids.push(...emit(b, est, String(t[0]), flow).ids);
    });
    const fin = attr(o, "final_estimator_");
    if (fin instanceof PyObj) return emit(b, fin, `${label || name}.final_estimator`, { ids, width: ids.length });
    const classes = arr(attr(o, "classes_"));
    const agg = b.add(title, name, "Mean", ids, classes?.shape[0] ?? null, { attr: desc });
    return { ids: [agg], width: classes?.shape[0] ?? null };
  }

  // MLP: one node per dense layer.
  const coefs = attr(o, "coefs_");
  if (Array.isArray(coefs) && coefs.length) {
    const intercepts = Array.isArray(attr(o, "intercepts_")) ? (attr(o, "intercepts_") as unknown[]) : [];
    const act = String(attr(o, "activation") ?? "relu");
    const outAct = String(attr(o, "out_activation_") ?? "identity");
    let ids = flow.ids;
    let width = flow.width;
    coefs.forEach((c, i) => {
      const W = arr(c), B = arr(intercepts[i]);
      if (!W) return;
      const [nin, nout] = W.shape;
      const last = i === coefs.length - 1;
      const macs = numel(W);
      const id = b.add(`${label || name}.layer${i + 1}`, last ? `Dense+${outAct}` : `Dense+${act}`, "Gemm", ids, nout, {
        params: numel(W) + numel(B), cost: { macs, flops: 2 * macs + 2 * nout }, w: arrStats(W),
        attr: { in_features: nin, out_features: nout, activation: last ? outAct : act, ...(i === 0 ? desc : {}) },
        insight: `${name} hidden layer ${i + 1}: ${nin}→${nout} dense (${last ? outAct : act}).`,
      });
      ids = [id]; width = nout;
    });
    return { ids, width };
  }

  const e = analyze(o, flow.width);
  const id = b.add(title, name, e.canon, flow.ids, e.out, { ...e, attr: desc });
  return { ids: [id], width: e.out };
}

function findEstimator(root: unknown, depth = 0): PyObj | null {
  if (isEstimator(root)) return root;
  if (depth > 3) return null;
  const kids = root instanceof Map ? [...root.values()] : Array.isArray(root) ? root : root instanceof PyObj ? [...root.attrs.values()] : [];
  for (const k of kids) { const f = findEstimator(k, depth + 1); if (f) return f; }
  return null;
}

function nFeatures(o: PyObj): number | null {
  const n = attr(o, "n_features_in_");
  if (typeof n === "number") return n;
  const steps = attr(o, "steps");
  if (Array.isArray(steps)) for (const s of steps) if (Array.isArray(s) && s[1] instanceof PyObj) { const r = nFeatures(s[1]); if (r != null) return r; }
  const t = attr(o, "transformers_");
  if (Array.isArray(t)) {
    const cols = t.flatMap((x) => (Array.isArray(x) && Array.isArray(x[2]) ? x[2] : []));
    if (cols.length) return new Set(cols).size;
  }
  return null;
}

function version(o: unknown, depth = 0): string {
  if (o instanceof PyObj) {
    const v = attr(o, "_sklearn_version");
    if (typeof v === "string") return v;
    if (depth < 3) for (const x of o.attrs.values()) { const r = version(x, depth + 1); if (r) return r; }
  } else if (Array.isArray(o) && depth < 3) for (const x of o) { const r = version(x, depth + 1); if (r) return r; }
  return "";
}

// ── Container handling ───────────────────────────────────────────────────────

type Payload = { pickle: Uint8Array | null; container: string; note?: string };

function unwrap(buf: Uint8Array): Payload {
  if (buf[0] === 0x80) return { pickle: buf, container: "pickle" };
  if (buf[0] === 0x78 && ((buf[0] << 8) | buf[1]) % 31 === 0) {
    return { pickle: inflateRaw(buf.subarray(2)), container: "joblib (zlib)" };
  }
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    const flg = buf[3];
    let p = 10;
    if (flg & 4) p += 2 + (buf[p] | (buf[p + 1] << 8));
    if (flg & 8) { while (buf[p]) p++; p++; }
    if (flg & 16) { while (buf[p]) p++; p++; }
    if (flg & 2) p += 2;
    return { pickle: inflateRaw(buf.subarray(p)), container: "joblib (gzip)" };
  }
  const head = String.fromCharCode(...buf.subarray(0, 6));
  if (head.startsWith("ZF")) return { pickle: null, container: "joblib (legacy zfile)", note: "Legacy joblib (<0.10) compressed file; re-save with a current joblib." };
  if (head.startsWith("\xfd7zXZ") || (buf[0] === 0x5d && buf[1] === 0 && buf[2] === 0)) return { pickle: null, container: "joblib (lzma/xz)", note: "joblib lzma/xz compression is not supported in the browser; re-save with joblib.dump(compress=('zlib', 3)) or uncompressed." };
  if (head.startsWith("BZh")) return { pickle: null, container: "joblib (bz2)", note: "joblib bz2 compression is not supported; re-save with zlib or no compression." };
  if (buf[0] === 0x04 && buf[1] === 0x22 && buf[2] === 0x4d && buf[3] === 0x18) return { pickle: null, container: "joblib (lz4)", note: "joblib lz4 compression is not supported; re-save with zlib or no compression." };
  return { pickle: buf, container: "pickle" };
}

/** True when the file is (or decompresses to) a pickle that references sklearn classes. */
export function isSklearnPickle(input: Uint8Array): boolean {
  try {
    const p = unwrap(input.subarray(0, Math.min(input.length, 1 << 20)));
    const b = p.pickle ?? input;
    const head = new TextDecoder("latin1").decode(b.subarray(0, Math.min(b.length, 1 << 20)));
    return /sklearn\.|sklearn\n/.test(head) && (b[0] === 0x80 || p.container !== "pickle");
  } catch {
    return false;
  }
}

function infoModel(buf: Uint8Array, name: string, format: string, note: string, attrs: Record<string, unknown> = {}): Model {
  const l = makeLayer({ id: 0, name, type: format, canon: "Detected", group: "input", attr: { sizeBytes: buf.byteLength, ...attrs }, insight: note, framework: FRAMEWORK });
  l.mem = buf.byteLength;
  return { name, format, framework: FRAMEWORK, opset: 0, sizeBytes: buf.byteLength, inputShape: [], outputShape: [], layers: [l], edges: [], doc: note };
}

/** Parse a scikit-learn model saved with pickle or joblib. */
export function parseSklearn(input: ArrayBuffer | Uint8Array, name = "model.pkl"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const payload = unwrap(buf);
  const isJoblibName = /\.joblib$/i.test(name);
  if (!payload.pickle) return infoModel(buf, name, `scikit-learn (${payload.container})`, payload.note ?? "Unsupported container.", { container: payload.container });

  let root: unknown;
  const seen = { joblib: false };
  try {
    root = unpickle(payload.pickle, handlers(seen));
  } catch (e) {
    return infoModel(buf, name, "Pickle", `Could not decode the pickle stream: ${(e as Error).message}`);
  }
  const est = findEstimator(root);
  if (!est) {
    const top = root instanceof PyObj ? `${clsModule(root)}.${clsName(root)}` : root instanceof Map ? "dict" : Array.isArray(root) ? "list" : typeof root;
    return infoModel(buf, name, "Pickle", `No scikit-learn estimator found (top-level object: ${top}).`, { topLevel: top });
  }

  const b = new Builder();
  const nf = nFeatures(est);
  const inId = b.add("X", "Input", "Input", [], nf, { group: "input", attr: { n_features_in: nf } });
  const res = emit(b, est, "", { ids: [inId], width: nf });
  for (const id of res.ids) b.layers[id].group = "output";
  const container = payload.container !== "pickle" ? payload.container : isJoblibName || seen.joblib ? "joblib" : "pickle";
  const ver = version(est);
  return {
    name, format: `scikit-learn (${container})`, framework: FRAMEWORK, opset: 0, sizeBytes: buf.byteLength,
    inputShape: nf != null ? [1, nf] : [], outputShape: res.width != null ? [1, res.width] : [],
    layers: b.layers, edges: b.edges.edges,
    producer: `${clsName(est)}${ver ? `, sklearn ${ver}` : ""}`,
    doc: "Predict-time graph reconstructed from the fitted estimator's state (no code executed).",
  };
}

