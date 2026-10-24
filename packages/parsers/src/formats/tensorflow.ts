import type { Model, ModelLayer } from "../types";
import { readTensorBundle } from "../common/tensorBundle";
import { ProtoReader, WT_LEN, WT_VARINT } from "../common/protobufWire";
import { parseProtoText, type PTMsg, type PTValue } from "../common/protoText";
import {
  broadcast, cleanShape, DYN, EdgeSet, elemCost, isFloatDtype, linearCost, makeLayer, NO_COST, numel,
  padOut, prod, sampleBytes, weightStats, type Cost, type Shape,
} from "../common/graphKit";

// TensorFlow graphs:
//   frozen GraphDef (.pb)      GraphDef { node=1 NodeDef, library=2, versions=4 }
//   SavedModel (saved_model.pb) SavedModel { saved_model_schema_version=1, meta_graphs=2 MetaGraphDef }
//   MetaGraphDef (.meta)       { meta_info_def=1, graph_def=2, signature_def=5 }
//   text GraphDef (.pbtxt)     the same messages in protobuf text format
// NodeDef { name=1, op=2, input=3, device=4, attr=5 map<string, AttrValue> }.
// TF2 SavedModels keep the real compute inside library functions reached via
// (Stateful)PartitionedCall; we follow the serving signature and inline those
// calls. Const/variable nodes are folded into their consumers as params.

// ── Decoded structures ───────────────────────────────────────────────────────

type TTensor = { dtype: string; shape: Shape; content?: Uint8Array; vals?: number[] };
type TList = { s: string[]; i: number[]; f: number[]; b: boolean[]; type: string[]; shape: (Shape | undefined)[]; tensor: TTensor[]; func: string[] };
type TAttr = {
  s?: string; i?: number; f?: number; b?: boolean; type?: string; shape?: Shape | null;
  tensor?: TTensor; list?: TList; func?: string; placeholder?: string;
};
type TNode = { name: string; op: string; inputs: string[]; device: string; attrs: Map<string, TAttr> };
type TFunc = { name: string; inArgs: string[]; outArgs: string[]; nodes: TNode[]; ret: Map<string, string> };
type TGraph = { nodes: TNode[]; funcs: Map<string, TFunc>; producer?: number };
type TSig = { key: string; inputs: { key: string; name: string }[]; outputs: { key: string; name: string }[] };
type TFile = {
  kind: "GraphDef" | "SavedModel" | "MetaGraphDef"; graph: TGraph; sigs: TSig[]; tfVersion?: string; tags?: string[];
  /** SavedModel object graph: how graph variables map to checkpoint keys. */
  objects?: ObjectGraph;
};

const DT: Record<number, string> = {
  1: "float32", 2: "float64", 3: "int32", 4: "uint8", 5: "int16", 6: "int8", 7: "string", 8: "complex64",
  9: "int64", 10: "bool", 11: "qint8", 12: "quint8", 13: "qint32", 14: "bfloat16", 15: "qint16", 16: "quint16",
  17: "uint16", 18: "complex128", 19: "float16", 20: "resource", 21: "variant", 22: "uint32", 23: "uint64",
};
const DT_TEXT: Record<string, string> = {
  DT_FLOAT: "float32", DT_DOUBLE: "float64", DT_INT32: "int32", DT_UINT8: "uint8", DT_INT16: "int16", DT_INT8: "int8",
  DT_STRING: "string", DT_INT64: "int64", DT_BOOL: "bool", DT_BFLOAT16: "bfloat16", DT_HALF: "float16",
  DT_UINT16: "uint16", DT_UINT32: "uint32", DT_UINT64: "uint64", DT_RESOURCE: "resource", DT_VARIANT: "variant",
  DT_QINT8: "qint8", DT_QUINT8: "quint8", DT_QINT32: "qint32", DT_COMPLEX64: "complex64",
};
// Ref dtypes (DT_FLOAT_REF = 101 …) map onto their base type.
const dtName = (n: number) => DT[n > 100 ? n - 100 : n] ?? `dtype_${n}`;

// ── Binary decoding ──────────────────────────────────────────────────────────

function readShape(r: ProtoReader): Shape | null {
  const dims: Shape = [];
  let unknownRank = false;
  for (let t; (t = r.next()); ) {
    if (t.field === 2) {
      const d = r.sub();
      let size = 0; // proto3 default: an omitted size is 0, not unknown
      for (let u; (u = d.next()); ) {
        if (u.field === 1) size = d.int64();
        else d.skip(u.wireType);
      }
      dims.push(size < 0 ? DYN : size);
    } else if (t.field === 3) unknownRank = r.bool();
    else r.skip(t.wireType);
  }
  return unknownRank ? null : dims;
}

function readTensor(r: ProtoReader): TTensor {
  const t: TTensor = { dtype: "float32", shape: [] };
  const vals: number[] = [];
  for (let f; (f = r.next()); ) {
    switch (f.field) {
      case 1: t.dtype = dtName(r.varint()); break;
      case 2: t.shape = readShape(r.sub()) ?? []; break;
      case 4: t.content = r.bytes(); break;
      case 5: vals.push(...r.packedFloats(f.wireType)); break;
      case 6:
        if (f.wireType === WT_LEN) { const s = r.sub(); while (!s.eof) vals.push(s.double()); }
        else vals.push(r.double());
        break;
      case 7: case 10: case 11: case 13: case 16: case 17:
        vals.push(...r.packedVarints(f.wireType)); break;
      default: r.skip(f.wireType);
    }
  }
  if (vals.length) t.vals = vals;
  return t;
}

function readList(r: ProtoReader): TList {
  const l: TList = { s: [], i: [], f: [], b: [], type: [], shape: [], tensor: [], func: [] };
  for (let t; (t = r.next()); ) {
    switch (t.field) {
      case 2: l.s.push(r.string()); break;
      case 3: l.i.push(...r.packedVarints(t.wireType)); break;
      case 4: l.f.push(...r.packedFloats(t.wireType).map(round6)); break;
      case 5: l.b.push(...r.packedVarints(t.wireType).map((x) => x !== 0)); break;
      case 6: l.type.push(...r.packedVarints(t.wireType).map(dtName)); break;
      case 7: l.shape.push(readShape(r.sub()) ?? undefined); break;
      case 8: l.tensor.push(readTensor(r.sub())); break;
      case 9: l.func.push(readFuncName(r.sub())); break;
      default: r.skip(t.wireType);
    }
  }
  return l;
}

function readFuncName(r: ProtoReader): string {
  let name = "";
  for (let t; (t = r.next()); ) {
    if (t.field === 1) name = r.string();
    else r.skip(t.wireType);
  }
  return name;
}

const round6 = (x: number) => Math.round(x * 1e6) / 1e6;

function readAttrValue(r: ProtoReader): TAttr {
  const a: TAttr = {};
  for (let t; (t = r.next()); ) {
    switch (t.field) {
      case 1: a.list = readList(r.sub()); break;
      case 2: a.s = r.string(); break;
      case 3: a.i = r.int64(); break;
      case 4: a.f = round6(r.float()); break;
      case 5: a.b = r.bool(); break;
      case 6: a.type = dtName(r.varint()); break;
      case 7: a.shape = readShape(r.sub()); break;
      case 8: a.tensor = readTensor(r.sub()); break;
      case 9: a.placeholder = r.string(); break;
      case 10: a.func = readFuncName(r.sub()); break;
      default: r.skip(t.wireType);
    }
  }
  return a;
}

function readAttrMap(r: ProtoReader, into: Map<string, TAttr>): void {
  let key = "";
  let val: TAttr = {};
  for (let t; (t = r.next()); ) {
    if (t.field === 1) key = r.string();
    else if (t.field === 2) val = readAttrValue(r.sub());
    else r.skip(t.wireType);
  }
  into.set(key, val);
}

function readNode(r: ProtoReader): TNode {
  const n: TNode = { name: "", op: "", inputs: [], device: "", attrs: new Map() };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) n.name = r.string();
    else if (t.field === 2) n.op = r.string();
    else if (t.field === 3) n.inputs.push(r.string());
    else if (t.field === 4) n.device = r.string();
    else if (t.field === 5) readAttrMap(r.sub(), n.attrs);
    else r.skip(t.wireType);
  }
  return n;
}

function readArgNames(r: ProtoReader): string {
  let name = "";
  for (let t; (t = r.next()); ) {
    if (t.field === 1) name = r.string();
    else r.skip(t.wireType);
  }
  return name;
}

function readFunction(r: ProtoReader): TFunc {
  const f: TFunc = { name: "", inArgs: [], outArgs: [], nodes: [], ret: new Map() };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) {
      // OpDef signature { name=1, input_arg=2, output_arg=3 }
      const s = r.sub();
      for (let u; (u = s.next()); ) {
        if (u.field === 1) f.name = s.string();
        else if (u.field === 2) f.inArgs.push(readArgNames(s.sub()));
        else if (u.field === 3) f.outArgs.push(readArgNames(s.sub()));
        else s.skip(u.wireType);
      }
    } else if (t.field === 3) f.nodes.push(readNode(r.sub()));
    else if (t.field === 4) {
      const e = r.sub();
      let k = "", v = "";
      for (let u; (u = e.next()); ) {
        if (u.field === 1) k = e.string();
        else if (u.field === 2) v = e.string();
        else e.skip(u.wireType);
      }
      f.ret.set(k, v);
    } else r.skip(t.wireType);
  }
  return f;
}

function readGraph(r: ProtoReader): TGraph {
  const g: TGraph = { nodes: [], funcs: new Map() };
  for (let t; (t = r.next()); ) {
    if (t.field === 1) g.nodes.push(readNode(r.sub()));
    else if (t.field === 2) {
      const lib = r.sub();
      for (let u; (u = lib.next()); ) {
        if (u.field === 1) { const f = readFunction(lib.sub()); g.funcs.set(f.name, f); }
        else lib.skip(u.wireType);
      }
    } else if (t.field === 4) {
      const v = r.sub();
      for (let u; (u = v.next()); ) {
        if (u.field === 1) g.producer = v.int64();
        else v.skip(u.wireType);
      }
    } else r.skip(t.wireType);
  }
  return g;
}

function readSigTensorMap(r: ProtoReader): { key: string; name: string } {
  let key = "", name = "";
  for (let t; (t = r.next()); ) {
    if (t.field === 1) key = r.string();
    else if (t.field === 2) {
      const ti = r.sub(); // TensorInfo { name=1, dtype=2, tensor_shape=3 }
      for (let u; (u = ti.next()); ) {
        if (u.field === 1) name = ti.string();
        else ti.skip(u.wireType);
      }
    } else r.skip(t.wireType);
  }
  return { key, name };
}

type ObjectGraph = {
  /** Object-graph node id → checkpoint key, for variable nodes. */
  keyOf: Map<number, string>;
  /** Variable name → checkpoint key(s); Keras exports track each variable twice. */
  byName: Map<string, string[]>;
  /** Concrete function name → object-graph node ids of its captured inputs, in order. */
  bound: Map<string, number[]>;
};

/**
 * SavedObjectGraph { nodes=1 SavedObject { children=1 {node_id=1, local_name=2},
 * variable=7 {name=6} }, concrete_functions=2 map<name, {bound_inputs=2}> }.
 * A variable's checkpoint key is its breadth-first path from the root object,
 * each step escaped the way TF does ("." → "..", "/" → ".S"), plus
 * "/.ATTRIBUTES/VARIABLE_VALUE".
 */
function readObjectGraph(r: ProtoReader): ObjectGraph {
  const nodes: { children: { id: number; name: string }[]; varName?: string }[] = [];
  const bound = new Map<string, number[]>();
  for (let t; (t = r.next()); ) {
    if (t.field === 2) {
      const e = r.sub(); // map entry { key=1, value=2 SavedConcreteFunction { bound_inputs=2 } }
      let fn = "";
      const ids: number[] = [];
      for (let u; (u = e.next()); ) {
        if (u.field === 1) fn = e.string();
        else if (u.field === 2) {
          const cf = e.sub();
          for (let w; (w = cf.next()); ) {
            if (w.field === 2) ids.push(...cf.packedVarints(w.wireType));
            else cf.skip(w.wireType);
          }
        } else e.skip(u.wireType);
      }
      if (fn) bound.set(fn, ids);
      continue;
    }
    if (t.field !== 1) { r.skip(t.wireType); continue; }
    const o = r.sub();
    const node: (typeof nodes)[number] = { children: [] };
    for (let u; (u = o.next()); ) {
      if (u.field === 1) {
        const c = o.sub();
        let id = -1, name = "";
        for (let w; (w = c.next()); ) {
          if (w.field === 1) id = c.varint();
          else if (w.field === 2) name = c.string();
          else c.skip(w.wireType);
        }
        node.children.push({ id, name });
      } else if (u.field === 7) {
        const v = o.sub();
        for (let w; (w = v.next()); ) {
          if (w.field === 6) node.varName = v.string();
          else v.skip(w.wireType);
        }
      } else o.skip(u.wireType);
    }
    nodes.push(node);
  }
  const path = new Map<number, string>([[0, ""]]);
  const queue = [0];
  while (queue.length) {
    const id = queue.shift()!;
    for (const c of nodes[id]?.children ?? []) {
      if (path.has(c.id) || !nodes[c.id]) continue;
      const step = c.name.replace(/\./g, "..").replace(/\//g, ".S");
      path.set(c.id, path.get(id) ? `${path.get(id)}/${step}` : step);
      queue.push(c.id);
    }
  }
  const keyOf = new Map<number, string>();
  const byName = new Map<string, string[]>();
  nodes.forEach((n, id) => {
    if (!n.varName || !path.has(id)) return;
    const key = `${path.get(id)}/.ATTRIBUTES/VARIABLE_VALUE`;
    keyOf.set(id, key);
    byName.set(n.varName, [...(byName.get(n.varName) ?? []), key]);
  });
  return { keyOf, byName, bound };
}

function readMetaGraph(r: ProtoReader): { graph: TGraph; sigs: TSig[]; tfVersion?: string; tags: string[]; objects?: ObjectGraph } {
  let graph: TGraph = { nodes: [], funcs: new Map() };
  let objects: ObjectGraph | undefined;
  const sigs: TSig[] = [];
  const tags: string[] = [];
  let tfVersion: string | undefined;
  for (let t; (t = r.next()); ) {
    if (t.field === 1) {
      const mi = r.sub(); // MetaInfoDef { tags=4, tensorflow_version=5 }
      for (let u; (u = mi.next()); ) {
        if (u.field === 4) tags.push(mi.string());
        else if (u.field === 5) tfVersion = mi.string();
        else mi.skip(u.wireType);
      }
    } else if (t.field === 2) graph = readGraph(r.sub());
    else if (t.field === 7) objects = readObjectGraph(r.sub());
    else if (t.field === 5) {
      const e = r.sub();
      const sig: TSig = { key: "", inputs: [], outputs: [] };
      for (let u; (u = e.next()); ) {
        if (u.field === 1) sig.key = e.string();
        else if (u.field === 2) {
          const sd = e.sub(); // SignatureDef { inputs=1, outputs=2, method_name=3 }
          for (let w; (w = sd.next()); ) {
            if (w.field === 1) sig.inputs.push(readSigTensorMap(sd.sub()));
            else if (w.field === 2) sig.outputs.push(readSigTensorMap(sd.sub()));
            else sd.skip(w.wireType);
          }
        } else e.skip(u.wireType);
      }
      sigs.push(sig);
    } else r.skip(t.wireType);
  }
  return { graph, sigs, tfVersion, tags, objects };
}

const OP_RE = /^[A-Za-z_>][A-Za-z0-9_>.]*$/;

/** Does `buf` decode as a GraphDef whose first node has a plausible op name? */
function looksLikeGraphDef(r: ProtoReader): boolean {
  try {
    for (let t; (t = r.next()); ) {
      if (t.field === 1 && t.wireType === WT_LEN) {
        const n = readNode(r.sub());
        return n.name.length > 0 && OP_RE.test(n.op);
      }
      r.skip(t.wireType);
    }
  } catch {
    return false;
  }
  return false;
}

/** Classify and decode a binary TF protobuf (SavedModel, MetaGraphDef or GraphDef). */
export function decodeTensorflow(buf: Uint8Array): TFile {
  const top = new ProtoReader(buf);
  const fields: { field: number; wt: number; start: number }[] = [];
  try {
    for (let t; (t = top.next()) && fields.length < 64; ) {
      fields.push({ field: t.field, wt: t.wireType, start: top.pos });
      top.skip(t.wireType);
    }
  } catch {
    throw new Error("Not a TensorFlow protobuf (malformed wire data).");
  }
  const has = (f: number, wt: number) => fields.some((x) => x.field === f && x.wt === wt);

  // SavedModel: only {1: varint schema version, 2: MetaGraphDef}.
  if (fields.length && fields.every((x) => (x.field === 1 && x.wt === WT_VARINT) || (x.field === 2 && x.wt === WT_LEN))
      && has(2, WT_LEN) && !looksLikeGraphDef(new ProtoReader(buf))) {
    const r = new ProtoReader(buf);
    for (let t; (t = r.next()); ) {
      if (t.field === 2) {
        const mg = readMetaGraph(r.sub());
        return { kind: "SavedModel", ...mg };
      }
      r.skip(t.wireType);
    }
  }
  if (looksLikeGraphDef(new ProtoReader(buf))) {
    return { kind: "GraphDef", graph: readGraph(new ProtoReader(buf)), sigs: [] };
  }
  // MetaGraphDef (.meta): field 2 holds a GraphDef.
  if (has(2, WT_LEN)) {
    const mg = readMetaGraph(new ProtoReader(buf));
    if (mg.graph.nodes.length) return { kind: "MetaGraphDef", ...mg };
  }
  const g = readGraph(new ProtoReader(buf));
  if (!g.nodes.length && !g.funcs.size) throw new Error("No TensorFlow graph nodes found.");
  return { kind: "GraphDef", graph: g, sigs: [] };
}

// ── Text format (.pbtxt) ─────────────────────────────────────────────────────

const ptStrBytes = (s: string) => Uint8Array.from(s, (c) => c.charCodeAt(0) & 0xff);
const ptArr = (m: PTMsg | undefined, k: string): PTValue[] => m?.[k] ?? [];
const ptS = (m: PTMsg | undefined, k: string) => { const v = ptArr(m, k)[0]; return v == null ? undefined : String(v); };
const ptM = (m: PTMsg | undefined, k: string) => ptArr(m, k).filter((v): v is PTMsg => typeof v === "object");
const ptType = (v: PTValue | undefined) => (typeof v === "number" ? dtName(v) : DT_TEXT[String(v)] ?? String(v).toLowerCase());

function textShape(m: PTMsg): Shape | null {
  if (ptArr(m, "unknown_rank")[0] === true) return null;
  return ptM(m, "dim").map((d) => { const s = Number(ptArr(d, "size")[0] ?? 0); return s < 0 ? DYN : s; });
}

function textTensor(m: PTMsg): TTensor {
  const t: TTensor = { dtype: ptType(ptArr(m, "dtype")[0] ?? "DT_FLOAT"), shape: textShape(ptM(m, "tensor_shape")[0] ?? {}) ?? [] };
  const c = ptS(m, "tensor_content");
  if (c != null) t.content = ptStrBytes(c);
  const vals = ["float_val", "double_val", "int_val", "int64_val", "bool_val", "half_val"]
    .flatMap((k) => ptArr(m, k).map((v) => (typeof v === "boolean" ? Number(v) : Number(v))));
  if (vals.length) t.vals = vals;
  return t;
}

function textAttr(v: PTMsg): TAttr {
  const a: TAttr = {};
  if (v.s) a.s = ptS(v, "s");
  if (v.i) a.i = Number(v.i[0]);
  if (v.f) a.f = Number(v.f[0]);
  if (v.b) a.b = v.b[0] === true || v.b[0] === "true";
  if (v.type) a.type = ptType(v.type[0]);
  if (v.shape) a.shape = textShape(ptM(v, "shape")[0] ?? {});
  if (v.tensor) a.tensor = textTensor(ptM(v, "tensor")[0] ?? {});
  if (v.func) a.func = ptS(ptM(v, "func")[0], "name");
  if (v.placeholder) a.placeholder = ptS(v, "placeholder");
  if (v.list) {
    const l = ptM(v, "list")[0] ?? {};
    a.list = {
      s: ptArr(l, "s").map(String), i: ptArr(l, "i").map(Number), f: ptArr(l, "f").map(Number),
      b: ptArr(l, "b").map((x) => x === true || x === "true"), type: ptArr(l, "type").map(ptType),
      shape: ptM(l, "shape").map((s) => textShape(s) ?? undefined), tensor: ptM(l, "tensor").map(textTensor),
      func: ptM(l, "func").map((f) => ptS(f, "name") ?? ""),
    };
  }
  return a;
}

function textNode(m: PTMsg): TNode {
  const attrs = new Map<string, TAttr>();
  for (const e of ptM(m, "attr")) attrs.set(ptS(e, "key") ?? "", textAttr(ptM(e, "value")[0] ?? {}));
  return { name: ptS(m, "name") ?? "", op: ptS(m, "op") ?? "", inputs: ptArr(m, "input").map(String), device: ptS(m, "device") ?? "", attrs };
}

/** Decode a text-format GraphDef (or MetaGraphDef/SavedModel in text form). */
export function decodeTensorflowText(text: string): TFile {
  const root = parseProtoText(text);
  const graphMsg = (m: PTMsg): TGraph => {
    const g: TGraph = { nodes: ptM(m, "node").map(textNode), funcs: new Map() };
    for (const lib of ptM(m, "library")) {
      for (const f of ptM(lib, "function")) {
        const sig = ptM(f, "signature")[0];
        const fn: TFunc = {
          name: ptS(sig, "name") ?? "", inArgs: ptM(sig, "input_arg").map((a) => ptS(a, "name") ?? ""),
          outArgs: ptM(sig, "output_arg").map((a) => ptS(a, "name") ?? ""), nodes: ptM(f, "node_def").map(textNode), ret: new Map(),
        };
        for (const e of ptM(f, "ret")) fn.ret.set(ptS(e, "key") ?? "", ptS(e, "value") ?? "");
        g.funcs.set(fn.name, fn);
      }
    }
    return g;
  };
  if (root.node) return { kind: "GraphDef", graph: graphMsg(root), sigs: [] };
  const mgs = root.meta_graphs ? ptM(root, "meta_graphs") : root.graph_def ? [root] : [];
  const mg = mgs[0];
  if (!mg) throw new Error("No TensorFlow graph found in text proto.");
  const sigs: TSig[] = ptM(mg, "signature_def").map((e) => {
    const sd = ptM(e, "value")[0];
    const io = (k: string) => ptM(sd, k).map((x) => ({ key: ptS(x, "key") ?? "", name: ptS(ptM(x, "value")[0], "name") ?? "" }));
    return { key: ptS(e, "key") ?? "", inputs: io("inputs"), outputs: io("outputs") };
  });
  return { kind: root.meta_graphs ? "SavedModel" : "MetaGraphDef", graph: graphMsg(ptM(mg, "graph_def")[0] ?? {}), sigs };
}

// ── Graph flattening (function inlining) ─────────────────────────────────────

type Scope = { prefix: string; nodes: Map<string, TNode>; args: Map<string, Ref> | null; depth: number };
type Ref = { scope: Scope; name: string; idx: number };
type FlatNode = { key: string; node: TNode; scope: Scope; ins: (Resolved | null)[] };
type Resolved = { fn: FlatNode; idx: number };

const CALL_OPS = new Set(["PartitionedCall", "StatefulPartitionedCall"]);
const PASSTHROUGH = new Set(["Identity", "IdentityN", "Snapshot", "StopGradient", "PreventGradient", "CheckNumerics", "EnsureShape"]);
const WEIGHT_SRC = new Set(["Const", "VariableV2", "Variable", "VarHandleOp", "HostConst"]);
const WEIGHT_PASS = new Set(["ReadVariableOp", "Identity", "Cast", "Enter", "Dequantize"]);
// Ops that must stay visible even when fed only by constants.
const NEVER_FOLD = new Set(["Placeholder", "PlaceholderWithDefault", "RandomUniform", "RandomStandardNormal", "Print", "PrintV2"]);
const HIDDEN = new Set([
  "NoOp", "Assert", "AssignVariableOp", "Assign", "VarIsInitializedOp", "SaveV2", "RestoreV2", "MergeV2Checkpoints",
  "ShardedFilename", "StringJoin", "StaticRegexFullMatch", "SaveSlices", "RestoreSlice", "DisableCopyOnRead",
]);
const MAX_DEPTH = 8;

/** Parse "name", "name:1" (graph) or "name:out_arg:1" (function body) refs; control deps return null. */
function parseRef(s: string, scope: Scope): Ref | null {
  if (s.startsWith("^")) return null;
  const parts = s.split(":");
  if (parts.length === 1) return { scope, name: parts[0], idx: 0 };
  if (parts.length === 2) return { scope, name: parts[0], idx: Number(parts[1]) || 0 };
  // Function bodies use "node:output_arg:i"; i indexes within that arg, which
  // matches the flat output index for the common single-list-arg ops.
  return { scope, name: parts[0], idx: Number(parts[parts.length - 1]) || 0 };
}

class Flattener {
  private created: FlatNode[] = [];
  private queue = 0; // index into `created` of the next node whose inputs need resolving
  private memo = new Map<string, FlatNode>();
  private callScopes = new Map<string, { scope: Scope; fn: TFunc }>();
  constructor(private funcs: Map<string, TFunc>) {}

  private calleeOf(n: TNode): TFunc | undefined {
    if (CALL_OPS.has(n.op)) return this.funcs.get(n.attrs.get("f")?.func ?? "");
    return this.funcs.get(n.op); // a library function invoked directly as an op
  }

  resolve(ref: Ref | null, guard = 0): Resolved | null {
    if (!ref || guard > 256) return null;
    const { scope } = ref;
    // Function argument → caller-side tensor.
    if (scope.args && scope.args.has(ref.name) && !scope.nodes.has(ref.name)) {
      return this.resolve(scope.args.get(ref.name)!, guard + 1);
    }
    const node = scope.nodes.get(ref.name);
    if (!node) return null;
    const dataIns = node.inputs.filter((s) => !s.startsWith("^"));

    const callee = scope.depth < MAX_DEPTH ? this.calleeOf(node) : undefined;
    if (callee) {
      const key = scope.prefix + node.name;
      let cs = this.callScopes.get(key);
      if (!cs) {
        const args = new Map<string, Ref>();
        callee.inArgs.forEach((a, i) => { const r = parseRef(dataIns[i] ?? "", scope); if (r) args.set(a, r); });
        cs = {
          fn: callee,
          scope: { prefix: `${key}/`, nodes: new Map(callee.nodes.map((n) => [n.name, n])), args, depth: scope.depth + 1 },
        };
        this.callScopes.set(key, cs);
      }
      const out = cs.fn.ret.get(cs.fn.outArgs[ref.idx] ?? "");
      return out ? this.resolve(parseRef(out, cs.scope), guard + 1) : null;
    }
    if (PASSTHROUGH.has(node.op) && dataIns.length) {
      const which = node.op === "IdentityN" ? ref.idx : 0;
      return this.resolve(parseRef(dataIns[which] ?? dataIns[0], scope), guard + 1);
    }
    return { fn: this.flatten(node, scope), idx: ref.idx };
  }

  private flatten(node: TNode, scope: Scope): FlatNode {
    const key = scope.prefix + node.name;
    let f = this.memo.get(key);
    if (f) return f;
    f = { key, node, scope, ins: [] };
    this.memo.set(key, f);
    this.created.push(f); // inputs are resolved by drain(), iteratively (no deep recursion)
    return f;
  }

  /** Resolve inputs of every discovered node until the graph is closed. */
  drain(): void {
    while (this.queue < this.created.length) {
      const f = this.created[this.queue++];
      f.ins = f.node.inputs.filter((s) => !s.startsWith("^")).map((s) => this.resolve(parseRef(s, f.scope)));
    }
  }

  /** Nodes in topological order (producers first), stable w.r.t. discovery order. */
  get flat(): FlatNode[] {
    this.drain();
    const order: FlatNode[] = [];
    const state = new Map<FlatNode, number>(); // 1 = on stack, 2 = done
    for (const root of this.created) {
      if (state.has(root)) continue;
      const stack: { f: FlatNode; i: number }[] = [{ f: root, i: 0 }];
      state.set(root, 1);
      while (stack.length) {
        const top = stack[stack.length - 1];
        const nextIn = top.f.ins[top.i++];
        if (nextIn) {
          if (!state.has(nextIn.fn)) { state.set(nextIn.fn, 1); stack.push({ f: nextIn.fn, i: 0 }); }
          continue;
        }
        if (top.i <= top.f.ins.length) continue;
        state.set(top.f, 2);
        order.push(top.f);
        stack.pop();
      }
    }
    return order;
  }
}

// ── Op semantics ─────────────────────────────────────────────────────────────

const CANON: Record<string, string> = {
  Conv2D: "Conv", Conv3D: "Conv", DepthwiseConv2dNative: "Conv", Conv2DBackpropInput: "ConvTranspose",
  MatMul: "Gemm", BatchMatMul: "MatMul", BatchMatMulV2: "MatMul", BatchMatMulV3: "MatMul", Einsum: "MatMul",
  BiasAdd: "Add", Add: "Add", AddV2: "Add", AddN: "Add", Sub: "Sub", Mul: "Mul", RealDiv: "Div",
  Relu: "Relu", Relu6: "Relu", LeakyRelu: "LeakyRelu", Elu: "Elu", Selu: "Selu", Sigmoid: "Sigmoid", Tanh: "Tanh",
  Softmax: "Softmax", LogSoftmax: "LogSoftmax",
  FusedBatchNorm: "BatchNormalization", FusedBatchNormV2: "BatchNormalization", FusedBatchNormV3: "BatchNormalization",
  MaxPool: "MaxPool", AvgPool: "AveragePool", MaxPoolV2: "MaxPool",
  ConcatV2: "Concat", Concat: "Concat", Reshape: "Reshape", Squeeze: "Reshape", ExpandDims: "Reshape",
  Transpose: "Transpose", Pad: "Pad", PadV2: "Pad", MirrorPad: "Pad", Placeholder: "Input", PlaceholderWithDefault: "Input",
  ResizeBilinear: "Resize", ResizeNearestNeighbor: "Resize",
};

type Ctx = { ins: (Shape | undefined)[]; constIn: (i: number) => number[] | undefined; attrs: Map<string, TAttr> };

const attrInts = (a: Map<string, TAttr>, k: string) => a.get(k)?.list?.i;
const attrStr = (a: Map<string, TAttr>, k: string) => a.get(k)?.s;

/** Spatial axes for a 4-D/5-D tensor in this node's data_format. */
const isNCHW = (a: Map<string, TAttr>) => /^NC/.test(attrStr(a, "data_format") ?? "NHWC");

function inferShape(op: string, c: Ctx): Shape | undefined {
  const [x, y] = c.ins;
  const a = c.attrs;
  switch (op) {
    case "Placeholder": case "PlaceholderWithDefault": {
      const s = a.get("shape")?.shape;
      return s ?? (op === "PlaceholderWithDefault" ? x : undefined);
    }
    case "Conv2D": case "DepthwiseConv2dNative": {
      if (!x || !y || x.length !== 4) return undefined;
      const nchw = isNCHW(a);
      const st = attrInts(a, "strides") ?? [1, 1, 1, 1];
      const dl = attrInts(a, "dilations") ?? [1, 1, 1, 1];
      const pad = attrStr(a, "padding") ?? "VALID";
      const [hA, wA] = nchw ? [2, 3] : [1, 2];
      let oh: number, ow: number;
      if (pad === "EXPLICIT") {
        const ep = attrInts(a, "explicit_paddings") ?? [];
        const eff = (k: number, d: number) => d * (k - 1) + 1;
        oh = x[hA] < 0 ? DYN : Math.floor((x[hA] + (ep[2 * hA] ?? 0) + (ep[2 * hA + 1] ?? 0) - eff(y[0], dl[hA])) / st[hA]) + 1;
        ow = x[wA] < 0 ? DYN : Math.floor((x[wA] + (ep[2 * wA] ?? 0) + (ep[2 * wA + 1] ?? 0) - eff(y[1], dl[wA])) / st[wA]) + 1;
      } else {
        oh = padOut(x[hA], y[0], st[hA], pad, dl[hA]);
        ow = padOut(x[wA], y[1], st[wA], pad, dl[wA]);
      }
      const cout = op === "Conv2D" ? y[3] : y[2] * y[3];
      return nchw ? [x[0], cout, oh, ow] : [x[0], oh, ow, cout];
    }
    case "Conv2DBackpropInput": {
      const s = c.constIn(0);
      return s ? s.map((d) => (d < 0 ? DYN : d)) : undefined;
    }
    case "MaxPool": case "AvgPool": {
      if (!x || x.length !== 4) return undefined;
      const nchw = isNCHW(a);
      const k = attrInts(a, "ksize") ?? [1, 1, 1, 1];
      const st = attrInts(a, "strides") ?? [1, 1, 1, 1];
      const pad = attrStr(a, "padding") ?? "VALID";
      const [hA, wA] = nchw ? [2, 3] : [1, 2];
      const out = [...x];
      out[hA] = padOut(x[hA], k[hA], st[hA], pad);
      out[wA] = padOut(x[wA], k[wA], st[wA], pad);
      return out;
    }
    case "MatMul": {
      if (!x || !y) return undefined;
      const ta = a.get("transpose_a")?.b, tb = a.get("transpose_b")?.b;
      return [ta ? x[1] : x[0], tb ? y[0] : y[1]];
    }
    case "BatchMatMul": case "BatchMatMulV2": case "BatchMatMulV3": {
      if (!x || !y || x.length < 2 || y.length < 2) return undefined;
      const ax = a.get("adj_x")?.b, ay = a.get("adj_y")?.b;
      const M = ax ? x[x.length - 1] : x[x.length - 2];
      const N = ay ? y[y.length - 2] : y[y.length - 1];
      return [...broadcast(x.slice(0, -2), y.slice(0, -2)), M, N];
    }
    case "Add": case "AddV2": case "Sub": case "Mul": case "RealDiv": case "Maximum": case "Minimum":
    case "SquaredDifference": case "Pow": case "BiasAdd": case "FloorDiv": case "Less": case "Greater": case "Equal":
      return x && y ? (op === "BiasAdd" ? [...x] : broadcast(x, y)) : x;
    case "Mean": case "Sum": case "Max": case "Min": case "Prod": case "ArgMax": case "ArgMin": {
      const axes = c.constIn(1);
      if (!x || !axes) return undefined;
      const ax = new Set(axes.map((v) => (v < 0 ? v + x.length : v)));
      const keep = op.startsWith("Arg") ? false : a.get("keep_dims")?.b === true;
      return x.flatMap((d, i) => (ax.has(i) ? (keep ? [1] : []) : [d]));
    }
    case "Reshape": {
      const s = c.constIn(1);
      if (!s) return undefined;
      const out = s.map((d) => (d < 0 ? DYN : d));
      if (x && out.filter((d) => d === DYN).length === 1 && !x.includes(DYN)) {
        const total = prod(x);
        const rest = prod(out.filter((d) => d !== DYN));
        out[out.indexOf(DYN)] = rest ? total / rest : DYN;
      }
      return out;
    }
    case "Squeeze": {
      if (!x) return undefined;
      const dims = attrInts(a, "squeeze_dims") ?? [];
      const ax = new Set(dims.map((v) => (v < 0 ? v + x.length : v)));
      return x.filter((d, i) => (ax.size ? !ax.has(i) : d !== 1));
    }
    case "ExpandDims": {
      const d = c.constIn(1)?.[0];
      if (!x || d == null) return undefined;
      const out = [...x];
      out.splice(d < 0 ? x.length + d + 1 : d, 0, 1);
      return out;
    }
    case "ConcatV2": {
      const axisArr = c.constIn(c.ins.length - 1);
      const parts = c.ins.slice(0, -1).filter((s): s is Shape => !!s);
      if (!axisArr || !parts.length || parts.length !== c.ins.length - 1) return undefined;
      const out = [...parts[0]];
      const ax = axisArr[0] < 0 ? out.length + axisArr[0] : axisArr[0];
      out[ax] = parts.some((p) => p[ax] < 0) ? DYN : parts.reduce((s, p) => s + p[ax], 0);
      return out;
    }
    case "Transpose": {
      const perm = c.constIn(1);
      return x && perm ? perm.map((p) => x[p]) : undefined;
    }
    case "Pad": case "PadV2": case "MirrorPad": {
      const p = c.constIn(1);
      return x && p ? x.map((d, i) => (d < 0 ? d : d + p[2 * i] + p[2 * i + 1])) : undefined;
    }
    case "ResizeBilinear": case "ResizeNearestNeighbor": {
      const s = c.constIn(1);
      return x && s && x.length === 4 ? [x[0], s[0], s[1], x[3]] : undefined;
    }
    case "Shape":
      return x ? [x.length] : undefined;
    case "Tile": {
      const m = c.constIn(1);
      return x && m && m.length === x.length ? x.map((d, i) => (d < 0 ? DYN : d * m[i])) : undefined;
    }
    case "Pack": {
      const parts = c.ins.filter((s): s is Shape => !!s);
      if (!parts.length || parts.length !== c.ins.length) return undefined;
      const out = [...parts[0]];
      const ax0 = a.get("axis")?.i ?? 0;
      out.splice(ax0 < 0 ? out.length + ax0 + 1 : ax0, 0, parts.length);
      return out;
    }
    case "Relu": case "Relu6": case "LeakyRelu": case "Elu": case "Selu": case "Sigmoid": case "Tanh": case "Softmax":
    case "LogSoftmax": case "FusedBatchNorm": case "FusedBatchNormV2": case "FusedBatchNormV3": case "Neg": case "Sqrt":
    case "Rsqrt": case "Exp": case "Log": case "Square": case "Abs": case "Erf": case "Cast": case "Softplus": case "AddN":
    case "Floor": case "Round": case "Sign": case "Reciprocal": case "LRN":
      return x ? [...x] : undefined;
    default:
      return undefined;
  }
}

/**
 * Partial constant evaluation for small int vectors/scalars. Returns the value
 * (scalars as a 1-element array) or undefined when it isn't statically known.
 */
function evalInts(op: string, c: Ctx, nIn: number): number[] | undefined {
  const v = (i: number) => c.constIn(i);
  const all = (n: number) => { const r: number[][] = []; for (let i = 0; i < n; i++) { const x = v(i); if (!x) return undefined; r.push(x); } return r; };
  switch (op) {
    case "Shape": {
      const x = c.ins[0];
      return x && !x.includes(DYN) ? [...x] : undefined;
    }
    case "Identity": case "Cast": case "StopGradient": case "Reshape": case "Squeeze": case "ExpandDims":
      return v(0); // element order is unchanged
    case "Pack": case "ConcatV2": {
      const n = op === "ConcatV2" ? nIn - 1 : nIn;
      return all(n)?.flat();
    }
    case "Mul": case "Add": case "AddV2": case "Sub": case "FloorDiv": case "Maximum": case "Minimum": {
      const [a, b] = [v(0), v(1)];
      if (!a || !b) return undefined;
      const n = Math.max(a.length, b.length);
      const f = (x: number, y: number) => op === "Mul" ? x * y : op === "Sub" ? x - y : op === "FloorDiv" ? Math.floor(x / y)
        : op === "Maximum" ? Math.max(x, y) : op === "Minimum" ? Math.min(x, y) : x + y;
      return Array.from({ length: n }, (_, i) => f(a[a.length === 1 ? 0 : i], b[b.length === 1 ? 0 : i]));
    }
    case "Prod": {
      const a = v(0);
      return a ? [a.reduce((x, y) => x * y, 1)] : undefined;
    }
    case "StridedSlice": {
      // 1-D slices only (the shape-vector case).
      const [x, b, e, s] = [v(0), v(1), v(2), v(3)];
      if (!x || !b || !e || !s || b.length !== 1) return undefined;
      const bm = c.attrs.get("begin_mask")?.i ?? 0, em = c.attrs.get("end_mask")?.i ?? 0;
      const shrink = c.attrs.get("shrink_axis_mask")?.i ?? 0;
      const norm = (i: number) => (i < 0 ? x.length + i : i);
      const st = s[0] || 1;
      const lo = bm & 1 ? (st > 0 ? 0 : x.length - 1) : norm(b[0]);
      if (shrink & 1) return x[lo] == null ? undefined : [x[lo]];
      const hi = em & 1 ? (st > 0 ? x.length : -1) : norm(e[0]);
      const out: number[] = [];
      for (let i = lo; st > 0 ? i < Math.min(hi, x.length) : i > hi; i += st) out.push(x[i]);
      return out;
    }
    default:
      return undefined;
  }
}

function opCost(op: string, out: Shape | undefined, c: Ctx): Cost {
  const [x, w] = c.ins;
  switch (op) {
    case "Conv2D": return w ? linearCost(out, w[0] * w[1] * w[2]) : NO_COST;
    case "DepthwiseConv2dNative": return w ? linearCost(out, w[0] * w[1]) : NO_COST;
    case "Conv3D": return w ? linearCost(out, w[0] * w[1] * w[2] * w[3]) : NO_COST;
    case "Conv2DBackpropInput": {
      const f = c.ins[1], g = c.ins[2];
      return f && g ? linearCost(g, f[0] * f[1] * f[2]) : NO_COST;
    }
    case "MatMul": {
      if (!x) return NO_COST;
      return linearCost(out, c.attrs.get("transpose_a")?.b ? x[0] : x[1]);
    }
    case "BatchMatMul": case "BatchMatMulV2": case "BatchMatMulV3":
      if (!x) return NO_COST;
      return linearCost(out, c.attrs.get("adj_x")?.b ? x[x.length - 2] : x[x.length - 1]);
    case "MaxPool": case "AvgPool": {
      const k = attrInts(c.attrs, "ksize") ?? [1];
      return elemCost(out, prod(k));
    }
    case "Mean": case "Sum": case "Max": return elemCost(x);
    case "FusedBatchNorm": case "FusedBatchNormV2": case "FusedBatchNormV3": return elemCost(out, 2);
    case "Softmax": case "LogSoftmax": return elemCost(out, 3);
    case "BiasAdd": case "Add": case "AddV2": case "Sub": case "Mul": case "RealDiv": case "Relu": case "Relu6":
    case "LeakyRelu": case "Elu": case "Selu": case "Sigmoid": case "Tanh": case "Maximum": case "Minimum":
    case "SquaredDifference": case "Pow": case "Sqrt": case "Rsqrt": case "Exp": case "Erf": case "AddN":
      return elemCost(out);
    default:
      return NO_COST;
  }
}

function canonOf(op: string, c: Ctx): string {
  if (op === "Mean" || op === "Max") {
    const axes = c.constIn(1);
    const x = c.ins[0];
    if (axes && x && x.length === 4 && axes.length === 2) return op === "Mean" ? "GlobalAveragePool" : "GlobalMaxPool";
    return op === "Mean" ? "ReduceMean" : "ReduceMax";
  }
  return CANON[op] ?? op;
}

/** Display-friendly attributes (drops internal `_…` keys and bulky tensors). */
function displayAttrs(a: Map<string, TAttr>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of a) {
    if (k.startsWith("_") || k === "f" && v.func == null) continue;
    if (v.list) {
      const l = v.list;
      const val = l.i.length ? l.i : l.f.length ? l.f : l.s.length ? l.s : l.type.length ? l.type
        : l.shape.length ? l.shape.map((s) => (s ? s.join("×") : "?")) : l.b.length ? l.b : l.func.length ? l.func : [];
      out[k] = val;
    } else if (v.tensor) out[k] = `${v.tensor.dtype}[${v.tensor.shape.join("×")}]`;
    else if (v.shape !== undefined) out[k] = v.shape ? v.shape.map((d) => (d < 0 ? "?" : d)).join("×") : "?";
    else if (v.type != null) out[k] = v.type;
    else if (v.func != null) out[k] = v.func;
    else if (v.s != null) out[k] = v.s;
    else if (v.i != null) out[k] = v.i;
    else if (v.f != null) out[k] = v.f;
    else if (v.b != null) out[k] = v.b;
  }
  return out;
}

/** Integer values of a small constant tensor (for shape/axis operands). */
function tensorInts(t: TTensor): number[] | undefined {
  if (!/^u?int/.test(t.dtype) || t.shape.includes(DYN)) return undefined;
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (n > 4096) return undefined;
  if (t.content && t.content.length) return sampleBytes(t.content, t.dtype) ?? undefined;
  // proto3 drops default values: no data means zeros; one value is splatted.
  const vals = t.vals ?? [];
  return Array.from({ length: n }, (_, i) => vals[Math.min(i, vals.length - 1)] ?? 0);
}

function tensorValues(t: TTensor): number[] | null {
  if (t.content && t.content.length) return sampleBytes(t.content, t.dtype);
  if (t.vals && t.vals.length) {
    // Splat-encoded constants: a single value stands for the whole tensor.
    return t.vals;
  }
  return null;
}

// ── Model assembly ───────────────────────────────────────────────────────────

const FW = "TensorFlow";

/** Parse a TensorFlow `.pb` (frozen GraphDef / SavedModel / MetaGraphDef) or `.pbtxt`. */
/**
 * Parse a TensorFlow `.pb` (frozen GraphDef / SavedModel / MetaGraphDef) or
 * `.pbtxt`. `variables` is the SavedModel's checkpoint (variables.index +
 * data shards); with it, variable weights get real value statistics.
 */
export function parseTensorflow(input: ArrayBuffer | Uint8Array, name = "model.pb", variables?: { index: Uint8Array; shards: (Uint8Array | undefined)[] }): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const isText = /\.pbtxt$|\.prototxt$/i.test(name) || looksLikeText(buf);
  const file = isText ? decodeTensorflowText(new TextDecoder().decode(buf)) : decodeTensorflow(buf);
  const bundle = variables ? readTensorBundle(variables.index, variables.shards) : null;
  const byHandle = bundle && file.objects ? capturedVariableKeys(file.graph, file.objects) : new Map<string, string>();
  // A variable's checkpoint key, most exact first: the concrete function's
  // bound input (TF2 SavedModel), the object graph by variable name, then the
  // name itself (TF1 checkpoints key variables by name).
  const varValues = (n: TNode, shape: Shape): number[] | null => {
    if (!bundle) return null;
    const vname = n.attrs.get("shared_name")?.s || n.name;
    const exact = byHandle.get(n.name);
    const keys = exact ? [exact] : file.objects?.byName.get(vname) ?? [vname];
    const found = keys.map((k) => bundle.get(k));
    const t = found[0];
    // Several objects can share a variable name; use it only if every copy is
    // byte-identical (Keras tracks each variable twice), never guess between them.
    if (!t || found.some((o) => !o || !sameBytes(o.bytes, t.bytes))) return null;
    if (t.shape.length !== shape.length || t.shape.some((d, i) => shape[i] >= 0 && d !== shape[i])) return null;
    return sampleBytes(t.bytes, t.dt);
  };
  return tfFileToModel(file, name, buf.byteLength, isText, varValues);
}

function looksLikeText(buf: Uint8Array): boolean {
  const head = new TextDecoder().decode(buf.subarray(0, 256)).trimStart();
  return /^(node|meta_graphs|saved_model_schema_version|meta_info_def|graph_def|versions|library)\s*[{:]/.test(head);
}

function tfFileToModel(
  file: TFile, name: string, sizeBytes: number, isText: boolean,
  varValues: (n: TNode, shape: Shape) => number[] | null = () => null,
): Model {
  const g = file.graph;
  const top: Scope = { prefix: "", nodes: new Map(g.nodes.map((n) => [n.name, n])), args: null, depth: 0 };
  const fl = new Flattener(g.funcs);

  // Roots: the serving signature's outputs if there is one, else every node.
  const sig = file.sigs.find((s) => s.key === "serving_default") ?? file.sigs.find((s) => s.key !== "__saved_model_init_op");
  const outputRefs: (Resolved | null)[] = [];
  const inputNames: string[] = [];
  if (sig && sig.outputs.length) {
    for (const i of sig.inputs) inputNames.push(i.name.split(":")[0]);
    for (const o of sig.outputs) outputRefs.push(fl.resolve(parseRef(o.name, top)));
    for (const n of inputNames) fl.resolve(parseRef(n, top));
  } else {
    for (const n of g.nodes) if (!HIDDEN.has(n.op)) fl.resolve({ scope: top, name: n.name, idx: 0 });
  }
  const flat = fl.flat; // post-order: producers before consumers

  // Weight detection: constants/variables, and any node computed only from
  // them (ReadVariableOp, bias Reshape/Squeeze, BN rsqrt(var+eps)·γ, …) is
  // constant-folded into its consumers. Returns the source tensors, or null.
  const weightMemo = new Map<FlatNode, FlatNode[] | null>();
  const weightSources = (f: FlatNode | undefined): FlatNode[] | null => {
    if (!f) return null;
    if (weightMemo.has(f)) return weightMemo.get(f)!;
    weightMemo.set(f, null); // cycle guard
    let src: FlatNode[] | null = null;
    if (WEIGHT_SRC.has(f.node.op)) src = [f];
    else if (f.ins.length && !NEVER_FOLD.has(f.node.op)) {
      const parts = f.ins.map((r) => (r ? weightSources(r.fn) : null));
      if (parts.every((p) => p)) src = [...new Set(parts.flatMap((p) => p!))];
    }
    weightMemo.set(f, src);
    return src;
  };
  // A single pass-through chain back to one Const (for shape/axis operands).
  const constTensor = (f: FlatNode, guard = 0): TTensor | undefined => {
    if (f.node.op === "Const") return f.node.attrs.get("value")?.tensor;
    if (guard < 8 && WEIGHT_PASS.has(f.node.op) && f.ins.length === 1 && f.ins[0]) return constTensor(f.ins[0].fn, guard + 1);
    return undefined;
  };

  // Shapes per (flat node, output index).
  const shapes = new Map<string, Shape | undefined>();
  const shapeKey = (r: Resolved) => `${r.fn.key}:${r.idx}`;
  // Small integer tensors (shape vectors, axes, multiples) evaluated at parse
  // time so dynamic-shape plumbing (Shape → StridedSlice → Pack → Reshape) folds.
  const values = new Map<string, number[] | undefined>();
  const constInts = (r: Resolved | null | undefined): number[] | undefined => {
    if (!r) return undefined;
    const t = constTensor(r.fn);
    if (t) return tensorInts(t);
    return values.get(shapeKey(r));
  };
  const declared = (f: FlatNode, idx: number): Shape | undefined => {
    const n = f.node;
    // Variables: the `shape` attr is the tensor; _output_shapes is the scalar handle.
    if (n.op === "VarHandleOp" || n.op === "VariableV2" || n.op === "Variable") return n.attrs.get("shape")?.shape ?? undefined;
    const os = n.attrs.get("_output_shapes")?.list?.shape;
    if (os && os[idx]) return os[idx];
    if (n.op === "Const") return n.attrs.get("value")?.tensor?.shape;
    return undefined;
  };
  for (const f of flat) {
    const ctx: Ctx = {
      ins: f.ins.map((r) => (r ? shapes.get(shapeKey(r)) : undefined)),
      constIn: (i) => constInts(f.ins[i]),
      attrs: f.node.attrs,
    };
    const s0 = declared(f, 0) ?? inferShape(f.node.op, ctx);
    shapes.set(`${f.key}:0`, s0);
    for (let i = 1; i < 4; i++) { const si = declared(f, i); if (si) shapes.set(`${f.key}:${i}`, si); }
    const v = evalInts(f.node.op, ctx, f.ins.length);
    if (v) values.set(`${f.key}:0`, v);
  }

  // Visible layers: everything that isn't a weight, a pass-through of one, or bookkeeping.
  const visible = flat.filter((f) => !weightSources(f) && !HIDDEN.has(f.node.op));
  const idOf = new Map<FlatNode, number>();
  visible.forEach((f, i) => idOf.set(f, i));
  const consumed = new Set<FlatNode>();
  for (const f of visible) for (const r of f.ins) if (r && idOf.has(r.fn)) consumed.add(r.fn);
  const outputSet = new Set<FlatNode>(outputRefs.filter((r): r is Resolved => !!r).map((r) => r.fn));
  const inputSet = new Set(inputNames.map((n) => top.nodes.get(n)).filter(Boolean));

  const edges = new EdgeSet();
  const counted = new Set<FlatNode>();
  const layers: ModelLayer[] = visible.map((f, id) => {
    const n = f.node;
    const ctx: Ctx = {
      ins: f.ins.map((r) => (r ? shapes.get(shapeKey(r)) : undefined)),
      constIn: (i) => constInts(f.ins[i]),
      attrs: n.attrs,
    };
    let params = 0;
    let big: FlatNode | null = null;
    const ins: { n: string; s?: Shape }[] = [];
    for (const r of f.ins) {
      if (!r) continue;
      const srcs = weightSources(r.fn);
      if (srcs) {
        for (const w of srcs) {
          const t = w.node.attrs.get("value")?.tensor;
          const dt = t?.dtype ?? w.node.attrs.get("dtype")?.type ?? "float32";
          const s = shapes.get(`${w.key}:0`) ?? [];
          // Parameters are float tensors of rank ≥ 1: int consts are shapes/axes,
          // float scalars are hyper-parameters (BN epsilon, leaky alpha).
          if (!isFloatDtype(dt) || s.length === 0) continue;
          if (!counted.has(w)) { counted.add(w); params += numel(s); }
          if (!big || numel(s) > numel(shapes.get(`${big.key}:0`) ?? [])) big = w;
        }
        continue;
      }
      ins.push({ n: r.fn.key + (r.idx ? `:${r.idx}` : ""), s: shapes.get(shapeKey(r)) });
      edges.add(idOf.get(r.fn), id);
    }
    const out = shapes.get(`${f.key}:0`);
    const canon = canonOf(n.op, ctx);
    const isInput = n.op === "Placeholder" || n.op === "PlaceholderWithDefault" || inputSet.has(n);
    const isOutput = outputSet.size ? outputSet.has(f) : !consumed.has(f);
    const dt = n.attrs.get("T")?.type ?? n.attrs.get("dtype")?.type ?? "float32";
    let w: ModelLayer["w"] = null;
    if (big) {
      const s = shapes.get(`${big.key}:0`) ?? [];
      const t = big.node.attrs.get("value")?.tensor;
      w = weightStats(s, t ? tensorValues(t) : varValues(big.node, s));
    }
    const layer = makeLayer({
      id, name: f.key, type: n.op, canon, shape: out, dt, params, cost: opCost(n.op, out, ctx),
      group: isInput ? "input" : isOutput ? "output" : undefined, ins,
      outs: [{ n: f.key, s: out }], attr: { ...displayAttrs(n.attrs), ...(n.device ? { device: n.device } : {}) }, w, framework: FW,
      insight: isInput ? "Graph input (placeholder). Apply preprocessing (resize/normalize) upstream." : undefined,
    });
    if (isInput) layer.math = "model input";
    return layer;
  });

  const firstIn = layers.find((l) => l.group === "input");
  const outLayer = outputRefs[0] ? layers[idOf.get(outputRefs[0].fn) ?? -1] : layers.filter((l) => l.group === "output").pop();
  const inShape = firstIn ? shapes.get(`${visible[firstIn.id].key}:0`) : undefined;
  const outShape = outLayer ? shapes.get(`${visible[outLayer.id].key}:0`) : undefined;
  const kindLabel = file.kind === "GraphDef" ? (isText ? "GraphDef (text)" : "GraphDef") : file.kind;
  return {
    name,
    format: `TensorFlow ${kindLabel}`,
    framework: FW,
    opset: g.producer ?? 0,
    sizeBytes,
    inputShape: cleanShape(inShape),
    outputShape: cleanShape(outShape),
    layers,
    edges: edges.edges,
    producer: file.tfVersion ? `TensorFlow ${file.tfVersion}` : undefined,
    doc: sig ? `signature "${sig.key}"${file.tags?.length ? ` · tags: ${file.tags.join(", ")}` : ""}` : undefined,
  };
}

/** True if the bytes look like a binary TF graph (GraphDef / SavedModel / MetaGraphDef). */
export function isTensorflowGraph(buf: Uint8Array): boolean {
  try {
    const f = decodeTensorflow(buf);
    return f.graph.nodes.length > 0 || f.graph.funcs.size > 0;
  } catch {
    return false;
  }
}

/**
 * Graph variable handle (VarHandleOp node name) → checkpoint key. A call node
 * (StatefulPartitionedCall, ...) passes its captured variables as its trailing
 * inputs, in the order of that concrete function's bound_inputs.
 */
function capturedVariableKeys(g: TGraph, objects: ObjectGraph): Map<string, string> {
  const out = new Map<string, string>();
  const isHandle = new Set(g.nodes.filter((n) => n.op === "VarHandleOp").map((n) => n.name));
  for (const n of g.nodes) {
    const fn = n.attrs.get("f")?.func;
    const ids = fn ? objects.bound.get(fn) : undefined;
    if (!ids?.length) continue;
    const ins = n.inputs.filter((i) => !i.startsWith("^"));
    const captured = ins.slice(ins.length - ids.length);
    captured.forEach((inp, k) => {
      const handle = inp.replace(/:\d+$/, "");
      const key = objects.keyOf.get(ids[k]);
      if (key && isHandle.has(handle)) out.set(handle, key);
    });
  }
  return out;
}

function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
