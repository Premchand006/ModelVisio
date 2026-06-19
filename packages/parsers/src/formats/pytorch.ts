import type { Model, ModelLayer } from "../types";
import { extractStored, isZip, readZipEntries } from "../common/zip";
import { isGlobal, PyObj, unpickle, type GlobalRef, type Handlers } from "../common/pickle";
import { parseContainer } from "./container";

// Parses modern PyTorch `.pt`/`.pth`/`.bin` archives (torch.save → ZIP since
// 1.6). We unpickle `data.pkl` and walk the nn.Module hierarchy
// (_modules/_parameters/_buffers) to recover module classes + weight shapes —
// the same structure Netron shows. No compute graph exists in a pickle, so the
// "edges" are the parent→child module tree (renders as a tree in the graph).

const STORAGE_DTYPE: Record<string, string> = {
  FloatStorage: "float32", HalfStorage: "float16", DoubleStorage: "float64",
  LongStorage: "int64", IntStorage: "int32", ShortStorage: "int16",
  CharStorage: "int8", ByteStorage: "uint8", BoolStorage: "bool", BFloat16Storage: "bfloat16",
};

type Tensor = { __tensor__: true; shape: number[]; dtype: string };
const isTensor = (x: unknown): x is Tensor => typeof x === "object" && x !== null && (x as Tensor).__tensor__ === true;
const prod = (s: number[]) => s.reduce((a, b) => a * b, s.length ? 1 : 1);
const asMap = (x: unknown): Map<unknown, unknown> | null => (x instanceof Map ? x : null);

function handlers(): Handlers {
  return {
    persistentLoad(pid) {
      const a = pid as unknown[];
      if (Array.isArray(a) && a[0] === "storage") {
        const st = a[1];
        const dtype = isGlobal(st) ? STORAGE_DTYPE[st.name] ?? "float32" : "float32";
        return { __storage__: true, dtype };
      }
      return { __storage__: true, dtype: "float32" };
    },
    findClass: (module, name): GlobalRef => ({ __global__: true, module, name }),
    reduce(func, args) {
      if (isGlobal(func)) {
        const n = func.name;
        if (/^_rebuild_tensor/.test(n)) {
          const storage = args[0] as { dtype?: string } | undefined;
          const size = args[2];
          return { __tensor__: true, shape: Array.isArray(size) ? (size as unknown[]).map(Number) : [], dtype: storage?.dtype ?? "float32" } as Tensor;
        }
        if (n === "_rebuild_parameter") { const tens = args[0]; return isTensor(tens) ? { ...tens, __param__: true } : tens; }
        if (n === "OrderedDict") return new Map();
        if (n === "_reconstructor" || n === "__newobj__") return new PyObj(isGlobal(args[0]) ? args[0] : null);
        return new PyObj(func);
      }
      return new PyObj(null);
    },
  };
}

function classify(cls: string): ModelLayer["group"] {
  if (/Detect|Classify|Segment|Pose|Linear|head/i.test(cls)) return "head";
  if (/Norm/.test(cls)) return "neck";
  if (/Pool|Upsample|Concat|Sequential|ModuleList|Identity/.test(cls)) return "neck";
  return "backbone";
}

function tensorEntries(m: Map<unknown, unknown> | null): { name: string; t: Tensor }[] {
  const out: { name: string; t: Tensor }[] = [];
  if (m) for (const [k, v] of m) if (isTensor(v)) out.push({ name: String(k), t: v });
  return out;
}

function walk(obj: PyObj, path: string, parentId: number, layers: ModelLayer[], edges: [number, number][], counter: { id: number }) {
  const cls = obj.cls?.name || "Module";
  const id = counter.id++;
  const params = tensorEntries(asMap(obj.attrs.get("_parameters")));
  const buffers = tensorEntries(asMap(obj.attrs.get("_buffers")));
  const weight = params.find((p) => p.name === "weight")?.t ?? params[0]?.t ?? null;
  const own = params.reduce((s, p) => s + prod(p.t.shape), 0);
  const attr: Record<string, unknown> = {};
  for (const p of [...params, ...buffers]) attr[p.name] = p.t.shape.length ? p.t.shape.join("×") : "scalar";

  layers.push({
    id, name: path, type: cls, op: cls,
    shape: weight ? (weight.shape.length ? weight.shape.join("×") : "scalar") : "",
    dt: weight?.dtype ?? "float32", params: own, flops: 0, macs: 0, mem: own * 4,
    group: classify(cls), ins: [], outs: [], attr,
    w: weight ? { shape: weight.shape, size: prod(weight.shape) } : null,
    math: cls, insight: `PyTorch ${cls} module.`, qSens: /Conv|Linear/.test(cls) ? 0.5 : 0.1, compIssues: [],
  });
  if (parentId >= 0) edges.push([parentId, id]);

  const modules = asMap(obj.attrs.get("_modules"));
  if (modules) for (const [k, child] of modules) if (child instanceof PyObj) walk(child, path ? `${path}.${k}` : String(k), id, layers, edges, counter);
}

function pickModel(root: unknown): PyObj | Map<unknown, unknown> | null {
  if (root instanceof PyObj && root.attrs.get("_modules")) return root;
  if (root instanceof Map) {
    for (const key of ["model", "ema", "module", "net"]) {
      const m = root.get(key);
      if (m instanceof PyObj && m.attrs.get("_modules")) return m;
    }
    let tensors = 0;
    for (const v of root.values()) if (isTensor(v)) tensors++;
    if (tensors > 0) return root; // looks like a state_dict
    for (const v of root.values()) if (v instanceof PyObj && v.attrs.get("_modules")) return v;
  }
  return null;
}

export function parsePytorch(input: ArrayBuffer | Uint8Array, name = "model.pt"): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  try {
    if (!isZip(buf)) return parseContainer(buf, name); // legacy tar/pickle — not yet supported
    const entries = readZipEntries(buf);
    const pkl = entries.find((e) => /(^|\/)data\.pkl$/.test(e.name)) ?? entries.find((e) => e.name.endsWith(".pkl"));
    const bytes = pkl ? extractStored(buf, pkl) : null;
    if (!bytes) return parseContainer(buf, name);

    const root = unpickle(bytes, handlers());
    const model = pickModel(root);
    const layers: ModelLayer[] = [];
    const edges: [number, number][] = [];

    if (model instanceof PyObj) {
      walk(model, model.cls?.name ?? "model", -1, layers, edges, { id: 0 });
    } else if (model instanceof Map) {
      let id = 0;
      for (const [k, v] of model) {
        if (!isTensor(v)) continue;
        const t = v as Tensor;
        layers.push({
          id: id++, name: String(k), type: "Tensor", op: "Weight",
          shape: t.shape.length ? t.shape.join("×") : "scalar", dt: t.dtype,
          params: prod(t.shape), flops: 0, macs: 0, mem: prod(t.shape) * 4, group: "backbone",
          ins: [], outs: [{ n: String(k), s: t.shape }], attr: {},
          w: { shape: t.shape, size: prod(t.shape) }, math: "weight",
          insight: "State-dict tensor.", qSens: 0.3, compIssues: [],
        });
      }
      for (let i = 1; i < layers.length; i++) edges.push([i - 1, i]);
    }

    if (layers.length === 0) return parseContainer(buf, name);
    return {
      name, format: "PyTorch", framework: "PyTorch", opset: 0, sizeBytes: buf.byteLength,
      inputShape: [], outputShape: [], layers, edges, producer: `${layers.length} modules`,
    };
  } catch {
    return parseContainer(buf, name);
  }
}
