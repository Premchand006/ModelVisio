import type { Model, ModelLayer } from "./types";

// Rule-based per-op compatibility notes for the edge-NPU compilers the app
// scores against. Fills ModelLayer.compIssues for every parsed format, so the
// compiler pre-flight, op-support scoring and CI checks see real findings
// instead of only the demo's hand-written ones.
//
// Deliberately conservative: rules fire only on ops whose name we recognize
// (across ONNX / TFLite / Keras / Caffe / OpenVINO / ... spellings) and that the
// vendor documentation lists as unsupported or constrained. An op we don't
// recognize produces no finding, so this never invents an incompatibility; it
// can miss one. The vendor compiler remains the final word.
//
// Sources:
//   Coral:  coral.ai/docs/edgetpu/models-intro ("Supported operations",
//           "Model requirements": static tensor sizes, at most 3 non-unit
//           dimensions, a single CPU/TPU partition point).
//   RKNN / Hailo DFC / Kneron toolchain: fixed-function NPU compilers that
//           require static input shapes and do not lower data-dependent or
//           control-flow ops (NMS, NonZero, TopK, Loop/If, Scatter...).

export type CompatTarget = "coral_tpu" | "rk3588" | "hailo" | "kneron_kl720";
export const COMPAT_TARGETS: CompatTarget[] = ["coral_tpu", "rk3588", "hailo", "kneron_kl720"];

type Issue = ModelLayer["compIssues"][number];

const DYN = -1;

/** Lower-case, drop dialect prefixes and punctuation: "stablehlo.dot_general" → "dotgeneral". */
export function normOp(op: string): string {
  return op
    .toLowerCase()
    .replace(/^(stablehlo|mhlo|tfl|tf|aten|onnx|torch|ai\.onnx)[.:]+/, "")
    .replace(/[^a-z0-9]/g, "");
}

const set = (...xs: string[]) => new Set(xs);

// Ops whose output size or control flow depends on the data: no static NPU
// schedule exists, so every fixed-function compiler here rejects them or runs
// them on the host CPU.
const DATA_DEPENDENT = set(
  "nonmaxsuppression", "nonmaxsuppressionv4", "nonmaxsuppressionv5", "nms", "multiclassnms", "multiclassnms3",
  "nonzero", "unique", "topk", "topkv2", "loop", "if", "scan", "while", "scatternd", "scatterelements",
  "scatter", "detectionoutput",
);

// Not in the Edge TPU supported-operations list (after ONNX/Keras → TFLite
// lowering these become DIV, POW, EXP, LOG, SQRT, GATHER, CAST, ... which the
// Edge TPU compiler leaves on the CPU).
const CORAL_UNSUPPORTED = set(
  "div", "realdiv", "floordiv", "pow", "exp", "log", "sqrt", "erf", "gelu", "gather", "gatherv2", "gathernd",
  "gatherelements", "cast", "einsum", "cumsum", "layernormalization", "layernorm", "instancenormalization",
  "instancenorm", "groupnormalization", "groupnorm", "mish", "softplus", "elu", "selu", "where", "select", "selectv2",
  "argmin", "mod", "sin", "cos", "reciprocal",
);

const SOFTMAX = set("softmax", "softmaxnd", "logsoftmax");
const RESIZE = set(
  "resize", "upsample", "upsampling2d", "resizenearestneighbor", "resizebilinear", "interp", "interpolate",
  "nearestinterpv2", "bilinearinterpv2", "nearestinterp", "bilinearinterp", "resizearea", "resizenearest",
);
const MATMUL = set("matmul", "batchmatmul", "batchmatmulv2", "batchmatmulv3", "matmulv2", "einsum", "dotgeneral");
const INPUT = set("input", "inputlayer", "placeholder", "parameter", "feed", "data");

// Integer tensors in a float graph are shape arithmetic (Shape → Gather → Cast →
// Concat → Reshape) that converters constant-fold away for a static input, so
// they never reach the NPU and must not be reported against it.
const isShapeMath = (l: ModelLayer) => /^u?int(8|16|32|64)$/.test(l.dt) && !/^u?int8$/.test(l.dt);

const parseShape = (s: string | undefined) => {
  if (!s || s === "?") return undefined;
  const dims = s.split("×").map((d) => (d === "?" ? DYN : Number(d)));
  return dims.every((d) => Number.isFinite(d)) ? dims : undefined;
};
const sameShape = (a: unknown, b: number[]) => Array.isArray(a) && a.length === b.length && a.every((d, i) => d === b[i]);

/**
 * The layer's output activation shape, or undefined when the parser has none.
 * Checkpoint and weights-only parsers (PyTorch state dicts, safetensors) put
 * the weight shape in `shape`; those must never be judged as activations.
 */
function activationShape(l: ModelLayer): number[] | undefined {
  const s = l.outs?.[0]?.s;
  if (!s || !s.length || l.op === "Weight" || sameShape(l.w?.shape, s)) return undefined;
  return s;
}

const issue = (target: CompatTarget, severity: Issue["severity"], msg: string): Issue => ({ target, severity, msg });

/** Rule-based compatibility findings for one layer. */
export function layerCompat(l: ModelLayer): Issue[] {
  const op = normOp(l.op || l.type);
  const out: Issue[] = [];
  const shape = activationShape(l);

  if (l.group === "input" || INPUT.has(op)) {
    // Batch (dim 0) is fixed at export; any other unknown dim is a real problem.
    // Prefer the file's declared shape when the parser filled dims in for counting.
    const declared = typeof l.attr?.declaredShape === "string" ? l.attr.declaredShape : undefined;
    const inShape = parseShape(declared) ?? l.outs?.[0]?.s ?? parseShape(l.shape);
    if (inShape && inShape.slice(1).some((d) => d === DYN || d === 0)) {
      const dims = inShape.map((d) => (d <= 0 ? "?" : d)).join("×");
      out.push(issue("coral_tpu", "error", `Dynamic input shape ${dims}: the Edge TPU compiler needs every tensor size fixed at compile time. Re-export with a static input size.`));
      out.push(issue("rk3588", "warn", `Dynamic input shape ${dims}: RKNN builds for fixed shapes (or a preset list via dynamic_input). Re-export with a static input size.`));
      out.push(issue("hailo", "error", `Dynamic input shape ${dims}: the Hailo Dataflow Compiler requires static shapes. Re-export with a static input size.`));
      out.push(issue("kneron_kl720", "error", `Dynamic input shape ${dims}: the Kneron toolchain requires static shapes. Re-export with a static input size.`));
    }
    return out;
  }

  if (l.constFolded || (isShapeMath(l) && !DATA_DEPENDENT.has(op))) return out;

  if (DATA_DEPENDENT.has(op)) {
    const what = `${l.op} is data-dependent (its output size or control flow depends on the input values)`;
    out.push(issue("coral_tpu", "error", `${what}: not supported by the Edge TPU, runs on CPU. Cut the graph before it and do this step in post-processing.`));
    out.push(issue("rk3588", "warn", `${what}: RKNN runs it on the CPU or rejects it. Cut the graph before it and do this step in post-processing.`));
    out.push(issue("hailo", "error", `${what}: not compiled by the Hailo DFC. Cut the graph before it (Hailo handles NMS in its own post-process).`));
    out.push(issue("kneron_kl720", "error", `${what}: not supported on the KL720 NPU. Cut the graph before it and do this step on the host.`));
    return out;
  }

  // ── Coral Edge TPU specifics ──────────────────────────────────────────────
  if (CORAL_UNSUPPORTED.has(op)) {
    out.push(issue("coral_tpu", "error", `${l.op} is not in the Edge TPU supported-ops list: it runs on CPU, and since the compiler maps only up to the first unsupported op, everything after it runs on CPU too.`));
  }
  if (MATMUL.has(op) && (l.ins?.length ?? 0) >= 2 && l.params === 0) {
    out.push(issue("coral_tpu", "error", `${l.op} multiplies two activations (attention-style batch matmul), which the Edge TPU does not support; only FullyConnected with constant weights maps.`));
  }
  if (SOFTMAX.has(op) && shape) {
    const nonUnit = shape.filter((d) => d !== 1).length;
    const n = shape.reduce((a, d) => a * Math.max(1, d), 1);
    if (nonUnit > 1 || n > 16000 || shape.includes(DYN)) {
      out.push(issue("coral_tpu", "error", `Softmax over ${shape.map((d) => (d < 0 ? "?" : d)).join("×")}: the Edge TPU supports Softmax only on a 1-D tensor of at most 16,000 elements, so this runs on CPU.`));
    }
  }
  if (RESIZE.has(op)) {
    out.push(issue("coral_tpu", "warn", `${l.op} (resize/upsample): the Edge TPU compiler may leave it on CPU depending on input/output size, to avoid precision loss. Check the compiler log.`));
  }
  if (shape && shape.length > 3 && !shape.includes(DYN)) {
    const outer = shape.slice(0, shape.length - 3).reduce((a, d) => a * d, 1);
    if (outer > 1) {
      out.push(issue("coral_tpu", "error", `Output tensor ${shape.join("×")} has more than 3 non-unit dimensions; the Edge TPU allows size > 1 only in the 3 innermost dimensions.`));
    }
  }
  return out;
}

/**
 * Fill compIssues on every layer that has none yet, in place. Layers that
 * already carry issues (e.g. the curated demo model) are left untouched, so
 * the pass is idempotent. Returns the same model for chaining.
 */
export function annotateCompat<T extends Pick<Model, "layers">>(model: T): T {
  annotateLayers(model.layers);
  return model;
}

/** In-place variant over a layer list (used per streamed chunk). */
export function annotateLayers(layers: ModelLayer[]): void {
  for (const l of layers) {
    if (l.compIssues && l.compIssues.length) continue;
    l.compIssues = layerCompat(l);
  }
}
