import type { Model, ModelLayer } from "../types";
import {
  DT_SIZE, EdgeSet, NO_COST, cleanShape, elemCost, linearCost, makeLayer, numel, sampleBytes, weightStats,
  type Cost, type Shape, type WStat,
} from "../common/graphKit";

// MLIR textual IR (`.mlir`): a tokenizer/line-based reader, not a full MLIR
// parser. It understands both the generic form
//   %0 = "tosa.conv2d"(%arg0, %w, %b) <{…}> : (…) -> tensor<…>
// and the common pretty forms
//   %1 = stablehlo.add %0, %c : tensor<…>
//   %2 = linalg.matmul ins(%a, %b : …) outs(%c : …) -> tensor<…>
// for any dialect (func, stablehlo/mhlo, tosa, tfl/tf, onnx, torch, linalg, arith,
// tensor, …). SSA values become edges, result types become shapes, constant ops
// (`*.constant`, `tosa.const`, `tfl.pseudo_const`, `onnx.Constant`, …) fold into
// their consumers as parameters, multi-result ops (`%0:2`) are supported, and
// nested region bodies (reductions, linalg.generic, scf.for …) collapse into
// the op that owns them. One function is rendered (`main`/first public).

const FRAMEWORK = "MLIR";

type ValInfo = { shape?: Shape; dt: string };
type Constant = { shape: Shape; dt: string; params: number; splat: boolean; w: WStat | null };
type Op = {
  results: string[]; // expanded: "%0" or "%0#1"
  op: string;
  text: string; // op line (without results)
  operands: string[];
  resultTypes: ValInfo[];
  operandTypes: ValInfo[];
  line: number;
};
type Func = { name: string; visibility: string; args: { name: string; t: ValInfo }[]; ops: Op[]; returns: string[]; returnTypes: ValInfo[] };

// ── Lexical helpers ──────────────────────────────────────────────────────────

/** Remove `{-# … #-}` resource blobs and `//` comments (string-aware). */
function preprocess(text: string): string[] {
  let t = text.replace(/\{-#[\s\S]*?#-\}/g, "");
  const out: string[] = [];
  for (const line of t.split(/\r?\n/)) {
    let inStr = false;
    let cut = line.length;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (inStr) { if (c === "\\") i++; else if (c === '"') inStr = false; continue; }
      if (c === '"') inStr = true;
      else if (c === "/" && line[i + 1] === "/") { cut = i; break; }
    }
    out.push(line.slice(0, cut));
  }
  t = "";
  return out;
}

/** Net ( and [ depth of a line, string-aware. */
function parenDepth(s: string): number {
  let d = 0, inStr = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inStr) { if (c === "\\") i++; else if (c === '"') inStr = false; continue; }
    if (c === '"') inStr = true;
    else if (c === "(" || c === "[") d++;
    else if (c === ")" || c === "]") d--;
  }
  return d;
}

/** Join physical lines into logical statements (open ( / [ lists continue). */
function logicalLines(lines: string[]): { s: string; line: number }[] {
  const out: { s: string; line: number }[] = [];
  let buf = "", start = 0, depth = 0;
  lines.forEach((raw, i) => {
    const l = raw.trim();
    if (!l && !buf) return;
    if (!buf) start = i + 1;
    buf = buf ? `${buf} ${l}` : l;
    depth += parenDepth(l);
    if (depth > 0 && !/\{$/.test(l)) return;
    out.push({ s: buf, line: start });
    buf = ""; depth = 0;
  });
  if (buf) out.push({ s: buf, line: start });
  return out;
}

/** Split at top-level commas (respecting (), [], {}, <> and strings). */
function splitTop(s: string, sep = ","): string[] {
  const parts: string[] = [];
  let d = 0, inStr = false, cur = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inStr) { cur += c; if (c === "\\") { cur += s[++i] ?? ""; } else if (c === '"') inStr = false; continue; }
    if (c === '"') inStr = true;
    else if (c === "-" && s[i + 1] === ">") { cur += "->"; i++; continue; }
    else if ("([{<".includes(c)) d++;
    else if (")]}>".includes(c)) d--;
    if (c === sep && d === 0) { parts.push(cur.trim()); cur = ""; continue; }
    cur += c;
  }
  if (cur.trim()) parts.push(cur.trim());
  return parts;
}

/** Index of the last top-level occurrence of `tok` (":" or "->"), or -1. */
function lastTop(s: string, tok: ":" | "->"): number {
  let d = 0, inStr = false, found = -1;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inStr) { if (c === "\\") i++; else if (c === '"') inStr = false; continue; }
    if (c === '"') { inStr = true; continue; }
    if (c === "-" && s[i + 1] === ">") { if (d === 0 && tok === "->") found = i; i++; continue; }
    if ("([{<".includes(c)) d++;
    else if (")]}>".includes(c)) d--;
    else if (d === 0 && tok === ":" && c === ":" && s[i + 1] !== ":" && s[i - 1] !== ":") found = i;
  }
  return found;
}

const ELEM: Record<string, string> = {
  f32: "float32", f16: "float16", bf16: "bfloat16", f64: "float64", f8E4M3FN: "float8", f8E5M2: "float8",
  i1: "bool", i4: "int4", i8: "int8", i16: "int16", i32: "int32", i64: "int64",
  si8: "int8", si16: "int16", si32: "int32", si64: "int64", ui8: "uint8", ui16: "uint16", ui32: "uint32", ui64: "uint64",
  index: "int64", "complex<f32>": "complex64", "complex<f64>": "complex128",
};
function elemDt(e: string): string {
  const s = e.trim();
  if (ELEM[s]) return ELEM[s];
  const q = s.match(/^!quant\.\w+<\s*([a-z]+\d+)/);
  if (q) return ELEM[q[1]] ?? q[1];
  return s.replace(/^!/, "") || "float32";
}

/** Parse one MLIR type into shape + dtype (dynamic dims → -1). */
export function parseMlirType(t: string): ValInfo {
  const s = t.trim().replace(/\s*\{.*\}\s*$/, "");
  const m = s.match(/^(?:tensor|memref|vector)<(.*)>$/s);
  if (m) {
    const body = splitTop(m[1])[0]; // drop encoding / layout
    if (body.startsWith("*")) return { dt: elemDt(body.replace(/^\*x/, "")) };
    const dims: number[] = [];
    let rest = body;
    for (let dm = rest.match(/^(\d+|\?|\[\d+\])x/); dm; dm = rest.match(/^(\d+|\?|\[\d+\])x/)) {
      dims.push(dm[1] === "?" ? -1 : Number(dm[1].replace(/[[\]]/g, "")));
      rest = rest.slice(dm[0].length);
    }
    return { shape: dims, dt: elemDt(rest) };
  }
  const torch = s.match(/^!torch\.(?:v)?tensor<\[([^\]]*)\]\s*,\s*([^>]+)>$/);
  if (torch) return { shape: torch[1].split(",").filter((x) => x.trim()).map((x) => (Number(x) >= 0 ? Number(x) : -1)), dt: elemDt(torch[2]) };
  if (ELEM[s] || /^[isu]?i\d+$|^f\d+$/.test(s)) return { shape: [], dt: elemDt(s) };
  return { dt: s.slice(0, 24) || "?" };
}

function parseTypeList(s: string): ValInfo[] {
  let t = s.trim();
  if (t.startsWith("(") && t.endsWith(")")) t = t.slice(1, -1);
  return t ? splitTop(t).map(parseMlirType) : [];
}

/** Types from a signature tail: "(a, b) -> c" or "t" or "-> t". */
function parseSignature(sig: string): { ins: ValInfo[]; outs: ValInfo[] } {
  const arrow = lastTop(sig, "->");
  if (arrow >= 0) return { ins: parseTypeList(sig.slice(0, arrow)), outs: parseTypeList(sig.slice(arrow + 2)) };
  return { ins: [], outs: parseTypeList(sig) };
}

// ── Constants ────────────────────────────────────────────────────────────────

const CONST_OP = /(^|\.)(constant|const|pseudo_const|pseudo_qconst|Constant|ConstantOfShape|literal|Const)$/;
const isConstOp = (op: string) => CONST_OP.test(op) || op === "ml_program.global_load_const" || op === "tensor.empty";

function hexSample(hex: string, dt: string, count: number): number[] | null {
  const bpe = DT_SIZE[dt];
  if (!bpe) return null;
  const step = Math.max(1, Math.ceil(count / 100_000));
  const n = Math.ceil(count / step);
  const bytes = new Uint8Array(n * bpe);
  for (let k = 0, i = 0; i < count; i += step, k++) {
    for (let b = 0; b < bpe; b++) {
      const o = (i * bpe + b) * 2;
      bytes[k * bpe + b] = parseInt(hex.substr(o, 2), 16);
    }
  }
  return sampleBytes(bytes, dt);
}

function readConstant(text: string, t: ValInfo): Constant {
  const shape = t.shape ?? [];
  const n = numel(shape);
  const dt = t.dt;
  const res = /dense_resource</.test(text);
  const hex = text.match(/dense<"0x([0-9A-Fa-f]*)"/);
  const list = text.match(/dense<(\[[\s\S]*?\])>/);
  let splat = !res;
  let vals: number[] | null = null;
  if (hex) {
    const bpe = DT_SIZE[dt] ?? 0;
    splat = !bpe || hex[1].length / 2 < n * bpe;
    if (!splat) vals = hexSample(hex[1], dt, n);
  } else if (list) {
    const nums = list[1].match(/[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|true|false/g) ?? [];
    splat = nums.length < n || n <= 1;
    if (!splat) vals = nums.slice(0, 200_000).map((x) => (x === "true" ? 1 : x === "false" ? 0 : Number(x)));
  }
  if (!res && !hex && !list) splat = true; // dense<0.0>, scalar attrs, …
  if (n <= 1) splat = true;
  const params = splat ? 0 : n;
  return { shape, dt, params, splat, w: params ? weightStats(shape, vals) : null };
}

// ── Op classification and cost ───────────────────────────────────────────────

const short = (op: string) => op.split(".").pop()!.toLowerCase();

function canonOf(op: string): string {
  const s = short(op);
  if (/conv.*transpose|transpose_conv|deconv|conv_transpose|backprop/.test(s)) return "ConvTranspose";
  if (/conv/.test(s) && !/convert/.test(s)) return "Conv";
  if (/^(fully_connected|gemm|linear|addmm)$/.test(s)) return "Gemm";
  if (/dot|matmul|^mm$|^bmm$|einsum/.test(s)) return "MatMul";
  if (/max_?pool|pooling_.*max/.test(s)) return "MaxPool";
  if (/avg_?pool|average_?pool|pooling_.*sum/.test(s)) return "AveragePool";
  if (/softmax/.test(s)) return /log/.test(s) ? "LogSoftmax" : "Softmax";
  if (/^(add|addf|addi|add_v2)$/.test(s)) return "Add";
  if (/^(sub|subtract|subf|subi)$/.test(s)) return "Sub";
  if (/^(mul|multiply|mulf|muli)$/.test(s)) return "Mul";
  if (/^(div|divide|divf|divsi|divui|real_div)$/.test(s)) return "Div";
  if (/^relu6?$/.test(s)) return "Relu";
  if (/leaky_?relu/.test(s)) return "LeakyRelu";
  if (/^(logistic|sigmoid)$/.test(s)) return "Sigmoid";
  if (/^tanh$/.test(s)) return "Tanh";
  if (/^(reshape|view|expand_shape|collapse_shape|flatten)$/.test(s)) return "Reshape";
  if (/^(transpose|permute)$/.test(s)) return "Transpose";
  if (/^(concatenate|concat|cat)$/.test(s)) return "Concat";
  if (/batch_norm/.test(s)) return "BatchNormalization";
  if (/^(broadcast_in_dim|broadcast|broadcast_to|expand)$/.test(s)) return "Expand";
  if (/^(maximum|max|maxf|maximumf|maxnumf)$/.test(s)) return "Max";
  if (/^(minimum|min|minf|minimumf|minnumf)$/.test(s)) return "Min";
  return s;
}

const ELEMENTWISE = /^(add|addf|addi|add_v2|sub|subtract|subf|subi|mul|multiply|mulf|muli|div|divide|divf|real_div|maximum|minimum|max|min|maxf|minf|maximumf|minimumf|relu|relu6|leaky_relu|tanh|logistic|sigmoid|exp|exponential|log|sqrt|rsqrt|negate|neg|abs|clamp|clip|select|compare|cmpf|pow|power|erf|gelu|silu|hard_swish|hardswish|square|squared_difference|floor|ceil|round|sign|cosine|sine|reciprocal|elu)$/;

function windowArea(text: string): number {
  const k = text.match(/kernel\s*=\s*array<i64:\s*([\d,\s]+)>/) ?? text.match(/window_dimensions\s*=\s*array<i64:\s*([\d,\s]+)>/);
  if (k) return k[1].split(",").map(Number).reduce((a, b) => a * b, 1);
  const fh = text.match(/filter_height\s*=\s*(\d+)/), fw = text.match(/filter_width\s*=\s*(\d+)/);
  if (fh && fw) return Number(fh[1]) * Number(fw[1]);
  const ks = text.match(/kernel_shape\s*=\s*\[([\d,\s]+)\]/);
  if (ks) return ks[1].split(",").map(Number).reduce((a, b) => a * b, 1);
  return 0;
}

/** Output-channel axis for conv results (NCHW-style dialects use 1, NHWC-style use last). */
function channelAxis(op: string, text: string, rank: number): number {
  const dn = text.match(/->\s*\[([^\]]*)\]/);
  if (dn) { const ax = dn[1].split(",").map((x) => x.trim()).indexOf("f"); if (ax >= 0) return ax; }
  const s = short(op);
  if (/nchw|ncw|ncdhw/.test(s)) return 1;
  if (/nhwc|nwc|ndhwc/.test(s)) return rank - 1;
  if (/^(tosa|tfl|tf)\./.test(op)) return rank - 1;
  return 1;
}

function costOf(op: Op, canon: string, inS: (Shape | undefined)[], out: Shape | undefined): Cost {
  const s = short(op.op);
  if ((canon === "Conv" || canon === "ConvTranspose") && out && inS[1]) {
    const cout = out[channelAxis(op.op, op.text, out.length)] ?? 0;
    return cout > 0 ? linearCost(out, numel(inS[1]) / cout) : NO_COST;
  }
  if ((canon === "MatMul" || canon === "Gemm") && out) {
    const lhs = inS[0];
    if (!lhs || !lhs.length) return NO_COST;
    const cd = op.text.match(/contracting_dims\s*=\s*\[([\d,\s]*)\]/) ?? op.text.match(/lhs_contracting_dimensions\s*=\s*\[([\d,\s]*)\]/);
    const k = cd && cd[1].trim()
      ? cd[1].split(",").map(Number).reduce((a, d) => a * Math.max(1, lhs[d] ?? 1), 1)
      : Math.max(1, lhs[lhs.length - 1]);
    return linearCost(out, k);
  }
  if (canon === "MaxPool" || canon === "AveragePool" || /reduce_window/.test(s)) {
    const a = windowArea(op.text);
    return a ? elemCost(out, a) : elemCost(inS[0]);
  }
  if (canon === "Softmax" || canon === "LogSoftmax") return elemCost(out, 3);
  if (canon === "BatchNormalization") return elemCost(out, 2);
  if (/layer_norm|rms_norm/.test(s)) return elemCost(out, 5);
  if (/^(reduce|reduce_\w+|mean|sum|argmax|argmin)$/.test(s)) return elemCost(inS[0]);
  if (ELEMENTWISE.test(s)) return elemCost(out);
  return NO_COST;
}

const ATTR_KEYS = /\b(stride|strides|window_strides|pad|padding|dilation|dilations|lhs_dilation|rhs_dilation|group|feature_group_count|axis|dimensions|broadcast_dimensions|perm|permutation|kernel|kernel_shape|dim_numbers|dimension_numbers|contracting_dims)\s*=\s*/g;

/** Selected op attributes (value read with bracket balancing up to a top-level , } or >). */
function keyAttrs(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of text.matchAll(ATTR_KEYS)) {
    let d = 0, i = m.index! + m[0].length;
    const start = i;
    for (; i < text.length; i++) {
      const c = text[i];
      if (c === "-" && text[i + 1] === ">") { i++; continue; }
      if ("([{<".includes(c)) d++;
      else if (")]}>".includes(c)) { if (d === 0) break; d--; }
      else if (c === "," && d === 0) break;
    }
    const v = text.slice(start, i).trim();
    if (v && v.length <= 200) out[m[1]] = v;
  }
  return out;
}

// ── Statement parsing ────────────────────────────────────────────────────────

const RESULT_RE = /^((?:%[\w.$\-]+(?::\d+)?)(?:\s*,\s*%[\w.$\-]+(?::\d+)?)*)\s*=\s*/;
const OPNAME_RE = /^(?:"([^"]+)"|([A-Za-z_][\w.$]*))/;

function parseOpStatement(s: string, line: number): Op | null {
  let rest = s;
  const results: string[] = [];
  const rm = rest.match(RESULT_RE);
  if (rm) {
    for (const r of rm[1].split(",").map((x) => x.trim())) {
      const mm = r.match(/^(%[\w.$\-]+)(?::(\d+))?$/);
      if (!mm) continue;
      const n = mm[2] ? Number(mm[2]) : 1;
      if (n === 1) results.push(mm[1]);
      else for (let k = 0; k < n; k++) results.push(`${mm[1]}#${k}`);
    }
    rest = rest.slice(rm[0].length);
  }
  const om = rest.match(OPNAME_RE);
  if (!om) return null;
  const op = om[1] ?? om[2];
  // Ops are dialect-qualified; bare words are keywords / continuation lines.
  if (!om[1] && !op.includes(".") && !["return", "call", "call_indirect"].includes(op)) return null;
  const body = rest.slice(om[0].length);
  const operands: string[] = [];
  for (const m of body.matchAll(/%[\w.$\-]+(?:#\d+)?/g)) if (!operands.includes(m[0])) operands.push(m[0]);
  const op0: Op = { results, op: op === "call" ? "func.call" : op === "return" ? "func.return" : op, text: body, operands, resultTypes: [], operandTypes: [], line };
  applySignature(op0, body);
  return op0;
}

function applySignature(op: Op, tail: string): void {
  // Drop a trailing region opener so it doesn't confuse the signature split.
  const t = tail.replace(/\(?\{\s*$/, "").replace(/\sloc\([^)]*\)\s*$/, "");
  const colon = lastTop(t, ":");
  let sig: { ins: ValInfo[]; outs: ValInfo[] } | null = null;
  if (colon >= 0) sig = parseSignature(t.slice(colon + 1));
  else {
    const arrow = lastTop(t, "->");
    if (arrow >= 0) sig = { ins: [], outs: parseTypeList(t.slice(arrow + 2)) };
  }
  if (!sig || (!sig.outs.length && !sig.ins.length)) return;
  if (sig.ins.length) op.operandTypes = sig.ins;
  // Pretty elementwise form ("op %a, %b : tensor<…>"): one type for all.
  if (!sig.ins.length && sig.outs.length === 1 && colon >= 0 && op.results.length !== 1) op.operandTypes = op.operands.map(() => sig!.outs[0]);
  if (sig.outs.length && op.results.length) {
    op.resultTypes = sig.outs.length >= op.results.length ? sig.outs : op.results.map(() => sig!.outs[0]);
  }
}

function parseFuncHeader(s: string): Func | null {
  const pretty = s.match(/^(?:func\.func|func)\s+(?:(public|private|nested)\s+)?@([\w.$\-"]+)\s*\(/);
  if (pretty) {
    const open = s.indexOf("(", pretty[0].length - 1);
    let d = 0, close = -1;
    for (let i = open; i < s.length; i++) {
      if (s[i] === "(") d++;
      else if (s[i] === ")") { d--; if (d === 0) { close = i; break; } }
    }
    const args = close > open ? splitTop(s.slice(open + 1, close)) : [];
    const fn: Func = { name: pretty[2].replace(/"/g, ""), visibility: pretty[1] ?? "public", args: [], ops: [], returns: [], returnTypes: [] };
    for (const a of args) {
      const m = a.match(/^(%[\w.$\-]+)\s*:\s*(.*)$/s);
      if (m) fn.args.push({ name: m[1], t: parseMlirType(splitTop(m[2], " ")[0] === m[2] ? m[2].replace(/\s*\{.*$/, "") : m[2].replace(/\s*\{.*$/, "")) });
    }
    const after = s.slice(close + 1);
    const arrow = after.match(/^\s*->\s*(.*?)(?:\s+attributes\s*\{.*)?\s*\{?\s*$/s);
    if (arrow) fn.returnTypes = parseTypeList(arrow[1].replace(/\{[^{}]*\}/g, ""));
    return fn;
  }
  const generic = s.match(/^"func\.func"\s*\(\)/);
  if (generic) {
    const name = s.match(/sym_name\s*=\s*"([^"]+)"/)?.[1] ?? "func";
    const vis = s.match(/sym_visibility\s*=\s*"([^"]+)"/)?.[1] ?? "public";
    const fn: Func = { name, visibility: vis, args: [], ops: [], returns: [], returnTypes: [] };
    const ft = s.match(/function_type\s*=\s*(\(.*?\)\s*->\s*(?:\([^)]*\)|[^,}]+?))\s*,\s*(?:res_attrs|sym_name|arg_attrs|sym_visibility)/);
    if (ft) fn.returnTypes = parseSignature(ft[1]).outs;
    return fn;
  }
  return null;
}

type Frame = { kind: "module" | "func" | "op" | "other"; func?: Func; owner?: Op };

/** Parse MLIR text into functions (graph-level ops only). */
function parseFunctions(text: string): Func[] {
  const funcs: Func[] = [];
  const stack: Frame[] = [];
  let lastPopped: Frame | null = null;
  for (const { s, line } of logicalLines(preprocess(text))) {
    const top = stack[stack.length - 1];
    const fnFrame = top?.kind === "func" ? top : null;

    let lineFrame: Frame = { kind: "other" };
    let lineOp: Op | null = null;

    if (/^\}/.test(s)) {
      // Closing line: types for a region-holding op often follow `})`/`}`.
      const tail = s.replace(/^[})\s,]+/, "");
      const owner = top?.owner;
      if (owner && !owner.resultTypes.length) applySignature(owner, tail);
    } else if (/^\^[\w.$\-]+/.test(s)) {
      // Block label: entry-block args of a generic-form function are its inputs.
      if (fnFrame?.func && fnFrame.func.args.length === 0 && fnFrame.func.ops.length === 0) {
        const m = s.match(/\((.*)\)/);
        for (const a of m ? splitTop(m[1]) : []) {
          const am = a.match(/^(%[\w.$\-]+)\s*:\s*(.*)$/s);
          if (am) fnFrame.func.args.push({ name: am[1], t: parseMlirType(am[2].replace(/\s*loc\(.*$/, "")) });
        }
      }
    } else {
      const fn = parseFuncHeader(s);
      if (fn) {
        funcs.push(fn);
        lineFrame = { kind: "func", func: fn };
      } else if (/^(?:"?builtin\.module"?|module|"?\w+\.module"?)\b/.test(s)) {
        lineFrame = { kind: "module" };
      } else if (fnFrame?.func) {
        lineOp = parseOpStatement(s, line);
        if (lineOp) {
          if (/^(func\.return|return|tfl\.yield|tf\.Yield|stablehlo\.return|mhlo\.return|onnx\.Return|torch\.return)$/.test(lineOp.op) || /\.return$/.test(lineOp.op)) {
            fnFrame.func.returns = lineOp.operands;
            if (lineOp.operandTypes.length) fnFrame.func.returnTypes = lineOp.operandTypes;
            lineOp = null;
          } else {
            fnFrame.func.ops.push(lineOp);
            lineFrame = { kind: "op", owner: lineOp };
          }
        } else {
          // Continuation (e.g. a `reducer(...) {` line): regions belong to the previous op.
          // Also handles pretty ops wrapped over lines (`linalg.matmul ins(…)` / `outs(…) -> t`).
          const prev = fnFrame.func.ops[fnFrame.func.ops.length - 1];
          if (prev && !/^\^/.test(s)) {
            const body = s.replace(/\{\s*$/, "");
            // `reducer(%a: t, %b: t) {` declares region block args: not operands.
            const declares = /\{\s*$/.test(s) && /^\w+\s*\(\s*%[\w.$\-]+\s*:/.test(s);
            if (!declares) for (const m of body.matchAll(/%[\w.$\-]+(?:#\d+)?/g)) if (!prev.operands.includes(m[0])) prev.operands.push(m[0]);
            prev.text += ` ${body}`;
            if (!prev.resultTypes.length) applySignature(prev, body);
          }
          lineFrame = { kind: "op", owner: prev };
        }
      } else if (top?.kind === "op") {
        lineFrame = { kind: "op", owner: top.owner };
      }
    }

    // Walk braces to maintain the region stack.
    let inStr = false;
    let leading = true;
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (inStr) { if (c === "\\") i++; else if (c === '"') inStr = false; continue; }
      if (c === '"') { inStr = true; continue; }
      if (c === "}") { lastPopped = stack.pop() ?? null; continue; }
      if (c === "{") {
        // `} else {` / `}, {` reopen a region of the same owner.
        const frame = leading && lastPopped && /^\}/.test(s) ? lastPopped : lineFrame;
        stack.push(frame.kind === "other" && top?.kind === "op" ? { kind: "op", owner: top.owner } : frame);
        leading = false;
        continue;
      }
      if (c !== " " && c !== ")" && c !== ",") leading = false;
    }
  }
  return funcs;
}

function pickFunction(funcs: Func[]): Func | undefined {
  return funcs.find((f) => /^(main|main_graph|forward|serving_default)$/.test(f.name) && f.ops.length)
    ?? funcs.find((f) => f.visibility !== "private" && f.ops.length)
    ?? [...funcs].sort((a, b) => b.ops.length - a.ops.length)[0];
}

const DIALECT_NAMES: Record<string, string> = {
  stablehlo: "StableHLO", mhlo: "MHLO", chlo: "CHLO", tosa: "TOSA", tfl: "TFLite", tf: "TensorFlow",
  onnx: "ONNX", torch: "Torch", linalg: "Linalg", arith: "Arith", tensor: "Tensor", math: "Math", scf: "SCF",
  memref: "MemRef", vector: "Vector", ttir: "TTIR", iree: "IREE", flow: "IREE Flow",
};

/** True when the text looks like MLIR (module/func/op syntax). */
export function isMlir(text: string): boolean {
  const head = text.slice(0, 64 * 1024);
  return /(^|\n)\s*(module\b|"builtin\.module"|func\.func\b|"func\.func"|func\s+@)/.test(head) && /tensor<|memref<|!torch\.|%\w+\s*=/.test(head);
}

/** Parse textual MLIR into the normalized graph (one function rendered). */
export function parseMlir(input: string | ArrayBuffer | Uint8Array, name = "model.mlir"): Model {
  const text = typeof input === "string" ? input : new TextDecoder().decode(input instanceof Uint8Array ? input : new Uint8Array(input));
  const funcs = parseFunctions(text);
  const fn = pickFunction(funcs);
  if (!fn) throw new Error("MLIR: no func.func with operations found.");

  const vals = new Map<string, ValInfo>();
  const producer = new Map<string, number>();
  const constants = new Map<string, Constant>();
  const constUsed = new Set<string>();
  const layers: ModelLayer[] = [];
  const edges = new EdgeSet();
  const dialects = new Map<string, number>();
  const base = (v: string) => v.replace(/#\d+$/, "");
  const lookup = (v: string) => vals.get(v) ?? vals.get(`${v}#0`) ?? vals.get(base(v));
  const prodOf = (v: string) => producer.get(v) ?? producer.get(`${v}#0`) ?? producer.get(base(v));

  for (const a of fn.args) {
    const id = layers.length;
    vals.set(a.name, a.t);
    producer.set(a.name, id);
    layers.push(makeLayer({
      id, name: a.name, type: "Input", canon: "Input", shape: a.t.shape, dt: a.t.dt, group: "input",
      outs: [{ n: a.name, s: a.t.shape }], framework: FRAMEWORK,
    }));
  }
  const returned = new Set(fn.returns.map(base));

  for (const op of fn.ops) {
    op.operands.forEach((v, k) => { if (!lookup(v) && op.operandTypes[k]) vals.set(v, op.operandTypes[k]); });
    op.results.forEach((r, k) => vals.set(r, op.resultTypes[k] ?? op.resultTypes[0] ?? { dt: "?" }));
    const dialect = op.op.split(".")[0];
    dialects.set(dialect, (dialects.get(dialect) ?? 0) + 1);

    if (isConstOp(op.op) && op.results.length === 1) {
      const t = op.resultTypes[0] ?? { dt: "?" };
      constants.set(op.results[0], readConstant(op.text, t));
      continue;
    }

    const id = layers.length;
    const inS = op.operands.map((v) => lookup(v)?.shape);
    const out = op.resultTypes[0];
    const canon = canonOf(op.op);
    let params = 0;
    let w: WStat | null = null;
    const attr: Record<string, unknown> = {};
    const constIns: string[] = [];
    for (const v of op.operands) {
      const c = constants.get(v);
      if (c) {
        constIns.push(`${v} ${c.shape.join("×") || "scalar"}${c.splat ? " (splat)" : ""}`);
        if (!constUsed.has(v)) { params += c.params; constUsed.add(v); if (c.w && (!w || (c.w.size ?? 0) > (w.size ?? 0))) w = c.w; }
        continue;
      }
      edges.add(prodOf(v), id);
    }
    if (constIns.length) attr.constants = constIns;
    const callee = op.text.match(/@([\w.$\-]+)/);
    if (callee && /call/.test(op.op)) attr.callee = callee[1];
    Object.assign(attr, keyAttrs(op.text));
    if (op.results.length > 1) attr.results = op.results.length;
    attr.line = op.line;

    const isOut = op.results.some((r) => returned.has(base(r)) || returned.has(r));
    for (const r of op.results) producer.set(r, id);
    producer.set(base(op.results[0] ?? `%${id}`), id);
    layers.push(makeLayer({
      id, name: op.results[0] ? base(op.results[0]) : `${short(op.op)}_${id}`,
      type: op.op, canon, shape: out?.shape, dt: out?.dt && out.dt !== "?" ? out.dt : "float32",
      params, cost: costOf(op, canon, inS, out?.shape),
      group: isOut ? "output" : undefined,
      ins: op.operands.map((v) => ({ n: v, s: lookup(v)?.shape })),
      outs: op.results.map((r, k) => ({ n: r, s: op.resultTypes[k]?.shape })),
      attr, w, framework: FRAMEWORK,
    }));
  }

  const ranked = [...dialects.entries()].filter(([d]) => !["func", "arith", "tensor", "builtin"].includes(d)).sort((a, b) => b[1] - a[1]);
  const main = ranked[0]?.[0] ?? [...dialects.keys()][0] ?? "";
  const label = DIALECT_NAMES[main] ?? main;
  const others = funcs.filter((f) => f !== fn).map((f) => `@${f.name} (${f.ops.length} ops)`);
  const retShape = fn.returns.length ? lookup(fn.returns[0])?.shape : fn.returnTypes[0]?.shape;
  return {
    name, format: label ? `MLIR (${label})` : "MLIR", framework: label ? `MLIR / ${label}` : FRAMEWORK, opset: 0,
    sizeBytes: new TextEncoder().encode(text).byteLength,
    inputShape: cleanShape(fn.args[0]?.t.shape), outputShape: cleanShape(retShape ?? fn.returnTypes[0]?.shape),
    layers, edges: edges.edges,
    producer: `@${fn.name}: ${fn.ops.length} ops, dialects ${[...dialects.keys()].join(", ")}`,
    doc: others.length ? `Other functions: ${others.join(", ")}` : undefined,
  };
}
