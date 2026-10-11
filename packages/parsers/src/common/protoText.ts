// Protobuf *text format* parser (Caffe .prototxt, TF .pbtxt, …). Schema-less:
// every field becomes an array of values on the message object so repeated
// and singular fields read the same way. Scalars are kept as numbers when they
// look numeric, booleans for true/false, and strings otherwise (enum
// identifiers stay as their identifier string, e.g. "MAX").

export type PTValue = string | number | boolean | PTMsg;
export type PTMsg = { [field: string]: PTValue[] };

type Tok = { t: "id" | "str" | "num" | "sym"; v: string };

function tokenize(text: string): Tok[] {
  const toks: Tok[] = [];
  const n = text.length;
  let i = 0;
  while (i < n) {
    const c = text[i];
    if (c === "#") { while (i < n && text[i] !== "\n") i++; continue; }
    if (c === " " || c === "\t" || c === "\r" || c === "\n" || c === "\f" || c === "\v") { i++; continue; }
    if ("{}<>[]:,;".includes(c)) { toks.push({ t: "sym", v: c }); i++; continue; }
    if (c === '"' || c === "'") {
      let s = "";
      let j = i + 1;
      while (j < n && text[j] !== c) {
        if (text[j] === "\\" && j + 1 < n) {
          const e = text[j + 1];
          const map: Record<string, string> = { n: "\n", t: "\t", r: "\r", "\\": "\\", "'": "'", '"': '"', "0": "\0" };
          if (/[0-7]/.test(e)) {
            const m = text.slice(j + 1, j + 4).match(/^[0-7]{1,3}/)![0];
            s += String.fromCharCode(parseInt(m, 8));
            j += 1 + m.length;
            continue;
          }
          if (e === "x") {
            const m = text.slice(j + 2, j + 4).match(/^[0-9a-fA-F]{1,2}/)?.[0] ?? "";
            s += String.fromCharCode(parseInt(m || "0", 16));
            j += 2 + m.length;
            continue;
          }
          s += map[e] ?? e;
          j += 2;
          continue;
        }
        s += text[j++];
      }
      if (j >= n) throw new Error("prototxt: unterminated string");
      // Adjacent string literals concatenate ("a" "b" == "ab").
      const prev = toks[toks.length - 1];
      if (prev && prev.t === "str") prev.v += s;
      else toks.push({ t: "str", v: s });
      i = j + 1;
      continue;
    }
    // Identifier or number (numbers may start with - or . and include e/E, +, -).
    let j = i;
    while (j < n && !/[\s{}<>[\]:,;#"']/.test(text[j])) j++;
    if (j === i) throw new Error(`prototxt: unexpected '${c}' at ${i}`);
    const word = text.slice(i, j);
    toks.push({ t: /^[-+]?(\d|\.\d)/.test(word) || /^-?(inf|infinity|nan)$/i.test(word) ? "num" : "id", v: word });
    i = j;
  }
  return toks;
}

function toNumber(w: string): number {
  const s = w.replace(/[fFuUlL]+$/, "");
  if (/^-?inf(inity)?$/i.test(s)) return s.startsWith("-") ? -Infinity : Infinity;
  if (/^-?nan$/i.test(s)) return NaN;
  if (/^-?0x[0-9a-f]+$/i.test(s)) return parseInt(s, 16);
  const v = Number(s);
  if (Number.isNaN(v)) throw new Error(`prototxt: bad number '${w}'`);
  return v;
}

/** Parse protobuf text format into a schema-less message object. */
export function parseProtoText(text: string): PTMsg {
  const toks = tokenize(text);
  let p = 0;
  const peek = () => toks[p];

  const scalar = (): PTValue => {
    const t = toks[p++];
    if (!t) throw new Error("prototxt: unexpected end of input");
    if (t.t === "str") return t.v;
    if (t.t === "num") return toNumber(t.v);
    if (t.t === "id") {
      if (t.v === "true" || t.v === "True") return true;
      if (t.v === "false" || t.v === "False") return false;
      return t.v;
    }
    throw new Error(`prototxt: unexpected '${t.v}'`);
  };

  const message = (close: string | null): PTMsg => {
    const msg: PTMsg = {};
    for (;;) {
      const t = peek();
      if (!t) {
        if (close) throw new Error(`prototxt: missing '${close}'`);
        return msg;
      }
      if (t.t === "sym" && t.v === close) { p++; return msg; }
      if (t.t === "sym" && (t.v === "," || t.v === ";")) { p++; continue; }
      if (t.t !== "id") throw new Error(`prototxt: expected field name, got '${t.v}'`);
      p++;
      const key = t.v;
      const vals = (msg[key] ??= []);
      let nt = peek();
      if (nt && nt.t === "sym" && nt.v === ":") { p++; nt = peek(); }
      if (!nt) throw new Error(`prototxt: missing value for '${key}'`);
      if (nt.t === "sym" && (nt.v === "{" || nt.v === "<")) {
        p++;
        vals.push(message(nt.v === "{" ? "}" : ">"));
      } else if (nt.t === "sym" && nt.v === "[") {
        p++;
        for (;;) {
          const x = peek();
          if (!x) throw new Error("prototxt: unterminated list");
          if (x.t === "sym" && x.v === "]") { p++; break; }
          if (x.t === "sym" && x.v === ",") { p++; continue; }
          if (x.t === "sym" && (x.v === "{" || x.v === "<")) { p++; vals.push(message(x.v === "{" ? "}" : ">")); continue; }
          vals.push(scalar());
        }
      } else {
        vals.push(scalar());
      }
    }
  };

  return message(null);
}

// ── Accessors ────────────────────────────────────────────────────────────────

export function ptFirst(m: PTMsg | undefined, k: string): PTValue | undefined {
  return m?.[k]?.[0];
}
export function ptNum(m: PTMsg | undefined, k: string, d: number): number {
  const v = ptFirst(m, k);
  return typeof v === "number" ? v : typeof v === "boolean" ? Number(v) : d;
}
export function ptNums(m: PTMsg | undefined, k: string): number[] {
  return (m?.[k] ?? []).filter((v): v is number => typeof v === "number");
}
export function ptStr(m: PTMsg | undefined, k: string): string | undefined {
  const v = ptFirst(m, k);
  return typeof v === "string" ? v : typeof v === "number" || typeof v === "boolean" ? String(v) : undefined;
}
export function ptStrs(m: PTMsg | undefined, k: string): string[] {
  return (m?.[k] ?? []).filter((v): v is string => typeof v === "string");
}
export function ptBool(m: PTMsg | undefined, k: string, d: boolean): boolean {
  const v = ptFirst(m, k);
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (v === "true") return true;
  if (v === "false") return false;
  return d;
}
export function ptMsg(m: PTMsg | undefined, k: string): PTMsg | undefined {
  const v = ptFirst(m, k);
  return v && typeof v === "object" ? v : undefined;
}
export function ptMsgs(m: PTMsg | undefined, k: string): PTMsg[] {
  return (m?.[k] ?? []).filter((v): v is PTMsg => typeof v === "object");
}
