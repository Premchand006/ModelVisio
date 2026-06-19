import { num, numArr, type RawNode } from "./raw";

const td = new TextDecoder();

/** Decode a node's attributes into a plain, JSON-friendly record. */
export function decodeAttrs(node: RawNode): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const a of node.attribute ?? []) {
    if (!a.name) continue;
    if (a.ints && a.ints.length) out[a.name] = numArr(a.ints);
    else if (a.floats && a.floats.length) out[a.name] = a.floats;
    else if (a.strings && a.strings.length) out[a.name] = a.strings.map((s) => td.decode(s));
    else if (a.s && a.s.length) out[a.name] = td.decode(a.s);
    else if (a.i != null && a.type === 2) out[a.name] = num(a.i);
    else if (a.f != null && a.type === 1) out[a.name] = a.f;
    else if (a.i != null) out[a.name] = num(a.i);
    else if (a.f != null) out[a.name] = a.f;
  }
  return out;
}

export function attrInts(attrs: Record<string, unknown>, key: string): number[] | undefined {
  const v = attrs[key];
  return Array.isArray(v) ? (v as number[]) : undefined;
}

export function attrInt(attrs: Record<string, unknown>, key: string, dflt: number): number {
  const v = attrs[key];
  return typeof v === "number" ? v : dflt;
}

export function attrStr(attrs: Record<string, unknown>, key: string): string | undefined {
  const v = attrs[key];
  return typeof v === "string" ? v : undefined;
}
