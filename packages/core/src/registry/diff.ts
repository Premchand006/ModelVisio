// Version diff: compare two model snapshots. Pure, no React, no I/O.
// Used by the registry UI (diff any two versions) and `modelvisio diff`.

import type { CompatIssueRecord, CompatResult, DeviceEstimate, LayerFingerprint, ModelSnapshot } from "./types";

export type Delta = { before: number; after: number; delta: number; /** null when before is 0 */ pct: number | null };

export type LayerChange = {
  name: string;
  op: string;
  before: LayerFingerprint;
  after: LayerFingerprint;
  fields: ("shape" | "params" | "flops")[];
};

export type CompatChange = {
  target: string;
  label: string;
  before: { errors: number; warnings: number } | null;
  after: { errors: number; warnings: number } | null;
  status: "regression" | "improvement" | "unchanged" | "added" | "removed";
  newIssues: string[];
  resolvedIssues: string[];
};

export type LatencyChange = {
  deviceId: string;
  deviceName: string;
  beforeMs: number;
  afterMs: number;
  pct: number | null;
  fitsBefore: boolean;
  fitsAfter: boolean;
  status: "regression" | "improvement" | "unchanged";
};

export type Regression = {
  kind: "compat" | "latency" | "memory-fit" | "compile" | "size";
  target: string;
  severity: "error" | "warn";
  msg: string;
};

export type VersionDiff = {
  /** Both snapshots hash to the same bytes. */
  identical: boolean;
  formatChanged: boolean;
  stats: Record<"params" | "flops" | "macs" | "sizeBytes" | "layers" | "memBytes", Delta>;
  layers: { added: LayerFingerprint[]; removed: LayerFingerprint[]; changed: LayerChange[]; unchanged: number };
  compat: CompatChange[];
  latency: LatencyChange[];
  regressions: Regression[];
};

export type DiffOptions = {
  /** Latency increase (percent) tolerated before it counts as a regression. Default 5. */
  latencyTolerancePct?: number;
  /** File-size increase (percent) that counts as a regression. Default: size growth is never a regression. */
  sizeTolerancePct?: number;
};

export function delta(before: number, after: number): Delta {
  return { before, after, delta: after - before, pct: before !== 0 ? ((after - before) / Math.abs(before)) * 100 : null };
}

const layerKey = (l: LayerFingerprint) => `${l.name}\u0000${l.op}`;
const issueKey = (i: CompatIssueRecord) => (i.layer ? `${i.layer}: ${i.msg}` : i.msg);

/** Structural layer diff, matching layers by (name, op). Duplicate keys pair up in order. */
export function diffLayers(a: LayerFingerprint[], b: LayerFingerprint[]): VersionDiff["layers"] {
  const pool = new Map<string, LayerFingerprint[]>();
  for (const l of a) {
    const k = layerKey(l);
    const list = pool.get(k);
    if (list) list.push(l);
    else pool.set(k, [l]);
  }
  const added: LayerFingerprint[] = [];
  const changed: LayerChange[] = [];
  let unchanged = 0;
  for (const l of b) {
    const prev = pool.get(layerKey(l))?.shift();
    if (!prev) { added.push(l); continue; }
    const fields: LayerChange["fields"] = [];
    if (prev.shape !== l.shape) fields.push("shape");
    if (prev.params !== l.params) fields.push("params");
    if (prev.flops !== l.flops) fields.push("flops");
    if (fields.length) changed.push({ name: l.name, op: l.op, before: prev, after: l, fields });
    else unchanged++;
  }
  const removed = [...pool.values()].flat();
  return { added, removed, changed, unchanged };
}

function diffCompat(a: CompatResult[], b: CompatResult[]): CompatChange[] {
  const ids = [...new Set([...a.map((c) => c.target), ...b.map((c) => c.target)])];
  return ids.map((id) => {
    const x = a.find((c) => c.target === id);
    const y = b.find((c) => c.target === id);
    const label = (y ?? x)!.label;
    if (!x || !y) {
      return {
        target: id, label,
        before: x ? { errors: x.errors, warnings: x.warnings } : null,
        after: y ? { errors: y.errors, warnings: y.warnings } : null,
        status: x ? "removed" : "added",
        newIssues: y ? y.issues.map(issueKey) : [],
        resolvedIssues: x ? x.issues.map(issueKey) : [],
      } satisfies CompatChange;
    }
    const before = new Set(x.issues.map(issueKey));
    const after = new Set(y.issues.map(issueKey));
    const newIssues = [...after].filter((k) => !before.has(k));
    const resolvedIssues = [...before].filter((k) => !after.has(k));
    let status: CompatChange["status"] = "unchanged";
    if (y.errors > x.errors || (y.errors === x.errors && y.warnings > x.warnings)) status = "regression";
    else if (y.errors < x.errors || y.warnings < x.warnings) status = "improvement";
    return { target: id, label, before: { errors: x.errors, warnings: x.warnings }, after: { errors: y.errors, warnings: y.warnings }, status, newIssues, resolvedIssues };
  });
}

function diffLatency(a: DeviceEstimate[], b: DeviceEstimate[], tolPct: number): LatencyChange[] {
  const out: LatencyChange[] = [];
  for (const y of b) {
    const x = a.find((e) => e.deviceId === y.deviceId);
    if (!x) continue;
    const d = delta(x.latencyMs, y.latencyMs);
    let status: LatencyChange["status"] = "unchanged";
    if (d.pct != null && d.pct > tolPct) status = "regression";
    else if (d.pct != null && d.pct < -tolPct) status = "improvement";
    out.push({ deviceId: y.deviceId, deviceName: y.deviceName, beforeMs: x.latencyMs, afterMs: y.latencyMs, pct: d.pct, fitsBefore: x.fits, fitsAfter: y.fits, status });
  }
  return out;
}

const pctStr = (p: number | null) => (p == null ? "n/a" : `${p >= 0 ? "+" : ""}${p.toFixed(1)}%`);

/** Compare `before` → `after`. */
export function diffSnapshots(before: ModelSnapshot, after: ModelSnapshot, opts: DiffOptions = {}): VersionDiff {
  const tol = opts.latencyTolerancePct ?? 5;
  const stats = {
    params: delta(before.stats.params, after.stats.params),
    flops: delta(before.stats.flops, after.stats.flops),
    macs: delta(before.stats.macs, after.stats.macs),
    sizeBytes: delta(before.sizeBytes, after.sizeBytes),
    layers: delta(before.stats.layers, after.stats.layers),
    memBytes: delta(before.stats.memBytes, after.stats.memBytes),
  };
  const compat = diffCompat(before.compat, after.compat);
  const latency = diffLatency(before.estimates, after.estimates, tol);

  const regressions: Regression[] = [];
  for (const c of compat) {
    if (c.status !== "regression" || !c.before || !c.after) continue;
    const errUp = c.after.errors > c.before.errors;
    regressions.push({
      kind: "compat", target: c.target, severity: errUp ? "error" : "warn",
      msg: errUp
        ? `${c.label}: compat errors ${c.before.errors} → ${c.after.errors}`
        : `${c.label}: compat warnings ${c.before.warnings} → ${c.after.warnings}`,
    });
  }
  for (const l of latency) {
    const b = before.estimates.find((e) => e.deviceId === l.deviceId)!;
    const a = after.estimates.find((e) => e.deviceId === l.deviceId)!;
    if (l.fitsBefore && !l.fitsAfter) regressions.push({ kind: "memory-fit", target: l.deviceId, severity: "error", msg: `${l.deviceName}: model no longer fits in device memory` });
    if (!b.compileFail && a.compileFail) regressions.push({ kind: "compile", target: l.deviceId, severity: "error", msg: `${l.deviceName}: model no longer compiles (unsupported op, no CPU fallback)` });
    if (l.status === "regression") regressions.push({ kind: "latency", target: l.deviceId, severity: "warn", msg: `${l.deviceName}: est. latency ${l.beforeMs.toFixed(2)} → ${l.afterMs.toFixed(2)} ms (${pctStr(l.pct)})` });
  }
  if (opts.sizeTolerancePct != null && stats.sizeBytes.pct != null && stats.sizeBytes.pct > opts.sizeTolerancePct) {
    regressions.push({ kind: "size", target: "file", severity: "warn", msg: `File size grew ${pctStr(stats.sizeBytes.pct)} (tolerance ${opts.sizeTolerancePct}%)` });
  }

  return {
    identical: before.sha256 !== "" && before.sha256 === after.sha256,
    formatChanged: before.format !== after.format,
    stats,
    layers: diffLayers(before.layers, after.layers),
    compat,
    latency,
    regressions,
  };
}

/** Alias: registry versions are snapshots plus bookkeeping. */
export const diffVersions = diffSnapshots;
