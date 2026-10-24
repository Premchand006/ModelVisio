// Per-op profile parsing. With `enableProfiling`, onnxruntime-web's profiler
// prints Chrome-trace events to the console, one JSON object per line, when
// the session's profiling ends:
//   {"cat" : "Node", ..., "dur" :3025, "name" :"Relu_0_kernel_time", "args" : {"op_name" : "Relu", "provider" : "CPUExecutionProvider", ...}},
// The benchmark worker captures those lines; this module aggregates them.
// Times are host-side kernel times in microseconds. Pure: no React, no DOM.

export type OpProfileRow = {
  /** Node name in the graph. */
  node: string;
  op: string;
  provider: string;
  /** Mean kernel time per profiled inference (ms). */
  meanMs: number;
  calls: number;
  /** Fraction of the summed kernel time across all nodes (0..1). */
  share: number;
};

export type OpTypeRow = { op: string; meanMs: number; nodes: number; share: number };

type TraceEvent = { cat?: string; name?: string; dur?: number; args?: { op_name?: string; provider?: string } };

/** Pull trace events out of captured console lines; non-JSON lines are skipped. */
export function parseTraceLines(lines: readonly string[]): TraceEvent[] {
  const out: TraceEvent[] = [];
  for (const raw of lines) {
    for (const piece of raw.split(/\r?\n/)) {
      let s = piece.trim();
      if (!s.startsWith("{")) continue;
      if (s.endsWith(",")) s = s.slice(0, -1);
      try {
        const ev = JSON.parse(s) as TraceEvent;
        if (ev && typeof ev === "object") out.push(ev);
      } catch {
        // not a trace event line
      }
    }
  }
  return out;
}

const SUFFIX = "_kernel_time";

/**
 * Aggregate node kernel events into per-node rows, sorted by time (desc).
 * `skipFirst` drops each node's first event (the cold run) from the averages.
 */
export function aggregateProfile(events: readonly TraceEvent[], profiledRuns: number, skipFirst = false): OpProfileRow[] {
  const by = new Map<string, { op: string; provider: string; us: number; calls: number }>();
  const seen = new Set<string>();
  for (const ev of events) {
    if (ev.cat !== "Node" || typeof ev.name !== "string" || !ev.name.endsWith(SUFFIX)) continue;
    const node = ev.name.slice(0, -SUFFIX.length);
    if (skipFirst && !seen.has(node)) {
      seen.add(node);
      continue;
    }
    const cur = by.get(node) ?? { op: ev.args?.op_name ?? "?", provider: (ev.args?.provider ?? "").replace(/ExecutionProvider$/, ""), us: 0, calls: 0 };
    cur.us += typeof ev.dur === "number" ? ev.dur : 0;
    cur.calls += 1;
    by.set(node, cur);
  }
  let total = 0;
  by.forEach((v) => { total += v.us; });
  const runs = Math.max(1, profiledRuns);
  return [...by.entries()]
    .map(([node, v]) => ({ node, op: v.op, provider: v.provider, meanMs: v.us / 1000 / runs, calls: v.calls, share: total > 0 ? v.us / total : 0 }))
    .sort((a, b) => b.meanMs - a.meanMs);
}

/** Roll per-node rows up to op types (e.g. all Conv nodes together). */
export function byOpType(rows: readonly OpProfileRow[]): OpTypeRow[] {
  const m = new Map<string, OpTypeRow>();
  for (const r of rows) {
    const cur = m.get(r.op) ?? { op: r.op, meanMs: 0, nodes: 0, share: 0 };
    cur.meanMs += r.meanMs;
    cur.nodes += 1;
    cur.share += r.share;
    m.set(r.op, cur);
  }
  return [...m.values()].sort((a, b) => b.meanMs - a.meanMs);
}
