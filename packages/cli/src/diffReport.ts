// Rendering for `modelvisio diff` (text / markdown). JSON is the VersionDiff itself.

import type { Delta, VersionDiff } from "@modelvisio/core/registry";
import { bytesStr, escapeMd, human, pctStr } from "./load";

type Row = [string, Delta, (n: number) => string];

const rows = (d: VersionDiff): Row[] => [
  ["Params", d.stats.params, human],
  ["FLOPs", d.stats.flops, human],
  ["Size", d.stats.sizeBytes, bytesStr],
  ["Layers", d.stats.layers, String],
  ["Act. memory", d.stats.memBytes, bytesStr],
];

const signed = (x: Delta, f: (n: number) => string) => (x.delta === 0 ? "=" : `${x.delta > 0 ? "+" : "-"}${f(Math.abs(x.delta))} (${pctStr(x.pct)})`);

export function diffText(d: VersionDiff, oldName: string, newName: string): string {
  const out = [`${oldName} -> ${newName}${d.identical ? "  (identical bytes)" : ""}${d.formatChanged ? "  (format changed)" : ""}`, ""];
  for (const [k, x, f] of rows(d)) out.push(`  ${k.padEnd(12)} ${f(x.before).padStart(10)} -> ${f(x.after).padStart(10)}  ${signed(x, f)}`);
  out.push("", `Layers: +${d.layers.added.length} added, -${d.layers.removed.length} removed, ~${d.layers.changed.length} changed, ${d.layers.unchanged} unchanged`);
  for (const l of d.layers.added.slice(0, 20)) out.push(`  + ${l.name} (${l.op}) ${l.shape}`);
  for (const l of d.layers.removed.slice(0, 20)) out.push(`  - ${l.name} (${l.op}) ${l.shape}`);
  for (const c of d.layers.changed.slice(0, 20)) out.push(`  ~ ${c.name} (${c.op}): ${c.fields.join(", ")}`);
  out.push("", "Compiler compatibility:");
  for (const c of d.compat) {
    const b = c.before ? `${c.before.errors}E/${c.before.warnings}W` : "-";
    const a = c.after ? `${c.after.errors}E/${c.after.warnings}W` : "-";
    out.push(`  ${c.label.padEnd(28)} ${b.padStart(7)} -> ${a.padEnd(7)} ${c.status}`);
    for (const i of c.newIssues.slice(0, 10)) out.push(`      new: ${i}`);
  }
  if (d.latency.length) {
    out.push("", "Estimated latency (roofline):");
    for (const l of d.latency) out.push(`  ${l.deviceName.padEnd(28)} ${l.beforeMs.toFixed(2).padStart(9)} -> ${l.afterMs.toFixed(2).padStart(9)} ms  ${pctStr(l.pct)}  ${l.status}${l.fitsAfter ? "" : " (does not fit)"}`);
  }
  out.push("", d.regressions.length ? `${d.regressions.length} regression(s):` : "No regressions.");
  for (const r of d.regressions) out.push(`  ${r.severity === "error" ? "ERROR" : "warn "} ${r.msg}`);
  return out.join("\n") + "\n";
}

export function diffMarkdown(d: VersionDiff, oldName: string, newName: string): string {
  const out = [`## ModelVisio diff: \`${escapeMd(oldName)}\` → \`${escapeMd(newName)}\``, ""];
  out.push(d.regressions.length ? `**${d.regressions.length} regression(s)**` : "No regressions.", "");
  for (const r of d.regressions) out.push(`- **${r.severity}** ${escapeMd(r.msg)}`);
  if (d.regressions.length) out.push("");
  out.push("| Metric | Before | After | Change |", "|---|---:|---:|---:|");
  for (const [k, x, f] of rows(d)) out.push(`| ${k} | ${f(x.before)} | ${f(x.after)} | ${signed(x, f)} |`);
  out.push("", `**Layers:** +${d.layers.added.length} added, -${d.layers.removed.length} removed, ~${d.layers.changed.length} changed, ${d.layers.unchanged} unchanged`, "");
  out.push("| Target | Before | After | Status |", "|---|---:|---:|---|");
  for (const c of d.compat) {
    const b = c.before ? `${c.before.errors}E / ${c.before.warnings}W` : "-";
    const a = c.after ? `${c.after.errors}E / ${c.after.warnings}W` : "-";
    out.push(`| ${escapeMd(c.label)} | ${b} | ${a} | ${c.status === "regression" ? "**regression**" : c.status} |`);
  }
  if (d.latency.length) {
    out.push("", "| Device | Est. latency before | after | Change |", "|---|---:|---:|---:|");
    for (const l of d.latency) out.push(`| ${escapeMd(l.deviceName)} | ${l.beforeMs.toFixed(2)} ms | ${l.afterMs.toFixed(2)} ms${l.fitsAfter ? "" : " (no fit)"} | ${l.status === "regression" ? `**${pctStr(l.pct)}**` : pctStr(l.pct)} |`);
  }
  out.push("");
  return out.join("\n");
}
