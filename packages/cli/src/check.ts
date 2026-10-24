// `modelvisio check`: compiler-compat pre-flight + budgets for CI.

import { scoreDevice } from "@modelvisio/core/scoring";
import { checkCompat, findDevice, type CompatResult, type CompatTarget } from "@modelvisio/core/registry";
import { bytesStr, escapeMd, escapeXml, type LoadedModel } from "./load";
import { summarize, type InspectSummary } from "./inspect";

export type Severity = "error" | "warn";
export type FailOn = "warn" | "error";

export type Budget = {
  kind: "size" | "latency";
  /** Device id for latency budgets; "file" for size. */
  subject: string;
  label: string;
  limit: number;
  actual: number | null;
  unit: "MB" | "ms";
  /** "unknown": the value can't be computed for this model (counted as a warning). */
  status: "ok" | "exceeded" | "unknown";
  passed: boolean;
  note?: string;
};

export type ModelCheck = {
  file: string;
  summary: InspectSummary;
  targets: CompatResult[];
  budgets: Budget[];
  notes: string[];
  errors: number;
  warnings: number;
  passed: boolean;
};

export type CheckReport = {
  tool: "modelvisio";
  failOn: FailOn;
  passed: boolean;
  models: ModelCheck[];
};

export type CheckOptions = {
  targets: CompatTarget[];
  /** Device for latency budgets; defaults to each target's reference device. */
  deviceId?: string;
  maxLatencyMs?: number;
  maxSizeMb?: number;
  failOn: FailOn;
};

const failing = (sev: Severity, failOn: FailOn) => sev === "error" || failOn === "warn";

export function checkModel(loaded: LoadedModel, opts: CheckOptions): ModelCheck {
  const { model } = loaded;
  const notes: string[] = [];
  const targets = loaded.graphParsed ? opts.targets.map((t) => checkCompat(model, t)) : [];
  if (!loaded.graphParsed) {
    // Metadata-only formats: no graph, so no op-level or roofline analysis. Say so loudly.
    notes.push(`${model.format}: graph not parsed (metadata only); op compatibility and latency can't be checked. Export to ONNX for a full check.`);
    for (const t of opts.targets) {
      targets.push({ target: t.id, label: t.label, deviceId: t.deviceId, errors: 0, warnings: 1, issues: [{ severity: "warn", msg: "Not analyzed: model graph not parsed for this format", source: "device" }] });
    }
  }

  const budgets: Budget[] = [];
  if (opts.maxSizeMb != null) {
    const mb = loaded.bytes.byteLength / 1e6;
    budgets.push({ kind: "size", subject: "file", label: "File size", limit: opts.maxSizeMb, actual: round(mb, 3), unit: "MB", status: mb <= opts.maxSizeMb ? "ok" : "exceeded", passed: mb <= opts.maxSizeMb });
  }
  if (opts.maxLatencyMs != null) {
    const ids = opts.deviceId ? [opts.deviceId] : [...new Set(opts.targets.map((t) => t.deviceId).filter((d): d is string => !!d))];
    if (ids.length === 0) notes.push("--max-latency-ms given but no device: pass --device <id>.");
    for (const id of ids) {
      const dev = findDevice(id)!;
      const noCompute = model.layers.every((l) => !l.flops && !l.macs);
      if (!loaded.graphParsed || noCompute) {
        budgets.push({
          kind: "latency", subject: id, label: `Latency on ${dev.n}`, limit: opts.maxLatencyMs, actual: null, unit: "ms", status: "unknown", passed: false,
          note: loaded.graphParsed ? `the ${model.format} parser reports no FLOPs, so latency can't be estimated; export to ONNX` : "cannot estimate without a parsed graph",
        });
        continue;
      }
      const s = scoreDevice(model, dev);
      const ms = Number.isFinite(s.latencyMs) ? s.latencyMs : Infinity;
      budgets.push({
        kind: "latency", subject: id, label: `Latency on ${dev.n}`, limit: opts.maxLatencyMs,
        actual: Number.isFinite(ms) ? round(ms, 3) : null, unit: "ms", status: ms <= opts.maxLatencyMs ? "ok" : "exceeded", passed: ms <= opts.maxLatencyMs,
        note: `roofline estimate, ${s.precision}, ${s.roofline.calibrated ? "calibrated" : "uncalibrated"}, ${s.confidence} confidence`,
      });
    }
  }

  const errors = targets.reduce((n, t) => n + t.errors, 0) + budgets.filter((b) => b.status === "exceeded").length;
  const warnings = targets.reduce((n, t) => n + t.warnings, 0) + budgets.filter((b) => b.status === "unknown").length;
  const passed = errors === 0 && (opts.failOn === "error" || warnings === 0);
  return { file: loaded.fileName, summary: summarize(loaded), targets, budgets, notes, errors, warnings, passed };
}

export function checkReport(models: ModelCheck[], failOn: FailOn): CheckReport {
  return { tool: "modelvisio", failOn, passed: models.every((m) => m.passed), models };
}

function round(n: number, d: number) {
  const f = 10 ** d;
  return Math.round(n * f) / f;
}

const fmtBudget = (b: Budget) => (b.actual == null ? "n/a" : b.unit === "MB" ? bytesStr(b.actual * 1e6) : `${b.actual.toFixed(2)} ms`);
const fmtLimit = (b: Budget) => (b.unit === "MB" ? `${b.limit} MB` : `${b.limit} ms`);

// ── text ────────────────────────────────────────────────────────────────────

export function checkText(r: CheckReport): string {
  const out: string[] = [];
  for (const m of r.models) {
    out.push(`${m.passed ? "PASS" : "FAIL"}  ${m.file}  (${m.summary.format}, ${m.summary.layers} layers, ${bytesStr(m.summary.sizeBytes)})`);
    for (const n of m.notes) out.push(`  note: ${n}`);
    for (const t of m.targets) {
      out.push(`  [${t.label}] ${t.errors} error(s), ${t.warnings} warning(s)`);
      for (const i of t.issues) out.push(`    ${i.severity === "error" ? "ERROR" : "warn "} ${i.layer ? i.layer + ": " : ""}${i.msg}`);
    }
    for (const b of m.budgets) {
      out.push(`  [budget] ${b.label}: ${fmtBudget(b)} (limit ${fmtLimit(b)}) ${b.status === "ok" ? "ok" : b.status === "unknown" ? "UNKNOWN" : "EXCEEDED"}${b.note ? `  (${b.note})` : ""}`);
    }
  }
  out.push("");
  out.push(`${r.passed ? "PASSED" : "FAILED"} (fail-on: ${r.failOn})`);
  return out.join("\n") + "\n";
}

// ── markdown (PR comment / $GITHUB_STEP_SUMMARY) ──────────────────────────────

export function checkMarkdown(r: CheckReport): string {
  const out: string[] = [`## ModelVisio check: ${r.passed ? "passed" : "failed"}`, ""];
  out.push("| Model | Format | Layers | Params | Size | Errors | Warnings | Result |", "|---|---|---:|---:|---:|---:|---:|---|");
  for (const m of r.models) {
    out.push(`| \`${escapeMd(m.file)}\` | ${escapeMd(m.summary.format)} | ${m.summary.layers} | ${fmtN(m.summary.params)} | ${bytesStr(m.summary.sizeBytes)} | ${m.errors} | ${m.warnings} | ${m.passed ? "PASS" : "**FAIL**"} |`);
  }
  out.push("");
  for (const m of r.models) {
    out.push(`### \`${escapeMd(m.file)}\``, "");
    for (const n of m.notes) out.push(`> ${escapeMd(n)}`, "");
    if (m.targets.length) {
      out.push("| Target | Errors | Warnings | Status |", "|---|---:|---:|---|");
      for (const t of m.targets) {
        const status = t.errors ? "**incompatible**" : t.warnings ? "warnings" : "ok";
        out.push(`| ${escapeMd(t.label)} | ${t.errors} | ${t.warnings} | ${status} |`);
      }
      out.push("");
      const issues = m.targets.flatMap((t) => t.issues.map((i) => ({ t, i })));
      if (issues.length) {
        out.push("<details><summary>Issues</summary>", "");
        for (const { t, i } of issues.slice(0, 200)) out.push(`- **${i.severity}** [${escapeMd(t.label)}] ${i.layer ? `\`${escapeMd(i.layer)}\`: ` : ""}${escapeMd(i.msg)}`);
        if (issues.length > 200) out.push(`- ... ${issues.length - 200} more`);
        out.push("", "</details>", "");
      }
    }
    if (m.budgets.length) {
      out.push("| Budget | Actual | Limit | Result |", "|---|---:|---:|---|");
      for (const b of m.budgets) out.push(`| ${escapeMd(b.label)} | ${fmtBudget(b)} | ${fmtLimit(b)} | ${b.status === "ok" ? "ok" : b.status === "unknown" ? "unknown" : "**exceeded**"} |`);
      out.push("");
      if (m.budgets.some((b) => b.kind === "latency")) out.push("_Latency is a roofline estimate from the model graph, not an on-device measurement._", "");
    }
  }
  out.push(`<sub>fail-on: ${r.failOn}</sub>`, "");
  return out.join("\n");
}

const fmtN = (n: number) => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n));

// ── JUnit XML (CI test reporting) ────────────────────────────────────────────

export function checkJunit(r: CheckReport): string {
  type Case = { name: string; classname: string; failure?: { message: string; body: string }; out?: string };
  const suites: { name: string; cases: Case[] }[] = [];
  for (const m of r.models) {
    const base = `modelvisio.${m.file}`;
    for (const t of m.targets) {
      const cases: Case[] = [];
      const fails = t.issues.filter((i) => failing(i.severity, r.failOn));
      const rest = t.issues.filter((i) => !failing(i.severity, r.failOn));
      if (t.issues.length === 0) cases.push({ name: "compatible", classname: `${base}.${t.target}` });
      for (const i of fails) {
        cases.push({
          name: `${i.source === "op" ? "op" : "device"}: ${i.layer ? i.layer + ": " : ""}${i.msg}`.slice(0, 200),
          classname: `${base}.${t.target}`,
          failure: { message: `${i.severity}: ${i.msg}`, body: `${t.label}${i.layer ? ` / ${i.layer}` : ""}: ${i.msg}` },
        });
      }
      if (rest.length) {
        cases.push({ name: `${rest.length} warning(s) below fail-on threshold`, classname: `${base}.${t.target}`, out: rest.map((i) => `${i.layer ? i.layer + ": " : ""}${i.msg}`).join("\n") });
      }
      suites.push({ name: `${m.file} [${t.label}]`, cases });
    }
    if (m.budgets.length || m.notes.length) {
      const cases: Case[] = m.budgets.map((b) => ({
        name: `${b.label} <= ${fmtLimit(b)}`,
        classname: `${base}.budget`,
        failure: b.status === "ok" || (b.status === "unknown" && r.failOn === "error") ? undefined
          : { message: b.status === "unknown" ? `${b.label}: cannot be estimated` : `${b.label}: ${fmtBudget(b)} exceeds ${fmtLimit(b)}`, body: b.note ?? "" },
        out: b.note,
      }));
      if (m.notes.length) cases.push({ name: "notes", classname: `${base}.notes`, out: m.notes.join("\n") });
      suites.push({ name: `${m.file} [budgets]`, cases });
    }
  }
  const total = suites.reduce((n, s) => n + s.cases.length, 0);
  const failures = suites.reduce((n, s) => n + s.cases.filter((c) => c.failure).length, 0);
  const x = escapeXml;
  const lines = ['<?xml version="1.0" encoding="UTF-8"?>', `<testsuites name="modelvisio check" tests="${total}" failures="${failures}">`];
  for (const s of suites) {
    const f = s.cases.filter((c) => c.failure).length;
    lines.push(`  <testsuite name="${x(s.name)}" tests="${s.cases.length}" failures="${f}" errors="0" skipped="0">`);
    for (const c of s.cases) {
      lines.push(`    <testcase name="${x(c.name)}" classname="${x(c.classname)}">`);
      if (c.failure) lines.push(`      <failure message="${x(c.failure.message)}">${x(c.failure.body)}</failure>`);
      if (c.out) lines.push(`      <system-out>${x(c.out)}</system-out>`);
      lines.push("    </testcase>");
    }
    lines.push("  </testsuite>");
  }
  lines.push("</testsuites>", "");
  return lines.join("\n");
}
