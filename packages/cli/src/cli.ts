// Command dispatch. `run()` is the whole CLI minus process wiring, so tests call
// it directly with captured output.

import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs, type ParseArgsConfig } from "node:util";
import {
  COMPAT_TARGETS, REGISTRY_SCHEMA, diffSnapshots, exportRegistry, knownTargetIds, parseRegistryFile, registerModel,
  resolveTarget, serializeRegistry, snapshotModel, storeFromFile, findDevice,
  type CompatTarget, type RegistryFile,
} from "@modelvisio/core/registry";
import { CliError, bytesStr, escapeMd, human, loadModel } from "./load";
import { inspectMarkdown, inspectText, summarize } from "./inspect";
import { checkJunit, checkMarkdown, checkModel, checkReport, checkText, type FailOn } from "./check";
import { diffMarkdown, diffText } from "./diffReport";

export const VERSION = "0.1.0";

export type Io = { stdout: (s: string) => void; stderr: (s: string) => void; cwd: string; env?: Record<string, string | undefined> };

const HELP = `modelvisio ${VERSION}: model analysis for CI

Usage:
  modelvisio inspect <file...> [--format text|json|markdown] [--json]
  modelvisio check <file...> [--target <id>[,<id>...]] [--device <id>]
                   [--max-latency-ms N] [--max-size-mb N] [--fail-on warn|error]
                   [--format text|json|markdown|junit] [--output <file>] [--summary <file>]
  modelvisio diff <old> <new> [--fail-on-regression] [--latency-tolerance PCT]
                  [--size-tolerance PCT] [--target <ids>] [--format text|json|markdown]
                  [--old-name LABEL] [--new-name LABEL]
                  [--output <file>] [--summary <file>]
  modelvisio registry add <file...> [--registry <path>] [--name N] [--model-id ID]
                  [--label L] [--tags a,b] [--notes TEXT] [--allow-duplicate]
  modelvisio registry export [--registry <path>] [--format json|markdown] [--output <file>]
  modelvisio registry list [--registry <path>] [--json]
  modelvisio targets

Targets: ${COMPAT_TARGETS.map((t) => t.id).join(", ")}, or any device id (\`modelvisio targets\`).
--output writes the report to a file instead of stdout; --summary also appends
the markdown report to a file (e.g. "$GITHUB_STEP_SUMMARY").
--registry defaults to ./modelvisio-registry.json.

Exit codes: 0 ok, 1 check failed / regression found, 2 usage or input error.
`;

const COMMON: ParseArgsConfig["options"] = {
  help: { type: "boolean", short: "h" },
  format: { type: "string" },
  json: { type: "boolean" },
  output: { type: "string", short: "o" },
  summary: { type: "string" },
};

function parse(args: string[], options: ParseArgsConfig["options"]) {
  try {
    return parseArgs({ args, options: { ...COMMON, ...options }, allowPositionals: true, strict: true });
  } catch (e) {
    throw new CliError((e as Error).message);
  }
}

function num(v: unknown, flag: string): number | undefined {
  if (v == null) return undefined;
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) throw new CliError(`${flag} must be a non-negative number, got "${String(v)}"`);
  return n;
}

function pickFormat(v: unknown, json: unknown, allowed: string[], dflt = "text"): string {
  const f = json ? "json" : (v as string | undefined) ?? dflt;
  if (!allowed.includes(f)) throw new CliError(`--format must be one of ${allowed.join(", ")}`);
  return f;
}

function parseTargets(v: unknown): CompatTarget[] {
  if (v == null) return COMPAT_TARGETS;
  const ids = String(v).split(",").map((s) => s.trim()).filter(Boolean);
  return ids.map((id) => {
    const t = resolveTarget(id);
    if (!t) throw new CliError(`Unknown target "${id}". Known: ${knownTargetIds().join(", ")}`);
    return t;
  });
}

function emit(io: Io, text: string, output: unknown, summary?: { file: unknown; markdown: () => string }) {
  if (output) writeFileSync(resolve(io.cwd, String(output)), text);
  else io.stdout(text);
  if (summary?.file) appendFileSync(resolve(io.cwd, String(summary.file)), summary.markdown() + "\n");
}

// ── commands ────────────────────────────────────────────────────────────────

function cmdInspect(args: string[], io: Io): number {
  const { values, positionals } = parse(args, {});
  if (values.help) { io.stdout(HELP); return 0; }
  if (!positionals.length) throw new CliError("inspect: missing <file>");
  const fmt = pickFormat(values.format, values.json, ["text", "json", "markdown"]);
  const sums = positionals.map((f) => summarize(loadModel(f, io.cwd)));
  const md = () => sums.map(inspectMarkdown).join("\n");
  const text = fmt === "json" ? JSON.stringify(sums.length === 1 ? sums[0] : sums, null, 2) + "\n"
    : fmt === "markdown" ? md()
    : sums.map(inspectText).join("\n");
  emit(io, text, values.output, { file: values.summary, markdown: md });
  return 0;
}

function cmdCheck(args: string[], io: Io): number {
  const { values, positionals } = parse(args, {
    target: { type: "string", short: "t" },
    targets: { type: "string" },
    device: { type: "string", short: "d" },
    "max-latency-ms": { type: "string" },
    "max-size-mb": { type: "string" },
    "fail-on": { type: "string" },
  });
  if (values.help) { io.stdout(HELP); return 0; }
  if (!positionals.length) throw new CliError("check: missing <file>");
  const fmt = pickFormat(values.format, values.json, ["text", "json", "markdown", "junit"]);
  const failOn = (values["fail-on"] ?? "error") as FailOn;
  if (failOn !== "error" && failOn !== "warn") throw new CliError("--fail-on must be warn or error");
  const deviceId = values.device as string | undefined;
  if (deviceId && !findDevice(deviceId)) throw new CliError(`Unknown device "${deviceId}". Run \`modelvisio targets\`.`);
  const opts = {
    targets: parseTargets(values.target ?? values.targets),
    deviceId: deviceId?.toLowerCase(),
    maxLatencyMs: num(values["max-latency-ms"], "--max-latency-ms"),
    maxSizeMb: num(values["max-size-mb"], "--max-size-mb"),
    failOn,
  };
  const report = checkReport(positionals.map((f) => checkModel(loadModel(f, io.cwd), opts)), failOn);
  const text = fmt === "json" ? JSON.stringify(report, null, 2) + "\n"
    : fmt === "markdown" ? checkMarkdown(report)
    : fmt === "junit" ? checkJunit(report)
    : checkText(report);
  emit(io, text, values.output, { file: values.summary, markdown: () => checkMarkdown(report) });
  return report.passed ? 0 : 1;
}

async function cmdDiff(args: string[], io: Io): Promise<number> {
  const { values, positionals } = parse(args, {
    "fail-on-regression": { type: "boolean" },
    "latency-tolerance": { type: "string" },
    "size-tolerance": { type: "string" },
    target: { type: "string", short: "t" },
    targets: { type: "string" },
    "old-name": { type: "string" },
    "new-name": { type: "string" },
  });
  if (values.help) { io.stdout(HELP); return 0; }
  if (positionals.length !== 2) throw new CliError("diff: expected <old> <new>");
  const fmt = pickFormat(values.format, values.json, ["text", "json", "markdown"]);
  const targets = parseTargets(values.target ?? values.targets);
  const [a, b] = positionals.map((f) => loadModel(f, io.cwd));
  const [sa, sb] = await Promise.all([a, b].map((m) => snapshotModel(m.model, { fileName: m.fileName, bytes: m.bytes, targets })));
  const d = diffSnapshots(sa, sb, {
    latencyTolerancePct: num(values["latency-tolerance"], "--latency-tolerance"),
    sizeTolerancePct: num(values["size-tolerance"], "--size-tolerance"),
  });
  const oldName = (values["old-name"] as string | undefined) ?? positionals[0];
  const newName = (values["new-name"] as string | undefined) ?? positionals[1];
  const md = () => diffMarkdown(d, oldName, newName);
  const text = fmt === "json" ? JSON.stringify(d, null, 2) + "\n" : fmt === "markdown" ? md() : diffText(d, oldName, newName);
  emit(io, text, values.output, { file: values.summary, markdown: md });
  return values["fail-on-regression"] && d.regressions.length > 0 ? 1 : 0;
}

function readRegistry(path: string): RegistryFile {
  if (!existsSync(path)) return { schema: REGISTRY_SCHEMA, exportedAt: "", models: [] };
  try {
    return parseRegistryFile(readFileSync(path, "utf8"));
  } catch (e) {
    throw new CliError(`${path}: ${(e as Error).message}`);
  }
}

/** Stable timestamp for the file: the newest change, so rewriting without changes is a no-op diff. */
const stamp = (f: RegistryFile) => f.models.reduce((m, x) => (x.updatedAt > m ? x.updatedAt : m), "");

function nowIso(io: Io): string | undefined {
  const epoch = io.env?.SOURCE_DATE_EPOCH;
  return epoch && /^\d+$/.test(epoch) ? new Date(Number(epoch) * 1000).toISOString() : undefined;
}

function registryMarkdown(f: RegistryFile): string {
  const out = ["## ModelVisio registry", ""];
  if (!f.models.length) return out.concat("_Empty._", "").join("\n");
  for (const m of f.models) {
    out.push(`### ${escapeMd(m.name)}${m.tags.length ? ` <sub>${m.tags.map((t) => "#" + t).join(" ")}</sub>` : ""}`, "");
    if (m.description) out.push(escapeMd(m.description), "");
    out.push("| Version | Date | File | Size | Params | FLOPs | Compat errors | SHA-256 |", "|---|---|---|---:|---:|---:|---|---|");
    for (const v of [...m.versions].reverse()) {
      const errs = v.compat.filter((c) => c.errors).map((c) => `${c.label} (${c.errors})`).join(", ") || "none";
      out.push(`| ${escapeMd(v.version)} | ${v.createdAt.slice(0, 10)} | \`${escapeMd(v.fileName)}\` | ${bytesStr(v.sizeBytes)} | ${human(v.stats.params)} | ${human(v.stats.flops)} | ${escapeMd(errs)} | \`${v.sha256.slice(0, 12)}\` |`);
    }
    out.push("");
  }
  return out.join("\n");
}

async function cmdRegistry(args: string[], io: Io): Promise<number> {
  const [sub, ...rest] = args;
  const { values, positionals } = parse(rest, {
    registry: { type: "string", short: "r" },
    name: { type: "string" },
    "model-id": { type: "string" },
    label: { type: "string" },
    tags: { type: "string" },
    notes: { type: "string" },
    "allow-duplicate": { type: "boolean" },
  });
  if (values.help || !sub) { io.stdout(HELP); return sub ? 0 : 2; }
  const path = resolve(io.cwd, (values.registry as string | undefined) ?? "modelvisio-registry.json");
  const file = readRegistry(path);

  if (sub === "add") {
    if (!positionals.length) throw new CliError("registry add: missing <file>");
    if (positionals.length > 1 && (values.label || values.name)) throw new CliError("--label/--name apply to a single file");
    const store = await storeFromFile(file);
    for (const f of positionals) {
      const m = loadModel(f, io.cwd);
      const res = await registerModel(store, m.model, {
        bytes: m.bytes, fileName: m.fileName,
        name: values.name as string | undefined, modelId: values["model-id"] as string | undefined,
        label: values.label as string | undefined, notes: values.notes as string | undefined,
        tags: values.tags ? String(values.tags).split(",") : undefined,
        allowDuplicate: !!values["allow-duplicate"], now: nowIso(io),
      }).catch((e: Error) => { throw new CliError(e.message); });
      io.stderr(res.duplicate
        ? `${m.fileName}: already registered as ${res.entry.name} ${res.version.version} (same SHA-256)\n`
        : `${m.fileName}: registered ${res.entry.name} ${res.version.version}\n`);
    }
    const out = await exportRegistry(store, "");
    out.exportedAt = stamp(out);
    writeFileSync(path, serializeRegistry(out));
    return 0;
  }
  if (sub === "export") {
    const fmt = pickFormat(values.format, values.json, ["json", "markdown"], "json");
    const store = await storeFromFile(file);
    const out = await exportRegistry(store, "");
    out.exportedAt = stamp(out);
    emit(io, fmt === "markdown" ? registryMarkdown(out) : serializeRegistry(out), values.output, { file: values.summary, markdown: () => registryMarkdown(out) });
    return 0;
  }
  if (sub === "list") {
    if (values.json) {
      io.stdout(JSON.stringify(file.models.map((m) => ({ id: m.id, name: m.name, tags: m.tags, versions: m.versions.map((v) => v.version) })), null, 2) + "\n");
      return 0;
    }
    if (!file.models.length) io.stdout(`No models in ${path}\n`);
    for (const m of file.models) {
      const latest = m.versions[m.versions.length - 1];
      io.stdout(`${m.name}  [${m.id}]  ${m.versions.length} version(s)${latest ? `, latest ${latest.version} (${latest.createdAt.slice(0, 10)})` : ""}${m.tags.length ? `  ${m.tags.map((t) => "#" + t).join(" ")}` : ""}\n`);
    }
    return 0;
  }
  throw new CliError(`registry: unknown subcommand "${sub}" (add, export, list)`);
}

function cmdTargets(io: Io): number {
  io.stdout("Compiler targets (op-level checks + reference device):\n");
  for (const t of COMPAT_TARGETS) io.stdout(`  ${t.id.padEnd(12)} ${t.label} (device: ${t.deviceId})\n`);
  io.stdout("Devices (memory-fit, format and latency checks):\n");
  for (const id of knownTargetIds().filter((x) => !COMPAT_TARGETS.some((t) => t.id === x))) io.stdout(`  ${id.padEnd(12)} ${findDevice(id)!.n}\n`);
  return 0;
}

export async function run(argv: string[], io: Io): Promise<number> {
  const [cmd, ...rest] = argv;
  try {
    switch (cmd) {
      case "inspect": return cmdInspect(rest, io);
      case "check": return cmdCheck(rest, io);
      case "diff": return await cmdDiff(rest, io);
      case "registry": return await cmdRegistry(rest, io);
      case "targets": return cmdTargets(io);
      case "--version": case "-v": io.stdout(VERSION + "\n"); return 0;
      case undefined: case "help": case "--help": case "-h": io.stdout(HELP); return cmd ? 0 : 2;
      default: throw new CliError(`Unknown command "${cmd}". Run \`modelvisio --help\`.`);
    }
  } catch (e) {
    if (e instanceof CliError) { io.stderr(`modelvisio: ${e.message}\n`); return e.exitCode; }
    io.stderr(`modelvisio: ${e instanceof Error ? e.stack ?? e.message : String(e)}\n`);
    return 2;
  }
}
