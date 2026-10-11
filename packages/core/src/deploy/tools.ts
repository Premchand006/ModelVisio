// Tool implementations for the deployment agent (packages/ai/src/deployAgent.ts).
// Each tool is a pure function over a DeploySession: a WORKING COPY of the model
// (fixes the agent applies land here, never on the app's model) plus the latest
// recipe / fleet result, which the package builder reads afterwards. Results are
// compact JSON so they fit in the model's context.

import type { Model } from "@modelvisio/parsers";
import type { AgentArgs, AgentTool } from "@modelvisio/ai";
import { HW, type DeviceSpec } from "../data/hardware";
import { FIXES, applicableLayers, applyFix, type FixId } from "../fixes/transforms";
import { classifyWorkload, modelCompute, scoreDevice } from "../scoring";
import { canRun, estimateDevice, type DeviceEstimate } from "../fleet/estimate";
import { simulateFleet, type FleetEntry, type FleetResult } from "../fleet/simulate";
import { FLEET_PRESETS, presetEntries } from "../fleet/presets";
import { generateRecipe, type RecipeBundle } from "./recipes";
import { resolveDevice } from "./goal";
import {
  RUNTIMES, parsePrecision, parseRuntime, precisionsFor, runtimesFor,
  type DeployPrecision, type RuntimeId,
} from "./runtimes";

export type DeploySession = {
  /** The model as loaded in the app (never modified). */
  base: Model;
  /** Working copy the agent's apply_fix calls modify. */
  working: Model;
  applied: FixId[];
  recipe: { device: DeviceSpec; bundle: RecipeBundle } | null;
  fleet: FleetResult | null;
  devices: DeviceSpec[];
};

export function createDeploySession(model: Model, devices: DeviceSpec[] = HW): DeploySession {
  return { base: model, working: model, applied: [], recipe: null, fleet: null, devices };
}

const r1 = (n: number) => (isFinite(n) ? Math.round(n * 10) / 10 : null);
const r2 = (n: number) => (isFinite(n) ? Math.round(n * 100) / 100 : null);

// ─── Compact result shapes (also consumed by the deterministic planner) ──────

export type ModelSummaryResult = {
  name: string; format: string; framework: string; workload: string;
  paramsM: number; gflops: number; sizeMB: number; layers: number;
  input: number[]; output: number[];
  quantSensitiveLayers: { name: string; qSens: number }[];
  compatIssueCounts: Record<string, { errors: number; warns: number }>;
  appliedFixes: FixId[];
};

export type DeviceRow = {
  id: string; name: string; vendor: string; score: number;
  runtimes: RuntimeId[]; precisions: DeployPrecision[];
  bestFps: number | null; bestPrecision: DeployPrecision;
  fits: boolean; compileFail: boolean; powerW: number; priceUSD: number | null;
  int8Only: boolean; ramGB: number;
};

export type EstimateResult = {
  device: string; name: string; precision: DeployPrecision; runtime: RuntimeId;
  supported: boolean; reason?: string; canRun: boolean;
  fps: number | null; latencyMs: number | null; bound: string;
  fits: boolean; footprintMB: number | null; budgetMB: number | null; headroom: number | null;
  compileFail: boolean; compatErrors: number; compatWarns: number;
  powerW: number; energyMj: number | null; fpsPerW: number | null;
  approx: boolean; confidence: string; meetsFps?: boolean;
};

export type CompatResult = {
  device: string; name: string; runtime: RuntimeId;
  errors: number; warns: number;
  issues: { layer: string; severity: string; msg: string }[];
  fits: boolean; compileFail: boolean; conversionNeeded: boolean; fallbackOps: number;
  banners: string[];
};

export type FixRow = {
  id: FixId; title: string; change: string; impact: string; layers: number;
  /** Compat issues on `device` before → after this fix (when a device was given). */
  issuesBefore?: number; issuesAfter?: number;
  applied: boolean;
};

export type RecipeResult = { device: string; runtime: RuntimeId; precision: DeployPrecision; artifact: string; files: string[]; steps: string[] };

export type FleetSummaryResult = {
  totalDevices: number; pctMeetingSlo: number | null; fleetP95Ms: number | null;
  cantRunDevices: number; capacityFps: number | null; demandFps: number | null;
  bottleneck: string | null;
  classes: { name: string; count: number; p95Ms: number | null; sloMs: number | null; meetsSlo: boolean; canRun: boolean; reasons: string[] }[];
};

// ─── Tool logic (pure, session-bound) ────────────────────────────────────────

export function modelSummary(s: DeploySession): ModelSummaryResult {
  const m = s.working;
  const mc = modelCompute(m);
  const counts: ModelSummaryResult["compatIssueCounts"] = {};
  for (const l of m.layers) for (const c of l.compIssues ?? []) {
    const e = (counts[c.target] ??= { errors: 0, warns: 0 });
    if (c.severity === "error") e.errors++; else e.warns++;
  }
  return {
    name: m.name, format: m.format, framework: m.framework, workload: classifyWorkload(m),
    paramsM: r2(mc.totalParams / 1e6) ?? 0, gflops: r2(mc.totalOps / 1e9) ?? 0, sizeMB: r2(m.sizeBytes / 1e6) ?? 0,
    layers: m.layers.length, input: m.inputShape, output: m.outputShape,
    quantSensitiveLayers: m.layers.filter((l) => l.qSens > 0.1).sort((a, b) => b.qSens - a.qSens).slice(0, 6).map((l) => ({ name: l.name, qSens: r2(l.qSens) ?? 0 })),
    compatIssueCounts: counts,
    appliedFixes: [...s.applied],
  };
}

export function listDevices(s: DeploySession, args: AgentArgs = {}): DeviceRow[] {
  let devs = s.devices;
  const ids = Array.isArray(args.ids) ? args.ids.map(String) : null;
  if (ids?.length) devs = devs.filter((d) => ids.includes(d.id));
  if (typeof args.vendor === "string" && args.vendor) devs = devs.filter((d) => d.v.toLowerCase().includes(String(args.vendor).toLowerCase()));
  const rt = parseRuntime(args.runtime);
  if (rt) devs = devs.filter((d) => runtimesFor(d).includes(rt));
  const limit = typeof args.limit === "number" && args.limit > 0 ? Math.floor(args.limit) : 10;
  return devs
    .map((d) => ({ d, sc: scoreDevice(s.working, d) }))
    .sort((a, b) => b.sc.overall - a.sc.overall)
    .slice(0, limit)
    .map(({ d, sc }) => ({
      id: d.id, name: d.n, vendor: d.v, score: sc.overall,
      runtimes: runtimesFor(d), precisions: precisionsFor(d),
      bestFps: r1(sc.fps), bestPrecision: precisionsFor(d)[0],
      fits: sc.memory.fits, compileFail: sc.opSupport.compileFail,
      powerW: d.power.typ, priceUSD: d.priceUSD ?? null, int8Only: d.quant.int8Only, ramGB: d.ramGB,
    }));
}

function requireDevice(s: DeploySession, ref: unknown): DeviceSpec {
  const d = resolveDevice(ref, s.devices);
  if (!d) throw new Error(`Unknown device "${String(ref ?? "")}". Use an id from list_devices: ${s.devices.map((x) => x.id).join(", ")}`);
  return d;
}

const devArg = (a: AgentArgs) => a.device ?? a.target ?? a.deviceId ?? a.id;

export function toEstimateResult(e: DeviceEstimate, minFps?: number): EstimateResult {
  return {
    device: e.device.id, name: e.device.n, precision: e.precision, runtime: e.runtime,
    supported: e.supported, reason: e.reason, canRun: canRun(e),
    fps: r1(e.fps), latencyMs: r2(e.latencyMs), bound: e.bound,
    fits: e.fits, footprintMB: r1(e.footprintBytes / 1e6), budgetMB: r1(e.budgetBytes / 1e6), headroom: r2(e.headroom),
    compileFail: e.compileFail, compatErrors: e.errors, compatWarns: e.warns,
    powerW: e.powerW, energyMj: r2(e.energyMj), fpsPerW: r2(e.fpsPerW),
    approx: e.approx, confidence: e.confidence,
    ...(minFps ? { meetsFps: canRun(e) && e.fps >= minFps } : {}),
  };
}

export function estimateLatency(s: DeploySession, args: AgentArgs): EstimateResult {
  const d = requireDevice(s, devArg(args));
  const prec = parsePrecision(args.precision) ?? precisionsFor(d)[0];
  const rt = parseRuntime(args.runtime) ?? undefined;
  const minFps = typeof args.minFps === "number" ? args.minFps : undefined;
  return toEstimateResult(estimateDevice(s.working, d, prec, rt), minFps);
}

export function checkCompat(s: DeploySession, args: AgentArgs): CompatResult {
  const d = requireDevice(s, devArg(args));
  const rt = parseRuntime(args.runtime) ?? runtimesFor(d)[0];
  const e = estimateDevice(s.working, d, precisionsFor(d)[0], rt);
  return {
    device: d.id, name: d.n, runtime: e.runtime, errors: e.errors, warns: e.warns,
    issues: e.issues.slice(0, 12).map((i) => ({ layer: i.layer, severity: i.severity, msg: i.msg })),
    fits: e.fits, compileFail: e.compileFail, conversionNeeded: e.conversionNeeded, fallbackOps: e.fallbackOps,
    banners: e.banners.map((b) => `${b.level}: ${b.msg}`),
  };
}

export function listFixes(s: DeploySession, args: AgentArgs = {}): FixRow[] {
  const d = devArg(args) != null ? resolveDevice(devArg(args), s.devices) : null;
  const rt = d ? parseRuntime(args.runtime) ?? runtimesFor(d)[0] : undefined;
  const issuesOn = (m: Model) => (d ? estimateDevice(m, d, precisionsFor(d)[0], rt).issues.length : 0);
  const before = d ? issuesOn(s.working) : undefined;
  return FIXES.map((f) => ({ f, layers: applicableLayers(s.working, f.id).length }))
    .filter((x) => x.layers > 0)
    .map(({ f, layers }) => ({
      id: f.id, title: f.title, change: f.change, impact: f.impact, layers,
      ...(d ? { issuesBefore: before, issuesAfter: issuesOn(applyFix(s.working, f.id)) } : {}),
      applied: s.applied.includes(f.id),
    }));
}

export function applyFixTool(s: DeploySession, args: AgentArgs) {
  const id = String(args.id ?? args.fix ?? "") as FixId;
  const def = FIXES.find((f) => f.id === id);
  if (!def) throw new Error(`Unknown fix "${id}". Valid: ${FIXES.map((f) => f.id).join(", ")}`);
  const layers = applicableLayers(s.working, id);
  if (layers.length === 0) throw new Error(`Fix "${id}" does not apply to the current working model.`);
  const issues = (m: Model) => m.layers.reduce((n, l) => n + l.compIssues.length, 0);
  const before = { issues: issues(s.working), mc: modelCompute(s.working) };
  s.working = applyFix(s.working, id);
  if (!s.applied.includes(id)) s.applied.push(id);
  const after = { issues: issues(s.working), mc: modelCompute(s.working) };
  return {
    applied: id, title: def.title, layers: layers.length,
    compatIssues: { before: before.issues, after: after.issues },
    paramsM: { before: r2(before.mc.totalParams / 1e6), after: r2(after.mc.totalParams / 1e6) },
    gflops: { before: r2(before.mc.totalOps / 1e9), after: r2(after.mc.totalOps / 1e9) },
    note: "Applied to the agent's working copy only (analysis model); apply it to the graph from the package view.",
  };
}

export function generateRecipeTool(s: DeploySession, args: AgentArgs): RecipeResult {
  const d = requireDevice(s, devArg(args));
  const prec = parsePrecision(args.precision) ?? precisionsFor(d)[0];
  if (!precisionsFor(d).includes(prec)) throw new Error(`${d.n} supports ${precisionsFor(d).join("/")} only.`);
  const rt = parseRuntime(args.runtime) ?? undefined;
  const targetFps = typeof args.targetFps === "number" ? args.targetFps : undefined;
  const bundle = generateRecipe({ model: s.working, device: d, precision: prec, runtime: rt, targetFps });
  s.recipe = { device: d, bundle };
  return { device: d.id, runtime: bundle.runtime, precision: prec, artifact: bundle.artifact, files: bundle.files.map((f) => f.path), steps: bundle.steps };
}

export function summarizeFleet(r: FleetResult): FleetSummaryResult {
  const a = r.aggregate;
  return {
    totalDevices: a.totalDevices, pctMeetingSlo: r1(a.pctMeetingSlo), fleetP95Ms: r2(a.fleetP95Ms),
    cantRunDevices: a.cantRunDevices, capacityFps: r1(a.capacityFps), demandFps: r1(a.demandFps),
    bottleneck: a.bottleneck ? `${a.bottleneck.name}: ${a.bottleneck.reason}` : null,
    classes: r.classes.map((c) => ({ name: c.name, count: c.entry.count, p95Ms: r2(c.p95Ms), sloMs: r2(c.sloMs), meetsSlo: c.meetsSlo, canRun: c.canRun, reasons: c.reasons.slice(0, 3) })),
  };
}

export function fleetSimulateTool(s: DeploySession, args: AgentArgs): FleetSummaryResult {
  let entries: FleetEntry[] = [];
  if (typeof args.preset === "string") {
    const p = FLEET_PRESETS.find((x) => x.id === args.preset || x.label.toLowerCase() === String(args.preset).toLowerCase());
    if (!p) throw new Error(`Unknown preset "${args.preset}". Valid: ${FLEET_PRESETS.map((x) => x.id).join(", ")}`);
    entries = presetEntries(p);
  } else if (Array.isArray(args.entries)) {
    entries = args.entries.map((raw, i) => {
      const e = (raw ?? {}) as AgentArgs;
      const d = requireDevice(s, devArg(e));
      return {
        id: `e${i}`, deviceId: d.id, count: Math.max(1, Number(e.count) || 1),
        runtime: parseRuntime(e.runtime) ?? runtimesFor(d)[0], precision: parsePrecision(e.precision) ?? precisionsFor(d)[0],
        targetFps: typeof e.targetFps === "number" ? e.targetFps : undefined,
        sloMs: typeof e.sloMs === "number" ? e.sloMs : undefined,
        requestRate: typeof e.requestRate === "number" ? e.requestRate : undefined,
        thermal: e.thermal === "hot" || e.thermal === "passive" ? e.thermal : undefined,
      };
    });
  } else {
    const d = requireDevice(s, devArg(args));
    entries = [{
      id: "e0", deviceId: d.id, count: Math.max(1, Number(args.count) || 1),
      runtime: parseRuntime(args.runtime) ?? runtimesFor(d)[0], precision: parsePrecision(args.precision) ?? precisionsFor(d)[0],
      targetFps: typeof args.targetFps === "number" ? args.targetFps : undefined,
      sloMs: typeof args.sloMs === "number" ? args.sloMs : undefined,
      thermal: args.thermal === "hot" || args.thermal === "passive" ? args.thermal : "passive",
    }];
  }
  if (entries.length === 0) throw new Error("fleet_simulate needs a device, entries[] or preset.");
  s.fleet = simulateFleet(s.working, entries, { queue: "md1" }, s.devices);
  return summarizeFleet(s.fleet);
}

// ─── AgentTool wiring ────────────────────────────────────────────────────────

const yes = (b: boolean) => (b ? "yes" : "no");

/** Build the injected tool set for one agent run over `session`. */
export function createDeployTools(session: DeploySession): AgentTool[] {
  const devIds = session.devices.map((d) => d.id).join("|");
  const rts = Object.keys(RUNTIMES).join("|");
  return [
    {
      name: "get_model_summary", description: "Model stats: params, GFLOPs, workload class, I/O shapes, quantization-sensitive layers, compat issue counts per target.",
      run: () => modelSummary(session),
      digest: (r) => { const m = r as ModelSummaryResult; return `${m.name}: ${m.paramsM}M params, ${m.gflops} GFLOPs, ${m.workload}, ${m.quantSensitiveLayers.length} quant-sensitive layer(s)`; },
    },
    {
      name: "list_devices", args: `{"vendor"?: string, "runtime"?: string, "ids"?: string[], "limit"?: number}`,
      description: "Devices from the hardware DB ranked by roofline fit score for this model (id, runtimes, precisions, best FPS, fit, power, price).",
      run: (a) => listDevices(session, a),
      digest: (r) => { const rows = r as DeviceRow[]; return rows.slice(0, 4).map((d) => `${d.id} ${d.score}/100 ~${d.bestFps} FPS`).join(", ") + (rows.length > 4 ? ` (+${rows.length - 4})` : ""); },
    },
    {
      name: "check_compat", args: `{"device": "${devIds}", "runtime"?: "${rts}"}`,
      description: "Op-compatibility issues, memory fit and compile risk for this model on a device/runtime.",
      run: (a) => checkCompat(session, a),
      digest: (r) => { const c = r as CompatResult; return `${c.device}/${c.runtime}: ${c.errors} error(s), ${c.warns} warning(s), fits ${yes(c.fits)}, compiles ${yes(!c.compileFail)}`; },
    },
    {
      name: "estimate_latency", args: `{"device": string, "precision": "int8"|"fp16"|"fp32", "runtime"?: string, "minFps"?: number}`,
      description: "Roofline latency/FPS, memory fit, energy per inference for a device at a precision and runtime.",
      run: (a) => estimateLatency(session, a),
      digest: (r) => { const e = r as EstimateResult; return e.supported ? `${e.device} ${e.precision}/${e.runtime}: ~${e.fps} FPS, ${e.latencyMs} ms, fits ${yes(e.fits)}${e.approx ? " (approx peak)" : ""}` : `${e.device}: ${e.reason}`; },
    },
    {
      name: "list_fixes", args: `{"device"?: string}`,
      description: "Graph auto-fixes applicable to the working model; with a device, shows its compat issue count before/after each fix.",
      run: (a) => listFixes(session, a),
      digest: (r) => { const f = r as FixRow[]; return f.length ? f.map((x) => `${x.id}${x.issuesBefore != null ? ` (${x.issuesBefore}→${x.issuesAfter})` : ""}`).join(", ") : "no applicable fixes"; },
    },
    {
      name: "apply_fix", args: `{"id": "${FIXES.map((f) => f.id).join("|")}"}`,
      description: "Apply an auto-fix to the agent's working copy of the model (the app's model is unchanged).",
      run: (a) => applyFixTool(session, a),
      digest: (r) => { const x = r as ReturnType<typeof applyFixTool>; return `${x.applied}: compat issues ${x.compatIssues.before}→${x.compatIssues.after}, params ${x.paramsM.before}M→${x.paramsM.after}M`; },
    },
    {
      name: "generate_recipe", args: `{"device": string, "precision": "int8"|"fp16"|"fp32", "runtime"?: string, "targetFps"?: number}`,
      description: "Generate the conversion script, calibration, inference/latency harness and accuracy check for the chosen deployment.",
      run: (a) => generateRecipeTool(session, a),
      digest: (r) => { const x = r as RecipeResult; return `${x.device} ${x.runtime}/${x.precision} → ${x.artifact} (${x.files.length} files)`; },
    },
    {
      name: "fleet_simulate", args: `{"device": string, "count"?: number, "precision"?: string, "runtime"?: string, "targetFps"?: number, "sloMs"?: number} | {"entries": [...]} | {"preset": "${FLEET_PRESETS.map((p) => p.id).join("|")}"}`,
      description: "Queueing + roofline fleet simulation: p95 latency, SLO attainment, devices that can't run it, bottleneck.",
      run: (a) => fleetSimulateTool(session, a),
      digest: (r) => { const f = r as FleetSummaryResult; return `${f.totalDevices} device(s): ${f.pctMeetingSlo}% meet SLO, fleet p95 ${f.fleetP95Ms ?? "∞"} ms, ${f.cantRunDevices} can't run`; },
    },
  ];
}

/** Compact model context for the agent's system prompt (saves a tool call). */
export function agentContext(model: Model): string {
  const mc = modelCompute(model);
  return `${model.name} | ${model.format} | ${model.framework} | input ${model.inputShape.join("x")} | ${(mc.totalParams / 1e6).toFixed(2)}M params | ${(mc.totalOps / 1e9).toFixed(2)} GFLOPs | ${model.layers.length} layers | workload ${classifyWorkload(model)}`;
}
