// Assemble the reviewable deployment package from an agent run: validated
// decision (device / runtime / precision), justification, plan transcript,
// conversion + runtime files, validation checklist and risks; as individual
// files or one ZIP. Pure (writeZip is pure); downloading is the caller's job.

import { writeZip } from "@modelvisio/parsers";
import type { AgentRun, AgentStep } from "@modelvisio/ai";
import type { DeviceSpec } from "../data/hardware";
import { FIXES, type FixId } from "../fixes/transforms";
import { scoreAll } from "../scoring";
import { estimateDevice, type DeviceEstimate } from "../fleet/estimate";
import { simulateFleet, type FleetResult } from "../fleet/simulate";
import { generateRecipe, type PackageFile, type RecipeBundle } from "./recipes";
import { parseGoal, resolveDevice } from "./goal";
import { parsePrecision, parseRuntime, precisionsFor, runtimesFor, RUNTIMES, type DeployPrecision, type RuntimeId } from "./runtimes";
import type { DeploySession } from "./tools";

export type ChecklistItem = { kind: "accuracy" | "latency" | "memory" | "compat" | "soak" | "rollout"; title: string; detail: string; command?: string };

export type DeployDecision = {
  device: DeviceSpec;
  runtime: RuntimeId;
  runtimeLabel: string;
  precision: DeployPrecision;
  justification: string;
  summary: string;
  source: "ai" | "planner";
  /** Corrections made while validating the AI's choice. */
  adjustments: string[];
};

export type DeployPackage = {
  goal: string;
  modelName: string;
  decision: DeployDecision;
  estimate: DeviceEstimate;
  fleet: FleetResult | null;
  fixes: { id: FixId; title: string }[];
  steps: AgentStep[];
  mode: AgentRun["mode"];
  fallbackReason?: string;
  recipe: RecipeBundle;
  checklist: ChecklistItem[];
  risks: string[];
  files: PackageFile[];
};

const stem = (n: string) => n.replace(/\.[^.]+$/, "") || "model";
const f1 = (n: number) => (isFinite(n) ? n.toFixed(1) : "∞");
const f2 = (n: number) => (isFinite(n) ? n.toFixed(2) : "∞");

/** Validate the run's final decision against the device DB; fill gaps from the session. */
export function resolveDecision(run: AgentRun, session: DeploySession, goal: string): DeployDecision {
  const adj: string[] = [];
  const g = parseGoal(goal, session.devices);
  let device = resolveDevice(run.final.target, session.devices);
  if (!device) {
    device = session.recipe?.device
      ?? (g.deviceIds[0] ? session.devices.find((d) => d.id === g.deviceIds[0]) ?? null : null)
      ?? scoreAll(session.working, session.devices)[0].device;
    adj.push(run.final.target ? `Target "${String(run.final.target)}" isn't in the hardware DB; used ${device.n}.` : `No target given; used ${device.n}.`);
  }
  const precs = precisionsFor(device);
  let precision = parsePrecision(run.final.precision) ?? (session.recipe?.device.id === device.id ? session.recipe.bundle.precision : null) ?? precs[0];
  if (!precs.includes(precision)) {
    adj.push(`${device.n} is ${precs.join("/").toUpperCase()}-only; ${precision.toUpperCase()} replaced by ${precs[0].toUpperCase()}.`);
    precision = precs[0];
  }
  const rts = runtimesFor(device);
  let runtime = parseRuntime(run.final.runtime) ?? (session.recipe?.device.id === device.id ? session.recipe.bundle.runtime : null) ?? rts[0];
  if (!rts.includes(runtime)) {
    adj.push(`${RUNTIMES[runtime].label} doesn't target ${device.n}; used ${RUNTIMES[rts[0]].label}.`);
    runtime = rts[0];
  }
  const source = run.mode === "ai" ? "ai" : "planner";
  const just = typeof run.final.justification === "string" && run.final.justification.trim() ? run.final.justification.trim() : "";
  const summary = typeof run.final.summary === "string" && run.final.summary.trim()
    ? run.final.summary.trim()
    : `${device.n} · ${RUNTIMES[runtime].label} · ${precision.toUpperCase()}`;
  return { device, runtime, runtimeLabel: RUNTIMES[runtime].label, precision, justification: just, summary, source, adjustments: adj };
}

function checklist(d: DeployDecision, e: DeviceEstimate, r: RecipeBundle, minFps?: number, maxMs?: number): ChecklistItem[] {
  const items: ChecklistItem[] = [
    {
      kind: "accuracy", title: "Accuracy vs FP32 reference",
      detail: `Run the FP32 ONNX reference and the deployed ${d.runtimeLabel} ${d.precision.toUpperCase()} model on the same 200+ held-out images; mean cosine >= 0.99 per output, then confirm the task metric (mAP / top-1) drops by <= 1 point.`,
      command: "python validate/reference.py val/ --out ref_outputs/ && python runtime/infer.py --images val/ --dump dev_outputs/ && python validate/accuracy_check.py ref_outputs/ dev_outputs/",
    },
    {
      kind: "latency", title: `Latency on the real ${d.device.n}`,
      detail: `Batch-1 p95 latency on the device after warm-up${minFps ? `; p95-based FPS must be >= ${minFps}` : maxMs ? `; p95 must be <= ${maxMs} ms` : ""}. Estimate: ~${f1(e.fps)} FPS / ${f2(e.latencyMs)} ms (roofline${e.approx ? ", approximate peak" : ""}; ${e.confidence} confidence).`,
      command: r.files.some((f) => f.path === "runtime/infer.py") ? `${minFps ? `TARGET_FPS=${minFps} ` : ""}python runtime/infer.py --bench --iters 500` : r.benchmarkCmd,
    },
    { kind: "latency", title: "Vendor benchmark cross-check", detail: "Compare the harness against the vendor's own profiler (excludes Python overhead).", command: r.benchmarkCmd },
    {
      kind: "memory", title: "Memory headroom",
      detail: `Estimated footprint ${(e.footprintBytes / 1e6).toFixed(1)} MB vs ${e.budgetBytes > 0 ? (e.budgetBytes / 1e6).toFixed(0) + " MB budget" : "unknown budget"} (${e.budgetBytes > 0 ? f2(e.headroom) + "x" : "?"} headroom). Measure peak RSS / device memory while the app pipeline runs, not just the model.`,
    },
    {
      kind: "compat", title: "Compiler log: no silent CPU fallback",
      detail: `Inspect the ${d.runtimeLabel} conversion log: every op should map to the accelerator.${e.issues.length ? ` Known issues: ${e.issues.slice(0, 3).map((i) => `${i.layer} (${i.msg})`).join("; ")}.` : ""}`,
    },
    { kind: "soak", title: "30-minute thermal soak", detail: "Run the benchmark continuously for 30 minutes in the production enclosure; p95 must stay within 15% of the cold-start figure (throttling check)." },
    { kind: "rollout", title: "Staged rollout", detail: "Ship to ~5% of devices first; monitor p95 latency, error rate and accuracy proxies before the full fleet." },
  ];
  return items;
}

function risks(d: DeployDecision, e: DeviceEstimate, session: DeploySession, aiRisks: unknown, minFps?: number): string[] {
  const out: string[] = [];
  if (Array.isArray(aiRisks)) for (const r of aiRisks) if (typeof r === "string" && r.trim()) out.push(r.trim());
  if (minFps && e.fps < minFps) out.push(`Estimated ${f1(e.fps)} FPS is below the ${minFps} FPS target.`);
  else if (minFps && e.fps < minFps * 1.2) out.push(`Thin margin: estimated ${f1(e.fps)} FPS vs ${minFps} FPS target (< 20% headroom for throttling and pre/post-processing).`);
  if (!e.fits) out.push(`Model does not fit ${d.device.n} memory (${(e.footprintBytes / 1e6).toFixed(1)} MB footprint).`);
  else if (e.budgetBytes > 0 && e.headroom < 1.3) out.push(`Low memory headroom (${f2(e.headroom)}x).`);
  if (e.compileFail) out.push(`Unsupported ops with no CPU fallback: ${d.device.n} will not compile this graph.`);
  if (e.errors > 0) out.push(`${e.errors} op(s) still fall back to CPU on ${d.runtimeLabel}.`);
  if (d.precision === "int8") {
    const sens = session.working.layers.filter((l) => l.qSens > 0.1);
    if (sens.length) out.push(`INT8 accuracy: ${sens.length} quantization-sensitive layer(s) (${sens.slice(0, 3).map((l) => l.name).join(", ")}); use per-channel quantization and keep them FP16 if the accuracy gate fails.`);
    out.push("INT8 needs a representative calibration set; a poor set is the most common cause of accuracy loss.");
  }
  if (e.approx) out.push(`${d.precision.toUpperCase()} peak compute for ${d.device.n} is derived, not from the datasheet; the estimate is approximate.`);
  if (e.confidence === "low") out.push(`Low-confidence device data for ${d.device.n} (estimated bandwidth or unsourced specs).`);
  if (d.device.eol) out.push(`${d.device.n} is end-of-life.`);
  if (d.device.hostDependency) out.push(`${d.device.n} is a host-attached accelerator: host CPU pre/post-processing and the bus can bottleneck.`);
  if (session.applied.includes("channel_prune")) out.push("Channel pruning changes the network: it needs a fine-tune and re-export before deployment.");
  if (session.applied.length) out.push(`Fixes (${session.applied.join(", ")}) were applied to the analysis model only; reproduce them in the training/export pipeline before converting.`);
  if (d.runtime === "vendor" || d.runtime === "qnn") out.push(`${d.runtimeLabel} commands are an outline; verify them against the vendor SDK version you use.`);
  return [...new Set(out)].slice(0, 14);
}

function stepLine(s: AgentStep): string {
  return `${s.index}. [${s.source}] ${s.tool}${Object.keys(s.args).length ? ` ${JSON.stringify(s.args)}` : ""}${s.thought ? `: ${s.thought}` : ""}\n   → ${s.ok ? s.digest : `ERROR: ${s.error}`}`;
}

/** Build the package. `minFps` / `maxLatencyMs` default to the parsed goal. */
export function buildDeployPackage(args: { goal: string; run: AgentRun; session: DeploySession }): DeployPackage {
  const { goal, run, session } = args;
  const g = parseGoal(goal, session.devices);
  const d = resolveDecision(run, session, goal);
  const recipe = session.recipe && session.recipe.device.id === d.device.id && session.recipe.bundle.runtime === d.runtime && session.recipe.bundle.precision === d.precision
    ? session.recipe.bundle
    : generateRecipe({ model: session.working, device: d.device, precision: d.precision, runtime: d.runtime, targetFps: g.minFps });
  const estimate = estimateDevice(session.working, d.device, d.precision, d.runtime);
  const fleet = session.fleet ?? simulateFleet(session.working, [{
    id: "e0", deviceId: d.device.id, count: 1, runtime: d.runtime, precision: d.precision, thermal: "passive",
    targetFps: g.minFps, sloMs: g.maxLatencyMs,
  }]);
  if (!d.justification) {
    d.justification = `${d.device.n} on ${d.runtimeLabel} at ${d.precision.toUpperCase()}: estimated ~${f1(estimate.fps)} FPS (${f2(estimate.latencyMs)} ms), ${estimate.bound}-bound, fits ${estimate.fits ? "yes" : "no"}.`;
  }
  const fixes = session.applied.map((id) => ({ id, title: FIXES.find((f) => f.id === id)?.title ?? id }));
  const checks = checklist(d, estimate, recipe, g.minFps, g.maxLatencyMs);
  const riskList = risks(d, estimate, session, run.final.risks, g.minFps);

  const plan = [
    `# Deployment plan`,
    "",
    `Goal: ${goal}`,
    `Mode: ${run.mode === "ai" ? "AI agent" : "deterministic planner"}${run.fallbackReason ? ` (${run.fallbackReason})` : ""}`,
    "",
    "## Steps",
    ...run.steps.map(stepLine),
  ].join("\n") + "\n";

  const readme = [
    `# ${stem(session.base.name)} → ${d.device.n}`,
    "",
    `**Decision:** ${d.device.n} · ${d.runtimeLabel} · ${d.precision.toUpperCase()}  `,
    `**Estimate:** ~${f1(estimate.fps)} FPS · ${f2(estimate.latencyMs)} ms · ${f2(estimate.energyMj)} mJ/inference · ${estimate.bound}-bound (roofline, ${estimate.confidence} confidence)  `,
    `**Artifact:** \`${recipe.artifact}\``,
    "",
    "## Why",
    d.justification,
    ...(d.adjustments.length ? ["", "Adjustments made while validating the decision:", ...d.adjustments.map((a) => `- ${a}`)] : []),
    ...(fixes.length ? ["", "## Graph fixes", ...fixes.map((f) => `- ${f.title} (\`${f.id}\`): reproduce in the export pipeline before converting.`)] : []),
    "",
    "## Run it",
    "```bash",
    ...recipe.steps,
    "```",
    "",
    "## Files",
    ...recipe.files.map((f) => `- \`${f.path}\`: ${f.description}`),
    "- `PLAN.md`: the agent's step-by-step transcript",
    "- `VALIDATION.md`: accuracy and latency gates",
    "- `RISKS.md`: known risks",
    "- `deploy.json`: machine-readable decision + estimates",
    "",
    ...(recipe.notes.length ? ["## Notes", ...recipe.notes.map((n) => `- ${n}`), ""] : []),
    "Estimates come from ModelVisio's roofline model; validate on the real device (VALIDATION.md).",
    "",
  ].join("\n");

  const validation = ["# Validation checklist", "", ...checks.map((c, i) => `${i + 1}. [ ] **${c.title}** (${c.kind})\n   ${c.detail}${c.command ? `\n   \`\`\`bash\n   ${c.command}\n   \`\`\`` : ""}`)].join("\n") + "\n";
  const riskMd = ["# Risks", "", ...(riskList.length ? riskList.map((r) => `- ${r}`) : ["- None identified beyond the validation checklist."])].join("\n") + "\n";
  const json = JSON.stringify({
    goal, model: session.base.name, mode: run.mode, fallbackReason: run.fallbackReason ?? null,
    decision: { device: d.device.id, deviceName: d.device.n, runtime: d.runtime, precision: d.precision, justification: d.justification, adjustments: d.adjustments },
    estimate: {
      fps: isFinite(estimate.fps) ? +estimate.fps.toFixed(2) : null, latencyMs: isFinite(estimate.latencyMs) ? +estimate.latencyMs.toFixed(3) : null,
      energyMj: isFinite(estimate.energyMj) ? +estimate.energyMj.toFixed(3) : null, bound: estimate.bound, fits: estimate.fits,
      headroom: isFinite(estimate.headroom) ? +estimate.headroom.toFixed(2) : null, approx: estimate.approx, confidence: estimate.confidence,
    },
    fixes: session.applied, risks: riskList,
    steps: run.steps.map((s) => ({ index: s.index, source: s.source, tool: s.tool, args: s.args, ok: s.ok, digest: s.digest, error: s.error ?? null, thought: s.thought })),
  }, null, 2) + "\n";

  const files: PackageFile[] = [
    { path: "README.md", content: readme, description: "Decision, justification and how to run" },
    ...recipe.files,
    { path: "PLAN.md", content: plan, description: "Agent transcript" },
    { path: "VALIDATION.md", content: validation, description: "Validation checklist" },
    { path: "RISKS.md", content: riskMd, description: "Risks" },
    { path: "deploy.json", content: json, description: "Machine-readable decision" },
  ];

  return {
    goal, modelName: session.base.name, decision: d, estimate, fleet, fixes,
    steps: run.steps, mode: run.mode, fallbackReason: run.fallbackReason,
    recipe, checklist: checks, risks: riskList, files,
  };
}

/** ZIP the package under a single top-level folder. */
export function packageToZip(pkg: DeployPackage): { filename: string; data: Uint8Array } {
  const root = `${stem(pkg.modelName)}-${pkg.decision.device.id}-${pkg.decision.runtime}-${pkg.decision.precision}`;
  const enc = new TextEncoder();
  return {
    filename: `${root}.zip`,
    data: writeZip(pkg.files.map((f) => ({ name: `${root}/${f.path}`, data: enc.encode(f.content) }))),
  };
}
