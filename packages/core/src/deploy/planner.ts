// Deterministic (rule-based) deployment planner: the fallback when there is no
// AI key, the proxy errors, or the AI misbehaves. Calls the SAME tools as the
// AI (through ctx.call, so every step is in the transcript) in a fixed order:
//   summary → devices → estimates → compat → fixes → precision/prune retry →
//   recipe → single-device SLO check → final decision.

import type { AgentFinal, PlannerFn } from "@modelvisio/ai";
import { parseGoal, type DeployGoal } from "./goal";
import { precisionsFor, RUNTIMES, type DeployPrecision, type RuntimeId } from "./runtimes";
import type {
  CompatResult, DeploySession, DeviceRow, EstimateResult, FixRow, FleetSummaryResult,
} from "./tools";

function choosePrecision(g: DeployGoal, precs: DeployPrecision[]): DeployPrecision | null {
  const want: DeployPrecision = g.precision ?? (g.int8Ok === false ? "fp16" : "int8");
  if (precs.includes(want)) return want;
  if (g.int8Ok === false || g.precision) return null; // constraint can't be met here
  return precs[0];
}

function chooseRuntime(g: DeployGoal, rts: RuntimeId[]): RuntimeId {
  return g.runtime && rts.includes(g.runtime) ? g.runtime : rts[0];
}

const meets = (g: DeployGoal, e: EstimateResult) =>
  e.canRun && (!g.minFps || (e.fps ?? 0) >= g.minFps) && (!g.maxLatencyMs || (e.latencyMs ?? Infinity) <= g.maxLatencyMs);

function rank(g: DeployGoal, rows: Map<string, DeviceRow>, a: EstimateResult, b: EstimateResult): number {
  const ma = meets(g, a) ? 1 : 0, mb = meets(g, b) ? 1 : 0;
  if (ma !== mb) return mb - ma;
  const ra = a.canRun ? 1 : 0, rb = b.canRun ? 1 : 0;
  if (ra !== rb) return rb - ra;
  if (!ma) return (b.fps ?? 0) - (a.fps ?? 0);                // nobody meets: closest wins
  const pa = rows.get(a.device), pb = rows.get(b.device);
  if (g.optimize === "cost") return (pa?.priceUSD ?? 1e9) - (pb?.priceUSD ?? 1e9);
  if (g.optimize === "power") return (b.fpsPerW ?? 0) - (a.fpsPerW ?? 0);
  if (g.optimize === "speed") return (b.fps ?? 0) - (a.fps ?? 0);
  return (pb?.score ?? 0) - (pa?.score ?? 0);
}

export function createDeployPlanner(session: DeploySession): PlannerFn {
  return async ({ goal, call }) => {
    const g = parseGoal(goal, session.devices);
    const risks: string[] = [];

    await call("get_model_summary", {}, "Read the model's size, compute and compat issues.");

    const named = g.deviceIds.length ? g.deviceIds : g.platformDeviceIds;
    const listArgs = named.length ? { ids: named, limit: named.length } : { limit: 10, ...(g.runtime ? { runtime: g.runtime } : {}) };
    let rows = (await call("list_devices", listArgs, named.length
      ? `Goal names ${g.deviceIds.length ? "device(s)" : g.platform}: ${named.join(", ")}.`
      : "No device named; rank the whole hardware DB for this model.")) as DeviceRow[];
    if (rows.length === 0) rows = (await call("list_devices", { limit: 10 }, "No device matched the runtime filter; rank all devices.")) as DeviceRow[];

    // Apply hard budget constraints from the goal before estimating.
    let pool = rows.filter((r) => (!g.maxPowerW || r.powerW <= g.maxPowerW) && (!g.maxPriceUSD || (r.priceUSD ?? Infinity) <= g.maxPriceUSD));
    if (pool.length === 0) { pool = rows; risks.push("No device satisfies the power/price budget in the goal; budget constraint relaxed."); }
    const candidates = pool
      .map((r) => ({ r, prec: choosePrecision(g, r.precisions) }))
      .filter((c) => c.prec != null)
      .slice(0, named.length === 1 ? 1 : 3);
    if (candidates.length === 0) {
      const r = pool[0];
      candidates.push({ r, prec: r.precisions[0] });
      risks.push(`${r.name} cannot run the requested precision; using ${r.precisions[0].toUpperCase()}.`);
    }

    const byId = new Map(rows.map((r) => [r.id, r]));
    const ests: EstimateResult[] = [];
    for (const c of candidates) {
      const rt = chooseRuntime(g, c.r.runtimes);
      ests.push((await call("estimate_latency", { device: c.r.id, precision: c.prec, runtime: rt, ...(g.minFps ? { minFps: g.minFps } : {}) },
        `Estimate ${c.r.name} at ${c.prec!.toUpperCase()} on ${RUNTIMES[rt].label}.`)) as EstimateResult);
    }
    let best = [...ests].sort((a, b) => rank(g, byId, a, b))[0];
    const device = session.devices.find((d) => d.id === best.device)!;
    let precision = best.precision;
    const runtime = best.runtime;

    const compat = (await call("check_compat", { device: device.id, runtime }, `Check op compatibility of the chosen target ${device.n}.`)) as CompatResult;

    if (compat.errors + compat.warns > 0) {
      const fixes = (await call("list_fixes", { device: device.id, runtime }, "Compat issues found; look for auto-fixes that resolve them.")) as FixRow[];
      for (const f of fixes) {
        if (f.id === "channel_prune" || f.applied) continue;
        if (f.issuesBefore != null && f.issuesAfter != null && f.issuesAfter < f.issuesBefore) {
          await call("apply_fix", { id: f.id }, `${f.title} removes ${f.issuesBefore - f.issuesAfter} compat issue(s) on ${device.n}.`);
        }
      }
    }

    // Re-estimate after fixes; if the FPS/latency target is still missed, try
    // INT8 (when allowed), then structured pruning.
    if (session.applied.length > 0 || !meets(g, best)) {
      best = (await call("estimate_latency", { device: device.id, precision, runtime, ...(g.minFps ? { minFps: g.minFps } : {}) },
        session.applied.length ? "Re-estimate on the fixed working model." : "Target missed; re-check before escalating.")) as EstimateResult;
    }
    if (!meets(g, best) && precision !== "int8" && g.int8Ok !== false && !g.precision && precisionsFor(device).includes("int8")) {
      const e8 = (await call("estimate_latency", { device: device.id, precision: "int8", runtime, ...(g.minFps ? { minFps: g.minFps } : {}) },
        "Target missed; try INT8.")) as EstimateResult;
      if ((e8.fps ?? 0) > (best.fps ?? 0)) { best = e8; precision = "int8"; }
    }
    if (!meets(g, best) && best.canRun && session.working.layers.some((l) => l.params > 200_000) && !session.applied.includes("channel_prune")) {
      await call("apply_fix", { id: "channel_prune" }, "Still short of the target; apply structured channel pruning (needs a fine-tune).");
      best = (await call("estimate_latency", { device: device.id, precision, runtime, ...(g.minFps ? { minFps: g.minFps } : {}) },
        "Re-estimate after pruning.")) as EstimateResult;
    }

    await call("generate_recipe", { device: device.id, precision, runtime, ...(g.minFps ? { targetFps: g.minFps } : {}) },
      `Generate the ${RUNTIMES[runtime].label} ${precision.toUpperCase()} package.`);

    const fleet = (await call("fleet_simulate", {
      device: device.id, count: 1, precision, runtime,
      ...(g.minFps ? { targetFps: g.minFps } : {}), ...(g.maxLatencyMs ? { sloMs: g.maxLatencyMs } : {}),
    }, "Single-device SLO check (passive cooling, M/D/1 queue).")) as FleetSummaryResult;

    // ── Decision + justification ───────────────────────────────────────────
    const fpsTxt = best.fps != null ? `~${best.fps} FPS (${best.latencyMs} ms)` : "an unknown rate";
    const target = g.minFps ? ` against a ${g.minFps} FPS target` : g.maxLatencyMs ? ` against a ${g.maxLatencyMs} ms budget` : "";
    const why = [
      `${device.n} on ${RUNTIMES[runtime].label} at ${precision.toUpperCase()} is estimated at ${fpsTxt}${target}, ${best.bound}-bound, with ${best.headroom != null ? `${best.headroom}x memory headroom` : "unknown memory headroom"}.`,
      ests.length > 1 ? `It ranked first of ${ests.length} candidates (${ests.map((e) => `${e.device} ~${e.fps} FPS`).join(", ")})${g.optimize ? ` optimizing for ${g.optimize}` : ""}.` : g.deviceIds.length ? "It is the device named in the goal." : "",
      session.applied.length ? `Applied fixes: ${session.applied.join(", ")}.` : "",
      fleet.pctMeetingSlo != null ? `Single-device SLO check: p95 ${fleet.fleetP95Ms ?? "∞"} ms, ${fleet.pctMeetingSlo >= 100 ? "meets" : "misses"} the SLO.` : "",
    ].filter(Boolean).join(" ");

    if (!meets(g, best)) risks.push(g.minFps || g.maxLatencyMs ? `Estimated performance misses the goal (${fpsTxt}${target}).` : "Model cannot run on the selected device as estimated.");
    if (compat.errors > 0) risks.push(`${compat.errors} op(s) have compile errors on ${device.n} before fixes.`);
    if (!g.deviceIds.length && !g.platformDeviceIds.length && !g.minFps) risks.push("Goal stated no device or performance target; chose the best overall roofline fit.");

    const final: AgentFinal = {
      target: device.id, runtime, precision, justification: why, risks,
      fixes: [...session.applied],
      summary: `${device.n} · ${RUNTIMES[runtime].label} · ${precision.toUpperCase()} · ${fpsTxt}`,
    };
    return final;
  };
}
