import { useMemo, useState } from "react";
import type { Model } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { HW } from "../data/hardware";
import { downloadFile } from "../utils/download";
import {
  simulateFleet, fleetResultToCsv, fleetResultToJson,
  type FleetEntry, type HistBin, type PowerMode, type QueueModel, type ThermalProfile,
} from "../fleet/simulate";
import { FLEET_PRESETS, newEntryId, presetEntries } from "../fleet/presets";
import { RUNTIMES, precisionsFor, runtimesFor, type DeployPrecision, type RuntimeId } from "../deploy/runtimes";

const base = (n: string) => n.replace(/\.[^.]+$/, "") || "model";
const ms = (n: number) => (!isFinite(n) ? "∞" : n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2));
const fpsTxt = (n: number) => (!isFinite(n) ? "∞" : n >= 1000 ? `${(n / 1000).toFixed(1)}K` : n.toFixed(n >= 10 ? 0 : 1));

/**
 * Fleet simulation: roofline latency per device class + single-server queueing
 * (M/D/1 or M/M/1) → p95, SLO attainment, devices that can't run the model,
 * energy, and a fleet-wide p95 (request-weighted mixture). Optional seeded
 * Monte-Carlo (thermal throttling, background load, unit variance) gives a
 * distribution of the fleet p95. Pure math lives in fleet/simulate.ts.
 */
export function FleetSim({ model }: { model: Model | null }) {
  const t = useT();
  const [presetId, setPresetId] = useState<string>("retail");
  const [entries, setEntries] = useState<FleetEntry[]>(() => presetEntries(FLEET_PRESETS.find((p) => p.id === "retail")!));
  const [queue, setQueue] = useState<QueueModel>("md1");
  const [mc, setMc] = useState(true);
  const [trials, setTrials] = useState(200);
  const [seed, setSeed] = useState(42);
  const [hover, setHover] = useState<number | null>(null);

  const result = useMemo(
    () => (model ? simulateFleet(model, entries, { queue, monteCarlo: mc ? { trials, seed } : null }) : null),
    [model, entries, queue, mc, trials, seed],
  );

  if (!model) return <div style={{ padding: 20, color: t.t3, fontSize: 11, textAlign: "center" }}>Load a model to simulate a fleet.</div>;

  const preset = FLEET_PRESETS.find((p) => p.id === presetId);
  const loadPreset = (id: string) => {
    const p = FLEET_PRESETS.find((x) => x.id === id);
    if (p) { setPresetId(id); setEntries(presetEntries(p)); }
  };
  const update = (id: string, patch: Partial<FleetEntry>) => {
    setPresetId("");
    setEntries((es) => es.map((e) => {
      if (e.id !== id) return e;
      const next = { ...e, ...patch };
      if (patch.deviceId) {
        const d = HW.find((x) => x.id === patch.deviceId)!;
        if (!runtimesFor(d).includes(next.runtime)) next.runtime = runtimesFor(d)[0];
        if (!precisionsFor(d).includes(next.precision)) next.precision = precisionsFor(d)[0];
        next.label = undefined;
      }
      return next;
    }));
  };
  const remove = (id: string) => { setPresetId(""); setEntries((es) => es.filter((e) => e.id !== id)); };
  const add = () => {
    setPresetId("");
    setEntries((es) => [...es, { id: newEntryId(), deviceId: "orin_nano", count: 10, runtime: "tensorrt", precision: "int8", thermal: "passive", targetFps: 30 }]);
  };

  const a = result!.aggregate;
  const mcr = result!.monteCarlo;
  const numInput = (v: number | undefined, on: (n: number | undefined) => void, w = 46, ph = "") => (
    <input type="number" value={v ?? ""} placeholder={ph} min={0} onChange={(e) => on(e.target.value === "" ? undefined : Math.max(0, +e.target.value))}
      style={{ width: w, padding: "2px 4px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 9.5, fontFamily: "'JetBrains Mono',monospace" }} />
  );
  const sel = <V extends string>(v: V, opts: { v: V; l: string }[], on: (v: V) => void, w?: number) => (
    <select value={v} onChange={(e) => on(e.target.value as V)}
      style={{ width: w, maxWidth: 170, padding: "2px 3px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 9.5 }}>
      {opts.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}
    </select>
  );
  const btn = (label: string, on: () => void, active = false, color = t.acc) => (
    <button type="button" onClick={on} style={{ padding: "3px 8px", borderRadius: 4, border: `1px solid ${active ? color : t.bdr}`, background: active ? color + "18" : "transparent", color: active ? color : t.t2, fontSize: 9.5, cursor: "pointer", whiteSpace: "nowrap" }}>{label}</button>
  );
  const th = { textAlign: "left" as const, padding: "3px 4px", color: t.t3, fontWeight: 600, fontSize: 8.5, textTransform: "uppercase" as const, letterSpacing: 0.4, whiteSpace: "nowrap" as const };
  const td = { padding: "3px 4px", color: t.t1, fontSize: 9.5, whiteSpace: "nowrap" as const, fontFamily: "'JetBrains Mono',monospace" };

  const card = (label: string, value: string, sub: string, color = t.t0) => (
    <div style={{ flex: "1 1 120px", minWidth: 110, background: t.glass, border: `1px solid ${t.glassBdr}`, borderRadius: 6, padding: "6px 8px" }}>
      <div style={{ fontSize: 8.5, color: t.t3, textTransform: "uppercase", letterSpacing: 0.6, fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 800, color, fontFamily: "'JetBrains Mono',monospace", marginTop: 1 }}>{value}</div>
      <div style={{ fontSize: 8.5, color: t.t2, marginTop: 1, lineHeight: 1.35 }}>{sub}</div>
    </div>
  );
  const pctColor = a.pctMeetingSlo >= 95 ? t.suc : a.pctMeetingSlo >= 70 ? t.wrn : t.err;

  return <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%", overflow: "auto" }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Fleet Simulation</div>
      <div style={{ display: "flex", gap: 4 }}>
        {btn("⬇ JSON", () => downloadFile(fleetResultToJson(result!, model.name), `${base(model.name)}.fleet.json`, "application/json"))}
        {btn("⬇ CSV", () => downloadFile(fleetResultToCsv(result!), `${base(model.name)}.fleet.csv`, "text/csv"))}
      </div>
    </div>

    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", alignItems: "center" }}>
      <span style={{ fontSize: 9, color: t.t3 }}>Presets:</span>
      {FLEET_PRESETS.map((p) => <span key={p.id}>{btn(p.label, () => loadPreset(p.id), presetId === p.id)}</span>)}
    </div>
    {preset && <div style={{ fontSize: 9, color: t.t2, lineHeight: 1.45 }}>{preset.note}</div>}

    {/* Editable fleet table */}
    <div style={{ overflowX: "auto", border: `1px solid ${t.bdr}`, borderRadius: 6 }}>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead><tr style={{ borderBottom: `1px solid ${t.bdr}` }}>
          {["Device class", "Count", "Runtime", "Prec.", "Thermal", "Power", "Target FPS", "SLO ms", "Req/s", ""].map((h) => <th key={h} style={th}>{h}</th>)}
        </tr></thead>
        <tbody>{entries.map((e) => {
          const d = HW.find((x) => x.id === e.deviceId);
          return <tr key={e.id} style={{ borderBottom: `1px solid ${t.bdr}` }}>
            <td style={{ padding: "3px 4px" }} title={e.label}>{sel(e.deviceId, HW.map((h) => ({ v: h.id, l: h.n })), (v) => update(e.id, { deviceId: v }))}</td>
            <td style={{ padding: "3px 4px" }}>{numInput(e.count, (n) => update(e.id, { count: Math.floor(n ?? 0) }), 54)}</td>
            <td style={{ padding: "3px 4px" }}>{sel<RuntimeId>(e.runtime, (d ? runtimesFor(d) : [e.runtime]).map((r) => ({ v: r, l: RUNTIMES[r].label })), (v) => update(e.id, { runtime: v }))}</td>
            <td style={{ padding: "3px 4px" }}>{sel<DeployPrecision>(e.precision, (d ? precisionsFor(d) : [e.precision]).map((p) => ({ v: p, l: p.toUpperCase() })), (v) => update(e.id, { precision: v }))}</td>
            <td style={{ padding: "3px 4px" }}>{sel<ThermalProfile>(e.thermal ?? "cooled", [{ v: "cooled", l: "Cooled" }, { v: "passive", l: "Passive" }, { v: "hot", l: "Hot / fanless" }], (v) => update(e.id, { thermal: v }))}</td>
            <td style={{ padding: "3px 4px" }}>{sel<PowerMode>(e.powerMode ?? "max", [{ v: "max", l: "Max" }, { v: "balanced", l: "Balanced" }, { v: "low", l: "Low" }], (v) => update(e.id, { powerMode: v }))}</td>
            <td style={{ padding: "3px 4px" }}>{numInput(e.targetFps, (n) => update(e.id, { targetFps: n || undefined }), 46, "-")}</td>
            <td style={{ padding: "3px 4px" }}>{numInput(e.sloMs, (n) => update(e.id, { sloMs: n || undefined }), 46, "auto")}</td>
            <td style={{ padding: "3px 4px" }}>{numInput(e.requestRate, (n) => update(e.id, { requestRate: n }), 46, "auto")}</td>
            <td style={{ padding: "3px 4px" }}><button type="button" onClick={() => remove(e.id)} aria-label="Remove device class" title="Remove" style={{ border: "none", background: "transparent", color: t.t3, cursor: "pointer", fontSize: 12 }}>×</button></td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", fontSize: 9.5, color: t.t2 }}>
      {btn("+ Add device class", add)}
      <span>Queue:</span>{btn("M/D/1", () => setQueue("md1"), queue === "md1")}{btn("M/M/1", () => setQueue("mm1"), queue === "mm1")}
      <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}><input type="checkbox" checked={mc} onChange={(e) => setMc(e.target.checked)} style={{ accentColor: t.acc }} />Monte-Carlo</label>
      {mc && <><span>trials</span>{numInput(trials, (n) => setTrials(Math.min(2000, Math.max(10, Math.floor(n ?? 200)))), 52)}<span>seed</span>{numInput(seed, (n) => setSeed(Math.floor(n ?? 0)), 52)}</>}
    </div>

    {/* Summary cards */}
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {card("Meeting SLO", `${a.pctMeetingSlo.toFixed(a.pctMeetingSlo % 1 ? 1 : 0)}%`, `${a.devicesMeetingSlo.toLocaleString()} of ${a.totalDevices.toLocaleString()} devices${mcr ? ` · MC p5–p95 ${mcr.pctMeetingSlo.p5.toFixed(0)}–${mcr.pctMeetingSlo.p95.toFixed(0)}%` : ""}`, pctColor)}
      {card("Fleet p95", `${ms(a.fleetP95Ms)} ms`, mcr && isFinite(mcr.p95Quantiles.p50) ? `MC median ${ms(mcr.p95Quantiles.p50)} ms (p5 ${ms(mcr.p95Quantiles.p5)}, p95 ${ms(mcr.p95Quantiles.p95)})` : `device-weighted ${ms(a.weightedP95Ms)} ms`)}
      {card("Can't run", a.cantRunDevices.toLocaleString(), a.cantRunDevices ? result!.classes.filter((c) => !c.canRun).map((c) => c.name).join(", ") : "every class can run the model", a.cantRunDevices ? t.err : t.suc)}
      {card("Capacity / demand", `${fpsTxt(a.capacityFps)} / ${fpsTxt(a.demandFps)}`, "inferences/s across runnable devices")}
      {card("Energy", isFinite(a.meanEnergyMj) ? `${a.meanEnergyMj.toFixed(1)} mJ` : "–", `per inference · fleet ${a.fleetPowerW >= 1000 ? (a.fleetPowerW / 1000).toFixed(1) + " kW" : a.fleetPowerW.toFixed(0) + " W"} typical`)}
    </div>
    {a.bottleneck && <div style={{ fontSize: 9.5, color: t.t1, background: t.wrn + "12", border: `1px solid ${t.wrn}33`, borderRadius: 5, padding: "5px 8px", lineHeight: 1.45 }}>
      <b style={{ color: t.wrn }}>Bottleneck:</b> {a.bottleneck.name}: {a.bottleneck.reason}
    </div>}

    {/* Per-class results */}
    <div style={{ overflowX: "auto", border: `1px solid ${t.bdr}`, borderRadius: 6 }}>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead><tr style={{ borderBottom: `1px solid ${t.bdr}` }}>
          {["Class", "Devices", "Latency", "Capacity", "Util.", "p95", "SLO", "Attain.", "mJ/inf", "Status"].map((h) => <th key={h} style={th}>{h}</th>)}
        </tr></thead>
        <tbody>{result!.classes.map((c) => <tr key={c.entry.id} style={{ borderBottom: `1px solid ${t.bdr}` }}>
          <td style={{ ...td, fontFamily: "inherit", color: t.t0, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis" }} title={c.name}>{c.name}<span style={{ color: t.t3 }}> · {c.entry.precision.toUpperCase()}{c.estimate?.approx ? "*" : ""}</span></td>
          <td style={td}>{c.entry.count.toLocaleString()}</td>
          <td style={td}>{ms(c.latencyMs)} ms</td>
          <td style={td}>{fpsTxt(c.capacityFps)} FPS</td>
          <td style={td}>{isFinite(c.utilization) ? `${(c.utilization * 100).toFixed(0)}%` : "–"}</td>
          <td style={td}>{ms(c.p95Ms)} ms</td>
          <td style={td}>{ms(c.sloMs)} ms</td>
          <td style={td}>{(c.sloAttainment * 100).toFixed(c.sloAttainment > 0.999 && c.sloAttainment < 1 ? 2 : 0)}%</td>
          <td style={td}>{isFinite(c.energyMj) ? c.energyMj.toFixed(1) : "–"}</td>
          <td style={{ ...td, fontFamily: "inherit", color: c.meetsSlo ? t.suc : c.canRun ? t.wrn : t.err }} title={c.reasons.join("\n")}>
            {c.meetsSlo ? "✓ meets SLO" : c.canRun ? "⚠ misses SLO" : "⛔ can't run"}{c.reasons.length > 0 && <span style={{ color: t.t3 }}> · {c.reasons[0]}</span>}
          </td>
        </tr>)}</tbody>
      </table>
    </div>

    {mcr && <Histogram bins={mcr.histogram} nominal={a.fleetP95Ms} hover={hover} setHover={setHover} trials={mcr.trials} unstable={mcr.unstableTrials} />}

    <div style={{ fontSize: 8.5, color: t.t3, lineHeight: 1.5 }}>
      Latency: per-device roofline estimate at the chosen precision/runtime (* = derived peak compute), slowed by thermal (passive ×1.1, hot ×1.3) and power mode. Queue: {queue === "md1" ? "M/D/1" : "M/M/1"} per device at the request rate (default = target FPS); p95 = S + k·S·ln(ρ/0.05)/(1−ρ). SLO defaults to the frame time (1000/target FPS). Fleet p95 mixes all runnable devices' response distributions by request volume. Monte-Carlo draws per-device throttling, background-load spikes and unit variance (seeded, reproducible).
    </div>
  </div>;
}

/** Monte-Carlo distribution of the fleet p95 (single series: one hue, no legend). */
function Histogram({ bins, nominal, hover, setHover, trials, unstable }: {
  bins: HistBin[]; nominal: number; hover: number | null; setHover: (i: number | null) => void; trials: number; unstable: number;
}) {
  const t = useT();
  if (bins.length === 0) return <div style={{ fontSize: 9.5, color: t.t3 }}>Monte-Carlo: every trial was unstable (overloaded fleet), so no finite p95 to plot.</div>;
  const W = 520, H = 130, padL = 8, padR = 8, padT = 14, padB = 22;
  const iw = W - padL - padR, ih = H - padT - padB;
  const max = Math.max(...bins.map((b) => b.count), 1);
  const lo = bins[0].lo, hi = bins[bins.length - 1].hi;
  const span = hi - lo || 1;
  const bw = iw / bins.length;
  const x = (v: number) => padL + ((v - lo) / span) * iw;
  const nomIn = isFinite(nominal) && nominal >= lo && nominal <= hi;
  const hb = hover != null ? bins[hover] : null;
  return <div style={{ border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "6px 8px", background: t.bg }}>
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9.5, color: t.t1, fontWeight: 600 }}>
      <span>Fleet p95 across {trials} Monte-Carlo trials</span>
      <span style={{ color: t.t2, fontWeight: 400, fontFamily: "'JetBrains Mono',monospace" }}>
        {hb ? `${ms(hb.lo)}–${ms(hb.hi)} ms: ${hb.count} trial${hb.count === 1 ? "" : "s"}` : unstable ? `${unstable} unstable trial(s) not shown` : "hover a bar"}
      </span>
    </div>
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Histogram of fleet p95 latency from ${ms(lo)} to ${ms(hi)} ms over ${trials} trials`} style={{ display: "block" }} onMouseLeave={() => setHover(null)}>
      <line x1={padL} x2={W - padR} y1={padT + ih} y2={padT + ih} stroke={t.bdr} strokeWidth={1} />
      {bins.map((b, i) => {
        const h = b.count === 0 ? 0 : Math.max(2, (b.count / max) * ih);
        const x0 = padL + i * bw + 1, w = Math.max(1, bw - 2), y0 = padT + ih - h;
        const r = Math.min(4, w / 2, h);
        const d = h <= 0 ? "" : `M${x0},${padT + ih} V${y0 + r} Q${x0},${y0} ${x0 + r},${y0} H${x0 + w - r} Q${x0 + w},${y0} ${x0 + w},${y0 + r} V${padT + ih} Z`;
        return <g key={i} onMouseEnter={() => setHover(i)}>
          <rect x={padL + i * bw} y={padT} width={bw} height={ih} fill="transparent" />
          {d && <path d={d} fill={t.acc} opacity={hover == null || hover === i ? 0.9 : 0.45} />}
          <title>{`${ms(b.lo)}–${ms(b.hi)} ms: ${b.count} trial(s)`}</title>
        </g>;
      })}
      {nomIn && <g>
        <line x1={x(nominal)} x2={x(nominal)} y1={padT - 4} y2={padT + ih} stroke={t.t2} strokeWidth={1} strokeDasharray="3 3" />
        <text x={x(nominal) + 3} y={padT + 2} fontSize={8.5} fill={t.t2}>nominal {ms(nominal)} ms</text>
      </g>}
      <text x={padL} y={H - 6} fontSize={8.5} fill={t.t3}>{ms(lo)} ms</text>
      <text x={W - padR} y={H - 6} fontSize={8.5} fill={t.t3} textAnchor="end">{ms(hi)} ms</text>
      <text x={W / 2} y={H - 6} fontSize={8.5} fill={t.t3} textAnchor="middle">fleet p95 latency</text>
    </svg>
  </div>;
}
