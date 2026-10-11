import { useEffect, useMemo, useRef, useState } from "react";
import type { Model } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { HW } from "../data/hardware";
import { countApplicable, fixById, type FixId } from "../fixes/transforms";
import { runSearch, type ConfigResult, type SearchResult } from "../optsearch/search";
import { buildRecipe } from "../optsearch/recipe";

const mono = "'JetBrains Mono',monospace";
const DEFAULT_DEVICES = ["orin_nano", "rk3588", "hailo8", "coral", "intel_npu"];
const msStr = (x: number) => (x >= 100 ? Math.round(x).toString() : x >= 1 ? x.toFixed(1) : x.toFixed(2));
const scaleLabel = (s: number) => (s === 1 ? "1×" : `${s}×`);

/**
 * Cross-compiler optimization search: enumerates device × runtime × precision ×
 * auto-fix set × input scale, estimates latency (roofline), size and accuracy
 * risk for each, and shows the Pareto frontier plus a ranked table. "Use this
 * config" shows the deploy recipe built from the Converter's script templates;
 * "Apply fixes" runs the config's auto-fixes on the analysis model one at a
 * time through `onApplyFix` (the same callback the Compiler tab uses).
 */
export function OptimizationSearch({
  model, onApplyFix, defaultDeviceIds = DEFAULT_DEVICES,
}: {
  model: Model;
  onApplyFix?: (id: FixId) => void;
  defaultDeviceIds?: string[];
}) {
  const t = useT();
  const [devs, setDevs] = useState<string[]>(() => defaultDeviceIds.filter((id) => HW.some((h) => h.id === id)));
  const [maxLat, setMaxLat] = useState("");
  const [maxSize, setMaxSize] = useState("");
  const [maxRisk, setMaxRisk] = useState("");
  const [tryScales, setTryScales] = useState(false);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [running, setRunning] = useState(false);
  const [selKey, setSelKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  // Fixes queued by "Apply fixes": App's onApplyFix closes over the current model,
  // so they are applied one per model update, never in the same tick.
  const pending = useRef<FixId[]>([]);
  const [appliedMsg, setAppliedMsg] = useState<string | null>(null);

  useEffect(() => {
    setResult(null); setSelKey(null);
    const next = pending.current.find((id) => countApplicable(model, id) > 0);
    pending.current = next ? pending.current.slice(pending.current.indexOf(next) + 1) : [];
    if (next && onApplyFix) onApplyFix(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  const num = (s: string) => { const v = parseFloat(s); return Number.isFinite(v) && v > 0 ? v : undefined; };
  const run = () => {
    setRunning(true); setAppliedMsg(null);
    setTimeout(() => {
      const r = runSearch(model, {
        devices: HW.filter((h) => devs.includes(h.id)),
        scales: tryScales ? [1, 0.75, 0.5] : [1],
        constraints: { maxLatencyMs: num(maxLat), maxSizeMB: num(maxSize), maxRisk: num(maxRisk) != null ? num(maxRisk)! / 100 : undefined },
      });
      setResult(r);
      setSelKey(r.ranked[0]?.key ?? null);
      setRunning(false);
    }, 0);
  };

  const sel = result?.all.find((c) => c.key === selKey) ?? null;
  const recipe = useMemo(() => (sel ? buildRecipe(model, sel) : null), [model, sel]);
  const toggle = (id: string) => setDevs((d) => (d.includes(id) ? d.filter((x) => x !== id) : [...d, id]));

  const applyFixes = (c: ConfigResult) => {
    if (!onApplyFix) return;
    const todo = c.fixes.filter((id) => countApplicable(model, id) > 0);
    if (todo.length === 0) return;
    pending.current = todo.slice(1);
    setAppliedMsg(`Applied ${todo.map((id) => fixById(id)?.title ?? id).join(", ")} to the analysis model. Re-run the search to see updated estimates.`);
    onApplyFix(todo[0]);
  };

  const inputStyle = { width: 64, padding: "3px 5px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 10 } as const;
  const chip = (on: boolean) => ({ padding: "2px 7px", borderRadius: 3, border: `1px solid ${on ? t.acc : t.bdr}`, background: on ? t.acc + "18" : "transparent", color: on ? t.acc : t.t2, fontSize: 9, cursor: "pointer" });

  return <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%" }}>
    <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Optimization Search</div>
    <div style={{ fontSize: 9, color: t.t2, lineHeight: 1.5, background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 5, padding: "5px 8px" }}>
      Searches device × runtime × precision × auto-fix set{tryScales ? " × input scale" : ""}. Latency is a roofline estimate, risk is an accuracy-risk proxy from quantization sensitivity: both are estimates, validate on-device.
    </div>

    <div style={{ display: "flex", gap: 3, flexWrap: "wrap", alignItems: "center" }}>
      <span style={{ fontSize: 9, color: t.t3 }}>Devices:</span>
      {HW.map((h) => <button key={h.id} type="button" onClick={() => toggle(h.id)} style={chip(devs.includes(h.id))} aria-pressed={devs.includes(h.id)}>{h.n}</button>)}
      <button type="button" onClick={() => setDevs(HW.map((h) => h.id))} style={chip(false)}>All</button>
      <button type="button" onClick={() => setDevs([])} style={chip(false)}>None</button>
    </div>

    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", fontSize: 9, color: t.t2 }}>
      <label>Max latency (ms) <input type="number" min={0} value={maxLat} onChange={(e) => setMaxLat(e.target.value)} placeholder="any" style={inputStyle} /></label>
      <label>Max size (MB) <input type="number" min={0} value={maxSize} onChange={(e) => setMaxSize(e.target.value)} placeholder="any" style={inputStyle} /></label>
      <label>Max risk (%) <input type="number" min={0} max={100} value={maxRisk} onChange={(e) => setMaxRisk(e.target.value)} placeholder="any" style={inputStyle} /></label>
      <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><input type="checkbox" checked={tryScales} onChange={(e) => setTryScales(e.target.checked)} /> Try 0.75× / 0.5× input</label>
      <button type="button" onClick={run} disabled={running || devs.length === 0}
        style={{ marginLeft: "auto", padding: "5px 14px", borderRadius: 5, border: "none", background: running || devs.length === 0 ? t.bg3 : t.acc, color: "#fff", fontSize: 10, fontWeight: 600, cursor: running || devs.length === 0 ? "default" : "pointer" }}>
        {running ? "Searching…" : "Run search"}</button>
    </div>

    {appliedMsg && <div style={{ fontSize: 9.5, color: t.acc3, background: t.acc3 + "12", border: `1px solid ${t.acc3}33`, borderRadius: 5, padding: "5px 8px" }}>{appliedMsg}</div>}

    <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
      {!result && !running && <div style={{ fontSize: 10, color: t.t3, padding: 12, textAlign: "center" }}>Pick devices and constraints, then run the search.</div>}
      {result && <>
        <div style={{ fontSize: 9, color: t.t3 }}>
          {result.evaluated} configs estimated from a space of {result.spaceSize} (device × runtime × precision × {result.fixesConsidered.length ? `${2 ** result.fixesConsidered.length} fix sets` : "no fixes"} × {result.scalesConsidered.length} scale{result.scalesConsidered.length > 1 ? "s" : ""}) ·
          pruned: {result.pruned["no-compile"]} won't compile, {result.pruned["no-memory"]} don't fit · {result.ranked.length} meet constraints · {result.pareto.length} on the Pareto front · {result.elapsedMs.toFixed(0)} ms
        </div>
        {result.ranked.length === 0
          ? <div style={{ fontSize: 10, color: t.wrn, padding: 8 }}>No configuration meets the constraints. Relax them or add devices.</div>
          : <>
            <Scatter result={result} selKey={selKey} onSelect={setSelKey} />
            <RankedTable rows={result.ranked.slice(0, 25)} selKey={selKey} onSelect={setSelKey} />
          </>}
        {sel && recipe && <div style={{ background: t.glass, border: `1px solid ${t.glassBdr}`, borderRadius: 6, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 5 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: t.t0 }}>Use this config: {sel.device.n} · {sel.runtime} · {sel.precision.toUpperCase()}{sel.scale !== 1 ? ` · ${scaleLabel(sel.scale)} input` : ""}</div>
          <div style={{ fontSize: 9.5, color: t.t2, fontFamily: mono }}>~{msStr(sel.latencyMs)} ms · {sel.sizeMB.toFixed(1)} MB · risk {(sel.risk * 100).toFixed(0)}% · {sel.errors} errors · {sel.warnings} warnings{sel.runtimes.length > 1 ? ` · also: ${sel.runtimes.filter((r) => r !== sel.runtime).join(", ")}` : ""}</div>
          {sel.notes.map((n, i) => <div key={i} style={{ fontSize: 9, color: t.t2 }}>• {n}</div>)}
          {sel.fixes.length > 0 && <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: 9.5, color: t.acc3 }}>Fixes: {sel.fixes.map((id) => fixById(id)?.title ?? id).join(", ")}</span>
            <button type="button" disabled={!onApplyFix} onClick={() => applyFixes(sel)}
              style={{ padding: "3px 10px", borderRadius: 4, border: `1px solid ${t.acc3}`, background: t.acc3 + "18", color: t.acc3, fontSize: 9.5, fontWeight: 600, cursor: onApplyFix ? "pointer" : "default", opacity: onApplyFix ? 1 : 0.5 }}>Apply fixes</button>
          </div>}
          <div style={{ position: "relative" }}>
            <button type="button" onClick={() => { navigator.clipboard?.writeText(recipe.text); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
              style={{ position: "absolute", top: 6, right: 6, padding: "3px 8px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t2, fontSize: 9, cursor: "pointer" }}>{copied ? "Copied" : "Copy"}</button>
            <pre style={{ background: t.bg, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: 10, fontSize: 9.5, color: t.suc, fontFamily: mono, lineHeight: 1.5, whiteSpace: "pre-wrap", margin: 0 }}>{recipe.text}</pre>
          </div>
        </div>}
      </>}
    </div>
  </div>;
}

/** Latency (log x) vs weight size (y) scatter; Pareto points highlighted. Inline SVG. */
function Scatter({ result, selKey, onSelect }: { result: SearchResult; selKey: string | null; onSelect: (k: string) => void }) {
  const t = useT();
  const paretoKeys = new Set(result.pareto.map((p) => p.key));
  // Plot the frontier plus the best-ranked rest (keeps the SVG light on big spaces).
  const pts = [...result.pareto, ...result.ranked.filter((c) => !paretoKeys.has(c.key)).slice(0, 400)];
  const W = 360, H = 170, L = 38, R = 8, T = 8, B = 26;
  const lat = pts.map((p) => Math.log10(Math.max(p.latencyMs, 1e-3)));
  const sz = pts.map((p) => p.sizeMB);
  const x0 = Math.floor(Math.min(...lat)), x1 = Math.max(x0 + 1, Math.ceil(Math.max(...lat)));
  const y1 = Math.max(...sz) * 1.08 || 1;
  const X = (ms: number) => L + ((Math.log10(Math.max(ms, 1e-3)) - x0) / (x1 - x0)) * (W - L - R);
  const Y = (mb: number) => T + (1 - mb / y1) * (H - T - B);
  const xt: number[] = [];
  for (let e = x0; e <= x1; e++) xt.push(10 ** e);
  const yt = [0, y1 / 2, y1].map((v) => +v.toPrecision(2));
  const front = [...result.pareto].sort((a, b) => a.latencyMs - b.latencyMs);
  return <div style={{ background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "6px 8px" }}>
    <div style={{ fontSize: 10.5, fontWeight: 600, color: t.t0 }}>Latency vs size</div>
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Pareto scatter of latency versus model size" style={{ display: "block" }}>
      {xt.map((v) => <g key={v}><line x1={X(v)} x2={X(v)} y1={T} y2={H - B} stroke={t.bdr} strokeWidth={0.5} /><text x={X(v)} y={H - B + 10} fontSize={8} textAnchor="middle" fill={t.t3}>{v >= 1 ? v : v.toString()} ms</text></g>)}
      {yt.map((v) => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke={t.bdr} strokeWidth={0.5} /><text x={L - 4} y={Y(v) + 3} fontSize={8} textAnchor="end" fill={t.t3}>{v}</text></g>)}
      <text x={(L + W - R) / 2} y={H - 3} fontSize={8} textAnchor="middle" fill={t.t2}>estimated latency (log scale)</text>
      <text x={9} y={(T + H - B) / 2} fontSize={8} textAnchor="middle" fill={t.t2} transform={`rotate(-90 9 ${(T + H - B) / 2})`}>weights MB</text>
      {front.length > 1 && <polyline points={front.map((p) => `${X(p.latencyMs)},${Y(p.sizeMB)}`).join(" ")} fill="none" stroke={t.acc} strokeWidth={1} strokeDasharray="3 2" opacity={0.6} />}
      {pts.filter((p) => !p.pareto).map((p) => <circle key={p.key} cx={X(p.latencyMs)} cy={Y(p.sizeMB)} r={3} fill={t.t3} opacity={0.35} onClick={() => onSelect(p.key)} style={{ cursor: "pointer" }}><title>{tip(p)}</title></circle>)}
      {result.pareto.map((p) => <circle key={p.key} cx={X(p.latencyMs)} cy={Y(p.sizeMB)} r={p.key === selKey ? 6 : 4.5} fill={t.acc} stroke={p.key === selKey ? t.t0 : t.bg2} strokeWidth={2} onClick={() => onSelect(p.key)} style={{ cursor: "pointer" }}><title>{tip(p)}</title></circle>)}
      {selKey && !paretoKeys.has(selKey) && (() => { const p = pts.find((c) => c.key === selKey); return p ? <circle cx={X(p.latencyMs)} cy={Y(p.sizeMB)} r={6} fill="none" stroke={t.t0} strokeWidth={2} /> : null; })()}
    </svg>
    <div style={{ display: "flex", gap: 10, fontSize: 8.5, color: t.t2 }}>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: 4, background: t.acc, display: "inline-block" }} />Pareto-optimal (latency, size, risk)</span>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: 4, background: t.t3, opacity: 0.5, display: "inline-block" }} />dominated</span>
    </div>
  </div>;
}

function tip(c: ConfigResult): string {
  return `${c.device.n} · ${c.runtime} · ${c.precision.toUpperCase()}${c.fixes.length ? ` · ${c.fixes.join("+")}` : ""}${c.scale !== 1 ? ` · ${c.scale}× input` : ""}\n`
    + `${msStr(c.latencyMs)} ms · ${c.sizeMB.toFixed(1)} MB · risk ${(c.risk * 100).toFixed(0)}%`;
}

function RankedTable({ rows, selKey, onSelect }: { rows: ConfigResult[]; selKey: string | null; onSelect: (k: string) => void }) {
  const t = useT();
  const th = { textAlign: "left" as const, padding: "3px 4px", color: t.t3, fontWeight: 600, borderBottom: `1px solid ${t.bdr}`, whiteSpace: "nowrap" as const };
  const td = { padding: "3px 4px", borderBottom: `1px solid ${t.bdr}`, whiteSpace: "nowrap" as const };
  return <div style={{ overflowX: "auto", border: `1px solid ${t.bdr}`, borderRadius: 6 }}>
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9, color: t.t1 }}>
      <thead><tr>{["#", "Device", "Runtime", "Prec", "Fixes", "Input", "Latency", "Size", "Risk", ""].map((h) => <th key={h} style={th}>{h}</th>)}</tr></thead>
      <tbody>{rows.map((c) => <tr key={c.key} onClick={() => onSelect(c.key)} style={{ cursor: "pointer", background: c.key === selKey ? t.acc + "18" : "transparent" }}>
        <td style={{ ...td, fontFamily: mono }}>{c.rank}</td>
        <td style={td}>{c.device.n}</td>
        <td style={td}>{c.runtime}</td>
        <td style={{ ...td, fontFamily: mono }}>{c.precision.toUpperCase()}{c.peakEstimated ? "*" : ""}</td>
        <td style={td} title={c.fixes.map((id) => fixById(id)?.title ?? id).join(", ")}>{c.fixes.length || "-"}</td>
        <td style={{ ...td, fontFamily: mono }}>{scaleLabel(c.scale)}</td>
        <td style={{ ...td, fontFamily: mono }}>{msStr(c.latencyMs)} ms</td>
        <td style={{ ...td, fontFamily: mono }}>{c.sizeMB.toFixed(1)} MB</td>
        <td style={{ ...td, fontFamily: mono }}>{(c.risk * 100).toFixed(0)}%</td>
        <td style={{ ...td, color: c.errors ? t.wrn : t.acc }} title={c.pareto ? "Pareto-optimal" : c.errors ? "CPU fallback" : ""}>{c.pareto ? "Pareto" : c.errors ? "fallback" : ""}</td>
      </tr>)}</tbody>
    </table>
    <div style={{ fontSize: 8, color: t.t3, padding: "3px 4px" }}>* precision peak derived (not in the datasheet).</div>
  </div>;
}
