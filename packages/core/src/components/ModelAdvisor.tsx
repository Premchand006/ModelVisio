import { useMemo, useState, type ReactElement } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { fmt, fB } from "../utils/format";
import { downloadFile } from "../utils/download";
import { HW } from "../data/hardware";
import { scoreAll } from "../scoring";
import { advise, type Advice, type AccuracyRisk, type Confidence } from "../advisor/advise";

/**
 * Hardware-aware model advisor. For one target device it ranks concrete
 * changes (quantization plan, op replacements, resolution, fusion, pruning)
 * with estimated latency / size / accuracy-risk deltas, and shows which layers
 * are compute- vs memory-bound on that device. All numbers are analytical
 * roofline + weight-statistics estimates, not on-device measurements.
 */
export function ModelAdvisor({
  model, onSelectLayer, initialDeviceId,
}: {
  model: Model;
  onSelectLayer?: (l: ModelLayer) => void;
  /** Device id from the hardware DB; defaults to the best-scoring device. */
  initialDeviceId?: string;
}) {
  const t = useT();
  const defaultId = useMemo(() => {
    if (initialDeviceId && HW.some((d) => d.id === initialDeviceId)) return initialDeviceId;
    return scoreAll(model, HW)[0]?.device.id ?? HW[0].id;
    // Only pick a default once per model.
  }, [model, initialDeviceId]);
  const [picked, setPicked] = useState<string | null>(null);
  const devId = picked && HW.some((d) => d.id === picked) ? picked : defaultId;
  const dev = HW.find((d) => d.id === devId) ?? HW[0];
  const [showAllLayers, setShowAllLayers] = useState(false);
  const [open, setOpen] = useState<string | null>(null);

  const r = useMemo(() => advise(model, dev), [model, dev]);
  const layerById = useMemo(() => new Map(model.layers.map((l) => [l.id, l])), [model]);
  const byName = useMemo(() => new Map(model.layers.map((l) => [l.name, l])), [model]);

  const mono = "'JetBrains Mono',monospace";
  const card = { background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "6px 8px" };
  const riskColor = (k: AccuracyRisk) => (k === "high" ? t.err : k === "medium" ? t.wrn : k === "low" ? t.t1 : t.suc);
  const confColor = (c: Confidence) => (c === "high" ? t.suc : c === "medium" ? t.wrn : t.t2);
  const ms = (v: number) => (!isFinite(v) ? "–" : v >= 100 ? Math.round(v) + " ms" : v.toFixed(2) + " ms");
  const fps = (v: number) => (!isFinite(v) || v <= 0 ? "0" : v >= 1000 ? fmt(Math.round(v)) : v >= 10 ? v.toFixed(0) : v.toFixed(1));

  const rows = useMemo(() => [...r.layers].filter((l) => l.bound !== "idle").sort((a, b) => b.share - a.share), [r]);
  const shownRows = showAllLayers ? rows : rows.slice(0, 15);
  const memCount = rows.filter((l) => l.bound === "memory").length;

  const perfs = [r.baseline, r.plan, r.fullInt8].filter((p): p is NonNullable<typeof p> => !!p);
  const base = (model.name || "model").replace(/\.[^.]+$/, "");
  const exportJson = () => {
    const { quant: _q, score: _s, device, ...rest } = r;
    downloadFile(JSON.stringify({ model: model.name, device: { id: device.id, name: device.n }, ...rest }, (_k, v) => (v === Infinity ? null : v), 2), `${base}.advisor.${device.id}.json`, "application/json");
  };

  const chip = (label: string, color: string) => (
    <span style={{ fontSize: 8.5, padding: "1px 5px", borderRadius: 3, border: `1px solid ${color}55`, color, fontFamily: mono, whiteSpace: "nowrap" }}>{label}</span>
  );
  const impactChips = (a: Advice) => {
    const out: ReactElement[] = [];
    const i = a.impact;
    if (i.speedup != null && isFinite(i.speedup)) out.push(<span key="s">{chip(`${i.speedup.toFixed(2)}× latency`, i.speedup >= 1.05 ? t.suc : i.speedup < 0.97 ? t.err : t.t2)}</span>);
    else if (i.latencyPct != null) out.push(<span key="l">{chip(`${i.latencyPct > 0 ? "+" : ""}${i.latencyPct}% latency`, i.latencyPct < 0 ? t.suc : t.err)}</span>);
    if (i.sizePct != null) out.push(<span key="z">{chip(`${i.sizePct > 0 ? "+" : ""}${i.sizePct}% size`, i.sizePct < 0 ? t.suc : t.wrn)}</span>);
    if (i.flopsPct != null) out.push(<span key="f">{chip(`${i.flopsPct > 0 ? "+" : ""}${i.flopsPct}% FLOPs`, i.flopsPct < 0 ? t.suc : t.wrn)}</span>);
    out.push(<span key="r">{chip(`accuracy risk: ${i.accuracyRisk}`, riskColor(i.accuracyRisk))}</span>);
    out.push(<span key="c">{chip(`confidence: ${a.confidence}`, confColor(a.confidence))}</span>);
    return out;
  };

  return <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Model advisor</div>
      <button type="button" onClick={exportJson}
        style={{ padding: "3px 10px", borderRadius: 4, border: `1px solid ${t.acc}`, background: t.acc + "18", color: t.acc, fontSize: 10, fontWeight: 600, cursor: "pointer" }}>Export JSON</button>
    </div>

    <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: t.t2 }}>
      Target
      <select value={dev.id} onChange={(e) => setPicked(e.target.value)} aria-label="Target device"
        style={{ flex: 1, minWidth: 0, padding: "4px 6px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 11 }}>
        {HW.map((d) => <option key={d.id} value={d.id}>{d.n}</option>)}
      </select>
    </label>

    {/* Performance summary */}
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${perfs.length}, minmax(0, 1fr))`, gap: 5 }}>
      {perfs.map((p) => (
        <div key={p.label} style={{ ...card, minWidth: 0, borderColor: p === r.plan ? t.acc + "88" : t.bdr }}>
          <div style={{ fontSize: 8.5, color: p === r.plan ? t.acc : t.t3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.label}</div>
          <div style={{ fontSize: 14, fontWeight: 800, color: t.t0, fontFamily: mono }}>~{fps(p.fps)}<span style={{ fontSize: 9, color: t.t2, fontWeight: 500 }}> FPS</span></div>
          <div style={{ fontSize: 8.5, color: t.t2, fontFamily: mono }}>{ms(p.latencyMs)} · {fB(Math.round(p.sizeBytes))}</div>
          <div style={{ fontSize: 8.5, color: p.memBoundShare > 0.5 ? t.wrn : t.t3 }}>{Math.round(p.memBoundShare * 100)}% memory-bound</div>
        </div>
      ))}
    </div>
    <div style={{ fontSize: 9, color: t.t2, lineHeight: 1.5, ...card }}>
      Roofline estimate · {r.workload === "transformer" ? "transformer" : r.workload === "cnnDet" ? "CNN detector/segmenter" : "CNN classifier"} · ridge point <b style={{ color: t.t1 }}>{isFinite(r.ridgeAI) ? r.ridgeAI.toFixed(0) : "∞"} FLOP/B</b> at {r.deployPrecision.toUpperCase()} · {dev.bandwidthGBs} GB/s{dev.bwConf === "?" ? " (estimated)" : ""} · fit score <b style={{ color: t.t1 }}>{r.score.overall}</b>/100.
    </div>

    {/* Recommendations */}
    <div style={{ fontSize: 10, fontWeight: 700, color: t.t1 }}>Recommendations ({r.advice.length})</div>
    {r.advice.length === 0 && <div style={{ ...card, fontSize: 10, color: t.t2 }}>No changes suggested for this device.</div>}
    {r.advice.map((a, idx) => {
      const isOpen = open === a.id;
      const names = a.layers ?? [];
      return (
        <div key={a.id} style={{ ...card, background: t.bg1, borderLeft: `3px solid ${a.required ? t.err : idx < 3 ? t.acc : t.bdr2}`, cursor: "pointer" }} onClick={() => setOpen(isOpen ? null : a.id)}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span style={{ fontSize: 9, color: t.t3, fontFamily: mono }}>{idx + 1}</span>
            <span style={{ fontSize: 11, fontWeight: 600, color: t.t0, flex: 1, minWidth: 0 }}>{a.title}</span>
            {a.required && chip("required", t.err)}
          </div>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 4 }}>{impactChips(a)}</div>
          {isOpen && (
            <div style={{ marginTop: 5, paddingTop: 5, borderTop: `1px solid ${t.bdr}` }} onClick={(e) => e.stopPropagation()}>
              <div style={{ fontSize: 9.5, color: t.t2, lineHeight: 1.5 }}>{a.rationale}</div>
              {names.length > 0 && (
                <div style={{ display: "flex", gap: 3, flexWrap: "wrap", marginTop: 5 }}>
                  {names.slice(0, 24).map((n) => {
                    const l = byName.get(n);
                    return (
                      <button key={n} type="button" disabled={!l || !onSelectLayer} onClick={() => l && onSelectLayer?.(l)}
                        style={{ fontSize: 8.5, padding: "1px 5px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.acc, fontFamily: mono, cursor: l && onSelectLayer ? "pointer" : "default", maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{n}</button>
                    );
                  })}
                  {names.length > 24 && <span style={{ fontSize: 8.5, color: t.t3 }}>+{names.length - 24} more</span>}
                </div>
              )}
            </div>
          )}
        </div>
      );
    })}

    {/* Per-layer bottleneck */}
    <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginTop: 4 }}>
      <span style={{ fontSize: 10, fontWeight: 700, color: t.t1 }}>Per-layer bottleneck on {dev.n}</span>
      <span style={{ fontSize: 9, color: t.t3 }}>{memCount} memory-bound / {rows.length - memCount} compute-bound</span>
    </div>
    <div style={{ border: `1px solid ${t.bdr}`, borderRadius: 6, overflow: "auto", maxHeight: 360 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9.5 }}>
        <thead>
          <tr>
            {["Layer", "Bound", "FLOP/B", "Time", "Share"].map((h, i) => (
              <th key={h} style={{ textAlign: i === 0 || i === 1 ? "left" : "right", padding: "4px 5px", fontSize: 9, fontWeight: 600, color: t.t3, borderBottom: `1px solid ${t.bdr}`, position: "sticky", top: 0, background: t.bg1 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shownRows.map((l) => {
            const layer = layerById.get(l.id);
            return (
              <tr key={l.id} onClick={() => layer && onSelectLayer?.(layer)} style={{ cursor: onSelectLayer ? "pointer" : "default", borderBottom: `1px solid ${t.bdr}` }}>
                <td style={{ padding: "4px 5px", maxWidth: 140 }}>
                  <div style={{ color: t.t0, fontWeight: 600, fontFamily: mono, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{l.op} <span style={{ color: t.t3, fontWeight: 400 }}>{l.precision.toUpperCase()}</span></div>
                  <div style={{ color: t.t3, fontSize: 8.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{l.name}</div>
                </td>
                <td style={{ padding: "4px 5px", color: l.bound === "memory" ? t.wrn : t.acc3, whiteSpace: "nowrap" }}>{l.bound}</td>
                <td style={{ padding: "4px 5px", textAlign: "right", fontFamily: mono, color: t.t2 }}>{isFinite(l.intensity) ? l.intensity.toFixed(1) : "∞"}</td>
                <td style={{ padding: "4px 5px", textAlign: "right", fontFamily: mono, color: t.t2, whiteSpace: "nowrap" }}>{l.ms < 0.01 ? "<0.01" : l.ms.toFixed(2)} ms</td>
                <td style={{ padding: "4px 5px", textAlign: "right", width: 70 }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                    <span style={{ width: 34, height: 5, background: t.bg3, borderRadius: 3, overflow: "hidden", display: "inline-block" }}>
                      <span style={{ display: "block", width: `${Math.min(100, l.share * 100)}%`, height: "100%", background: l.bound === "memory" ? t.wrn : t.acc3 }} />
                    </span>
                    <span style={{ fontFamily: mono, color: t.t2 }}>{Math.round(l.share * 100)}%</span>
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
    {rows.length > 15 && (
      <button type="button" onClick={() => setShowAllLayers((v) => !v)}
        style={{ alignSelf: "center", padding: "3px 9px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 10, cursor: "pointer" }}>
        {showAllLayers ? "Show top 15" : `Show all ${rows.length}`}
      </button>
    )}

    {r.notes.length > 0 && (
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {r.notes.map((n, i) => <div key={i} style={{ fontSize: 8.5, color: t.t3, lineHeight: 1.45 }}>• {n}</div>)}
      </div>
    )}
  </div>;
}
