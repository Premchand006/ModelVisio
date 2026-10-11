import { useMemo, useState } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { fmt, fB } from "../utils/format";
import { downloadFile } from "../utils/download";
import { analyzeQuantization, REC_LABEL, type LayerSensitivity, type QuantPrecision, type Recommendation } from "../quant/sensitivity";
import { ortQuantizeScript, planHeadline, planJsonString, tfliteQuantizeScript } from "../quant/exportPlan";
import { viridis, viridisGradient } from "./Graph/colorBy";

type SortKey = "risk" | "sqnr" | "params" | "name" | "order";

const REC_ORDER: Recommendation[] = ["int4", "int8", "int8-per-channel", "fp16", "keep-fp32"];

/**
 * Quantization heatmap panel: model-level summary, a sortable per-layer risk
 * table, and exports of the mixed-precision plan (JSON, ONNX Runtime
 * quantize_static script with nodes_to_exclude, TFLite selective quantization).
 * Every number is a statistical estimate, not measured accuracy.
 */
export function QuantHeatmap({
  model, onSelectLayer, selected, onShowOnGraph,
}: {
  model: Model;
  onSelectLayer?: (l: ModelLayer) => void;
  selected?: ModelLayer | null;
  /** Optional: switch the graph's "Color by" overlay to this precision. */
  onShowOnGraph?: (mode: "q8" | "q4") => void;
}) {
  const t = useT();
  const [prec, setPrec] = useState<QuantPrecision>("int8");
  const [sort, setSort] = useState<SortKey>("risk");
  const [desc, setDesc] = useState(true);
  const [showAll, setShowAll] = useState(false);

  const a = useMemo(() => analyzeQuantization(model, { target: prec }), [model, prec]);
  const s = a.summary;
  const layerById = useMemo(() => new Map(model.layers.map((l) => [l.id, l])), [model]);

  const est = (r: LayerSensitivity) => (prec === "int8" ? r.int8 : r.int4);
  const rows = useMemo(() => {
    const q = a.layers.filter((r) => r.quantizable && (prec === "int8" || r.kind === "weighted"));
    const key = (r: LayerSensitivity): number | string => {
      const e = prec === "int8" ? r.int8 : r.int4;
      switch (sort) {
        case "risk": return e.perChannelRisk;
        case "sqnr": return e.sqnrDb ?? Infinity;
        case "params": return r.params;
        case "name": return r.name.toLowerCase();
        default: return r.id;
      }
    };
    return [...q].sort((x, y) => {
      const kx = key(x), ky = key(y);
      const c = kx < ky ? -1 : kx > ky ? 1 : 0;
      return desc ? -c : c;
    });
  }, [a, sort, desc, prec]);
  const shown = showAll ? rows : rows.slice(0, 25);

  const setSortKey = (k: SortKey) => {
    if (k === sort) setDesc((d) => !d);
    else { setSort(k); setDesc(k === "risk" || k === "params"); }
  };

  const base = (model.name || "model").replace(/\.[^.]+$/, "");
  const btn = (active = false) => ({
    padding: "3px 9px", borderRadius: 4, border: `1px solid ${active ? t.acc : t.bdr}`,
    background: active ? t.acc + "18" : "transparent", color: active ? t.acc : t.t2,
    fontSize: 10, fontWeight: 600 as const, cursor: "pointer",
  });
  const card = { background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "6px 8px" };
  const mono = "'JetBrains Mono',monospace";
  const recColor = (r: Recommendation) => (r === "fp16" || r === "keep-fp32" ? t.wrn : r === "int8-per-channel" ? t.acc : t.suc);

  const sizes: { k: string; v: number }[] = [
    { k: "FP32", v: s.sizeBytes.fp32 }, { k: "FP16", v: s.sizeBytes.fp16 },
    { k: "INT8", v: s.sizeBytes.int8 }, { k: "INT4", v: s.sizeBytes.int4 },
    { k: "Plan", v: s.sizeBytes.mixed },
  ];
  const maxSize = Math.max(1, s.sizeBytes.fp32);
  const totalRec = REC_ORDER.reduce((n, r) => n + s.counts[r], 0);

  const th = (k: SortKey, label: string, align: "left" | "right" = "left") => (
    <th
      onClick={() => setSortKey(k)}
      aria-sort={sort === k ? (desc ? "descending" : "ascending") : "none"}
      style={{ textAlign: align, padding: "4px 5px", fontSize: 9, fontWeight: 600, color: sort === k ? t.acc : t.t3, cursor: "pointer", whiteSpace: "nowrap", borderBottom: `1px solid ${t.bdr}`, position: "sticky", top: 0, background: t.bg1 }}
    >{label}{sort === k ? (desc ? " ↓" : " ↑") : ""}</th>
  );

  return <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Quantization heatmap</div>
      <div style={{ display: "flex", gap: 3 }}>
        {(["int8", "int4"] as QuantPrecision[]).map((p) => (
          <button key={p} type="button" style={btn(prec === p)} onClick={() => setPrec(p)}>{p === "int8" ? "INT8" : "INT4 (W4A8)"}</button>
        ))}
        {onShowOnGraph && <button type="button" style={btn()} onClick={() => onShowOnGraph(prec === "int8" ? "q8" : "q4")}>Show on graph</button>}
      </div>
    </div>

    <div style={{ fontSize: 9, color: t.t2, lineHeight: 1.5, ...card }}>
      <b style={{ color: t.t1 }}>Statistical estimate, not measured accuracy.</b> Weight SQNR from min/max/σ (uniform quantizer noise Δ²/12), op-type priors and graph position.
      Weight stats available for <b style={{ color: t.t1 }}>{Math.round(s.statsCoverage * 100)}%</b> of {s.weighted} weighted layers · confidence <b style={{ color: s.confidence === "medium" ? t.wrn : t.t2 }}>{s.confidence}</b>.
    </div>

    {s.quantizable === 0 ? (
      <div style={{ ...card, fontSize: 10, color: t.t2, textAlign: "center", padding: 16 }}>No layers with arithmetic to quantize in this model.</div>
    ) : <>
      {/* Summary */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(92px, 1fr))", gap: 5 }}>
        <Stat label="Safe for INT8" value={`${s.safeInt8Pct}%`} sub={`${s.highRisk8} high-risk`} color={s.safeInt8Pct >= 80 ? t.suc : s.safeInt8Pct >= 50 ? t.wrn : t.err} />
        <Stat label="Safe for INT4 (weights)" value={`${s.safeInt4Pct}%`} sub={`${s.highRisk4} high-risk`} color={s.safeInt4Pct >= 80 ? t.suc : s.safeInt4Pct >= 50 ? t.wrn : t.err} />
        <Stat label="Plan size" value={fB(s.sizeBytes.mixed)} sub={s.sizeBytes.fp32 > 0 ? `${Math.round((1 - s.sizeBytes.mixed / s.sizeBytes.fp32) * 100)}% smaller than FP32` : "no weights"} color={t.acc} />
        <Stat label="Kept in float" value={String(s.excluded.length)} sub={`${s.perChannel.length} need per-channel`} color={s.excluded.length ? t.wrn : t.suc} />
      </div>

      {/* Size per precision */}
      <div style={card}>
        <div style={{ fontSize: 9, color: t.t3, marginBottom: 4 }}>Estimated weight storage ({fmt(s.totalParams)} params)</div>
        {sizes.map((z) => (
          <div key={z.k} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 9, padding: "1px 0" }}>
            <span style={{ width: 30, color: z.k === "Plan" ? t.acc : t.t2, fontWeight: z.k === "Plan" ? 700 : 500 }}>{z.k}</span>
            <span style={{ flex: 1, height: 6, background: t.bg3, borderRadius: 3, overflow: "hidden" }}>
              <span style={{ display: "block", width: `${(z.v / maxSize) * 100}%`, height: "100%", background: z.k === "Plan" ? t.acc : t.t3, borderRadius: 3 }} />
            </span>
            <span style={{ width: 56, textAlign: "right", fontFamily: mono, color: t.t1 }}>{fB(Math.round(z.v))}</span>
          </div>
        ))}
      </div>

      {/* Plan distribution */}
      <div style={card}>
        <div style={{ fontSize: 9, color: t.t3, marginBottom: 4 }}>Mixed-precision plan: {planHeadline(a)}</div>
        <div style={{ display: "flex", height: 7, borderRadius: 4, overflow: "hidden", background: t.bg3 }}>
          {REC_ORDER.filter((r) => s.counts[r] > 0).map((r) => (
            <div key={r} title={`${REC_LABEL[r]}: ${s.counts[r]}`} style={{ width: `${(s.counts[r] / Math.max(1, totalRec)) * 100}%`, background: recColor(r), opacity: r === "int8" ? 0.7 : 1 }} />
          ))}
        </div>
        <div style={{ display: "flex", gap: 4, marginTop: 6, flexWrap: "wrap" }}>
          <button type="button" style={btn()} onClick={() => downloadFile(planJsonString(model, a), `${base}.quant-plan.json`, "application/json")}>Plan JSON</button>
          <button type="button" style={btn()} onClick={() => downloadFile(ortQuantizeScript(model, a), `${base}_quantize_ort.py`, "text/x-python")}>ONNX Runtime script</button>
          <button type="button" style={btn()} onClick={() => downloadFile(tfliteQuantizeScript(model, a), `${base}_quantize_tflite.py`, "text/x-python")}>TFLite script</button>
        </div>
      </div>

      {/* Per-layer table */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 9, color: t.t3 }}>
        <span>Risk</span>
        <span aria-hidden style={{ width: 90, height: 6, borderRadius: 3, background: viridisGradient() }} />
        <span>0 → 100% (per-channel)</span>
        <span style={{ flex: 1 }} />
        <span>{rows.length} layers</span>
      </div>
      <div style={{ border: `1px solid ${t.bdr}`, borderRadius: 6, overflow: "auto", maxHeight: 420 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9.5 }}>
          <thead>
            <tr>
              {th("name", "Layer")}
              {th("risk", "Risk")}
              {th("sqnr", "SQNR dB", "right")}
              {th("params", "Params", "right")}
              <th style={{ textAlign: "left", padding: "4px 5px", fontSize: 9, fontWeight: 600, color: t.t3, borderBottom: `1px solid ${t.bdr}`, position: "sticky", top: 0, background: t.bg1 }}>Plan</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const e = est(r);
              const isSel = selected?.id === r.id;
              const layer = layerById.get(r.id);
              return (
                <tr
                  key={r.id}
                  onClick={() => layer && onSelectLayer?.(layer)}
                  title={[r.reason, ...r.factors].join("\n")}
                  style={{ cursor: onSelectLayer ? "pointer" : "default", background: isSel ? t.bg2 : "transparent", borderBottom: `1px solid ${t.bdr}` }}
                >
                  <td style={{ padding: "4px 5px", maxWidth: 150 }}>
                    <div style={{ color: t.t0, fontWeight: 600, fontFamily: mono, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.op}</div>
                    <div style={{ color: t.t3, fontSize: 8.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.name}</div>
                  </td>
                  <td style={{ padding: "4px 5px", whiteSpace: "nowrap" }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <span aria-hidden style={{ width: 10, height: 10, borderRadius: 2, background: viridis(e.perChannelRisk), border: `1px solid ${t.bdr}` }} />
                      <span style={{ fontFamily: mono, color: e.level === "high" ? t.err : e.level === "medium" ? t.wrn : t.t1 }}>{Math.round(e.perChannelRisk * 100)}%</span>
                      {e.risk - e.perChannelRisk > 0.05 && <span style={{ color: t.t3, fontFamily: mono }} title="Per-tensor risk">({Math.round(e.risk * 100)}%)</span>}
                    </span>
                  </td>
                  <td style={{ padding: "4px 5px", textAlign: "right", fontFamily: mono, color: t.t2, whiteSpace: "nowrap" }}>
                    {e.sqnrDb == null ? "–" : `${e.sqnrDb}${e.perChannelSqnrDb != null && e.perChannelSqnrDb - e.sqnrDb >= 1 ? ` / ${e.perChannelSqnrDb}` : ""}`}
                    {r.statsSource === "assumed" && <span title="No weight stats: Gaussian assumed" style={{ color: t.t3 }}>*</span>}
                  </td>
                  <td style={{ padding: "4px 5px", textAlign: "right", fontFamily: mono, color: t.t2 }}>{r.params ? fmt(r.params) : "–"}</td>
                  <td style={{ padding: "4px 5px", whiteSpace: "nowrap", color: recColor(r.recommendation), fontWeight: 600 }}>{REC_LABEL[r.recommendation]}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length > 25 && (
        <button type="button" style={{ ...btn(), alignSelf: "center" }} onClick={() => setShowAll((v) => !v)}>
          {showAll ? "Show top 25" : `Show all ${rows.length}`}
        </button>
      )}
      <div style={{ fontSize: 8.5, color: t.t3, lineHeight: 1.5 }}>
        SQNR shows per-tensor / per-channel weight estimates. * = no weight statistics in the file (Gaussian assumed). INT4 means 4-bit weights with 8-bit activations. Hover a row for the reasoning; click to select the layer.
      </div>
    </>}
  </div>;
}

function Stat({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  const t = useT();
  return (
    <div style={{ background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "6px 8px", minWidth: 0 }}>
      <div style={{ fontSize: 8.5, color: t.t3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 800, color, fontFamily: "'JetBrains Mono',monospace" }}>{value}</div>
      <div style={{ fontSize: 8.5, color: t.t2 }}>{sub}</div>
    </div>
  );
}
