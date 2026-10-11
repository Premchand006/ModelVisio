import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import {
  investigate, ruleBasedReport,
  type EvidenceItem, type EvidencePack, type Hypothesis, type InvestigationReport,
} from "@modelvisio/ai";
import { useT } from "../theme/ThemeContext";
import { HW } from "../data/hardware";
import { fixById, type FixId } from "../fixes/transforms";
import { buildEvidence } from "../investigate/evidence";
import { isDesktop, getUserApiKey } from "../utils/apiKey";
import { fmt } from "../utils/format";

const mono = "'JetBrains Mono',monospace";
const pct = (x: number) => `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
const msStr = (x: number) => (x >= 100 ? Math.round(x).toString() : x >= 1 ? x.toFixed(1) : x.toFixed(2));

/**
 * AI performance investigator: "why is this model slow on <device>?". The
 * evidence (per-layer roofline time shares, bottlenecks, compiler issues,
 * patterns) is computed deterministically and shown immediately with a
 * rule-based diagnosis; "Investigate" asks the AI copilot (via the /api/chat
 * proxy) for ranked, evidence-cited hypotheses, validated against the evidence.
 * Without AI (desktop with no key, proxy down) the rule-based report stands.
 */
export function Investigator({
  model, onSelectLayer, onAskInChat, defaultDeviceId = "orin_nano",
}: {
  model: Model;
  /** Jump to a layer in the graph/inspector. */
  onSelectLayer?: (layer: ModelLayer) => void;
  /** Optional: hand a follow-up question to the AI chat. */
  onAskInChat?: (prompt: string) => void;
  defaultDeviceId?: string;
}) {
  const t = useT();
  const [deviceId, setDeviceId] = useState(() => (HW.some((d) => d.id === defaultDeviceId) ? defaultDeviceId : HW[0].id));
  const [question, setQuestion] = useState("");
  const [aiReport, setAiReport] = useState<InvestigationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const abort = useRef<AbortController | null>(null);

  const device = HW.find((d) => d.id === deviceId) ?? HW[0];
  const evidence = useMemo(() => buildEvidence(model, device), [model, device]);
  const rules = useMemo(() => ruleBasedReport(evidence), [evidence]);
  const report = aiReport ?? rules;

  // A different model/device invalidates any AI answer.
  useEffect(() => { abort.current?.abort(); setAiReport(null); setLoading(false); }, [evidence]);
  useEffect(() => () => abort.current?.abort(), []);

  const desktop = isDesktop();
  const needsKey = desktop && !getUserApiKey();
  const vendors = [...new Set(HW.map((h) => h.v))];
  const layerByName = useMemo(() => new Map(model.layers.map((l) => [l.name, l])), [model]);
  const jump = onSelectLayer ? (name: string) => { const l = layerByName.get(name); if (l) onSelectLayer(l); } : undefined;
  const evById = useMemo(() => new Map(evidence.evidence.map((e) => [e.id, e])), [evidence]);

  const run = async () => {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setLoading(true);
    try {
      const r = await investigate({ evidence, question, aiEnabled: !needsKey, signal: ctrl.signal });
      if (!ctrl.signal.aborted) setAiReport(r);
    } catch {
      // aborted: a newer run or unmount superseded this one
    } finally {
      if (!ctrl.signal.aborted) setLoading(false);
    }
  };

  const tot = evidence.totals;
  return <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%" }}>
    <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Performance Investigator</div>

    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
      <label style={{ fontSize: 9, color: t.t3 }} htmlFor="mv-inv-device">Device</label>
      <select id="mv-inv-device" value={deviceId} onChange={(e) => setDeviceId(e.target.value)}
        style={{ flex: 1, minWidth: 160, padding: "4px 6px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 10 }}>
        {vendors.map((v) => <optgroup key={v} label={v}>{HW.filter((h) => h.v === v).map((h) => <option key={h.id} value={h.id}>{h.n}</option>)}</optgroup>)}
      </select>
    </div>
    <div style={{ display: "flex", gap: 4 }}>
      <input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !loading && run()}
        placeholder={`Why is this model slow on ${device.n}?`} aria-label="Question for the investigator"
        style={{ flex: 1, padding: "6px 8px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t0, fontSize: 10, outline: "none" }} />
      <button type="button" onClick={run} disabled={loading}
        style={{ padding: "6px 12px", borderRadius: 5, border: "none", background: loading ? t.bg3 : t.acc, color: "#fff", fontSize: 10, fontWeight: 600, cursor: loading ? "default" : "pointer", whiteSpace: "nowrap" }}>
        {loading ? "Investigating…" : "Investigate"}</button>
    </div>

    {needsKey && <div style={{ fontSize: 9.5, color: t.t2, lineHeight: 1.5, background: t.acc + "10", border: `1px solid ${t.acc}33`, borderRadius: 5, padding: "5px 8px" }}>
      Add your Gemini API key in the <b>AI</b> tab (key icon) for an AI diagnosis. Free key at <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer noopener" style={{ color: t.acc }}>aistudio.google.com/apikey</a> (stored only on this device). The rule-based diagnosis below works without it.
    </div>}

    <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
      {/* Headline numbers */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {[
          { l: "Est. latency", v: `${msStr(tot.latencyMs)} ms` },
          { l: "FPS", v: fmt(Math.round(tot.fps)) },
          { l: "Precision", v: evidence.device.precision.toUpperCase() },
          { l: "Regime", v: `${tot.bound}-bound` },
          { l: "Utilization", v: `${pct(tot.util)}${tot.calibrated ? " cal." : ""}` },
        ].map((s) => <div key={s.l} style={{ background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 5, padding: "4px 8px" }}>
          <div style={{ fontSize: 8, color: t.t3, textTransform: "uppercase", letterSpacing: .5 }}>{s.l}</div>
          <div style={{ fontSize: 12, fontWeight: 700, color: t.t0, fontFamily: mono }}>{s.v}</div>
        </div>)}
      </div>

      <Section title="Where the time goes: top layers" note="Estimated share of latency (roofline). Click a bar to jump to the layer.">
        <ShareBars
          rows={evidence.bottlenecks.map((b) => ({ key: b.id, label: b.layer, share: b.share, color: b.bound === "compute" ? t.acc : t.acc2, tip: `${b.layer} (${b.op}): ${msStr(b.timeMs)} ms, ${pct(b.share)}, ${b.bound}-bound` }))}
          onClick={jump ? (r) => jump(r.label) : undefined} />
        <div style={{ display: "flex", gap: 10, fontSize: 8.5, color: t.t2, marginTop: 2 }}>
          <Legend color={t.acc} label="compute-bound" /><Legend color={t.acc2} label="memory-bound" />
          <span style={{ marginLeft: "auto" }}>memory-bound layers: {pct(tot.memoryBoundShare)} of time</span>
        </div>
      </Section>

      <Section title="By op type">
        <ShareBars rows={evidence.opBreakdown.map((o) => ({ key: o.op, label: `${o.op} ×${o.count}`, share: o.share, color: t.acc3, tip: `${o.op}: ${o.count} layer(s), ${msStr(o.timeMs)} ms, ${pct(o.share)}` }))} />
      </Section>

      {/* Diagnosis */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: t.t0 }}>Diagnosis</span>
        <span style={{ fontSize: 8, fontWeight: 700, textTransform: "uppercase", letterSpacing: .5, padding: "1px 5px", borderRadius: 3, color: report.source === "ai" ? t.acc : t.t2, border: `1px solid ${report.source === "ai" ? t.acc : t.bdr2}` }}>{report.source === "ai" ? "AI · evidence-checked" : "Rule-based"}</span>
        {onAskInChat && <button type="button" onClick={() => onAskInChat(followUp(evidence, report))}
          style={{ marginLeft: "auto", padding: "2px 8px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 9, cursor: "pointer" }}>Ask follow-up in Chat</button>}
      </div>
      {report.note && <div style={{ fontSize: 9, color: t.wrn, lineHeight: 1.45 }}>{report.note}</div>}
      <div style={{ fontSize: 10.5, color: t.t1, lineHeight: 1.5 }}>{report.summary}</div>
      {report.rejected.hypotheses + report.rejected.layers.length + report.rejected.evidence.length > 0 && <div style={{ fontSize: 8.5, color: t.t3 }}>
        Validation removed {[report.rejected.hypotheses ? `${report.rejected.hypotheses} uncited hypothesis(es)` : "", report.rejected.layers.length ? `unknown layer(s): ${report.rejected.layers.join(", ")}` : "", report.rejected.evidence.length ? `unknown evidence id(s): ${report.rejected.evidence.join(", ")}` : ""].filter(Boolean).join(" · ")}.
      </div>}

      {report.hypotheses.map((h) => <HypothesisCard key={h.rank + h.title} h={h} evById={evById} onLayer={jump} />)}

      {report.sources.length > 0 && <div style={{ padding: "5px 8px", borderRadius: 6, background: t.acc + "0E", border: `1px solid ${t.acc}33` }}>
        <div style={{ fontSize: 8, fontWeight: 700, color: t.acc, textTransform: "uppercase", letterSpacing: 1, marginBottom: 3 }}>Sources</div>
        {report.sources.map((s, j) => <a key={s.url} href={s.url} target="_blank" rel="noreferrer noopener" style={{ display: "block", fontSize: 10, color: t.acc, textDecoration: "none", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{j + 1}. {s.title}</a>)}
      </div>}

      <details>
        <summary style={{ fontSize: 10, color: t.t2, cursor: "pointer" }}>Evidence pack ({evidence.evidence.length} items)</summary>
        <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 4 }}>
          {evidence.evidence.map((e) => <div key={e.id} style={{ fontSize: 9, lineHeight: 1.45, color: t.t2 }}>
            <b style={{ color: e.severity === "error" ? t.err : e.severity === "warn" ? t.wrn : t.t1, fontFamily: mono }}>{e.id}</b> <span style={{ color: t.t1 }}>{e.title}</span>: {e.detail}
          </div>)}
        </div>
      </details>
    </div>
  </div>;
}

function Section({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  const t = useT();
  return <div style={{ background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "6px 8px" }}>
    <div style={{ fontSize: 10.5, fontWeight: 600, color: t.t0 }}>{title}</div>
    {note && <div style={{ fontSize: 8.5, color: t.t3, marginBottom: 4 }}>{note}</div>}
    {children}
  </div>;
}

function Legend({ color, label }: { color: string; label: string }) {
  return <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: 2, background: color, display: "inline-block" }} />{label}</span>;
}

type BarRow = { key: string; label: string; share: number; color: string; tip: string };

/** Horizontal time-share bars in inline SVG (theme colors, hover tooltips via <title>). */
function ShareBars({ rows, onClick }: { rows: BarRow[]; onClick?: (r: BarRow) => void }) {
  const t = useT();
  if (rows.length === 0) return <div style={{ fontSize: 9, color: t.t3 }}>No timed layers.</div>;
  const W = 360, labelW = 130, valW = 40, rowH = 16, barH = 10;
  const max = Math.max(...rows.map((r) => r.share), 1e-9);
  const barW = W - labelW - valW;
  const clip = (s: string) => (s.length > 22 ? s.slice(0, 21) + "…" : s);
  return <svg viewBox={`0 0 ${W} ${rows.length * rowH}`} width="100%" role="img" aria-label="Estimated latency share" style={{ display: "block" }}>
    {rows.map((r, i) => {
      const y = i * rowH;
      const w = Math.max(2, (r.share / max) * barW);
      return <g key={r.key} onClick={onClick ? () => onClick(r) : undefined} style={{ cursor: onClick ? "pointer" : "default" }}>
        <title>{r.tip}</title>
        <rect x={0} y={y} width={W} height={rowH} fill="transparent" />
        <text x={labelW - 6} y={y + rowH / 2 + 3} textAnchor="end" fontSize={8.5} fill={t.t1}>{clip(r.label)}</text>
        <rect x={labelW} y={y + (rowH - barH) / 2} width={w} height={barH} rx={2} fill={r.color} />
        <text x={labelW + w + 4} y={y + rowH / 2 + 3} fontSize={8.5} fill={t.t2} fontFamily="'JetBrains Mono',monospace">{pct(r.share)}</text>
      </g>;
    })}
  </svg>;
}

function HypothesisCard({ h, evById, onLayer }: { h: Hypothesis; evById: Map<string, EvidenceItem>; onLayer?: (name: string) => void }) {
  const t = useT();
  const cc = h.confidence === "high" ? t.acc3 : h.confidence === "medium" ? t.t1 : t.t3;
  const chip = { padding: "1px 6px", borderRadius: 3, fontSize: 8.5, border: `1px solid ${t.bdr2}`, background: "transparent" } as const;
  return <div style={{ background: t.glass, border: `1px solid ${t.glassBdr}`, borderRadius: 6, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 4 }}>
    <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
      <span style={{ fontSize: 11, fontWeight: 800, color: t.acc, fontFamily: mono }}>#{h.rank}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: t.t0, flex: 1 }}>{h.title}</span>
      <span style={{ fontSize: 8, color: cc, textTransform: "uppercase", letterSpacing: .4 }}>{h.confidence}</span>
    </div>
    {h.explanation && <div style={{ fontSize: 10, color: t.t1, lineHeight: 1.5 }}>{h.explanation}</div>}
    <div style={{ display: "flex", gap: 3, flexWrap: "wrap", alignItems: "center" }}>
      <span style={{ fontSize: 8.5, color: t.t3 }}>Evidence:</span>
      {h.evidence.map((id) => { const e = evById.get(id); return <span key={id} title={e ? `${e.title}: ${e.detail}` : id} style={{ ...chip, color: t.t2, fontFamily: mono }}>{id}</span>; })}
    </div>
    {h.layers.length > 0 && <div style={{ display: "flex", gap: 3, flexWrap: "wrap", alignItems: "center" }}>
      <span style={{ fontSize: 8.5, color: t.t3 }}>Layers:</span>
      {h.layers.map((n) => onLayer
        ? <button key={n} type="button" onClick={() => onLayer(n)} title="Jump to layer" style={{ ...chip, color: t.acc, cursor: "pointer" }}>{n} ↗</button>
        : <span key={n} style={{ ...chip, color: t.t2 }}>{n}</span>)}
    </div>}
    {h.experiments.length > 0 && <ul style={{ margin: 0, paddingLeft: 14, display: "flex", flexDirection: "column", gap: 2 }}>
      {h.experiments.map((x, i) => <li key={i} style={{ fontSize: 9.5, color: t.t1, lineHeight: 1.45 }}>{x.action}{x.expectedGain && <span style={{ color: t.acc3 }}> · {x.expectedGain}</span>}</li>)}
    </ul>}
    {h.fixIds.length > 0 && <div style={{ fontSize: 9, color: t.acc3 }}>Auto-fix available (Compiler tab): {h.fixIds.map((id) => fixById(id as FixId)?.title ?? id).join(", ")}</div>}
  </div>;
}

function followUp(ev: EvidencePack, r: InvestigationReport): string {
  const top = r.hypotheses[0];
  return `On ${ev.device.name}, the investigator's top hypothesis is "${top?.title ?? r.summary}". `
    + `Give me a step-by-step plan to verify and fix it for ${ev.model.name}.`;
}
