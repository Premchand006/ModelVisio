import { useMemo, useState } from "react";
import type { Model } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { fmt, fB } from "../utils/format";
import { HW } from "../data/hardware";
import { scoreAll, type Profile, type ScoreResult } from "../scoring";

const PROFILE_LABELS: Record<Profile, string> = { throughput: "Max throughput", power: "Power-constrained", cost: "Cost-sensitive" };

/**
 * Hardware fit report. The 0-100 score is COMPUTED per model by the roofline
 * scoring engine (see packages/core/src/scoring): per-layer min(compute, BW·AI)
 * with a calibrated utilization de-rating, a memory-fit hard-fail guard, op/
 * CPU-fallback coverage, and an efficiency term. Every row is explainable and
 * carries a confidence badge tied to the device's data provenance.
 */
export function HWReport({ model }: { model: Model }) {
  const t = useT();
  const [vendor, setVendor] = useState("all");
  const [profile, setProfile] = useState<Profile>("throughput");
  const [open, setOpen] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const vendors = ["all", ...[...new Set(HW.map((h) => h.v))].sort()];

  const scored = useMemo(() => {
    const fil = vendor === "all" ? HW : HW.filter((h) => h.v === vendor);
    return scoreAll(model, fil, profile);
  }, [model, vendor, profile]);

  const totalP = model.layers.reduce((s, l) => s + l.params, 0);
  const totalF = model.layers.reduce((s, l) => s + (l.flops || l.macs * 2), 0);
  const wc = scored[0]?.roofline.workload ?? "cnnCls";
  const wcLabel = wc === "transformer" ? "transformer" : wc === "cnnDet" ? "CNN detector/segmenter" : "CNN classifier";

  const report = useMemo(() => buildReport(model, scored, profile, totalP, totalF, wcLabel), [model, scored, profile, totalP, totalF, wcLabel]);

  const confColor = (c: ScoreResult["confidence"]) => (c === "high" ? t.suc : c === "medium" ? t.wrn : t.t2);

  return <div style={{ display: "flex", flexDirection: "column", gap: 6, height: "100%" }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Hardware — {scored.length} devices</div>
      <button type="button" onClick={() => { navigator.clipboard?.writeText(report); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
        style={{ padding: "3px 10px", borderRadius: 4, border: `1px solid ${t.acc}`, background: t.acc + "18", color: t.acc, fontSize: 10, fontWeight: 600, cursor: "pointer" }}>{copied ? "✓ Copied" : "Export Report"}</button>
    </div>

    {/* Why this score — methodology line */}
    <div style={{ fontSize: 9, color: t.t2, lineHeight: 1.5, background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 5, padding: "5px 8px" }}>
      Roofline estimate · workload classified as <b style={{ color: t.t1 }}>{wcLabel}</b> · per-layer min(compute, bandwidth×intensity) with calibrated utilization, memory-fit guard, op-support & efficiency. Scores reflect <b style={{ color: t.t1 }}>this model</b>, not fixed ratings.
    </div>

    {/* Profile selector */}
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      <span style={{ fontSize: 9, color: t.t3 }}>Profile:</span>
      {(Object.keys(PROFILE_LABELS) as Profile[]).map((p) => <button key={p} type="button" onClick={() => setProfile(p)}
        style={{ padding: "2px 7px", borderRadius: 3, border: `1px solid ${profile === p ? t.acc : t.bdr}`, background: profile === p ? t.acc + "18" : "transparent", color: profile === p ? t.acc : t.t2, fontSize: 9, cursor: "pointer" }}>{PROFILE_LABELS[p]}</button>)}
    </div>

    {/* Vendor filter */}
    <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>{vendors.map((v) => <button key={v} type="button" onClick={() => setVendor(v)}
      style={{ padding: "2px 7px", borderRadius: 3, border: `1px solid ${vendor === v ? t.acc : t.bdr}`, background: vendor === v ? t.acc + "18" : "transparent", color: vendor === v ? t.acc : t.t2, fontSize: 9, cursor: "pointer" }}>{v === "all" ? "All" : v}</button>)}</div>

    <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
      {scored.map((r) => {
        const hw = r.device;
        const bc = r.memory.hardFail || r.opSupport.compileFail ? t.err : r.overall >= 80 ? t.suc : r.overall >= 60 ? t.wrn : t.err;
        const isOpen = open === hw.id;
        const topBanner = r.banners.find((b) => b.level === "error") ?? r.banners.find((b) => b.level === "warn");
        return <div key={hw.id} style={{ background: t.glass, backdropFilter: "blur(8px)", border: `1px solid ${t.glassBdr}`, borderRadius: 6, padding: "8px 10px", cursor: "pointer" }}
          onClick={() => setOpen(isOpen ? null : hw.id)}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: t.t0 }}>{hw.n}</span>
              <span title={`Data confidence: ${r.confidence}`} style={{ fontSize: 7.5, fontWeight: 700, textTransform: "uppercase", letterSpacing: .4, padding: "1px 4px", borderRadius: 3, color: confColor(r.confidence), border: `1px solid ${confColor(r.confidence)}55` }}>{r.confidence === "high" ? "verified" : r.confidence === "medium" ? "benchmark" : "est."}</span>
            </span>
            <span style={{ fontSize: 13, fontWeight: 800, color: bc, fontFamily: "'JetBrains Mono',monospace" }}>{r.overall}</span>
          </div>

          {/* Stacked sub-score bar */}
          <StackBar r={r} />

          <div style={{ display: "flex", gap: 8, fontSize: 9, color: t.t2, marginTop: 4, flexWrap: "wrap" }}>
            <span style={{ color: t.acc }}>~{fmtFps(r.fps)} FPS</span>
            <span>{r.latencyMs < 1000 ? r.latencyMs.toFixed(1) : Math.round(r.latencyMs)} ms</span>
            <span>{r.fpsPerW.toFixed(1)} FPS/W</span>
            <span>{hw.peak.int8 ?? hw.peak.fp16}{hw.peak.int8 ? " TOPS" : " TFLOPS"}</span>
            <span>{hw.ramGB > 0 ? hw.ramGB + "GB" : (hw.sramMB ? hw.sramMB + "MB SRAM" : "SRAM")}</span>
            <span>{hw.power.typ}W</span>
            <span style={{ color: r.roofline.bound === "memory" ? t.wrn : t.acc3 }}>{r.roofline.bound}-bound</span>
          </div>

          {topBanner && <div style={{ marginTop: 4, fontSize: 8.5, color: topBanner.level === "error" ? t.err : t.wrn, lineHeight: 1.4 }}>{topBanner.level === "error" ? "⛔ " : "⚠ "}{topBanner.msg}</div>}

          {isOpen && <Detail r={r} />}
        </div>;
      })}
    </div>
  </div>;
}

/** Stacked, weighted contribution bar — shows WHY the number is what it is. */
function StackBar({ r }: { r: ScoreResult }) {
  const t = useT();
  const segs = [
    { k: "Latency", v: r.sub.latency, w: r.weights.latency, c: t.acc },
    { k: "Memory", v: r.sub.memory, w: r.weights.memory, c: t.acc3 },
    { k: "Op support", v: r.sub.opSupport, w: r.weights.opSupport, c: t.acc2 },
    { k: "Efficiency", v: r.sub.efficiency, w: r.weights.efficiency, c: t.cO },
  ];
  return <div style={{ display: "flex", height: 5, borderRadius: 3, overflow: "hidden", background: t.bg3 }}>
    {segs.map((s) => <div key={s.k} title={`${s.k}: ${s.v}/100 (weight ${Math.round(s.w * 100)}%)`}
      style={{ width: `${s.w * 100}%`, height: "100%", background: s.c, opacity: 0.25 + 0.75 * (s.v / 100) }} />)}
  </div>;
}

/** Expanded breakdown: sub-scores, roofline regime, footprint, sources. */
function Detail({ r }: { r: ScoreResult }) {
  const t = useT();
  const hw = r.device;
  const row = (label: string, value: string, color = t.t1) => <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, padding: "1px 0" }}><span style={{ color: t.t3 }}>{label}</span><span style={{ color, fontFamily: "'JetBrains Mono',monospace" }}>{value}</span></div>;
  return <div style={{ marginTop: 6, paddingTop: 6, borderTop: `1px solid ${t.bdr}`, display: "flex", flexDirection: "column", gap: 2 }} onClick={(e) => e.stopPropagation()}>
    {row("Latency", `${r.sub.latency}/100 · ${Math.round(r.weights.latency * 100)}% weight`, t.acc)}
    {row("Memory fit", `${r.sub.memory}/100 · ${Math.round(r.weights.memory * 100)}% weight`, t.acc3)}
    {row("Op support", `${r.sub.opSupport}/100 · ${Math.round(r.weights.opSupport * 100)}% weight`, t.acc2)}
    {row("Efficiency", `${r.sub.efficiency}/100 · ${Math.round(r.weights.efficiency * 100)}% weight`, t.cO)}
    <div style={{ height: 4 }} />
    {row("Deployment precision", r.precision.toUpperCase())}
    {row("Roofline regime", `${r.roofline.bound}-bound`, r.roofline.bound === "memory" ? t.wrn : t.acc3)}
    {row("Model intensity", `${finite(r.roofline.modelAI)} FLOP/B`)}
    {row("Device ridge point", `${finite(r.roofline.ridgeAI)} FLOP/B`)}
    {row("Utilization used", `${Math.round(r.roofline.util * 100)}% · ${r.roofline.workload} · ${r.roofline.calibrated ? "calibrated" : "estimated"}`, r.roofline.calibrated ? t.suc : t.t1)}
    {r.roofline.calibrated && r.roofline.utilNote && row("Calibration source", r.roofline.utilNote)}
    {row("Est. throughput", `~${fmtFps(r.fps)} FPS @ ${r.latencyMs < 1000 ? r.latencyMs.toFixed(2) : Math.round(r.latencyMs)} ms`, t.acc)}
    {row("Efficiency", `${r.fpsPerW.toFixed(1)} FPS/W${r.fpsPerDollar != null ? ` · ${r.fpsPerDollar.toFixed(2)} FPS/$` : ""}`)}
    <div style={{ height: 4 }} />
    {row("Model footprint", fB(r.memory.footprintBytes), r.memory.hardFail ? t.err : t.t1)}
    {row(hw.memType === "onchip" ? "On-chip budget" : "Device memory", fB(r.memory.budgetBytes))}
    {row("Headroom", r.memory.budgetBytes > 0 ? `${r.memory.headroom.toFixed(2)}×` : "unknown", r.memory.headroom >= 1.3 ? t.suc : r.memory.headroom >= 1 ? t.wrn : t.err)}
    {row("Op coverage", `${Math.round(r.opSupport.coverage * 100)}%${r.opSupport.fallbackOps ? ` · ${r.opSupport.fallbackOps} CPU fallback` : ""}`)}
    {r.banners.length > 0 && <div style={{ marginTop: 4, display: "flex", flexDirection: "column", gap: 2 }}>
      {r.banners.map((b, i) => <div key={i} style={{ fontSize: 8.5, lineHeight: 1.4, color: b.level === "error" ? t.err : b.level === "warn" ? t.wrn : t.t2 }}>• {b.msg}</div>)}
    </div>}
    {hw.notes && <div style={{ marginTop: 4, fontSize: 8.5, color: t.t2, lineHeight: 1.45 }}>{hw.notes}</div>}
    {hw.src.length > 0 && <div style={{ marginTop: 3, fontSize: 8, color: t.t3, lineHeight: 1.45 }}>Sources: {hw.src.join(" · ")}</div>}
  </div>;
}

function fmtFps(fps: number): string {
  if (!isFinite(fps) || fps <= 0) return "0";
  if (fps >= 1000) return fmt(Math.round(fps));
  if (fps >= 10) return Math.round(fps).toString();
  return fps.toFixed(1);
}
function finite(n: number): string { return isFinite(n) ? n.toFixed(1) : "∞"; }

function buildReport(model: Model, scored: ScoreResult[], profile: Profile, totalP: number, totalF: number, wcLabel: string): string {
  const line = "=".repeat(54);
  const head = `ModelVisio — Hardware Fit Report\n${line}\nModel: ${model.name} | ${model.format} | ${fB(model.sizeBytes)}\nParams: ${fmt(totalP)} | FLOPs: ${fmt(totalF)} | Workload: ${wcLabel}\nProfile: ${PROFILE_LABELS[profile]} | Method: per-layer roofline + memory-fit + op-support + efficiency\n${line}`;
  const rows = scored.map((r, i) => {
    const flags = [r.memory.hardFail ? "WON'T FIT" : "", r.opSupport.compileFail ? "NO COMPILE" : "", r.opSupport.conversionNeeded ? "convert" : "", r.confidence === "low" ? "low-conf" : ""].filter(Boolean).join(",");
    return `${(i + 1 + ".").padEnd(4)}${r.device.n.padEnd(30)} ${String(r.overall).padStart(3)}/100  ~${fmtFps(r.fps).padStart(6)} FPS  ${String(r.fpsPerW.toFixed(1)).padStart(5)} FPS/W  ${r.roofline.bound.padEnd(7)}${flags ? "  [" + flags + "]" : ""}`;
  }).join("\n");
  return `${head}\n${rows}\n${line}\nNote: roofline uses DENSE peak compute and calibrated per-workload utilization;\nFPS is an estimate — confidence varies by device (see 'low-conf' flag).`;
}
