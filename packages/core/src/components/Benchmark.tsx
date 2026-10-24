import { useMemo, useState, useSyncExternalStore } from "react";
import { GPU_WASM_MB } from "../bench/gpuWasm";
import type { Model } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import type { ThemePalette } from "../theme/theme";
import { fB } from "../utils/format";
import { downloadFile } from "../utils/download";
import { HW } from "../data/hardware";
import { availableEps, detectCapabilities, isBenchmarkable } from "../bench/capabilities";
import { defaultDynamicDim, dimKey, inputSpecsFromModel } from "../bench/inputs";
import { benchReport, chartGeometry, fmtMs, predictFor, runLabel } from "../bench/compare";
import { byOpType } from "../bench/profile";
import { benchStore, modelKey } from "../bench/store";
import type { BenchConfig, BenchEp, BenchPhase, BenchResult, WebnnDevice } from "../bench/types";

const EP_LABEL: Record<BenchEp, string> = { wasm: "WebAssembly (CPU)", webgpu: "WebGPU", webnn: "WebNN" };
const PHASE_LABEL: Record<BenchPhase, string> = {
  init: "Starting runtime", session: "Creating session", first: "First inference", warmup: "Warm-up",
  timed: "Timed runs", profile: "Profiling ops", memory: "Measuring memory",
};
const CHART_W = 230;
const LABEL_W = 92;

const base = (n: string) => n.replace(/\.[^.]+$/, "");

/**
 * Live on-device benchmark. Runs the ORIGINAL model file with onnxruntime-web
 * in a dedicated worker (WebAssembly, WebGPU or WebNN), reports measured
 * latency statistics next to the roofline prediction for a reference device,
 * and keeps a per-session history of runs for side-by-side comparison.
 */
export function Benchmark({ model, modelBytes }: { model: Model | null; modelBytes: ArrayBuffer | null }) {
  const t = useT();
  const st = useSyncExternalStore(benchStore.subscribe, benchStore.getSnapshot, benchStore.getSnapshot);
  const caps = useMemo(() => detectCapabilities(), []);
  const eps = availableEps(caps);
  const [showAll, setShowAll] = useState(false);

  const key = model ? modelKey(model) : "";
  const cfg = benchStore.configFor(key);
  const setCfg = (patch: Partial<BenchConfig>) => benchStore.setConfig(key, { ...cfg, ...patch });
  const specs = useMemo(() => (model ? inputSpecsFromModel(model) : []), [model]);
  const runs = st.runs[key] ?? [];
  const latest = runs[runs.length - 1];
  const active = st.active;
  const busyHere = active?.modelKey === key;
  const error = st.errors[key] ?? null;
  const prediction = useMemo(() => (model ? predictFor(model, st.refDevice) : null), [model, st.refDevice]);
  const geo = useMemo(() => chartGeometry(runs, prediction, { width: CHART_W }), [runs, prediction]);

  const card = { background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "8px 10px" } as const;
  const h = { fontSize: 10, fontWeight: 700, color: t.t1, textTransform: "uppercase", letterSpacing: .5, marginBottom: 6 } as const;
  const lbl = { fontSize: 9.5, color: t.t2 } as const;
  const num = { width: 64, padding: "3px 5px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t0, fontSize: 10.5, fontFamily: "'JetBrains Mono',monospace" } as const;
  const chip = (on: boolean, disabled = false) => ({
    padding: "3px 8px", borderRadius: 4, fontSize: 10, cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? .45 : 1,
    border: `1px solid ${on ? t.acc : t.bdr}`, background: on ? t.acc + "18" : "transparent", color: on ? t.acc : t.t2,
  } as const);

  const header = <div>
    <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Live benchmark</div>
    <div style={{ fontSize: 10, color: t.t2, lineHeight: 1.5, marginTop: 2 }}>Run this model on <b style={{ color: t.t1 }}>this device</b> with onnxruntime-web and measure real latency, off the main thread.</div>
  </div>;

  if (!model) {
    return <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>{header}<div style={{ ...card, fontSize: 11, color: t.t2 }}>Load a model to benchmark it.</div></div>;
  }

  if (!isBenchmarkable(model)) {
    return <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {header}
      <div style={{ ...card, fontSize: 11, color: t.t1, lineHeight: 1.6 }}>
        Live benchmarking runs the model with ONNX Runtime, so it needs an <b>ONNX</b> file. This model is <b>{model.format}</b>.
        <div style={{ marginTop: 6, color: t.t2 }}>
          Export it to ONNX first: the <b style={{ color: t.t1 }}>Convert</b> tab has ready-to-run export scripts and kits, and the <b style={{ color: t.t1 }}>Deploy</b> tab has per-target recipes. Then open the .onnx file here.
          The <b style={{ color: t.t1 }}>Hardware</b> tab still gives roofline-estimated latency for this model without running it.
        </div>
      </div>
    </div>;
  }

  if (!modelBytes) {
    return <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {header}
      <div style={{ ...card, fontSize: 11, color: t.t2, lineHeight: 1.6 }}>The original file isn't available (this is the built-in demo). Open a real <b style={{ color: t.t1 }}>.onnx</b> file to run it on this device.</div>
    </div>;
  }

  if (!caps.worker || eps.length === 0) {
    return <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {header}
      <div style={{ ...card, fontSize: 11, color: t.err, lineHeight: 1.6 }}>This environment can't run the benchmark: it needs Web Workers and WebAssembly with SIMD (or WebGPU / WebNN).</div>
    </div>;
  }

  const ep = eps.includes(cfg.ep) ? cfg.ep : eps[0];
  const start = () => {
    benchStore.run(key, {
      bytes: modelBytes,
      config: { ...cfg, ep },
      model: { name: model.name, format: model.format, sizeBytes: model.sizeBytes },
      fallbackSpecs: specs,
    });
  };
  const save = () => {
    const doc = benchReport(runs, prediction);
    downloadFile(JSON.stringify(doc, null, 2), `${base(model.name)}.benchmark.json`, "application/json");
  };
  const intField = (label: string, value: number, onChange: (v: number) => void, min: number, max: number, title?: string, disabled = false) =>
    <label title={title} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={lbl}>{label}</span>
      <input type="number" min={min} max={max} value={value} disabled={disabled || busyHere}
        onChange={(e) => { const v = Math.floor(Number(e.target.value)); if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v))); }} style={num} />
    </label>;

  const capBadge = (ok: boolean, text: string, title: string) =>
    <span title={title} style={{ fontSize: 9, padding: "1px 6px", borderRadius: 3, border: `1px solid ${(ok ? t.suc : t.t3)}55`, color: ok ? t.suc : t.t3, fontFamily: "'JetBrains Mono',monospace" }}>{ok ? "✓" : "✕"} {text}</span>;

  const pct = active?.progress && active.progress.total > 0 ? Math.min(1, active.progress.done / active.progress.total) : 0;

  return <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
    {header}

    {/* What this device can do */}
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
      {capBadge(caps.simd, "wasm SIMD", "WebAssembly fixed-width SIMD")}
      {capBadge(caps.threads, "threads", caps.threads ? "Cross-origin isolated: multi-threaded WebAssembly available" : "Page isn't cross-origin isolated (no COOP/COEP headers), so WebAssembly runs single-threaded")}
      {capBadge(caps.webgpu, "WebGPU", "navigator.gpu")}
      {capBadge(caps.webnn, "WebNN", "navigator.ml")}
      <span style={{ fontSize: 9, color: t.t2, padding: "1px 4px" }}>{caps.hardwareConcurrency} logical cores</span>
    </div>

    {/* Settings */}
    <div style={card}>
      <div style={h}>Execution provider</div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
        {(["wasm", "webgpu", "webnn"] as BenchEp[]).map((e) => {
          const ok = eps.includes(e);
          return <button key={e} type="button" disabled={!ok || busyHere} title={ok ? EP_LABEL[e] : `${EP_LABEL[e]} isn't available in this browser`}
            onClick={() => setCfg({ ep: e })} style={chip(ep === e, !ok)}>{EP_LABEL[e]}</button>;
        })}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
        {ep === "wasm" && intField("Threads (0 = auto)", cfg.threads, (v) => setCfg({ threads: v }), 0, 64,
          caps.threads ? "WebAssembly threads" : "Single-threaded here: multi-threading needs a cross-origin-isolated page")}
        {ep === "webnn" && <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={lbl}>Device</span>
          <select value={cfg.webnnDevice} disabled={busyHere} onChange={(e) => setCfg({ webnnDevice: e.target.value as WebnnDevice })} style={{ ...num, width: 72 }}>
            <option value="cpu">CPU</option><option value="gpu">GPU</option><option value="npu">NPU</option>
          </select>
        </label>}
        {intField("Warm-up", cfg.warmup, (v) => setCfg({ warmup: v }), 0, 100)}
        {intField("Timed runs", cfg.runs, (v) => setCfg({ runs: v }), 1, 10000)}
        {intField("Budget (s)", Math.round(cfg.timeBudgetMs / 1000), (v) => setCfg({ timeBudgetMs: v * 1000 }), 0, 3600, "Stop timed runs after this many seconds (0 = no limit)")}
      </div>
      {ep === "wasm" && !caps.threads && cfg.threads > 1 && <div style={{ fontSize: 9.5, color: t.wrn, marginTop: 6 }}>This page isn't cross-origin isolated, so the run will use 1 thread.</div>}
      {ep !== "wasm" && <div style={{ fontSize: 9.5, color: t.t2, marginTop: 6 }}>The first {EP_LABEL[ep]} run downloads ONNX Runtime's GPU build ({GPU_WASM_MB} MB, checksum-verified) from cdn.jsdelivr.net. WebAssembly (CPU) needs no download.</div>}

      {/* Inputs and dynamic dims */}
      <div style={{ ...h, marginTop: 10 }}>Inputs</div>
      {specs.length === 0 && <div style={{ fontSize: 10, color: t.t2 }}>Input shapes come from the runtime when the run starts.</div>}
      {specs.map((s) => <div key={s.name} style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap", marginBottom: 4 }}>
        <span style={{ fontSize: 10.5, color: t.t0, fontFamily: "'JetBrains Mono',monospace", maxWidth: 130, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={s.name}>{s.name}</span>
        <span style={{ fontSize: 9, color: t.t3 }}>{s.dtype}</span>
        <span style={{ fontSize: 10.5, color: t.t2 }}>[</span>
        {s.shape.map((d, axis) => typeof d === "number" && d > 0
          ? <span key={axis} style={{ fontSize: 10.5, color: t.t1, fontFamily: "'JetBrains Mono',monospace" }}>{d}{axis < s.shape.length - 1 ? "×" : ""}</span>
          : <span key={axis} style={{ display: "inline-flex", alignItems: "center" }}>
            <input type="number" min={1} aria-label={`${s.name} axis ${axis}`} title={axis === 0 ? "Dynamic batch dim" : `Dynamic dim (axis ${axis})`}
              disabled={busyHere}
              value={cfg.dimOverrides[dimKey(s.name, axis)] ?? defaultDynamicDim(s.shape, axis)}
              onChange={(e) => { const v = Math.floor(Number(e.target.value)); if (v >= 1) setCfg({ dimOverrides: { ...cfg.dimOverrides, [dimKey(s.name, axis)]: v } }); }}
              style={{ ...num, width: 52, borderColor: t.wrn + "88" }} />
            {axis < s.shape.length - 1 && <span style={{ fontSize: 10.5, color: t.t2 }}>×</span>}
          </span>)}
        <span style={{ fontSize: 10.5, color: t.t2 }}>]</span>
      </div>)}
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
        <span style={lbl}>Data</span>
        {(["random", "zeros"] as const).map((f) => <button key={f} type="button" disabled={busyHere} onClick={() => setCfg({ fill: f })} style={chip(cfg.fill === f)}>{f}</button>)}
        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: t.t2, cursor: "pointer" }}>
          <input type="checkbox" checked={cfg.profile} disabled={busyHere} onChange={(e) => setCfg({ profile: e.target.checked })} />
          Per-op profile
        </label>
      </div>
    </div>

    {/* Run / cancel + progress */}
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      {!busyHere
        ? <button type="button" onClick={start} disabled={!!active} title={active ? "Another benchmark is running" : undefined}
          style={{ padding: "6px 16px", borderRadius: 5, border: "none", background: t.acc, color: "#fff", fontSize: 11, fontWeight: 700, cursor: active ? "not-allowed" : "pointer", opacity: active ? .5 : 1 }}>Run benchmark</button>
        : <button type="button" onClick={() => benchStore.cancel()} disabled={active?.cancelling}
          style={{ padding: "6px 16px", borderRadius: 5, border: `1px solid ${t.err}`, background: t.err + "18", color: t.err, fontSize: 11, fontWeight: 700, cursor: "pointer" }}>{active?.cancelling ? "Cancelling…" : "Cancel"}</button>}
      {busyHere && <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 9.5, color: t.t2, marginBottom: 3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {active?.progress ? `${PHASE_LABEL[active.progress.phase]}${active.progress.total > 1 ? ` ${active.progress.done}/${active.progress.total}` : ""}${active.progress.lastMs != null ? ` · ${fmtMs(active.progress.lastMs)}` : ""}` : "Loading onnxruntime…"}
        </div>
        <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct * 100)} style={{ height: 4, borderRadius: 2, background: t.bg3, overflow: "hidden" }}>
          <div style={{ width: `${pct * 100}%`, height: "100%", background: t.acc, transition: "width .2s" }} />
        </div>
      </div>}
    </div>
    {error && <div role="alert" style={{ ...card, borderColor: t.err + "66", color: t.err, fontSize: 10.5, lineHeight: 1.5, wordBreak: "break-word" }}>{error}</div>}

    {/* Latest result */}
    {latest && resultTiles(latest, t)}

    {/* Prediction vs measured */}
    <div style={card}>
      <div style={h}>Measured vs predicted</div>
      <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <span style={lbl}>Reference device</span>
        <select value={st.refDevice} onChange={(e) => benchStore.setRefDevice(e.target.value)} style={{ ...num, width: "auto", maxWidth: 200, fontFamily: "inherit" }}>
          {HW.map((d) => <option key={d.id} value={d.id}>{d.n}</option>)}
        </select>
      </label>
      {prediction && <div style={{ fontSize: 10.5, color: t.t1, lineHeight: 1.6 }}>
        Roofline prediction on <b>{prediction.deviceName}</b> ({prediction.precision}, {prediction.bound}-bound{prediction.calibrated ? ", calibrated" : ", estimated"}):{" "}
        <b style={{ fontFamily: "'JetBrains Mono',monospace", color: t.t0 }}>{fmtMs(prediction.latencyMs)}</b> · {prediction.fps.toFixed(prediction.fps >= 100 ? 0 : 1)} FPS
        {latest && latest.latency.count > 0 && <div style={{ color: t.t2 }}>
          This device (median): <b style={{ fontFamily: "'JetBrains Mono',monospace", color: t.t0 }}>{fmtMs(latest.latency.median)}</b>{" "}
          → {ratioText(latest.latency.median, prediction.latencyMs)}
        </div>}
        <div style={{ fontSize: 9, color: t.t3, marginTop: 2 }}>The prediction is for the edge device at its deployment precision; the measurement is this machine running fp32/fp16 in the browser. Use it to sanity-check the estimate's order of magnitude.</div>
      </div>}

      {geo.bars.length > 0 && <svg width="100%" viewBox={`0 0 ${LABEL_W + CHART_W + 56} ${geo.height + 18}`} role="img" aria-label="Median latency per run, lower is better" style={{ marginTop: 8, display: "block" }}>
        {geo.ticks.map((k) => <g key={k.x}>
          <line x1={LABEL_W + k.x} x2={LABEL_W + k.x} y1={0} y2={geo.height} stroke={t.bdr} strokeWidth={1} />
          <text x={LABEL_W + k.x} y={geo.height + 12} fontSize={8} fill={t.t3} textAnchor="middle">{fmtMs(k.ms)}</text>
        </g>)}
        {geo.bars.map((b) => {
          const pred = b.kind === "predicted";
          return <g key={b.id}>
            <title>{`${b.label}: median ${fmtMs(b.value)}${pred ? " (roofline prediction)" : `, p90 ${fmtMs(b.p90)}`}`}</title>
            <text x={LABEL_W - 4} y={b.y + 10} fontSize={9} fill={t.t1} textAnchor="end">{b.label.length > 18 ? b.label.slice(0, 17) + "…" : b.label}</text>
            <rect x={LABEL_W} y={b.y} width={Math.max(2, b.w)} height={14} rx={4} ry={4}
              fill={pred ? "transparent" : t.acc} stroke={pred ? t.t2 : "none"} strokeWidth={pred ? 1.5 : 0} strokeDasharray={pred ? "4 3" : undefined} />
            {!pred && b.p90x > b.w && <line x1={LABEL_W + b.w} x2={LABEL_W + b.p90x} y1={b.y + 7} y2={b.y + 7} stroke={t.t2} strokeWidth={1.5} />}
            <text x={LABEL_W + Math.max(b.w, b.p90x) + 4} y={b.y + 10} fontSize={9} fill={t.t1} fontFamily="'JetBrains Mono',monospace">{fmtMs(b.value)}</text>
          </g>;
        })}
      </svg>}
    </div>

    {/* History */}
    {runs.length > 0 && <div style={card}>
      <div style={{ ...h, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span>Runs this session ({runs.length})</span>
        <span style={{ display: "flex", gap: 4, textTransform: "none", letterSpacing: 0 }}>
          <button type="button" onClick={save} style={chip(true)}>Save result</button>
          <button type="button" onClick={() => benchStore.clear(key)} disabled={busyHere} style={chip(false)}>Clear</button>
        </span>
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
        <thead><tr style={{ color: t.t2, textAlign: "right" }}>
          <th style={{ textAlign: "left", fontWeight: 600 }}>Run</th><th style={{ fontWeight: 600 }}>median</th><th style={{ fontWeight: 600 }}>p90</th><th style={{ fontWeight: 600 }}>inf/s</th><th style={{ fontWeight: 600 }}>session</th><th />
        </tr></thead>
        <tbody>{runs.map((r, i) => <tr key={r.id} style={{ color: t.t1, textAlign: "right", fontFamily: "'JetBrains Mono',monospace", borderTop: `1px solid ${t.bdr}` }}>
          <td style={{ textAlign: "left", padding: "3px 0" }} title={`${r.startedAt}${r.cancelled ? " (cancelled)" : ""}`}>{runLabel(r, i)}{r.cancelled ? "*" : ""}</td>
          <td>{fmtMs(r.latency.median)}</td><td>{fmtMs(r.latency.p90)}</td><td>{r.latency.throughput.toFixed(r.latency.throughput >= 100 ? 0 : 1)}</td><td>{fmtMs(r.sessionCreateMs)}</td>
          <td><button type="button" aria-label={`Remove run ${i + 1}`} onClick={() => benchStore.remove(key, r.id)} style={{ border: "none", background: "transparent", color: t.t3, cursor: "pointer", fontSize: 10 }}>✕</button></td>
        </tr>)}</tbody>
      </table>
    </div>}

    {/* Per-op profile of the latest run */}
    {latest?.profile && <div style={card}>
      <div style={h}>Per-op profile ({latest.profile.runs} runs, host-side kernel time)</div>
      {latest.profile.rows.length === 0
        ? <div style={{ fontSize: 10, color: t.t2 }}>{latest.profile.note}</div>
        : <>
          {byOpType(latest.profile.rows).slice(0, showAll ? undefined : 8).map((o) => <div key={o.op} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
            <span style={{ width: 92, fontSize: 10, color: t.t1, fontFamily: "'JetBrains Mono',monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={o.op}>{o.op}</span>
            <div style={{ flex: 1, height: 6, background: t.bg3, borderRadius: 3 }}><div style={{ width: `${Math.max(1, o.share * 100)}%`, height: "100%", background: t.acc2, borderRadius: 3 }} /></div>
            <span style={{ width: 92, textAlign: "right", fontSize: 9.5, color: t.t2, fontFamily: "'JetBrains Mono',monospace" }}>{fmtMs(o.meanMs)} · {(o.share * 100).toFixed(0)}% · {o.nodes}×</span>
          </div>)}
          {byOpType(latest.profile.rows).length > 8 && <button type="button" onClick={() => setShowAll((v) => !v)} style={{ ...chip(false), marginTop: 2 }}>{showAll ? "Show fewer" : "Show all op types"}</button>}
        </>}
    </div>}
  </div>;
}

function ratioText(measured: number, predicted: number): string {
  if (!(measured > 0) || !(predicted > 0)) return "–";
  const r = measured / predicted;
  return r >= 1 ? `${r.toFixed(r >= 10 ? 0 : 1)}× slower than predicted` : `${(1 / r).toFixed(1 / r >= 10 ? 0 : 1)}× faster than predicted`;
}

/** Stat tiles for one run (a render helper, not a component). */
function resultTiles(r: BenchResult, t: ThemePalette) {
  const L = r.latency;
  const tiles: [string, string, string?][] = [
    ["Median", fmtMs(L.median)], ["Mean", fmtMs(L.mean)], ["p90", fmtMs(L.p90)], ["p99", fmtMs(L.p99)],
    ["Min", fmtMs(L.min)], ["Max", fmtMs(L.max)], ["Std dev", fmtMs(L.std)], ["Throughput", `${L.throughput.toFixed(L.throughput >= 100 ? 0 : 1)}/s`, "Sequential inferences per second (1000 / mean)"],
    ["Session create", fmtMs(r.sessionCreateMs), "InferenceSession.create for this model"], ["First inference", fmtMs(r.firstInferenceMs), "Cold first run"],
    ["Runtime init", fmtMs(r.runtimeInitMs), "WebAssembly load/compile + execution-provider start-up"],
    ["Peak JS heap", r.memory.jsHeapPeakBytes != null ? fB(r.memory.jsHeapPeakBytes) : r.memory.uaSpecificBytes != null ? fB(r.memory.uaSpecificBytes) : "n/a", r.memory.note ?? "Sampled in the benchmark worker"],
  ];
  return <div style={{ background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "8px 10px" }}>
    <div style={{ fontSize: 10, fontWeight: 700, color: t.t1, textTransform: "uppercase", letterSpacing: .5, marginBottom: 6 }}>
      Latest: {r.config.ep === "webnn" ? `WebNN (${r.config.webnnDevice})` : r.config.ep === "webgpu" ? "WebGPU" : `WebAssembly · ${r.env.effectiveThreads} thread${r.env.effectiveThreads === 1 ? "" : "s"}`} · {L.count} runs{r.cancelled ? " (cancelled)" : ""}
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(82px, 1fr))", gap: 4 }}>
      {tiles.map(([k, v, title]) => <div key={k} title={title} style={{ background: t.bg1, border: `1px solid ${t.bdr}`, borderRadius: 4, padding: "4px 6px" }}>
        <div style={{ fontSize: 8.5, color: t.t2, textTransform: "uppercase", letterSpacing: .4 }}>{k}</div>
        <div style={{ fontSize: 12.5, fontWeight: 700, color: t.t0, fontFamily: "'JetBrains Mono',monospace" }}>{v}</div>
      </div>)}
    </div>
    <div style={{ fontSize: 9.5, color: t.t2, marginTop: 6, lineHeight: 1.5, wordBreak: "break-word" }}>
      Inputs: {r.inputs.map((i) => `${i.name}[${i.dims.join("×")}] ${i.dtype}`).join(", ") || "none"}
      {r.env.webgpuAdapter ? ` · GPU: ${r.env.webgpuAdapter}` : ""}{r.env.ortVersion ? ` · onnxruntime-web ${r.env.ortVersion}` : ""}
    </div>
    {r.warnings.map((w) => <div key={w} style={{ fontSize: 9.5, color: t.wrn, marginTop: 3 }}>{w}</div>)}
  </div>;
}
