import { useEffect, useRef, useState } from "react";
import type { Model } from "@modelvisio/parsers";
import { proxyLlm, type AgentRun, type AgentStep } from "@modelvisio/ai";
import { useT } from "../theme/ThemeContext";
import { Badge } from "./Badge";
import type { FixId } from "../fixes/transforms";
import { isDesktop, getUserApiKey, setUserApiKey } from "../utils/apiKey";
import { downloadFile } from "../utils/download";
import { runDeployment } from "../deploy/agent";
import { packageToZip, type DeployPackage } from "../deploy/package";

const QUICK_GOALS = [
  "Deploy to a Jetson Orin Nano at ≥30 FPS, INT8 OK",
  "Ship to Android phones via TFLite",
  "Raspberry Pi with Hailo-8L, lowest power",
  "Intel Core Ultra NPU via OpenVINO, FP16 only",
  "Cheapest device that hits 15 FPS",
  "Coral Edge TPU under 50 ms",
];

const mono = "'JetBrains Mono',monospace";
const f1 = (n: number) => (isFinite(n) ? (n >= 100 ? n.toFixed(0) : n.toFixed(1)) : "∞");

/**
 * AI deployment agent. The user states a goal; the agent (Gemini via the
 * server-side /api/chat proxy, JSON action protocol) calls ModelVisio's own
 * analysis tools step by step, then a reviewable package is built: decision +
 * justification, conversion/runtime scripts, validation checklist and risks.
 * Without AI (no desktop key, proxy error, AI off) the deterministic planner
 * runs the same tools in a fixed order, so a package is always produced.
 */
export function DeployAgent({ model, onApplyFix }: { model: Model | null; onApplyFix?: (id: FixId) => void }) {
  const t = useT();
  const [goal, setGoal] = useState("");
  const [useAi, setUseAi] = useState(true);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [run, setRun] = useState<AgentRun | null>(null);
  const [pkg, setPkg] = useState<DeployPackage | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [openStep, setOpenStep] = useState<number | null>(null);
  const [file, setFile] = useState<string>("README.md");
  const [copied, setCopied] = useState<string | null>(null);
  const [appliedToGraph, setAppliedToGraph] = useState<FixId[]>([]);
  const ctrl = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" }); }, [steps.length]);
  useEffect(() => () => ctrl.current?.abort(), []);

  // Desktop BYO-key (same storage + UX as the Chat copilot).
  const desktop = isDesktop();
  const [showKey, setShowKey] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [hasKey, setHasKey] = useState(() => !!getUserApiKey());
  const saveKey = () => { setUserApiKey(keyInput); setHasKey(!!keyInput.trim()); setShowKey(false); };
  const clearKey = () => { setUserApiKey(""); setHasKey(false); setKeyInput(""); };

  const start = async (text: string) => {
    const g = text.trim();
    if (!g || running || !model) return;
    setGoal(g);
    const noKey = desktop && !getUserApiKey();
    const llm = useAi && !noKey ? proxyLlm() : null;
    const noAiReason = !useAi ? "AI turned off: used the deterministic planner."
      : noKey ? "No Gemini API key set: used the deterministic planner. Add your key (key icon) to use the AI agent." : undefined;
    ctrl.current?.abort();
    const ac = new AbortController();
    ctrl.current = ac;
    setRunning(true); setErr(null); setSteps([]); setRun(null); setPkg(null); setOpenStep(null); setAppliedToGraph([]);
    try {
      const res = await runDeployment({
        model, goal: g, llm, noAiReason, signal: ac.signal,
        onStep: (s) => setSteps((p) => [...p, s]),
        onStatus: setStatus,
      });
      setRun(res.run); setPkg(res.pkg); setFile("README.md");
    } catch (e) {
      if (!(e instanceof Error && e.name === "AbortError")) setErr(e instanceof Error ? e.message : String(e));
      else setErr("Stopped.");
    } finally {
      if (ctrl.current === ac) ctrl.current = null;
      setRunning(false); setStatus("");
    }
  };
  const stop = () => ctrl.current?.abort();

  const copy = (key: string, text: string) => { navigator.clipboard?.writeText(text); setCopied(key); setTimeout(() => setCopied(null), 1500); };
  const current = pkg?.files.find((f) => f.path === file) ?? pkg?.files[0];

  const panel = { background: t.glass, border: `1px solid ${t.glassBdr}`, borderRadius: 6, padding: 8 } as const;
  const label = (s: string) => <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1, marginBottom: 5 }}>{s}</div>;
  const smallBtn = (txt: string, on: () => void, color = t.acc, disabled = false) => (
    <button type="button" disabled={disabled} onClick={on} style={{ padding: "3px 9px", borderRadius: 4, border: `1px solid ${color}55`, background: color + "14", color, fontSize: 9.5, fontWeight: 600, cursor: disabled ? "default" : "pointer", opacity: disabled ? 0.5 : 1, whiteSpace: "nowrap" }}>{txt}</button>
  );

  return <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%", overflow: "auto" }}>
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>AI Deployment Agent</div>
      <span style={{ fontSize: 8.5, color: t.t3 }}>plans · checks · converts · validates</span>
      <label style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 4, fontSize: 9.5, color: t.t2, cursor: "pointer" }}>
        <input type="checkbox" checked={useAi} onChange={(e) => setUseAi(e.target.checked)} style={{ accentColor: t.acc }} />Use AI
      </label>
      {desktop && <button type="button" onClick={() => { setKeyInput(getUserApiKey()); setShowKey((s) => !s); }} title={hasKey ? "Gemini API key set. Click to change" : "Set your Gemini API key"}
        style={{ padding: "2px 6px", borderRadius: 5, border: `1px solid ${hasKey ? t.bdr : t.acc}`, background: "transparent", color: hasKey ? t.t2 : t.acc, fontSize: 9, fontWeight: 600, cursor: "pointer" }}>🔑{hasKey ? "" : " Set key"}</button>}
    </div>

    {desktop && showKey && <div style={{ ...panel, display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ fontSize: 9, color: t.t2, lineHeight: 1.5 }}>Paste your own <b>Gemini API key</b>. It is stored only on this device. Free key at <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer noopener" style={{ color: t.acc }}>aistudio.google.com/apikey</a>. Without a key the deterministic planner runs instead.</div>
      <div style={{ display: "flex", gap: 4 }}>
        <input type="password" value={keyInput} onChange={(e) => setKeyInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && saveKey()} placeholder="AIza…" style={{ flex: 1, padding: "6px 8px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 10, outline: "none" }} />
        <button type="button" onClick={saveKey} style={{ padding: "6px 12px", borderRadius: 5, border: "none", background: t.acc, color: "#fff", fontSize: 10, fontWeight: 600, cursor: "pointer" }}>Save</button>
        {hasKey && <button type="button" onClick={clearKey} style={{ padding: "6px 10px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 10, cursor: "pointer" }}>Clear</button>}
      </div>
    </div>}

    {!model ? <div style={{ padding: 20, color: t.t3, fontSize: 11, textAlign: "center" }}>Load a model to plan a deployment.</div> : <>
      {/* Goal input */}
      <div style={{ display: "flex", gap: 4 }}>
        <textarea value={goal} onChange={(e) => setGoal(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void start(goal); }}
          rows={2} placeholder="Describe the deployment goal, e.g. “Jetson Orin Nano at ≥30 FPS, INT8 OK”" aria-label="Deployment goal"
          style={{ flex: 1, padding: "6px 8px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t0, fontSize: 10.5, outline: "none", resize: "vertical", fontFamily: "inherit" }} />
        {running
          ? <button type="button" onClick={stop} style={{ padding: "6px 12px", borderRadius: 5, border: `1px solid ${t.err}`, background: t.err + "18", color: t.err, fontSize: 10, fontWeight: 600, cursor: "pointer" }}>Stop</button>
          : <button type="button" onClick={() => void start(goal)} disabled={!goal.trim()} style={{ padding: "6px 12px", borderRadius: 5, border: "none", background: goal.trim() ? t.acc : t.bg3, color: "#fff", fontSize: 10, fontWeight: 600, cursor: goal.trim() ? "pointer" : "default" }}>Run agent</button>}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
        {QUICK_GOALS.map((q) => <button key={q} type="button" disabled={running} onClick={() => void start(q)} style={{ padding: "2px 8px", borderRadius: 8, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 9, cursor: running ? "default" : "pointer" }}>{q}</button>)}
      </div>
      {desktop && !hasKey && useAi && <div style={{ fontSize: 9, color: t.t2 }}>No Gemini key on this device: runs will use the deterministic planner (same tools, fixed order).</div>}

      {/* Live transcript */}
      {(steps.length > 0 || running) && <div style={panel}>
        {label(`Plan · ${steps.length} step${steps.length === 1 ? "" : "s"}${run ? ` · ${run.mode === "ai" ? "AI agent" : "deterministic planner"}` : ""}`)}
        {run?.fallbackReason && <div style={{ fontSize: 9.5, color: t.wrn, marginBottom: 5, lineHeight: 1.45 }}>⚠ {run.fallbackReason}</div>}
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          {steps.map((s) => <div key={s.index} onClick={() => setOpenStep(openStep === s.index ? null : s.index)} style={{ cursor: "pointer", borderLeft: `2px solid ${s.ok ? (s.source === "ai" ? t.acc : t.acc3) : t.err}`, paddingLeft: 6 }}>
            <div style={{ display: "flex", gap: 6, alignItems: "baseline", flexWrap: "wrap" }}>
              <span style={{ fontSize: 9, color: t.t3, fontFamily: mono }}>{s.index}</span>
              <span style={{ fontSize: 10, fontWeight: 600, color: t.t0, fontFamily: mono }}>{s.tool}</span>
              <Badge c={s.source === "ai" ? t.acc : t.acc3}>{s.source === "ai" ? "AI" : "planner"}</Badge>
              {s.thought && <span style={{ fontSize: 9, color: t.t2 }}>{s.thought}</span>}
            </div>
            <div style={{ fontSize: 9.5, color: s.ok ? t.t1 : t.err, fontFamily: mono, lineHeight: 1.45 }}>→ {s.ok ? s.digest : s.error}</div>
            {openStep === s.index && <pre style={{ margin: "3px 0 0", padding: 6, borderRadius: 4, background: t.bg, border: `1px solid ${t.bdr}`, fontSize: 9, color: t.t1, fontFamily: mono, whiteSpace: "pre-wrap", maxHeight: 220, overflow: "auto" }}>
              {`args: ${JSON.stringify(s.args)}\n${s.ok ? JSON.stringify(s.result, null, 2) : s.error}`}
            </pre>}
          </div>)}
          {running && <div style={{ fontSize: 10, color: t.t3 }}><span style={{ animation: "pulse 1.2s infinite", color: t.acc }}>●</span> {status || "Starting…"}</div>}
          <div ref={end} />
        </div>
      </div>}

      {err && <div style={{ fontSize: 10, color: err === "Stopped." ? t.t2 : t.err }}>{err}</div>}

      {/* Package */}
      {pkg && <>
        <div style={{ ...panel, borderColor: t.acc + "55" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: t.t0 }}>{pkg.decision.device.n} · {pkg.decision.runtimeLabel} · {pkg.decision.precision.toUpperCase()}</div>
            <div style={{ display: "flex", gap: 4 }}>
              {smallBtn("⬇ Download ZIP", () => { const z = packageToZip(pkg); downloadFile(z.data, z.filename, "application/zip"); }, t.acc2)}
            </div>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", fontSize: 9.5, color: t.t2, marginTop: 4, fontFamily: mono }}>
            <span style={{ color: t.acc }}>~{f1(pkg.estimate.fps)} FPS</span>
            <span>{isFinite(pkg.estimate.latencyMs) ? pkg.estimate.latencyMs.toFixed(2) : "∞"} ms</span>
            <span>{isFinite(pkg.estimate.energyMj) ? pkg.estimate.energyMj.toFixed(1) : "–"} mJ/inf</span>
            <span>{pkg.estimate.bound}-bound</span>
            <span style={{ color: pkg.estimate.fits ? t.t2 : t.err }}>{pkg.estimate.fits ? "fits" : "won't fit"}</span>
            <span>{pkg.estimate.confidence} confidence{pkg.estimate.approx ? " · approx peak" : ""}</span>
          </div>
          <div style={{ fontSize: 10, color: t.t1, lineHeight: 1.55, marginTop: 6 }}>{pkg.decision.justification}</div>
          {pkg.decision.adjustments.length > 0 && <div style={{ marginTop: 4, fontSize: 9, color: t.wrn, lineHeight: 1.45 }}>{pkg.decision.adjustments.map((a, i) => <div key={i}>↻ {a}</div>)}</div>}
          {pkg.fixes.length > 0 && <div style={{ marginTop: 6, display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 9, color: t.t3 }}>Fixes in this plan:</span>
            {pkg.fixes.map((f) => <span key={f.id}>{smallBtn(appliedToGraph.includes(f.id) ? `✓ ${f.title}` : `Apply ${f.title} to graph`, () => { onApplyFix?.(f.id); setAppliedToGraph((p) => [...p, f.id]); }, t.acc3, !onApplyFix || appliedToGraph.includes(f.id))}</span>)}
          </div>}
        </div>

        <div style={panel}>
          {label("Files")}
          <div style={{ display: "flex", gap: 3, flexWrap: "wrap", marginBottom: 6 }}>
            {pkg.files.map((f) => <button key={f.path} type="button" title={f.description} onClick={() => setFile(f.path)} style={{ padding: "2px 7px", borderRadius: 4, border: `1px solid ${current?.path === f.path ? t.acc : t.bdr}`, background: current?.path === f.path ? t.acc + "18" : "transparent", color: current?.path === f.path ? t.acc : t.t2, fontSize: 9, fontFamily: mono, cursor: "pointer" }}>{f.path}</button>)}
          </div>
          {current && <div style={{ position: "relative" }}>
            <div style={{ position: "absolute", top: 6, right: 6, display: "flex", gap: 4, zIndex: 2 }}>
              {smallBtn(copied === current.path ? "✓ Copied" : "Copy", () => copy(current.path, current.content), t.t2)}
              {smallBtn("⬇ File", () => downloadFile(current.content, current.path.split("/").pop() || current.path, "text/plain"), t.t2)}
            </div>
            <pre style={{ margin: 0, background: t.bg, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: 10, paddingTop: 30, fontSize: 9.5, color: t.t1, fontFamily: mono, lineHeight: 1.5, whiteSpace: "pre-wrap", maxHeight: 340, overflow: "auto" }}>{current.content}</pre>
          </div>}
        </div>

        <div style={panel}>
          {label("Validation checklist")}
          {pkg.checklist.map((c, i) => <div key={i} style={{ marginBottom: 6 }}>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}><Badge c={c.kind === "accuracy" ? t.acc2 : c.kind === "latency" ? t.acc : t.t2}>{c.kind}</Badge><span style={{ fontSize: 10, fontWeight: 600, color: t.t0 }}>{c.title}</span></div>
            <div style={{ fontSize: 9.5, color: t.t2, lineHeight: 1.45, marginTop: 2 }}>{c.detail}</div>
            {c.command && <div style={{ display: "flex", gap: 4, alignItems: "flex-start", marginTop: 2 }}>
              <code style={{ flex: 1, fontSize: 9, color: t.t1, fontFamily: mono, background: t.bg, border: `1px solid ${t.bdr}`, borderRadius: 4, padding: "3px 6px", whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{c.command}</code>
              {smallBtn(copied === `cmd${i}` ? "✓" : "Copy", () => copy(`cmd${i}`, c.command!), t.t2)}
            </div>}
          </div>)}
        </div>

        <div style={{ ...panel, borderColor: t.wrn + "44" }}>
          {label("Risks")}
          {pkg.risks.length === 0 ? <div style={{ fontSize: 9.5, color: t.t2 }}>None identified beyond the validation checklist.</div>
            : pkg.risks.map((r, i) => <div key={i} style={{ fontSize: 9.5, color: t.t1, lineHeight: 1.5 }}><span style={{ color: t.wrn }}>⚠</span> {r}</div>)}
        </div>
      </>}

      {!running && steps.length === 0 && !pkg && <div style={{ fontSize: 9.5, color: t.t3, lineHeight: 1.55 }}>
        The agent calls ModelVisio's own analysis tools (model summary, device ranking, compatibility, roofline latency, auto-fixes, recipe generator, fleet simulation) and returns a reviewable package: chosen target/runtime/precision with justification, conversion + inference scripts, an accuracy and latency validation checklist, and risks, as a ZIP or individual files. Fixes are applied to a working copy; you choose whether to apply them to the graph.
      </div>}
    </>}
  </div>;
}
