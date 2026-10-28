import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FORMAT_SUPPORT, pickPrimary, type Companions, type Model, type ModelLayer } from "@modelvisio/parsers";
import { parseBufferInWorker, parseBufferStreaming, shouldStream } from "./workers/parseClient";
import { PartialModelBuilder, throttleFrames } from "./workers/streaming";
import { ThemeCtx } from "./theme/ThemeContext";
import { TH, type ThemeName } from "./theme/theme";
import { fB, fmt } from "./utils/format";
import { FORMATS } from "./data/hardware";
import { demoModel } from "./demo/demoModel";
import { Badge } from "./components/Badge";
import { ToolStages, ToolTabs } from "./components/ToolNav";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { Icon } from "./components/Icon";
import { ALL_TOOLS, TOOL_GROUPS } from "./data/tools";
import { APP_VERSION } from "./data/version";
import { GraphPanel } from "./components/Graph/GraphPanel";
import { Inspector } from "./components/Inspector";
import { CompilerChecker } from "./components/CompilerChecker";
import { HWReport } from "./components/HWReport";
import { Converter } from "./components/Converter";
import { DeployRecipe } from "./components/DeployRecipe";
import { Chat } from "./components/Chat";
import { QuantHeatmap } from "./components/QuantHeatmap";
import { ModelAdvisor } from "./components/ModelAdvisor";
import { Benchmark } from "./components/Benchmark";
import { Investigator } from "./components/Investigator";
import { OptimizationSearch } from "./components/OptimizationSearch";
import { DeployAgent } from "./components/DeployAgent";
import { FleetSim } from "./components/FleetSim";
import { ModelRegistry } from "./components/ModelRegistry";
import type { ColorMode } from "./components/Graph/colorBy";
import { logUpload } from "@modelvisio/ai";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { BuildingPage } from "./components/BuildingPage";
import { applyFix, type FixId } from "./fixes/transforms";
import { useViewport } from "./hooks/useViewport";
import { GITHUB_URL, AUTHOR, DESKTOP_BUILDS, DESKTOP_RELEASES_URL } from "./data/links";
import { isDesktop, isVsCodeWebview } from "./utils/apiKey";
import { detectOS } from "./utils/platform";
import { OsIcon } from "./components/OsIcon";
import { DownloadPanel } from "./components/DownloadPanel";
import { MacInstallHelp } from "./components/MacInstallHelp";
import logoUrl from "./assets/logo.png";
import graphHexUrl from "./assets/graph-hex.png";

// Plus weight-only companions picked alongside their graph file.
const ACCEPT = [...FORMAT_SUPPORT.flatMap((f) => f.exts), "pdiparams", "index", "data-00000-of-00001"].map((e) => "." + e).join(",");

// GitHub "mark" glyph, reused in the header and landing page.
const GH_PATH = "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z";
const GhIcon = ({ size = 15 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d={GH_PATH} /></svg>
);

// One-line summary per workflow stage, for the landing page.
const STAGE_BLURB: Record<string, string> = {
  analyze: "An interactive graph with a per-layer inspector, compiler pre-flight for each target, and a hardware-fit score for 21 edge accelerators.",
  optimize: "Ranked fixes with estimated latency and size impact, bottleneck investigation, a Pareto search and on-device benchmarks.",
  ship: "Format conversion, ready-to-run TensorRT, HailoRT and RKNN scripts, a deployment agent, fleet simulation and a model registry.",
};

/** Imperative handle a shell can use to drive the app (e.g. native menus). */
/** `companions`: sibling files the format keeps its weights in (e.g. OpenVINO .bin). */
export type AppApi = { openFile: (f: File, companions?: File[]) => void };

/**
 * Top-level ModelVisio UI. Owns theme + loaded-model state and composes
 * the graph canvas with the analysis sidebar. Mounted by all three shells.
 */
export function App({
  initialTheme = "dark" as ThemeName,
  onReady,
  themeOverride,
}: {
  initialTheme?: ThemeName;
  onReady?: (api: AppApi) => void;
  /** Controlled theme from the host (e.g. VS Code editor theme). Live-synced. */
  themeOverride?: ThemeName;
}) {
  const [theme, setTheme] = useState<ThemeName>(themeOverride ?? initialTheme);
  useEffect(() => { if (themeOverride) setTheme(themeOverride); }, [themeOverride]);
  const t = TH[theme];
  const [model, setModel] = useState<Model | null>(null);
  const [modelBytes, setModelBytes] = useState<ArrayBuffer | null>(null);
  const [sel, setSel] = useState<ModelLayer | null>(null);
  // Undo stack of pre-fix model snapshots, so applied compiler fixes are reversible.
  const [fixUndo, setFixUndo] = useState<Model[]>([]);
  const [rightTab, setRightTab] = useState("inspector");
  const [graphColor, setGraphColor] = useState<ColorMode>("none");
  const [drag, setDrag] = useState(false);
  // When set, shows the full-screen "Building" (coming-soon) page for a surface
  // that isn't shipped yet, e.g. the desktop app or VS Code extension.
  const [building, setBuilding] = useState<string | null>(null);
  const [sidebarW, setSidebarW] = useState(400);
  const [resizeHover, setResizeHover] = useState(false);
  const { vw, narrow, compact } = useViewport();
  // Side-by-side: never let the sidebar starve the graph; keep at least 380px for it.
  const effSidebarW = Math.min(sidebarW, Math.max(260, vw - 380));

  const startResize = (e: React.MouseEvent) => {
    e.preventDefault();
    const move = (ev: MouseEvent) => setSidebarW(Math.max(300, Math.min(760, window.innerWidth - ev.clientX)));
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    document.body.style.userSelect = "none";
  };
  const [palette, setPalette] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const [parsing, setParsing] = useState(false);
  // Desktop download: one-click installer for the visitor's OS (SSR / mobile /
  // unknown → the releases page), plus an "Other platforms" panel.
  const os = useMemo(() => detectOS(), []);
  const build = DESKTOP_BUILDS.find((b) => b.os === os);
  const [dlPanel, setDlPanel] = useState(false);
  const [macHelp, setMacHelp] = useState(false);
  const closeDlPanel = useCallback(() => setDlPanel(false), []);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Progressive load of a large model: progress while chunks stream in, plus a
  // load token so a newer load (or closing the model) discards stale updates.
  const [streamProg, setStreamProg] = useState<{ pct: number; layers: number } | null>(null);
  const loadSeq = useRef(0);
  const cancelLoad = () => { loadSeq.current++; setStreamProg(null); setParsing(false); };

  const loadDemo = () => { cancelLoad(); setModel(demoModel); setModelBytes(null); setSel(null); setLoadError(null); setFixUndo([]); };
  const clearModel = () => { cancelLoad(); setModel(null); setModelBytes(null); setSel(null); setFixUndo([]); };

  // Apply a compiler auto-fix to the in-memory model (graph/stats/issues update
  // live), pushing a snapshot so it can be undone. Updaters stay pure (no nested
  // setState) so StrictMode's double-invoke can't double-push the undo stack.
  const onApplyFix = useCallback((id: FixId) => {
    if (!model) return;
    setFixUndo((h) => [...h, model]);
    setModel(applyFix(model, id));
    setSel(null);
  }, [model]);
  const onUndoFix = useCallback(() => {
    if (fixUndo.length === 0) return;
    setModel(fixUndo[fixUndo.length - 1]);
    setFixUndo((h) => h.slice(0, -1));
    setSel(null);
  }, [fixUndo]);

  const handleFile = useCallback(async (f?: File, extra: File[] = []) => {
    if (!f) return;
    const seq = ++loadSeq.current;
    const live = () => seq === loadSeq.current;
    setLoadError(null);
    setParsing(true);
    setStreamProg(null);
    let flush: ReturnType<typeof throttleFrames> | null = null;
    let shown = false; // a partial model of this load is on screen
    try {
      // Read the bytes once: parse them, and retain them for conversion/export.
      const buf = await f.arrayBuffer();
      const companions: Companions | undefined = extra.length
        ? Object.fromEntries(await Promise.all(extra.map(async (c) => [c.webkitRelativePath || c.name, await c.arrayBuffer()] as const)))
        : undefined;
      if (!live()) return;
      let m: Model;
      if (shouldStream(buf.byteLength)) {
        // Large model: show layers as they are parsed. Partial snapshots render
        // at most ~4x/sec; the final Model replaces them when parsing completes.
        const acc = new PartialModelBuilder();
        setStreamProg({ pct: 0, layers: 0 });
        flush = throttleFrames(() => {
          if (!live()) return;
          setStreamProg({ pct: Math.round(acc.progress * 100), layers: acc.layers.length });
          if (!acc.layers.length) return;
          if (!shown) { shown = true; setSel(null); setFixUndo([]); setModelBytes(null); }
          setModel(acc.snapshot());
        }, 250);
        const tick = flush;
        m = await parseBufferStreaming(buf, f.name, (c) => { if (live()) { acc.push(c); tick(); } }, companions);
        flush.cancel();
      } else {
        m = await parseBufferInWorker(buf, f.name, companions);
      }
      if (!live()) return;
      const final = m;
      setModel(final); setModelBytes(buf); setFixUndo([]);
      // Keep a node picked while streaming, re-pointed at the final layer object.
      setSel((s) => (s && shown ? final.layers.find((l) => l.id === s.id) ?? null : null));
      logUpload({
        modelName: m.name, format: m.format, framework: m.framework,
        sizeBytes: m.sizeBytes, params: m.layers.reduce((s, l) => s + l.params, 0),
        layers: m.layers.length,
      });
    } catch (err) {
      if (!live()) return;
      // Drop a half-streamed graph so the landing screen shows the error.
      if (shown) { setModel(null); setSel(null); }
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      flush?.cancel();
      if (live()) { setParsing(false); setStreamProg(null); }
    }
  }, []);

  // Expose an imperative handle so shells (e.g. the Tauri native menu) can open
  // files without reaching into the UI.
  // Several files opened together (multi-select / drop): parse the graph file,
  // pass the rest as its weight companions.
  const handleFiles = useCallback((list?: FileList | File[] | null) => {
    const files = Array.from(list ?? []);
    const primary = pickPrimary(files.map((x) => x.name));
    const f = files.find((x) => x.name === primary);
    void handleFile(f, files.filter((x) => x !== f));
  }, [handleFile]);

  useEffect(() => { onReady?.({ openFile: (f: File, companions?: File[]) => { void handleFile(f, companions); } }); }, [onReady, handleFile]);

  const totalP = model ? model.layers.reduce((s, l) => s + l.params, 0) : 0;
  const totalF = model ? model.layers.reduce((s, l) => s + l.flops, 0) : 0;

  // While a large model streams in, analysis tabs wait for the final model
  // (they'd show misleading partial numbers); the Inspector works on partial data.
  const sideTab = streamProg && rightTab !== "inspector" ? "parsing" : rightTab;

  // Layers with at least one compiler finding, surfaced in the stats strip.
  const issueLayers = model ? model.layers.filter((l) => l.compIssues.length > 0).length : 0;

  // Ctrl/Cmd+K opens the command palette from anywhere in the app.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPalette((p) => !p);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const commands = useMemo<Command[]>(() => {
    const out: Command[] = [];
    if (model) {
      for (const tl of ALL_TOOLS) out.push({ id: `tool:${tl.id}`, section: "Tools", label: tl.label, hint: tl.group.label, icon: tl.group.id, run: () => setRightTab(tl.id) });
    }
    out.push({ id: "act:open", section: "Actions", label: "Open model file…", icon: "upload", run: () => fileInput.current?.click() });
    out.push({ id: "act:demo", section: "Actions", label: "Load demo model (YOLO26n)", icon: "action", run: loadDemo });
    out.push({ id: "act:theme", section: "Actions", label: `Switch to ${theme === "dark" ? "light" : "dark"} theme`, icon: theme === "dark" ? "sun" : "moon", run: () => setTheme((p) => (p === "dark" ? "light" : "dark")) });
    if (model) {
      out.push({ id: "act:close", section: "Actions", label: "Close model", icon: "close", run: clearModel });
      for (const l of model.layers) {
        out.push({ id: `layer:${l.id}`, section: "Layers", label: l.name, hint: `${l.op} · ${l.group}`, icon: "layer", run: () => { setSel(l); setRightTab("inspector"); } });
      }
    }
    return out;
    // loadDemo/clearModel only touch state setters and refs, so they are stable in effect.
  }, [model, theme]);

  return <ThemeCtx.Provider value={t}>
    <ErrorBoundary>
    {dlPanel && <DownloadPanel detected={os} onClose={closeDlPanel} />}
    {palette && <CommandPalette commands={commands} onClose={() => setPalette(false)} />}
    <input ref={fileInput} type="file" accept={ACCEPT} multiple hidden onChange={(e) => { handleFiles(e.target.files); e.target.value = ""; }} />
    {building && <BuildingPage label={building} theme={theme} onToggleTheme={() => setTheme((p) => (p === "dark" ? "light" : "dark"))} onBack={() => setBuilding(null)} />}
    <div style={{ height: "100dvh", display: "flex", flexDirection: "column", background: t.bg, color: t.t0, fontFamily: "'Inter',-apple-system,sans-serif", overflow: "hidden", transition: "background .3s" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.bdr};border-radius:2px}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        ::selection{background:${t.acc}44}`}</style>

      {/* Header: full-width ribbon */}
      <header style={{ padding: compact ? "10px 14px" : "13px 24px", borderBottom: `1px solid ${t.bdr}`, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, rowGap: 8, flexWrap: "wrap", flexShrink: 0, background: t.bg1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 11, minWidth: 0 }}>
          <img src={logoUrl} alt="ModelVisio" width={34} height={34} style={{ width: 34, height: 34, borderRadius: 8, objectFit: "cover", flexShrink: 0 }} />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 17, fontWeight: 800, color: t.t0, fontFamily: "'Space Grotesk'", letterSpacing: -.3, display: "flex", alignItems: "center", gap: 7 }}>ModelVisio
              <span title={`Version ${APP_VERSION}`} style={{ fontSize: 10.5, fontWeight: 500, color: t.t2, fontFamily: "'JetBrains Mono'", border: `1px solid ${t.bdr}`, borderRadius: 4, padding: "0 5px", letterSpacing: 0 }}>{APP_VERSION}</span>
              {!compact && <span style={{ fontSize: 10.5, fontWeight: 600, color: t.t2, letterSpacing: 0 }}>by Premchand</span>}</div>
            {!compact && <div style={{ fontSize: 10, color: t.t2, letterSpacing: .5, textTransform: "uppercase", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>Model Analyzer · Edge Deployment Intelligence</div>}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {model && <Badge c={t.suc}>{model.format}</Badge>}
          {model && <span style={{ fontSize: 10, color: t.t2, fontFamily: "'JetBrains Mono'", maxWidth: compact ? 90 : 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{model.name}</span>}
          {model && <button type="button" onClick={clearModel} aria-label="Close model" title="Close model" style={{ width: 24, height: 24, borderRadius: 5, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon name="close" size={12} stroke={2} /></button>}
          <button type="button" onClick={() => setPalette(true)} aria-label="Search tools, layers and actions" aria-keyshortcuts="Control+K" title="Search tools, layers and actions (Ctrl+K)"
            style={{ height: 28, display: "flex", alignItems: "center", gap: 8, padding: compact ? "0 7px" : "0 6px 0 10px", borderRadius: 6, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t2, fontSize: 12, cursor: "pointer", flexShrink: 0 }}>
            <Icon name="analyze" size={14} stroke={2} />
            {!compact && <><span style={{ minWidth: narrow ? 0 : 110, textAlign: "left" }}>Search</span><span style={{ fontFamily: "'JetBrains Mono'", fontSize: 10, color: t.t1, border: `1px solid ${t.bdr2}`, borderRadius: 4, padding: "0 5px" }}>Ctrl K</span></>}
          </button>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener" title="View source on GitHub" style={{ width: 28, height: 28, borderRadius: 6, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t1, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}><GhIcon /></a>
          <button type="button" onClick={() => setTheme((p) => (p === "dark" ? "light" : "dark"))} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} style={{ width: 28, height: 28, borderRadius: 6, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t1, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}><Icon name={theme === "dark" ? "sun" : "moon"} size={15} /></button>
        </div>
      </header>

      {!model ? (
        <div style={{ flex: 1, overflow: "auto", background: t.bg }}>
          <div style={{ width: "100%", maxWidth: 1120, margin: "0 auto", padding: compact ? "28px 16px 24px" : "56px 24px 32px", display: "flex", flexDirection: "column", gap: compact ? 32 : 48 }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: compact ? 28 : 40, alignItems: "center" }}>
              <div style={{ flex: "1 1 420px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
                <span style={{ fontSize: 11.5, fontWeight: 600, letterSpacing: .8, textTransform: "uppercase", color: t.acc }}>Edge deployment, from the model file</span>
                <h1 style={{ fontFamily: "'Space Grotesk'", fontSize: "clamp(30px, 5vw, 48px)", lineHeight: 1.06, letterSpacing: "-0.025em", fontWeight: 700, color: t.t0 }}>See whether your model fits the edge before you flash it.</h1>
                <p style={{ fontSize: compact ? 14 : 15.5, lineHeight: 1.6, color: t.t1, maxWidth: 520 }}>Graph, compiler pre-flight, hardware fit for 21 accelerators and deploy scripts. Parsing runs in your browser; the model file never leaves your machine.</p>
                <div style={{ display: "flex", gap: 10, rowGap: 8, flexWrap: "wrap", paddingTop: 4 }}>
                  {/* Distribution downloads, web only. Hidden inside the desktop app and
                      the VS Code extension: you're already running one of them. */}
                  {!isDesktop() && !isVsCodeWebview() && <>
                    <a href={build ? build.url : DESKTOP_RELEASES_URL} target="_blank" rel="noreferrer noopener"
                      title={build ? `Download the ${build.label} desktop app (${build.file})` : "Download the desktop app (all platforms)"}
                      onClick={() => { if (os === "mac") setMacHelp(true); }}
                      style={{ display: "inline-flex", alignItems: "center", gap: 8, minHeight: 40, padding: "0 16px", borderRadius: 9, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t0, fontSize: 13, fontWeight: 600, textDecoration: "none", cursor: "pointer" }}>
                      <OsIcon os={os} size={14} />{build ? `Download for ${build.label}` : "Download Desktop App"}
                    </a>
                    <button type="button" onClick={() => setBuilding("VS Code Extension")}
                      style={{ display: "inline-flex", alignItems: "center", gap: 6, minHeight: 40, padding: "0 16px", borderRadius: 9, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t0, fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
                      VS Code Extension
                    </button>
                  </>}
                  <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener"
                    style={{ display: "inline-flex", alignItems: "center", gap: 7, minHeight: 40, padding: "0 16px", borderRadius: 9, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t0, fontSize: 13, fontWeight: 600, textDecoration: "none", cursor: "pointer" }}>
                    <GhIcon /> GitHub
                  </a>
                </div>
                {!isDesktop() && !isVsCodeWebview() && <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 10 }}>
                  <button type="button" onClick={() => setDlPanel(true)} aria-haspopup="dialog"
                    style={{ border: "none", background: "transparent", padding: "2px 0", color: t.t2, fontSize: 12, cursor: "pointer", textDecoration: "underline", textDecorationStyle: "dotted", textUnderlineOffset: 3 }}>
                    {build ? "Other platforms" : "macOS · Windows · Linux"}
                  </button>
                  {macHelp && <div role="status" style={{ width: "100%", border: `1px solid ${t.bdr}`, background: t.bg1, borderRadius: 10, padding: compact ? "10px 12px" : "12px 14px" }}>
                    <MacInstallHelp onDismiss={() => setMacHelp(false)} />
                  </div>}
                </div>}
              </div>

              <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); handleFiles(e.dataTransfer.files); }}
                style={{ flex: "1 1 400px", minWidth: 0, padding: compact ? "24px 18px" : 32, borderRadius: 16, border: `1.5px dashed ${drag ? t.acc : t.bdr2}`, background: drag ? t.acc + "10" : t.bg1, display: "flex", flexDirection: "column", alignItems: "center", gap: 12, textAlign: "center", transition: "border-color .2s, background .2s" }}>
                {/* Node-graph hexagon mark, tinted via CSS mask so it adopts the theme
                    color (works on both the dark and light dropzone). */}
                <div style={{ width: 72, height: 57, opacity: 0.75, background: drag ? t.acc : t.t2, WebkitMaskImage: `url(${graphHexUrl})`, maskImage: `url(${graphHexUrl})`, WebkitMaskRepeat: "no-repeat", maskRepeat: "no-repeat", WebkitMaskPosition: "center", maskPosition: "center", WebkitMaskSize: "contain", maskSize: "contain" }} />
                <div role="status" aria-live="polite" style={{ fontSize: 17, fontWeight: 600, color: t.t0 }}>{parsing ? (streamProg ? `Parsing model… ${streamProg.pct}%` : "Parsing model…") : "Drop a model file here"}</div>
                <details style={{ fontSize: 12.5, color: t.t2, lineHeight: 1.6, maxWidth: 400 }}>
                  <summary style={{ cursor: "pointer", listStyle: "none" }}>ONNX, TFLite, PyTorch, Core ML, OpenVINO, TensorRT, GGUF <span style={{ color: t.t1, textDecoration: "underline", textDecorationStyle: "dotted", textUnderlineOffset: 3 }}>and {FORMATS.length - 7} more</span></summary>
                  <div style={{ marginTop: 8, fontSize: 11.5 }}>{FORMATS.join(" · ")}</div>
                </details>
                {loadError && <div role="alert" style={{ fontSize: 12, color: t.err, lineHeight: 1.4 }}>{loadError}</div>}
                <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap", paddingTop: 4 }}>
                  <button type="button" onClick={() => fileInput.current?.click()} style={{ minHeight: 42, padding: "0 22px", borderRadius: 9, border: "none", background: t.acc, color: "#fff", fontSize: 13.5, fontWeight: 700, cursor: "pointer", whiteSpace: "nowrap" }}>Browse files</button>
                  <button type="button" onClick={loadDemo} style={{ minHeight: 42, padding: "0 18px", borderRadius: 9, border: `1px solid ${t.bdr2}`, background: "transparent", color: t.t0, fontSize: 13.5, fontWeight: 500, cursor: "pointer", whiteSpace: "nowrap" }}>Try the demo · YOLO26n</button>
                </div>
                <div style={{ fontSize: 11.5, color: t.t2, lineHeight: 1.5 }}>OpenVINO, ncnn, Paddle, TF.js: select or drop the weights file together with the graph file.</div>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 14 }}>
              {TOOL_GROUPS.filter((g) => g.id !== "copilot").map((g, i) => (
                <div key={g.id} style={{ padding: 18, border: `1px solid ${t.bdr}`, borderRadius: 12, background: t.bg1, display: "flex", flexDirection: "column", gap: 9 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ width: 30, height: 30, borderRadius: 8, background: t.bg2, color: t.acc, display: "flex", alignItems: "center", justifyContent: "center" }}><Icon name={g.id} size={16} /></span>
                    <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 11.5, color: t.t2 }}>0{i + 1}</span>
                    <span style={{ fontFamily: "'Space Grotesk'", fontSize: 17, fontWeight: 700, color: t.t0 }}>{g.label}</span>
                  </div>
                  <p style={{ fontSize: 12.5, lineHeight: 1.6, color: t.t1 }}>{STAGE_BLURB[g.id]}</p>
                  <div style={{ fontSize: 11.5, color: t.t2, fontFamily: "'JetBrains Mono'" }}>{g.tools.map((tl) => tl.label).join(" · ")}</div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 11.5, color: t.t2, textAlign: "center", letterSpacing: .3 }}>
              Developed by <span style={{ fontWeight: 700, color: t.t1 }}>Premchand</span> · © 2026{" "}
              <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener" style={{ color: t.t1, fontWeight: 600, textDecoration: "none" }}>@{AUTHOR}</a>
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Stats */}
          <div style={{ minHeight: 42, padding: compact ? "0 12px" : "0 24px", borderBottom: `1px solid ${t.bdr}`, display: "flex", alignItems: "center", gap: compact ? 16 : 22, overflowX: "auto", flexShrink: 0, background: t.bg1, whiteSpace: "nowrap" }}>
            {[{ l: "Params", v: fmt(totalP) }, { l: "FLOPs", v: fmt(totalF) }, { l: "Size", v: fB(model.sizeBytes) }, { l: "Input", v: model.inputShape.slice(2).join("×") || "-" }, { l: "Layers", v: model.layers.length }, { l: "Opset", v: model.opset }].map((s) =>
              <div key={s.l} style={{ display: "flex", alignItems: "baseline", gap: 7, flexShrink: 0 }}>
                <span style={{ fontSize: 10.5, color: t.t2, textTransform: "uppercase", letterSpacing: .6 }}>{s.l}</span>
                <span style={{ fontSize: 14.5, fontWeight: 600, color: t.t0, fontFamily: "'JetBrains Mono'" }}>{s.v}</span>
              </div>
            )}
            <span style={{ flex: 1 }} />
            {issueLayers > 0 && !streamProg && <button type="button" onClick={() => setRightTab("compiler")}
              style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0, minHeight: 28, padding: "0 10px", borderRadius: 6, border: `1px solid ${t.wrn}55`, background: t.wrn + "12", color: t.t0, fontSize: 12, cursor: "pointer" }}>
              <span style={{ color: t.wrn, display: "flex" }}><Icon name="warn" size={14} stroke={2} /></span>
              Compiler: {issueLayers} {issueLayers === 1 ? "layer needs" : "layers need"} attention
            </button>}
            {streamProg && <div role="status" aria-live="polite" style={{ background: t.bg1, border: `1px solid ${t.acc}66`, borderRadius: 6, padding: "5px 11px", minWidth: 180, flexShrink: 0, display: "flex", flexDirection: "column", justifyContent: "center", gap: 5 }}>
              <div style={{ fontSize: 11, color: t.t1, fontFamily: "'JetBrains Mono'", whiteSpace: "nowrap" }}>Parsing… {streamProg.pct}% · {streamProg.layers.toLocaleString("en-US")} layers</div>
              <div style={{ height: 3, borderRadius: 2, background: t.bdr, overflow: "hidden" }}><div style={{ width: `${streamProg.pct}%`, height: "100%", background: t.acc, transition: "width .25s" }} /></div>
            </div>}
          </div>
          {/* Parser caveats: numbers above that are assumed or incomplete. */}
          {!!model.notes?.length && <div role="note" style={{ padding: compact ? "4px 10px" : "5px 24px", borderBottom: `1px solid ${t.bdr}`, background: t.wrn + "12", fontSize: 11, color: t.t1, lineHeight: 1.45, flexShrink: 0 }}>
            {model.notes.map((n, i) => <div key={i}><span style={{ color: t.wrn, fontWeight: 700 }}>Note · </span>{n}</div>)}
          </div>}

          {/* Workspace: graph dominant (Netron-style), analysis sidebar on the right.
              On narrow viewports it stacks vertically and the splitter is hidden. */}
          <div style={{ flex: 1, display: "flex", flexDirection: narrow ? "column" : "row", overflow: "hidden" }}>
            <GraphPanel model={model} selected={sel} onSelect={setSel} streaming={!!streamProg} colorMode={graphColor} onColorModeChange={setGraphColor} />

            {!narrow && (
              <div
                onMouseDown={startResize}
                onMouseEnter={() => setResizeHover(true)}
                onMouseLeave={() => setResizeHover(false)}
                title="Drag to resize"
                style={{ width: 6, flexShrink: 0, cursor: "col-resize", background: resizeHover ? t.acc : t.bdr, transition: "background .15s" }}
              />
            )}

            <aside style={narrow
              ? { width: "100%", height: compact ? "52%" : "45%", flexShrink: 0, display: "flex", flexDirection: "column", overflow: "hidden", background: t.bg1, borderTop: `1px solid ${t.bdr}` }
              : { width: effSidebarW, flexShrink: 0, display: "flex", flexDirection: "column", overflow: "hidden", background: t.bg1 }}>
              {!narrow && <ToolStages active={rightTab} onChange={setRightTab} />}
              <ToolTabs active={rightTab} onChange={setRightTab} />
              <div role="tabpanel" style={{ flex: 1, overflow: "auto", padding: 10 }}>
                {streamProg && sideTab === "parsing" && <div style={{ fontSize: 11.5, color: t.t2, lineHeight: 1.6, padding: 6 }}>Parsing… {streamProg.pct}%. This panel opens once the whole model has loaded; the Inspector works now.</div>}
                {sideTab === "inspector" && <Inspector layer={sel} model={model} />}
                {sideTab === "compiler" && <CompilerChecker model={model} onApplyFix={onApplyFix} onUndo={onUndoFix} undoDepth={fixUndo.length} />}
                {sideTab === "hardware" && <HWReport model={model} />}
                {sideTab === "quant" && <QuantHeatmap model={model} selected={sel} onSelectLayer={setSel} onShowOnGraph={setGraphColor} />}
                {sideTab === "advisor" && <ModelAdvisor model={model} onSelectLayer={setSel} />}
                {sideTab === "bench" && <Benchmark model={model} modelBytes={modelBytes} />}
                {sideTab === "investigate" && <Investigator model={model} onSelectLayer={setSel} />}
                {sideTab === "optimize" && <OptimizationSearch model={model} onApplyFix={onApplyFix} />}
                {sideTab === "convert" && <Converter model={model} modelBytes={modelBytes} />}
                {sideTab === "deploy" && <DeployRecipe model={model} />}
                {sideTab === "agent" && <DeployAgent model={model} onApplyFix={onApplyFix} />}
                {sideTab === "fleet" && <FleetSim model={model} />}
                {sideTab === "registry" && <ModelRegistry model={model} modelBytes={modelBytes} />}
                {sideTab === "chat" && <div style={{ height: "100%" }}><Chat model={model} sel={sel} /></div>}
              </div>
            </aside>
          </div>

          {/* Narrow layouts dock the stage switcher at the bottom, in thumb reach, in place of the footer. */}
          {narrow ? <ToolStages active={rightTab} onChange={setRightTab} docked /> : <div style={{ padding: compact ? "4px 10px" : "5px 24px", borderTop: `1px solid ${t.bdr}`, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, fontSize: 9.5, color: t.t2, background: t.bg1, flexShrink: 0 }}>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>ModelVisio · {model.format} · opset {model.opset} · {model.framework}{model.producer ? ` · ${model.producer}` : ""}</span>
            <span style={{ display: "flex", alignItems: "center", gap: 10, whiteSpace: "nowrap", flexShrink: 0 }}>
              <span>© 2026 {AUTHOR}</span>
              <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener" title="View source on GitHub" style={{ color: t.t2, display: "inline-flex", alignItems: "center" }}><GhIcon size={12} /></a>
            </span>
          </div>}
        </>
      )}
    </div>
    </ErrorBoundary>
  </ThemeCtx.Provider>;
}
