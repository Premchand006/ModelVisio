import { useCallback, useEffect, useState } from "react";
import { FORMAT_SUPPORT, type Model, type ModelLayer } from "@modelvisio/parsers";
import { parseBufferInWorker } from "./workers/parseClient";
import { ThemeCtx } from "./theme/ThemeContext";
import { TH, type ThemeName } from "./theme/theme";
import { fB, fmt } from "./utils/format";
import { FORMATS } from "./data/hardware";
import { demoModel } from "./demo/demoModel";
import { Badge } from "./components/Badge";
import { Tabs } from "./components/Tabs";
import { GraphPanel } from "./components/Graph/GraphPanel";
import { Inspector } from "./components/Inspector";
import { CompilerChecker } from "./components/CompilerChecker";
import { HWReport } from "./components/HWReport";
import { Converter } from "./components/Converter";
import { DeployRecipe } from "./components/DeployRecipe";
import { Chat } from "./components/Chat";
import { logUpload } from "@modelvisio/ai";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { BuildingPage } from "./components/BuildingPage";
import { applyFix, type FixId } from "./fixes/transforms";
import { useViewport } from "./hooks/useViewport";
import { GITHUB_URL, AUTHOR } from "./data/links";
import logoUrl from "./assets/logo.png";
import graphHexUrl from "./assets/graph-hex.png";

const ACCEPT = FORMAT_SUPPORT.flatMap((f) => f.exts).map((e) => "." + e).join(",");

// GitHub "mark" glyph, reused in the header and landing page.
const GH_PATH = "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z";
const GhIcon = ({ size = 15 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d={GH_PATH} /></svg>
);

/** Imperative handle a shell can use to drive the app (e.g. native menus). */
export type AppApi = { openFile: (f: File) => void };

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
  const [drag, setDrag] = useState(false);
  // When set, shows the full-screen "Building" (coming-soon) page for a surface
  // that isn't shipped yet — e.g. the desktop app or VS Code extension.
  const [building, setBuilding] = useState<string | null>(null);
  const [sidebarW, setSidebarW] = useState(400);
  const [resizeHover, setResizeHover] = useState(false);
  const { vw, narrow, compact } = useViewport();
  // Side-by-side: never let the sidebar starve the graph — keep ≥380px for it.
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
  const [parsing, setParsing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const loadDemo = () => { setModel(demoModel); setModelBytes(null); setSel(null); setLoadError(null); setFixUndo([]); };
  const clearModel = () => { setModel(null); setModelBytes(null); setSel(null); setFixUndo([]); };

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

  const handleFile = useCallback(async (f?: File) => {
    if (!f) return;
    setLoadError(null);
    setParsing(true);
    try {
      // Read the bytes once: parse them, and retain them for conversion/export.
      const buf = await f.arrayBuffer();
      const m = await parseBufferInWorker(buf, f.name);
      setModel(m); setModelBytes(buf); setSel(null); setFixUndo([]);
      logUpload({
        modelName: m.name, format: m.format, framework: m.framework,
        sizeBytes: m.sizeBytes, params: m.layers.reduce((s, l) => s + l.params, 0),
        layers: m.layers.length,
      });
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      setParsing(false);
    }
  }, []);

  // Expose an imperative handle so shells (e.g. the Tauri native menu) can open
  // files without reaching into the UI.
  useEffect(() => { onReady?.({ openFile: (f: File) => { void handleFile(f); } }); }, [onReady, handleFile]);

  const totalP = model ? model.layers.reduce((s, l) => s + l.params, 0) : 0;
  const totalF = model ? model.layers.reduce((s, l) => s + l.flops, 0) : 0;

  const RTABS = [
    { id: "inspector", label: "Inspector" }, { id: "compiler", label: "Compiler" },
    { id: "hardware", label: "Hardware" }, { id: "convert", label: "Convert" },
    { id: "deploy", label: "Deploy" }, { id: "chat", label: "AI" },
  ];

  return <ThemeCtx.Provider value={t}>
    <ErrorBoundary>
    {building && <BuildingPage label={building} theme={theme} onToggleTheme={() => setTheme((p) => (p === "dark" ? "light" : "dark"))} onBack={() => setBuilding(null)} />}
    <div style={{ height: "100dvh", display: "flex", flexDirection: "column", background: t.bg, color: t.t0, fontFamily: "'Inter',-apple-system,sans-serif", overflow: "hidden", transition: "background .3s" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.bdr};border-radius:2px}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        ::selection{background:${t.acc}44}`}</style>

      {/* Header — a full-width ribbon with real presence */}
      <header style={{ padding: compact ? "10px 14px" : "13px 24px", borderBottom: `1px solid ${t.bdr}`, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, rowGap: 8, flexWrap: "wrap", flexShrink: 0, background: t.bg1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 11, minWidth: 0 }}>
          <img src={logoUrl} alt="ModelVisio" width={34} height={34} style={{ width: 34, height: 34, borderRadius: 8, objectFit: "cover", flexShrink: 0 }} />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 17, fontWeight: 800, color: t.t0, fontFamily: "'Space Grotesk'", letterSpacing: -.3 }}>ModelVisio <span style={{ fontSize: 10.5, fontWeight: 600, color: t.t3 }}>by Premchand</span></div>
            {!compact && <div style={{ fontSize: 10, color: t.t2, letterSpacing: .5, textTransform: "uppercase", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>Model Analyzer · Edge Deployment Intelligence</div>}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {model && <Badge c={t.suc}>{model.format}</Badge>}
          {model && <span style={{ fontSize: 10, color: t.t2, fontFamily: "'JetBrains Mono'", maxWidth: compact ? 90 : 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{model.name}</span>}
          {model && <button type="button" onClick={clearModel} style={{ padding: "2px 6px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t3, fontSize: 9, cursor: "pointer" }}>✕</button>}
          <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener" title="View source on GitHub" style={{ width: 28, height: 28, borderRadius: 6, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t1, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}><GhIcon /></a>
          <button type="button" onClick={() => setTheme((p) => (p === "dark" ? "light" : "dark"))} style={{ width: 28, height: 28, borderRadius: 6, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t1, fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>{theme === "dark" ? "☀" : "☾"}</button>
        </div>
      </header>

      {!model ? (
        <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: compact ? "24px 16px" : 40, gap: compact ? 20 : 28, background: t.bg, overflow: "auto" }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ display: "inline-block", fontSize: "clamp(30px, 9vw, 54px)", fontWeight: 800, fontFamily: "'Space Grotesk'", letterSpacing: "-0.03em", lineHeight: 1.08, paddingBottom: 6, color: t.t0 }}>Model<span style={{ color: t.acc }}>Visio</span></div>
            <div style={{ fontSize: "clamp(11px, 2.6vw, 14px)", color: t.t2, marginTop: 4 }}>Model analyzer · Compiler pre-flight · Edge deployment intelligence</div>
          </div>
          <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}
            style={{ width: "min(480px, 100%)", padding: compact ? "24px 18px" : 40, borderRadius: 14, border: `2px dashed ${drag ? t.acc : t.bdr}`, background: drag ? t.acc + "10" : t.bg1, textAlign: "center", transition: "border-color .2s, background .2s" }}>
            {/* Node-graph hexagon mark, tinted via CSS mask so it adopts the theme
                color (works on both the dark and light dropzone). */}
            <div style={{ width: 82, height: 65, margin: "0 auto 8px", opacity: 0.6, background: t.t2, WebkitMaskImage: `url(${graphHexUrl})`, maskImage: `url(${graphHexUrl})`, WebkitMaskRepeat: "no-repeat", maskRepeat: "no-repeat", WebkitMaskPosition: "center", maskPosition: "center", WebkitMaskSize: "contain", maskSize: "contain" }} />
            <div style={{ fontSize: 15, fontWeight: 600, color: t.t0, marginBottom: 3 }}>{parsing ? "Parsing model…" : "Drop your model file"}</div>
            <div style={{ fontSize: 11.5, color: t.t2, marginBottom: 16, lineHeight: 1.7 }}>
              {FORMATS.join(" · ")}
            </div>
            {loadError && <div style={{ fontSize: 11, color: t.err, marginBottom: 12, lineHeight: 1.4 }}>{loadError}</div>}
            <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
              <label style={{ padding: "9px 24px", borderRadius: 7, background: t.acc, color: "#fff", fontSize: 12, fontWeight: 600, cursor: "pointer", whiteSpace: "nowrap" }}>
                Browse Files<input type="file" accept={ACCEPT} hidden onChange={(e) => handleFile(e.target.files?.[0] || undefined)} />
              </label>
              <button type="button" onClick={loadDemo} style={{ padding: "9px 24px", borderRadius: 7, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 12, fontWeight: 500, cursor: "pointer", whiteSpace: "nowrap" }}>Load Demo · YOLO26n</button>
            </div>
          </div>
          <div style={{ display: "flex", gap: 10, rowGap: 8, flexWrap: "wrap", justifyContent: "center" }}>
            <button type="button" onClick={() => setBuilding("Desktop App")}
              style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 16px", borderRadius: 8, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t1, fontSize: 11.5, fontWeight: 600, cursor: "pointer" }}>
              ⬇ Download Desktop App
            </button>
            <button type="button" onClick={() => setBuilding("VS Code Extension")}
              style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 16px", borderRadius: 8, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t1, fontSize: 11.5, fontWeight: 600, cursor: "pointer" }}>
              VS Code Extension
            </button>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener"
              style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 16px", borderRadius: 8, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t1, fontSize: 11.5, fontWeight: 600, textDecoration: "none", cursor: "pointer" }}>
              <GhIcon /> GitHub
            </a>
          </div>
          <div style={{ marginTop: compact ? 4 : 10, fontSize: 11, color: t.t3, textAlign: "center", letterSpacing: .3 }}>
            Developed by <span style={{ fontWeight: 700, color: t.t1 }}>Premchand</span> · © 2026{" "}
            <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener" style={{ color: t.t2, fontWeight: 600, textDecoration: "none" }}>@{AUTHOR}</a>
          </div>
        </div>
      ) : (
        <>
          {/* Stats */}
          <div style={{ padding: compact ? "6px 10px" : "8px 24px", borderBottom: `1px solid ${t.bdr}`, display: "flex", gap: 6, overflowX: "auto", flexShrink: 0, background: t.bg1 }}>
            {[{ l: "Params", v: fmt(totalP), c: t.acc }, { l: "FLOPs", v: fmt(totalF), c: t.acc2 }, { l: "Size", v: fB(model.sizeBytes), c: t.suc }, { l: "Input", v: model.inputShape.slice(2).join("×") || "—", c: t.wrn }, { l: "Layers", v: model.layers.length, c: t.cH }, { l: "Opset", v: model.opset, c: t.t2 }].map((s) =>
              <div key={s.l} style={{ background: t.bg1, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: "5px 11px", minWidth: 84, flexShrink: 0 }}>
                <div style={{ fontSize: 9.5, color: t.t2, textTransform: "uppercase", letterSpacing: .6 }}>{s.l}</div>
                <div style={{ fontSize: 17, fontWeight: 700, color: s.c, fontFamily: "'JetBrains Mono'", marginTop: 1 }}>{s.v}</div>
              </div>
            )}
          </div>

          {/* Workspace: graph dominant (Netron-style), analysis sidebar on the right.
              On narrow viewports it stacks vertically and the splitter is hidden. */}
          <div style={{ flex: 1, display: "flex", flexDirection: narrow ? "column" : "row", overflow: "hidden" }}>
            <GraphPanel model={model} selected={sel} onSelect={setSel} />

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
              <div style={{ padding: "4px 8px", borderBottom: `1px solid ${t.bdr}` }}><Tabs tabs={RTABS} active={rightTab} onChange={setRightTab} /></div>
              <div style={{ flex: 1, overflow: "auto", padding: 10 }}>
                {rightTab === "inspector" && <Inspector layer={sel} model={model} />}
                {rightTab === "compiler" && <CompilerChecker model={model} onApplyFix={onApplyFix} onUndo={onUndoFix} undoDepth={fixUndo.length} />}
                {rightTab === "hardware" && <HWReport model={model} />}
                {rightTab === "convert" && <Converter model={model} modelBytes={modelBytes} />}
                {rightTab === "deploy" && <DeployRecipe model={model} />}
                {rightTab === "chat" && <div style={{ height: "100%" }}><Chat model={model} sel={sel} /></div>}
              </div>
            </aside>
          </div>

          {/* Footer */}
          <div style={{ padding: compact ? "4px 10px" : "5px 24px", borderTop: `1px solid ${t.bdr}`, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, fontSize: 9.5, color: t.t2, background: t.bg1, flexShrink: 0 }}>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>ModelVisio · {model.format} · opset {model.opset} · {model.framework}{model.producer ? ` · ${model.producer}` : ""}</span>
            <span style={{ display: "flex", alignItems: "center", gap: 10, whiteSpace: "nowrap", flexShrink: 0 }}>
              <span>© 2026 {AUTHOR}</span>
              <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener" title="View source on GitHub" style={{ color: t.t2, display: "inline-flex", alignItems: "center" }}><GhIcon size={12} /></a>
            </span>
          </div>
        </>
      )}
    </div>
    </ErrorBoundary>
  </ThemeCtx.Provider>;
}
