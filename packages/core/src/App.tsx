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
import { ErrorBoundary } from "./components/ErrorBoundary";
import { applyFix, type FixId } from "./fixes/transforms";
import { DESKTOP_DOWNLOAD_URL, VSCODE_MARKETPLACE_URL } from "./data/links";
import { useViewport } from "./hooks/useViewport";
import logoUrl from "./assets/logo.png";

const ACCEPT = FORMAT_SUPPORT.flatMap((f) => f.exts).map((e) => "." + e).join(",");

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
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: t.bg, color: t.t0, fontFamily: "'Inter',-apple-system,sans-serif", overflow: "hidden", transition: "background .3s" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.bdr};border-radius:2px}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        ::selection{background:${t.acc}44}`}</style>

      {/* Header */}
      <header style={{ padding: compact ? "6px 10px" : "6px 14px", borderBottom: `1px solid ${t.bdr}`, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap", flexShrink: 0, background: t.glass, backdropFilter: "blur(16px)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
          <img src={logoUrl} alt="ModelVisio" width={28} height={28} style={{ width: 28, height: 28, borderRadius: 7, objectFit: "cover", boxShadow: `0 0 20px ${t.acc}40`, flexShrink: 0 }} />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 800, color: t.t0, fontFamily: "'Space Grotesk'", letterSpacing: -.3 }}>ModelVisio</div>
            {!compact && <div style={{ fontSize: 9.5, color: t.t2, letterSpacing: .5, textTransform: "uppercase", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>AI-Native Model Analyzer · Edge Deployment Intelligence</div>}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
          {model && <Badge c={t.suc}>{model.format}</Badge>}
          {model && <span style={{ fontSize: 10, color: t.t2, fontFamily: "'JetBrains Mono'", maxWidth: compact ? 90 : 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{model.name}</span>}
          {model && <button type="button" onClick={clearModel} style={{ padding: "2px 6px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t3, fontSize: 9, cursor: "pointer" }}>✕</button>}
          <button type="button" onClick={() => setTheme((p) => (p === "dark" ? "light" : "dark"))} style={{ width: 28, height: 28, borderRadius: 6, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t1, fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>{theme === "dark" ? "☀" : "☾"}</button>
        </div>
      </header>

      {!model ? (
        <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: compact ? "24px 16px" : 40, gap: compact ? 20 : 28, background: `radial-gradient(ellipse at 50% 40%,${t.acc}08,transparent 70%)`, overflow: "auto" }}>
          <div style={{ textAlign: "center" }}>
            <div key={theme} style={{ display: "inline-block", fontSize: "clamp(30px, 9vw, 54px)", fontWeight: 800, fontFamily: "'Space Grotesk'", letterSpacing: "-0.03em", lineHeight: 1.08, paddingBottom: 6, backgroundImage: `linear-gradient(135deg,${t.acc},${t.acc2},${t.acc3})`, WebkitBackgroundClip: "text", backgroundClip: "text", WebkitTextFillColor: "transparent", color: "transparent" }}>ModelVisio</div>
            <div style={{ fontSize: "clamp(11px, 2.6vw, 14px)", color: t.t2, marginTop: 4 }}>AI-native model analyzer · Compiler pre-flight · Edge deployment intelligence</div>
          </div>
          <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}
            style={{ width: "min(480px, 100%)", padding: compact ? "24px 18px" : 40, borderRadius: 16, border: `2px dashed ${drag ? t.acc : t.bdr}`, background: drag ? t.acc + "08" : t.glass, backdropFilter: "blur(12px)", textAlign: "center", transition: "all .2s" }}>
            <div style={{ fontSize: 32, marginBottom: 6, opacity: .4 }}>⬡</div>
            <div style={{ fontSize: 15, fontWeight: 600, color: t.t0, marginBottom: 3 }}>{parsing ? "Parsing model…" : "Drop your model file"}</div>
            <div style={{ fontSize: 11.5, color: t.t2, marginBottom: 16, lineHeight: 1.7 }}>
              {FORMATS.slice(0, 12).join(" · ")}<br />{FORMATS.slice(12).join(" · ")}
            </div>
            {loadError && <div style={{ fontSize: 11, color: t.err, marginBottom: 12, lineHeight: 1.4 }}>{loadError}</div>}
            <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
              <label style={{ padding: "9px 24px", borderRadius: 7, background: `linear-gradient(135deg,${t.acc},${t.acc2})`, color: "#fff", fontSize: 12, fontWeight: 600, cursor: "pointer", boxShadow: `0 0 20px ${t.acc}30`, whiteSpace: "nowrap" }}>
                Browse Files<input type="file" accept={ACCEPT} hidden onChange={(e) => handleFile(e.target.files?.[0] || undefined)} />
              </label>
              <button type="button" onClick={loadDemo} style={{ padding: "9px 24px", borderRadius: 7, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 12, fontWeight: 500, cursor: "pointer", whiteSpace: "nowrap" }}>Load Demo · YOLO26n</button>
            </div>
          </div>
          <div style={{ display: "flex", gap: 10, rowGap: 8, flexWrap: "wrap", justifyContent: "center" }}>
            <a href={DESKTOP_DOWNLOAD_URL} target="_blank" rel="noreferrer noopener"
              style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 16px", borderRadius: 8, border: `1px solid ${t.bdr}`, background: t.glass, color: t.t1, fontSize: 11.5, fontWeight: 600, textDecoration: "none", cursor: "pointer" }}>
              ⬇ Download Desktop App
            </a>
            <a href={VSCODE_MARKETPLACE_URL} target="_blank" rel="noreferrer noopener"
              style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "7px 16px", borderRadius: 8, border: `1px solid ${t.bdr}`, background: t.glass, color: t.t1, fontSize: 11.5, fontWeight: 600, textDecoration: "none", cursor: "pointer" }}>
              VS Code Extension ↗
            </a>
          </div>
        </div>
      ) : (
        <>
          {/* Stats */}
          <div style={{ padding: compact ? "6px 10px" : "6px 14px", borderBottom: `1px solid ${t.bdr}`, display: "flex", gap: 6, overflowX: "auto", flexShrink: 0, background: t.glass, backdropFilter: "blur(8px)" }}>
            {[{ l: "Params", v: fmt(totalP), c: t.acc }, { l: "FLOPs", v: fmt(totalF), c: t.acc2 }, { l: "Size", v: fB(model.sizeBytes), c: t.suc }, { l: "Input", v: model.inputShape.slice(2).join("×") || "—", c: t.wrn }, { l: "Layers", v: model.layers.length, c: "#F472B6" }, { l: "Opset", v: model.opset, c: t.t2 }].map((s) =>
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
          <div style={{ padding: compact ? "4px 10px" : "4px 14px", borderTop: `1px solid ${t.bdr}`, display: "flex", justifyContent: "space-between", gap: 12, fontSize: 9.5, color: t.t2, background: t.glass, backdropFilter: "blur(8px)", flexShrink: 0 }}>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>ModelVisio · {model.format} · opset {model.opset} · {model.framework}{model.producer ? ` · ${model.producer}` : ""}</span>
            {!narrow && <span style={{ whiteSpace: "nowrap", flexShrink: 0 }}>Phase 1: Graph + Inspector + Compiler Checker + Auto-Fix + Deploy Recipes</span>}
          </div>
        </>
      )}
    </div>
    </ErrorBoundary>
  </ThemeCtx.Provider>;
}
