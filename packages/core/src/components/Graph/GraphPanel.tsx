import { useMemo, useRef, useState } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { useT } from "../../theme/ThemeContext";
import { DAGGraph, type DAGGraphHandle } from "./DAGGraph";
import { LayersList } from "./LayersList";
import { Tree } from "./Tree";
import { COLOR_MODES, computeNodeColors, viridisGradient, type ColorMode } from "./colorBy";
import { analyzeQuantization } from "../../quant/sensitivity";

type View = "graph" | "layers" | "tree";

export function GraphPanel({
  model, selected, onSelect, colorMode, onColorModeChange, streaming = false,
}: {
  model: Model;
  /** The model is a partial snapshot of a large file still being parsed. */
  streaming?: boolean;
  selected: ModelLayer | null;
  onSelect: (l: ModelLayer) => void;
  /** Optional controlled "Color by" mode (e.g. so the Quant panel can switch it on). */
  colorMode?: ColorMode;
  onColorModeChange?: (m: ColorMode) => void;
}) {
  const t = useT();
  const graphRef = useRef<DAGGraphHandle>(null);
  const [view, setView] = useState<View>("graph");
  const [direction, setDirection] = useState<"TB" | "LR">("TB");
  const [showWeights, setShowWeights] = useState(true);
  const [showNames, setShowNames] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [localColor, setLocalColor] = useState<ColorMode>("none");
  const color = colorMode ?? localColor;
  const setColor = (m: ColorMode) => { setLocalColor(m); onColorModeChange?.(m); };
  const quant = useMemo(() => (color === "q8" || color === "q4" ? analyzeQuantization(model) : null), [model, color]);
  const coloring = useMemo(() => computeNodeColors(model, color, t.bg3, quant), [model, color, t.bg3, quant]);
  const [zoom, setZoom] = useState(1);
  const [query, setQuery] = useState("");

  const isGraph = view === "graph";
  const btn = (active: boolean) => ({
    padding: "3px 8px", borderRadius: 4, border: `1px solid ${active ? t.acc : t.bdr}`,
    background: active ? t.acc + "1F" : "transparent", color: active ? t.acc : t.t2,
    fontSize: 11, cursor: "pointer", whiteSpace: "nowrap" as const,
  });
  const seg = (active: boolean) => ({
    padding: "3px 10px", borderRadius: 4, border: "none",
    background: active ? t.bg2 : "transparent", color: active ? t.t0 : t.t2,
    fontSize: 11, fontWeight: 600 as const, cursor: "pointer",
  });

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>
      {/* Toolbar */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", padding: "5px 8px", borderBottom: `1px solid ${t.bdr}`, background: t.bg1 }}>
        <div style={{ display: "flex", gap: 1, padding: 2, background: t.bg, borderRadius: 5 }}>
          {(["graph", "layers", "tree"] as View[]).map((v) => (
            <button key={v} type="button" style={seg(view === v)} onClick={() => setView(v)}>
              {v === "graph" ? "Graph" : v === "layers" ? "Layers" : "Tree"}
            </button>
          ))}
        </div>

        {isGraph && <>
          <span style={{ width: 1, height: 18, background: t.bdr }} />
          <button type="button" style={btn(false)} title="Zoom out" onClick={() => setZoom((z) => Math.max(0.1, +(z - 0.15).toFixed(2)))}>−</button>
          <span style={{ fontSize: 11, color: t.t2, width: 38, textAlign: "center", fontFamily: "'JetBrains Mono'" }}>{Math.round(zoom * 100)}%</span>
          <button type="button" style={btn(false)} title="Zoom in" onClick={() => setZoom((z) => Math.min(2, +(z + 0.15).toFixed(2)))}>+</button>
          <button type="button" style={btn(false)} title="Fit to width" onClick={() => graphRef.current?.fit()}>⤢ Fit</button>
          <button type="button" style={btn(false)} title="Actual size" onClick={() => setZoom(1)}>1:1</button>
          <span style={{ width: 1, height: 18, background: t.bdr }} />
          <button type="button" style={btn(direction === "LR")} title="Layout direction" onClick={() => setDirection((d) => (d === "TB" ? "LR" : "TB"))}>{direction === "TB" ? "↓ Vert" : "→ Horiz"}</button>
          <button type="button" style={btn(showWeights)} title="Show weights" onClick={() => setShowWeights((v) => !v)}>W</button>
          <button type="button" style={btn(showNames)} title="Show names" onClick={() => setShowNames((v) => !v)}>Names</button>
          <button type="button" style={btn(showMetrics)} title="Show shapes & FLOPs" onClick={() => setShowMetrics((v) => !v)}>∑</button>
        </>}

        {view !== "tree" && (
          <select
            value={color}
            onChange={(e) => setColor(e.target.value as ColorMode)}
            title="Color nodes by"
            aria-label="Color nodes by"
            style={{ padding: "3px 6px", borderRadius: 4, border: `1px solid ${color !== "none" ? t.acc : t.bdr}`, background: t.bg, color: color !== "none" ? t.acc : t.t2, fontSize: 11, cursor: "pointer" }}
          >
            {COLOR_MODES.map((m) => <option key={m.id} value={m.id}>{m.id === "none" ? "Color: None" : `Color: ${m.label}`}</option>)}
          </select>
        )}

        <span style={{ flex: 1 }} />

        {view !== "tree" && (
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Find node…"
            style={{ width: 160, padding: "4px 9px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 11, outline: "none" }}
          />
        )}

        {isGraph && <>
          <button type="button" style={btn(false)} title="Export SVG" onClick={() => graphRef.current?.exportSvg()}>SVG</button>
          <button type="button" style={btn(false)} title="Export PNG" onClick={() => graphRef.current?.exportPng()}>PNG</button>
        </>}
      </div>

      {/* Color-by legend */}
      {coloring && view !== "tree" && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 10px", borderBottom: `1px solid ${t.bdr}`, background: t.bg1, fontSize: 10, color: t.t2, flexWrap: "wrap" }}>
          <span style={{ fontWeight: 600, color: t.t1 }}>{coloring.legend.label}</span>
          <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>{coloring.legend.min}</span>
          <span aria-hidden style={{ width: 140, height: 8, borderRadius: 4, background: viridisGradient(), border: `1px solid ${t.bdr}` }} />
          <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>{coloring.legend.max}</span>
          {coloring.legend.scale === "log" && <span style={{ color: t.t3 }}>(log scale)</span>}
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span aria-hidden style={{ width: 10, height: 10, borderRadius: 2, background: t.bg3, border: `1px solid ${t.bdr2}` }} />{coloring.legend.naLabel}
          </span>
          {(color === "q8" || color === "q4") && <span style={{ color: t.t3 }}>Statistical estimate, not measured accuracy</span>}
        </div>
      )}

      {/* View */}
      {view === "graph" && (
        <DAGGraph
          ref={graphRef}
          model={model}
          selected={selected}
          onSelect={onSelect}
          direction={direction}
          showWeights={showWeights}
          showNames={showNames}
          showMetrics={showMetrics}
          nodeColors={coloring?.colors ?? null}
          zoom={zoom}
          onZoomChange={setZoom}
          query={query}
          streaming={streaming}
        />
      )}
      {view === "layers" && <LayersList model={model} selected={selected} onSelect={onSelect} nodeColors={coloring?.colors ?? null} query={query} />}
      {view === "tree" && <Tree model={model} selected={selected} onSelect={onSelect} />}
    </div>
  );
}
