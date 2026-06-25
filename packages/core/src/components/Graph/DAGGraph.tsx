import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { useT } from "../../theme/ThemeContext";
import { GC } from "../../theme/theme";
import { computeGraph, edgePath, type LayoutOpts, type NodeLine } from "./dagreLayout";
import { exportPng, exportSvg } from "./export";

const HEADER = 22;

export type DAGGraphHandle = {
  fit: () => void;
  exportSvg: () => void;
  exportPng: () => void;
};

type Props = LayoutOpts & {
  model: Model;
  selected: ModelLayer | null;
  onSelect: (l: ModelLayer) => void;
  qHeatmap: boolean;
  zoom: number;
  onZoomChange: (z: number) => void;
  query: string;
};

// Netron-style category hues (the colored node header). The body fill is derived
// per-theme from this hue so nodes read clearly on BOTH the dark and light canvas
// — the old fixed dark-navy fills looked like black boxes in light mode.
const CAT_DEFS: { re: RegExp; hue: string }[] = [
  { re: /conv/, hue: "#2563EB" },
  { re: /gemm|matmul|linear|fullyconnected|dense/, hue: "#7C3AED" },
  { re: /sigmoid|silu|relu|tanh|gelu|swish|hardswish|elu|clip|softmax|logsoftmax|^act/, hue: "#DC2626" },
  { re: /maxpool|averagepool|globalaverage|globalmax|pool/, hue: "#0D9488" },
  { re: /batchnorm|instancenorm|layernorm|\bnorm\b|^bn/, hue: "#059669" },
  { re: /concat|split|route|slice|reshape|transpose|flatten|gather|squeeze|pad|resize|upsample/, hue: "#B45309" },
  { re: /add|sub|mul|div|sum|mean|max|min/, hue: "#64748B" },
];
function catHue(l: ModelLayer): string {
  const s = (l.op + " " + l.type).toLowerCase();
  for (const c of CAT_DEFS) if (c.re.test(s)) return c.hue;
  return "#64748B"; // input/output/tensor/weight/identity
}
// Blend `hex` over `base` by `ratio` (0..1) → an opaque solid color.
function mix(hex: string, base: string, ratio: number): string {
  const h = parseInt(hex.slice(1), 16), b = parseInt(base.slice(1), 16);
  const ch = (sh: number) => Math.round(((h >> sh) & 255) * ratio + ((b >> sh) & 255) * (1 - ratio));
  return `#${((1 << 24) + (ch(16) << 16) + (ch(8) << 8) + ch(0)).toString(16).slice(1)}`;
}

function lineColor(kind: NodeLine["kind"], t: ReturnType<typeof useT>): string {
  switch (kind) {
    case "weight": return t.t1;
    case "name": return t.t3;
    case "shape": return t.acc;
    case "metric": return t.acc2;
  }
}

export const DAGGraph = forwardRef<DAGGraphHandle, Props>(function DAGGraph(
  { model, selected, onSelect, direction, showWeights, showNames, showMetrics, qHeatmap, zoom, onZoomChange, query },
  ref,
) {
  const t = useT();
  const scroller = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const layout = useMemo(
    () => computeGraph(model, { direction, showWeights, showNames, showMetrics }),
    [model, direction, showWeights, showNames, showMetrics],
  );

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return new Set<number>();
    return new Set(model.layers.filter((l) => (l.name + " " + l.op + " " + l.type).toLowerCase().includes(q)).map((l) => l.id));
  }, [query, model]);

  const fit = () => {
    const c = scroller.current;
    if (!c || !layout.width) return;
    // Fit to WIDTH only: scale so the graph fills the available width and the
    // user scrolls vertically — instead of shrinking it to cram the full height
    // (which makes a tall, Netron-style vertical graph tiny).
    const z = c.clientWidth / (layout.width + 20);
    onZoomChange(Math.max(0.1, Math.min(1.5, z)));
    c.scrollTo({ left: 0, top: 0 });
  };

  useImperativeHandle(ref, () => ({
    fit,
    exportSvg: () => svgRef.current && exportSvg(svgRef.current, model.name || "graph"),
    exportPng: () => svgRef.current && exportPng(svgRef.current, model.name || "graph", t.bg),
  }));

  // Center the first search match.
  useEffect(() => {
    if (matches.size === 0) return;
    const first = layout.nodes.find((n) => matches.has(n.id));
    const c = scroller.current;
    if (first && c) c.scrollTo({ left: first.x * zoom - c.clientWidth / 2, top: first.y * zoom - c.clientHeight / 2, behavior: "smooth" });
  }, [matches, layout, zoom]);

  // Drag-to-pan.
  const pan = useRef<{ x: number; y: number; l: number; t: number } | null>(null);
  const onDown = (e: React.MouseEvent) => {
    if ((e.target as Element).closest("[data-node]")) return; // let node clicks through
    const c = scroller.current; if (!c) return;
    pan.current = { x: e.clientX, y: e.clientY, l: c.scrollLeft, t: c.scrollTop };
  };
  const onMove = (e: React.MouseEvent) => {
    const c = scroller.current; if (!c || !pan.current) return;
    c.scrollLeft = pan.current.l - (e.clientX - pan.current.x);
    c.scrollTop = pan.current.t - (e.clientY - pan.current.y);
  };
  const endPan = () => { pan.current = null; };

  return (
    <div
      ref={scroller}
      onMouseDown={onDown}
      onMouseMove={onMove}
      onMouseUp={endPan}
      onMouseLeave={endPan}
      style={{
        flex: 1, overflow: "auto", position: "relative", cursor: pan.current ? "grabbing" : "default",
        background: t.bg,
      }}
    >
      <div style={{ position: "absolute", inset: 0, backgroundImage: `radial-gradient(${t.bdr} 1px, transparent 1px)`, backgroundSize: "22px 22px", opacity: 0.25, pointerEvents: "none" }} />
      <svg
        ref={svgRef}
        width={layout.width * zoom}
        height={layout.height * zoom}
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        style={{ display: "block", position: "relative" }}
      >
        <defs>
          <marker id="emarr" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={t.t3} />
          </marker>
          <filter id="emsel" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor={t.acc} floodOpacity="0.45" />
          </filter>
        </defs>

        {/* Edges */}
        <g fill="none">
          {layout.edges.map((e, i) => {
            const active = selected?.id === e.from || selected?.id === e.to;
            const showLabel = !!e.label && e.lx != null && e.ly != null;
            const lw = e.label.length * 5.3 + 8; // ≤ the box dagre reserved for it
            return (
              <g key={i}>
                <path d={edgePath(e.points)} stroke={active ? t.acc : t.bdr2} strokeWidth={active ? 2 : 1.25} markerEnd="url(#emarr)" />
                {showLabel && (
                  <g>
                    <rect x={(e.lx as number) - lw / 2} y={(e.ly as number) - 7} width={lw} height={14} rx={3} fill={t.bg} opacity={0.9} />
                    <text x={e.lx as number} y={e.ly as number} textAnchor="middle" dominantBaseline="central" fontSize={8.5} fill={active ? t.acc : t.t2} fontFamily="'JetBrains Mono',monospace">{e.label}</text>
                  </g>
                )}
              </g>
            );
          })}
        </g>

        {/* Nodes */}
        {layout.nodes.map((n) => {
          const l = n.layer;
          const x = n.x - n.w / 2, y = n.y - n.h / 2;
          const hue = catHue(l);
          const light = t.mode === "light";
          const sel = selected?.id === l.id;
          const matched = matches.has(l.id);
          const gc = t[GC[l.group]] || t.t3;
          const qc = qHeatmap && l.qSens > 0 ? (l.qSens > 0.12 ? "#EF4444" : l.qSens > 0.06 ? "#F59E0B" : "#22C55E") : null;
          const body = qc ? mix(qc, t.bg1, light ? 0.14 : 0.16) : mix(hue, t.bg1, light ? 0.10 : 0.20);
          const stroke = sel ? t.acc : matched ? t.wrn : qc || hue + (light ? "66" : "88");
          return (
            <g key={l.id} data-node transform={`translate(${x},${y})`} style={{ cursor: "pointer" }} filter={sel ? "url(#emsel)" : undefined} onClick={() => onSelect(l)}>
              <rect width={n.w} height={n.h} rx={7} fill={body} stroke={stroke} strokeWidth={sel || matched ? 2 : 1.25} />
              <path d={`M0,7 a7,7 0 0 1 7,-7 h${n.w - 14} a7,7 0 0 1 7,7 v13 h-${n.w} z`} fill={hue} />
              <text x={9} y={15} fontSize={11.5} fontWeight={700} fill="#fff" fontFamily="'JetBrains Mono',monospace">{l.op}</text>
              {(l.compIssues?.length ?? 0) > 0 && <circle cx={n.w - 9} cy={10} r={3.5} fill={t.err} />}
              {n.lines.map((ln, i) => (
                <text key={i} x={9} y={HEADER + 14 + i * 15} fontSize={10} fill={lineColor(ln.kind, t)} fontFamily="'JetBrains Mono',monospace">{ln.text}</text>
              ))}
              <rect x={0} y={0} width={3} height={n.h} rx={1.5} fill={gc} />
            </g>
          );
        })}
      </svg>
    </div>
  );
});
