// "Color by" overlay for the graph and layer list (pure, no React).
// Maps a per-layer metric onto viridis, a perceptually uniform sequential scale
// whose lightness rises monotonically, so it reads in both the dark and light
// themes (and in grayscale / for common color-vision deficiencies).

import type { Model, ModelLayer } from "@modelvisio/parsers";
import type { QuantAnalysis } from "../../quant/sensitivity";

export type ColorMode = "none" | "q8" | "q4" | "flops" | "params" | "mem";

export const COLOR_MODES: { id: ColorMode; label: string }[] = [
  { id: "none", label: "None" },
  { id: "q8", label: "Quant sensitivity (INT8)" },
  { id: "q4", label: "Quant sensitivity (INT4)" },
  { id: "flops", label: "FLOPs" },
  { id: "params", label: "Params" },
  { id: "mem", label: "Memory" },
];

// matplotlib viridis, 9 evenly spaced stops.
const VIRIDIS = ["#440154", "#472c7a", "#3b518b", "#2c718e", "#21908d", "#27ad81", "#5cc863", "#aadc32", "#fde725"];

const rgb = (hex: string) => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const hex = (r: number, g: number, b: number) =>
  `#${((1 << 24) + (Math.round(r) << 16) + (Math.round(g) << 8) + Math.round(b)).toString(16).slice(1)}`;

/** Viridis color at t ∈ [0, 1]. */
export function viridis(t: number): string {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = rgb(VIRIDIS[i]), b = rgb(VIRIDIS[i + 1]);
  return hex(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f);
}

/** CSS linear-gradient of the scale, for legends. */
export function viridisGradient(dir = "to right"): string {
  return `linear-gradient(${dir}, ${VIRIDIS.join(", ")})`;
}

/** Readable text color (near-black or white) on a solid background. */
export function textOn(bg: string): string {
  const [r, g, b] = rgb(bg).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.32 ? "#111111" : "#FFFFFF";
}

export type ColorLegend = { label: string; min: string; max: string; scale: "linear" | "log"; naLabel: string };
export type NodeColoring = { colors: Map<number, string>; legend: ColorLegend };

const fmtN = (n: number) => (n >= 1e9 ? (n / 1e9).toFixed(1) + "G" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(Math.round(n)));
const fmtB = (n: number) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "MB" : n >= 1e3 ? (n / 1e3).toFixed(1) + "KB" : Math.round(n) + "B");

/**
 * Per-layer colors for a mode. Layers without a value (no weights, zero FLOPs,
 * layout-only ops for quant modes) get `naColor`. Returns null for "none".
 * Quant modes use the per-channel risk (0..1, linear); size metrics use a log
 * scale over the model's own nonzero range.
 */
export function computeNodeColors(model: Model, mode: ColorMode, naColor: string, quant?: QuantAnalysis | null): NodeColoring | null {
  if (mode === "none") return null;
  const colors = new Map<number, string>();

  if (mode === "q8" || mode === "q4") {
    for (const l of model.layers) {
      const s = quant?.byId.get(l.id);
      if (!s || !s.quantizable) { colors.set(l.id, naColor); continue; }
      colors.set(l.id, viridis((mode === "q8" ? s.int8 : s.int4).perChannelRisk));
    }
    return {
      colors,
      legend: { label: mode === "q8" ? "INT8 risk (estimate)" : "INT4 W4A8 risk (estimate)", min: "0%", max: "100%", scale: "linear", naLabel: "no arithmetic" },
    };
  }

  const get = (l: ModelLayer) => (mode === "flops" ? l.flops || l.macs * 2 : mode === "params" ? l.params : l.mem);
  const vals = model.layers.map(get).filter((v) => v > 0);
  const vMin = vals.reduce((m, v) => Math.min(m, v), Infinity);
  const vMax = vals.reduce((m, v) => Math.max(m, v), 0);
  if (vals.length === 0) {
    for (const l of model.layers) colors.set(l.id, naColor);
  } else {
    const lo = Math.log10(vMin);
    const hi = Math.log10(vMax);
    const span = hi - lo || 1;
    for (const l of model.layers) {
      const v = get(l);
      colors.set(l.id, v > 0 ? viridis(hi === lo ? 1 : (Math.log10(v) - lo) / span) : naColor);
    }
  }
  const f = mode === "mem" ? fmtB : fmtN;
  const min = vals.length ? f(vMin) : "0";
  const max = vals.length ? f(vMax) : "0";
  const label = mode === "flops" ? "FLOPs" : mode === "params" ? "Params" : "Output memory";
  return { colors, legend: { label, min, max, scale: "log", naLabel: mode === "params" ? "no weights" : "zero" } };
}
