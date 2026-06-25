// Design tokens for ModelVisio. Dark/light palettes; components read the
// active palette via useT(). Shared across the web, desktop, and VS Code shells.

export type ThemeName = "dark" | "light";

export interface ThemePalette {
  mode: ThemeName;
  bg: string; bg1: string; bg2: string; bg3: string;
  bdr: string; bdr2: string;
  t0: string; t1: string; t2: string; t3: string;
  acc: string; acc2: string; acc3: string;
  suc: string; wrn: string; err: string;
  glow: string;
  cI: string; cB: string; cN: string; cH: string; cO: string;
  nodeGrad1: string; nodeGrad2: string;
  glass: string; glassBdr: string;
}

export const TH: Record<ThemeName, ThemePalette> = {
  dark: {
    // Neutral near-black surfaces (zinc/neutral ramp) — no blue tint. Reads like a
    // serious dev tool (GitHub/Vercel dark), not a glossy "AI" product.
    mode: "dark",
    bg: "#0A0A0B", bg1: "#141417", bg2: "#1C1C20", bg3: "#2A2A30",
    bdr: "#27272E", bdr2: "#3A3A44",
    t0: "#F4F4F5", t1: "#D2D2D8", t2: "#9A9AA4", t3: "#6E6E78",
    acc: "#4D8DFF", acc2: "#9B8AE6", acc3: "#3FB6A8",
    suc: "#34D399", wrn: "#FBBF24", err: "#F87171",
    glow: "rgba(255,255,255,0.06)",
    cI: "#6EE7B7", cB: "#60A5FA", cN: "#C084FC", cH: "#F472B6", cO: "#FBBF24",
    nodeGrad1: "#141417", nodeGrad2: "#1C1C20",
    glass: "rgba(20,20,23,0.88)", glassBdr: "#27272E",
  },
  light: {
    mode: "light",
    bg: "#F1F5F9", bg1: "#FFFFFF", bg2: "#F8FAFC", bg3: "#E2E8F0",
    bdr: "#CBD5E1", bdr2: "#94A3B8",
    t0: "#0F172A", t1: "#1E293B", t2: "#475569", t3: "#64748B",
    acc: "#0284C7", acc2: "#6366F1", acc3: "#0D9488",
    suc: "#059669", wrn: "#D97706", err: "#DC2626",
    glow: "rgba(2,132,199,0.08)",
    cI: "#059669", cB: "#2563EB", cN: "#7C3AED", cH: "#DB2777", cO: "#D97706",
    nodeGrad1: "#FFFFFF", nodeGrad2: "#F1F5F9",
    glass: "rgba(255,255,255,0.8)", glassBdr: "rgba(2,132,199,0.15)",
  },
};

// Layer-group → palette color key.
export const GC: Record<"input" | "backbone" | "neck" | "head" | "output", keyof ThemePalette> = {
  input: "cI", backbone: "cB", neck: "cN", head: "cH", output: "cO",
};
