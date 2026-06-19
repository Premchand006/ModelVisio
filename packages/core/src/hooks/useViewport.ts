import { useEffect, useState } from "react";

export type Viewport = {
  /** Live viewport width in px. */
  vw: number;
  /** < 860px — stack the graph + sidebar vertically instead of side-by-side. */
  narrow: boolean;
  /** < 560px — phone-class: drop secondary chrome, tighten padding. */
  compact: boolean;
};

/**
 * Tracks viewport width and the app's two layout breakpoints. SSR-safe (assumes
 * a desktop width when `window` is absent). Shared so any component can adapt to
 * the surface it's mounted in — web, the resizable desktop window, or the VS
 * Code editor pane.
 */
export function useViewport(): Viewport {
  const [vw, setVw] = useState(() => (typeof window !== "undefined" ? window.innerWidth : 1280));
  useEffect(() => {
    const on = () => setVw(window.innerWidth);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return { vw, narrow: vw < 860, compact: vw < 560 };
}
