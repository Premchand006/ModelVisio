import { createContext, useContext } from "react";
import type { ThemePalette } from "./theme";

// The active palette is provided by <App>, which owns the dark/light state.
export const ThemeCtx = createContext<ThemePalette | null>(null);

/** Read the active theme palette. */
export function useT(): ThemePalette {
  const ctx = useContext(ThemeCtx);
  if (!ctx) throw new Error("useT must be used within the App theme provider");
  return ctx;
}
