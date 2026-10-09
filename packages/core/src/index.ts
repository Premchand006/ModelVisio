// @modelvisio/core — the product. ALL app UI lives here.
// The three apps (web, desktop, vscode) are thin shells that mount these exports.

export { App, type AppApi } from "./App";
export { useT } from "./theme/ThemeContext";
export { useViewport, type Viewport } from "./hooks/useViewport";
export { TH, GC, type ThemeName, type ThemePalette } from "./theme/theme";
export { demoModel } from "./demo/demoModel";

// Bring-your-own-key helpers (desktop AI copilot). The provider/key getters are
// also used by the desktop proxy (apps/web/src/tauri.ts).
export {
  GEMINI_KEY_STORAGE, GROK_KEY_STORAGE, AI_PROVIDER_STORAGE, isDesktop, isVsCodeWebview,
  getAiProvider, setAiProvider, getUserApiKey, setUserApiKey, type AiProvider,
} from "./utils/apiKey";

// File saves. Shells where `<a download>` is a no-op (VS Code WebView) install
// a save handler to route every export to a native Save dialog.
export { downloadFile, setSaveHandler, type SaveHandler } from "./utils/download";

// Roofline-grounded hardware scoring (pure; usable headless).
export {
  scoreDevice, scoreAll, estimateRoofline, modelCompute, peakActivationBytes,
  classifyWorkload, type ScoreResult, type SubScores, type Banner, type Profile,
  type Roofline, type WorkloadClass,
} from "./scoring";
export { HW, FORMATS, BYTES, type DeviceSpec, type Precision } from "./data/hardware";
export { FIXES, applyFix, countApplicable, applicableLayers, fixById, type FixId, type FixDef } from "./fixes/transforms";

// Individual components, exported so shells can compose custom layouts if needed.
export { GraphPanel } from "./components/Graph/GraphPanel";
export { DAGGraph } from "./components/Graph/DAGGraph";
export { Inspector } from "./components/Inspector";
export { CompilerChecker } from "./components/CompilerChecker";
export { HWReport } from "./components/HWReport";
export { Converter } from "./components/Converter";
export { DeployRecipe } from "./components/DeployRecipe";
export { Chat } from "./components/Chat";
export { BuildingPage } from "./components/BuildingPage";
export { Badge } from "./components/Badge";
export { Tabs } from "./components/Tabs";
export { ErrorBoundary } from "./components/ErrorBoundary";
