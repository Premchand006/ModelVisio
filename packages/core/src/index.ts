// @modelvisio/core: the product. ALL app UI lives here.
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

// Model registry (pure; also served headless as `@modelvisio/core/registry`).
export {
  MemoryRegistryStore, IdbRegistryStore, openRegistryStore, registerModel, exportRegistry, importRegistry,
  parseRegistryFile, serializeRegistry, diffVersions, snapshotModel, COMPAT_TARGETS, REGISTRY_SCHEMA,
  type RegistryStore, type RegistryFile, type ModelEntry, type ModelVersion, type ModelSnapshot, type VersionDiff,
} from "./registry";
export { ModelRegistry } from "./components/ModelRegistry";

// Phase 3: AI performance investigator + cross-compiler optimization search.
export {
  buildEvidence, fixesResolving, layerRoofline, prepareLayers, rooflineFromPrepared,
  type EvidenceOptions, type LayerTime, type LayerRooflineResult, type PreparedLayers,
} from "./investigate";
export {
  runSearch, paretoFront, dominates, runtimesFor, supportedPrecisions, applicableFixes, fixSubsets,
  scaleResolution, buildRecipe, defaultRisk, heuristicRisk, sensitivityRisk,
  type SearchOptions, type SearchResult, type ConfigResult, type Constraints, type RiskEstimator, type RiskContext,
} from "./optsearch";
export { Investigator } from "./components/Investigator";
export { OptimizationSearch } from "./components/OptimizationSearch";

// Phase 2: quantization heatmap + hardware-aware model advisor (pure logic + panels).
export {
  analyzeQuantization, quantSqnrDb, sqnrRisk, QUANT_DISCLAIMER, REC_LABEL,
  planToJson, planJsonString, ortQuantizeScript, tfliteQuantizeScript,
  type QuantAnalysis, type QuantSummary, type QuantOptions, type QuantPrecision,
  type LayerSensitivity, type PrecisionEstimate, type Recommendation, type PlanJson,
} from "./quant";
export {
  advise, simulate,
  type AdvisorResult, type Advice, type AdviceKind, type LayerBottleneck, type Perf, type AdviseOptions,
} from "./advisor/advise";
export { COLOR_MODES, computeNodeColors, viridis, type ColorMode } from "./components/Graph/colorBy";
export { QuantHeatmap } from "./components/QuantHeatmap";
export { ModelAdvisor } from "./components/ModelAdvisor";

// Phase 4: AI deployment agent + fleet simulation.
export { runDeployment, type RunDeploymentArgs } from "./deploy/agent";
export { createDeploySession, createDeployTools, agentContext, type DeploySession } from "./deploy/tools";
export { createDeployPlanner } from "./deploy/planner";
export {
  buildDeployPackage, packageToZip, resolveDecision,
  type DeployPackage, type DeployDecision, type ChecklistItem,
} from "./deploy/package";
export { generateRecipe, type RecipeBundle, type PackageFile } from "./deploy/recipes";
export { deployRecipes, deployRecipeText, DEPLOY_RECIPE_TARGETS } from "./deploy/recipeTemplates";
export { parseGoal, resolveDevice, type DeployGoal } from "./deploy/goal";
export {
  RUNTIMES, DEPLOY_PRECISIONS, runtimesFor as deployRuntimesFor, precisionsFor, runtimeEff, parseRuntime, parsePrecision,
  type RuntimeId, type DeployPrecision, type RuntimeInfo,
} from "./deploy/runtimes";
export {
  simulateFleet, queueStats, responseCdf, sloAttainment, mixtureQuantile, mulberry32, histogram,
  fleetResultToJson, fleetResultToCsv,
  type FleetEntry, type FleetOptions, type FleetResult, type FleetAggregate, type ClassResult,
  type MonteCarloResult, type HistBin, type ThermalProfile, type PowerMode, type QueueModel,
} from "./fleet/simulate";
export { estimateDevice, canRun, type DeviceEstimate } from "./fleet/estimate";
export { FLEET_PRESETS, presetEntries, type FleetPreset } from "./fleet/presets";
export { DeployAgent } from "./components/DeployAgent";
export { FleetSim } from "./components/FleetSim";

// Phase 2: live on-device benchmarking (onnxruntime-web in a dedicated worker,
// loaded only when a run starts).
export { Benchmark } from "./components/Benchmark";
export { startBenchmark, setBenchWasmUrl, benchWasmUrl, type BenchHandle, type StartArgs as BenchStartArgs } from "./bench/benchClient";
export { runBenchmark, type OrtLike, type OrtSessionLike } from "./bench/runner";
export { summarize as summarizeLatency, percentile, type LatencyStats } from "./bench/stats";
export {
  inputSpecsFromModel, resolveInputs, resolveDims, synthesizeData, dimKey,
  type InputSpec, type ResolvedInput, type DimOverrides,
} from "./bench/inputs";
export { detectCapabilities as detectBenchCapabilities, isBenchmarkable, type BenchCapabilities } from "./bench/capabilities";
export { predictFor, benchReport, type Prediction } from "./bench/compare";
export { DEFAULT_CONFIG as DEFAULT_BENCH_CONFIG, type BenchConfig, type BenchResult, type BenchEp, type BenchProgress } from "./bench/types";
