// Cross-compiler optimization search (pure TS).
export {
  runSearch, paretoFront, dominates, runtimesFor, supportedPrecisions, scriptKeyFor,
  applicableFixes, fixSubsets, canRescale, scaleResolution, CPU_FALLBACK_SLOWDOWN, PARTITION_SWITCH_MS,
  type SearchOptions, type SearchResult, type ConfigResult, type Constraints, type PrecisionSupport, type PruneReason,
} from "./search";
export { defaultRisk, heuristicRisk, sensitivityRisk, structuralRisk, FIX_RISK, type RiskEstimator, type RiskContext } from "./risk";
export { buildRecipe, convOptsFor, type ConfigRecipe, type RecipeStep } from "./recipe";
