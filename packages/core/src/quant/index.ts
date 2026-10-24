// Quantization-sensitivity analysis and mixed-precision plan export (pure TS).
export {
  analyzeQuantization, quantSqnrDb, sqnrRisk, gaussianCrest, isDepthwise,
  DISCLAIMER as QUANT_DISCLAIMER, REC_LABEL, REC_BYTES,
  type QuantAnalysis, type QuantSummary, type QuantOptions, type QuantPrecision,
  type LayerSensitivity, type PrecisionEstimate, type Recommendation, type RiskLevel,
  type LayerKind, type StatsSource,
} from "./sensitivity";
export { planToJson, planJsonString, ortQuantizeScript, tfliteQuantizeScript, planHeadline, type PlanJson } from "./exportPlan";
