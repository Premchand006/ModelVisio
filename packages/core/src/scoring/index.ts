// Roofline-grounded model→hardware scoring. Pure TS, usable from any shell.
export {
  estimateRoofline, modelCompute, peakActivationBytes, pickPrecision,
  classifyWorkload, peakOpsPerSec, utilFor, calibratedUtil,
  type Roofline, type WorkloadClass, type ModelCompute,
} from "./roofline";
export {
  scoreDevice, scoreAll,
  type ScoreResult, type SubScores, type Banner, type Profile,
} from "./scoreDevice";
