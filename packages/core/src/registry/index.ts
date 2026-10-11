// Model registry: local-first history of analyzed models and their versions.
// Pure (no React). Also served as the `@modelvisio/core/registry` subpath so the
// CLI can use it without pulling in the UI.

export * from "./types";
export {
  COMPAT_TARGETS, REFERENCE_DEVICES, resolveTarget, knownTargetIds, findDevice, checkCompat, compatMatrix,
  type CompatTarget,
} from "./compat";
export { sha256Hex, snapshotModel, modelStats, layerFingerprints, deviceEstimates, type SnapshotOptions } from "./snapshot";
export {
  diffSnapshots, diffVersions, diffLayers, delta,
  type VersionDiff, type DiffOptions, type Delta, type LayerChange, type CompatChange, type LatencyChange, type Regression,
} from "./diff";
export { MemoryRegistryStore, type RegistryStore } from "./store";
export { IdbRegistryStore, openRegistryStore, sharedRegistryStore, openIdb, checkStorageQuota, type QuotaCheck } from "./idbStore";
export {
  registerModel, registerSnapshot, updateModelEntry, nextVersion, normalizeTags, newId,
  exportRegistry, serializeRegistry, parseRegistryFile, importRegistry, storeFromFile,
  searchModels, allTags,
  type RegisterOptions, type RegisterResult, type ImportResult,
} from "./registry";
