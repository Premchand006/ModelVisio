// Model registry data model. Pure types, shared by the UI (IndexedDB store),
// the CLI (JSON file in a repo) and the import/export format.
//
// A registry holds model ENTRIES (a named model a team tracks over time), each
// with an ordered list of VERSIONS. A version is an analysis SNAPSHOT of one
// model file: hash, stats, per-layer fingerprint, compiler-compat matrix and
// hardware estimates. The model bytes are NOT part of a snapshot; storing them
// is opt-in (see RegistryStore.putBlob) for privacy and storage-quota reasons.

/** Schema tag written into exported registry JSON. Bump on breaking changes. */
export const REGISTRY_SCHEMA = "modelvisio.registry/v1";

export type ModelEntry = {
  id: string;
  name: string;
  tags: string[];
  description?: string;
  createdAt: string; // ISO-8601
  updatedAt: string; // ISO-8601
};

/** Compact per-layer record used for structural diffs between versions. */
export type LayerFingerprint = {
  name: string;
  op: string;
  shape: string;
  params: number;
  flops: number;
};

export type CompatIssueRecord = {
  severity: "warn" | "error";
  msg: string;
  /** Layer the issue is attached to; absent for device-level findings (memory fit, format). */
  layer?: string;
  source: "op" | "device";
};

/** Compiler-compat result for one target runtime. */
export type CompatResult = {
  target: string;
  label: string;
  deviceId?: string;
  errors: number;
  warnings: number;
  issues: CompatIssueRecord[];
};

/** Hardware estimate for one reference device (from the roofline scorer). */
export type DeviceEstimate = {
  deviceId: string;
  deviceName: string;
  overall: number;
  fps: number;
  latencyMs: number;
  fits: boolean;
  compileFail: boolean;
  precision: string;
  confidence: "high" | "medium" | "low";
};

export type ModelStats = {
  layers: number;
  params: number;
  flops: number;
  macs: number;
  /** Sum of per-layer activation memory reported by the parser (bytes). */
  memBytes: number;
};

/** Everything derived from one model file; the content of a version minus bookkeeping. */
export type ModelSnapshot = {
  fileName: string;
  sizeBytes: number;
  sha256: string;
  format: string;
  framework: string;
  opset: number;
  producer?: string;
  inputShape: number[];
  outputShape: number[];
  stats: ModelStats;
  layers: LayerFingerprint[];
  compat: CompatResult[];
  estimates: DeviceEstimate[];
};

export type ModelVersion = ModelSnapshot & {
  id: string;
  modelId: string;
  /** Auto semver-ish ("1.0.0", "1.1.0", ...) or a user label ("v2-int8"). Unique per model. */
  version: string;
  createdAt: string; // ISO-8601
  notes?: string;
  /** True when the original model bytes were stored alongside (opt-in). */
  fileStored?: boolean;
};

export type RegistryModel = ModelEntry & { versions: ModelVersion[] };

/** The import/export format (UI "Export" button and the CLI's repo JSON file). */
export type RegistryFile = {
  schema: typeof REGISTRY_SCHEMA;
  exportedAt: string;
  models: RegistryModel[];
};
