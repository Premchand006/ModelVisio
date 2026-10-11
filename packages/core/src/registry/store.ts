// Storage abstraction for the model registry. The UI uses the IndexedDB store
// (idbStore.ts); tests and the CLI use the in-memory store (the CLI loads and
// saves it as a JSON file). Everything is async so both look the same.

import type { ModelEntry, ModelVersion } from "./types";

export interface RegistryStore {
  /** "indexeddb" | "memory" | a shell-provided kind. */
  readonly kind: string;
  /** False when data is lost on reload (memory fallback). */
  readonly persistent: boolean;

  listModels(): Promise<ModelEntry[]>;
  getModel(id: string): Promise<ModelEntry | undefined>;
  putModel(entry: ModelEntry): Promise<void>;
  /** Deletes the model, its versions and any stored files. */
  deleteModel(id: string): Promise<void>;

  /** Versions of one model, oldest first. */
  listVersions(modelId: string): Promise<ModelVersion[]>;
  getVersion(id: string): Promise<ModelVersion | undefined>;
  putVersion(v: ModelVersion): Promise<void>;
  /** Deletes the version and its stored file, if any. */
  deleteVersion(id: string): Promise<void>;

  /** Optional original model bytes, keyed by version id. */
  putBlob(versionId: string, bytes: ArrayBuffer): Promise<void>;
  getBlob(versionId: string): Promise<ArrayBuffer | undefined>;
  deleteBlob(versionId: string): Promise<void>;

  /** Remove everything. */
  clear(): Promise<void>;
}

export const byCreated = (a: { createdAt: string }, b: { createdAt: string }) => a.createdAt.localeCompare(b.createdAt);

/** Non-persistent store: tests, the CLI, and the fallback when IndexedDB is unavailable. */
export class MemoryRegistryStore implements RegistryStore {
  readonly kind = "memory";
  readonly persistent = false;
  private models = new Map<string, ModelEntry>();
  private versions = new Map<string, ModelVersion>();
  private blobs = new Map<string, ArrayBuffer>();

  async listModels() { return [...this.models.values()].map(clone); }
  async getModel(id: string) { const m = this.models.get(id); return m && clone(m); }
  async putModel(entry: ModelEntry) { this.models.set(entry.id, clone(entry)); }
  async deleteModel(id: string) {
    this.models.delete(id);
    for (const v of [...this.versions.values()]) if (v.modelId === id) await this.deleteVersion(v.id);
  }

  async listVersions(modelId: string) {
    return [...this.versions.values()].filter((v) => v.modelId === modelId).sort(byCreated).map(clone);
  }
  async getVersion(id: string) { const v = this.versions.get(id); return v && clone(v); }
  async putVersion(v: ModelVersion) { this.versions.set(v.id, clone(v)); }
  async deleteVersion(id: string) { this.versions.delete(id); this.blobs.delete(id); }

  async putBlob(versionId: string, bytes: ArrayBuffer) { this.blobs.set(versionId, bytes.slice(0)); }
  async getBlob(versionId: string) { return this.blobs.get(versionId)?.slice(0); }
  async deleteBlob(versionId: string) { this.blobs.delete(versionId); }

  async clear() { this.models.clear(); this.versions.clear(); this.blobs.clear(); }
}

// Records are plain JSON, so a JSON round-trip is an exact deep copy and keeps
// callers from mutating stored state (matches IndexedDB's structured clone).
function clone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T;
}
