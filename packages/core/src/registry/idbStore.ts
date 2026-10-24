// IndexedDB-backed registry store (web, Tauri desktop, VS Code webview).
//
// VS Code webviews are Chromium iframes and expose IndexedDB, but VS Code does
// not document it as durable webview storage (its documented persistence is
// getState/setState and the extension host's globalState). Treat the registry
// there as best-effort local storage and use Export JSON for anything that must
// survive. openRegistryStore() falls back to memory when IndexedDB is missing or
// blocked (private mode, sandboxed iframe), and the UI shows that.

import { MemoryRegistryStore, byCreated, type RegistryStore } from "./store";
import type { ModelEntry, ModelVersion } from "./types";

const DB_NAME = "modelvisio-registry";
const DB_VERSION = 1;
const MODELS = "models";
const VERSIONS = "versions";
const BLOBS = "blobs";

function req<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function done(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error("IndexedDB transaction aborted"));
  });
}

export function openIdb(factory: IDBFactory = indexedDB, name = DB_NAME): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = factory.open(name, DB_VERSION);
    r.onupgradeneeded = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains(MODELS)) db.createObjectStore(MODELS, { keyPath: "id" });
      if (!db.objectStoreNames.contains(VERSIONS)) db.createObjectStore(VERSIONS, { keyPath: "id" }).createIndex("modelId", "modelId");
      if (!db.objectStoreNames.contains(BLOBS)) db.createObjectStore(BLOBS);
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.onblocked = () => reject(new Error("IndexedDB open blocked by another tab"));
  });
}

export class IdbRegistryStore implements RegistryStore {
  readonly kind = "indexeddb";
  readonly persistent = true;
  constructor(private db: IDBDatabase) {}

  private store(name: string, mode: IDBTransactionMode = "readonly") {
    return this.db.transaction(name, mode).objectStore(name);
  }

  async listModels() { return req<ModelEntry[]>(this.store(MODELS).getAll()); }
  async getModel(id: string) { return req<ModelEntry | undefined>(this.store(MODELS).get(id)); }
  async putModel(entry: ModelEntry) { await req(this.store(MODELS, "readwrite").put(entry)); }

  async deleteModel(id: string) {
    const versions = await this.listVersions(id);
    const tx = this.db.transaction([MODELS, VERSIONS, BLOBS], "readwrite");
    tx.objectStore(MODELS).delete(id);
    for (const v of versions) {
      tx.objectStore(VERSIONS).delete(v.id);
      tx.objectStore(BLOBS).delete(v.id);
    }
    await done(tx);
  }

  async listVersions(modelId: string) {
    const all = await req<ModelVersion[]>(this.store(VERSIONS).index("modelId").getAll(modelId));
    return all.sort(byCreated);
  }
  async getVersion(id: string) { return req<ModelVersion | undefined>(this.store(VERSIONS).get(id)); }
  async putVersion(v: ModelVersion) { await req(this.store(VERSIONS, "readwrite").put(v)); }
  async deleteVersion(id: string) {
    const tx = this.db.transaction([VERSIONS, BLOBS], "readwrite");
    tx.objectStore(VERSIONS).delete(id);
    tx.objectStore(BLOBS).delete(id);
    await done(tx);
  }

  async putBlob(versionId: string, bytes: ArrayBuffer) { await req(this.store(BLOBS, "readwrite").put(bytes, versionId)); }
  async getBlob(versionId: string) { return req<ArrayBuffer | undefined>(this.store(BLOBS).get(versionId)); }
  async deleteBlob(versionId: string) { await req(this.store(BLOBS, "readwrite").delete(versionId)); }

  async clear() {
    const tx = this.db.transaction([MODELS, VERSIONS, BLOBS], "readwrite");
    tx.objectStore(MODELS).clear();
    tx.objectStore(VERSIONS).clear();
    tx.objectStore(BLOBS).clear();
    await done(tx);
  }
}

/** IndexedDB store when available, else a (non-persistent) memory store. */
export async function openRegistryStore(): Promise<RegistryStore> {
  try {
    if (typeof indexedDB === "undefined") return new MemoryRegistryStore();
    return new IdbRegistryStore(await openIdb());
  } catch {
    return new MemoryRegistryStore();
  }
}

let shared: Promise<RegistryStore> | null = null;

/** One store per page: the registry tab can remount without reopening the database. */
export function sharedRegistryStore(): Promise<RegistryStore> {
  shared ??= openRegistryStore();
  return shared;
}

export type QuotaCheck ={ ok: boolean; usage?: number; quota?: number; reason?: string };

/**
 * Can `bytes` more be stored? Uses navigator.storage.estimate() and keeps a 10%
 * safety margin. When the estimate API is missing the answer is ok=true with no
 * numbers (the write itself will fail with QuotaExceededError if it must).
 */
export async function checkStorageQuota(bytes: number): Promise<QuotaCheck> {
  const est = typeof navigator !== "undefined" ? navigator.storage?.estimate : undefined;
  if (!est) return { ok: true, reason: "Storage estimate API unavailable" };
  try {
    const { usage = 0, quota = 0 } = await navigator.storage.estimate();
    if (quota <= 0) return { ok: true, usage, quota, reason: "Quota unknown" };
    const free = quota * 0.9 - usage;
    return free >= bytes
      ? { ok: true, usage, quota }
      : { ok: false, usage, quota, reason: `Needs ${fmtMB(bytes)}, about ${fmtMB(Math.max(0, free))} free of ${fmtMB(quota)}` };
  } catch (e) {
    return { ok: true, reason: e instanceof Error ? e.message : String(e) };
  }
}

const fmtMB = (b: number) => `${(b / 1e6).toFixed(1)} MB`;
