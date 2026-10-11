// High-level registry operations over any RegistryStore: register a model
// version, import/export the registry JSON, search and tag. Pure logic (no
// React); the UI and the CLI share it.

import type { Model } from "@modelvisio/parsers";
import { snapshotModel, type SnapshotOptions } from "./snapshot";
import { MemoryRegistryStore, type RegistryStore } from "./store";
import { REGISTRY_SCHEMA, type ModelEntry, type ModelSnapshot, type ModelVersion, type RegistryFile, type RegistryModel } from "./types";

export function newId(): string {
  const c = globalThis.crypto;
  if (c?.randomUUID) return c.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

const SEMVER = /^v?(\d+)\.(\d+)\.(\d+)$/;

/**
 * Next auto version: one minor above the highest semver-ish version present
 * ("1.0.0" for the first). User labels that aren't semver are ignored for
 * ordering, and the result never collides with an existing version string.
 */
export function nextVersion(existing: string[]): string {
  let best: [number, number, number] | null = null;
  for (const v of existing) {
    const m = SEMVER.exec(v);
    if (!m) continue;
    const t: [number, number, number] = [+m[1], +m[2], +m[3]];
    if (!best || t[0] > best[0] || (t[0] === best[0] && (t[1] > best[1] || (t[1] === best[1] && t[2] > best[2])))) best = t;
  }
  let [maj, min] = best ? [best[0], best[1] + 1] : [1, 0];
  const taken = new Set(existing);
  while (taken.has(`${maj}.${min}.0`)) min++;
  return `${maj}.${min}.0`;
}

export function normalizeTags(tags: string[] | string | undefined): string[] {
  const list = typeof tags === "string" ? tags.split(",") : tags ?? [];
  return [...new Set(list.map((t) => t.trim().toLowerCase()).filter(Boolean))].sort();
}

export type RegisterOptions = {
  /** Existing model entry to add the version to. Otherwise matched by name, else created. */
  modelId?: string;
  /** Model entry name (new entries). Defaults to the file name without extension. */
  name?: string;
  tags?: string[];
  description?: string;
  /** User version label; auto semver-ish when omitted. */
  label?: string;
  notes?: string;
  /** Add even if a version with the same SHA-256 already exists for this model. */
  allowDuplicate?: boolean;
  /** ISO timestamp override (tests, reproducible CLI output). */
  now?: string;
};

export type RegisterResult = {
  entry: ModelEntry;
  version: ModelVersion;
  /** True when the bytes matched an existing version and nothing was added. */
  duplicate: boolean;
};

const stripExt = (n: string) => n.replace(/\.[^.\\/]+$/, "");

/** Add a snapshot as a new version (creating the model entry if needed). */
export async function registerSnapshot(store: RegistryStore, snap: ModelSnapshot, opts: RegisterOptions = {}): Promise<RegisterResult> {
  const now = opts.now ?? new Date().toISOString();
  let entry: ModelEntry | undefined;
  if (opts.modelId) {
    entry = await store.getModel(opts.modelId);
    if (!entry) throw new Error(`Unknown model id "${opts.modelId}"`);
  } else {
    const name = (opts.name ?? stripExt(snap.fileName)).trim();
    const all = await store.listModels();
    entry = all.find((m) => m.name.toLowerCase() === name.toLowerCase());
    if (!entry) {
      entry = { id: newId(), name, tags: normalizeTags(opts.tags), description: opts.description, createdAt: now, updatedAt: now };
      await store.putModel(entry);
    }
  }

  const versions = await store.listVersions(entry.id);
  if (!opts.allowDuplicate && snap.sha256) {
    const dup = versions.find((v) => v.sha256 === snap.sha256);
    if (dup) return { entry, version: dup, duplicate: true };
  }

  const label = opts.label?.trim();
  if (label && versions.some((v) => v.version === label)) throw new Error(`Version "${label}" already exists for ${entry.name}`);
  // Keep createdAt strictly increasing so ordering is stable even within one millisecond.
  const last = versions[versions.length - 1];
  const createdAt = last && last.createdAt >= now ? new Date(Date.parse(last.createdAt) + 1).toISOString() : now;
  const version: ModelVersion = {
    ...snap,
    id: newId(),
    modelId: entry.id,
    version: label || nextVersion(versions.map((v) => v.version)),
    createdAt,
    notes: opts.notes?.trim() || undefined,
  };
  await store.putVersion(version);

  const tags = opts.tags ? normalizeTags([...entry.tags, ...opts.tags]) : entry.tags;
  entry = { ...entry, tags, description: opts.description ?? entry.description, updatedAt: createdAt };
  await store.putModel(entry);
  return { entry, version, duplicate: false };
}

/** Analyze `model` and register it; optionally keep the original bytes. */
export async function registerModel(
  store: RegistryStore,
  model: Model,
  opts: RegisterOptions & SnapshotOptions & { storeFile?: boolean } = {},
): Promise<RegisterResult> {
  const snap = await snapshotModel(model, opts);
  const res = await registerSnapshot(store, snap, opts);
  if (opts.storeFile && opts.bytes && !res.duplicate) {
    const buf = opts.bytes instanceof Uint8Array ? (opts.bytes.slice().buffer as ArrayBuffer) : opts.bytes;
    await store.putBlob(res.version.id, buf);
    res.version = { ...res.version, fileStored: true };
    await store.putVersion(res.version);
  }
  return res;
}

export async function updateModelEntry(store: RegistryStore, id: string, patch: Partial<Pick<ModelEntry, "name" | "tags" | "description">>): Promise<ModelEntry> {
  const e = await store.getModel(id);
  if (!e) throw new Error(`Unknown model id "${id}"`);
  const next: ModelEntry = {
    ...e, ...patch,
    tags: patch.tags ? normalizeTags(patch.tags) : e.tags,
    updatedAt: new Date().toISOString(),
  };
  await store.putModel(next);
  return next;
}

// ── Import / export ─────────────────────────────────────────────────────────

/** Whole registry as the shareable JSON document (stored files are not included). */
export async function exportRegistry(store: RegistryStore, now = new Date().toISOString()): Promise<RegistryFile> {
  const models: RegistryModel[] = [];
  const entries = (await store.listModels()).sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  for (const e of entries) {
    const versions = (await store.listVersions(e.id)).map(({ fileStored: _f, ...v }) => v);
    models.push({ ...e, versions });
  }
  return { schema: REGISTRY_SCHEMA, exportedAt: now, models };
}

export function serializeRegistry(file: RegistryFile): string {
  return JSON.stringify(file, null, 2) + "\n";
}

/** Parse and validate registry JSON (text or an already-parsed object). Throws with a readable message. */
export function parseRegistryFile(input: string | unknown): RegistryFile {
  let o: unknown = input;
  if (typeof input === "string") {
    try { o = JSON.parse(input); } catch (e) { throw new Error(`Registry file is not valid JSON: ${(e as Error).message}`); }
  }
  if (!o || typeof o !== "object") throw new Error("Registry file must be a JSON object");
  const f = o as Partial<RegistryFile>;
  if (f.schema !== REGISTRY_SCHEMA) throw new Error(`Unsupported registry schema "${String(f.schema)}" (expected "${REGISTRY_SCHEMA}")`);
  if (!Array.isArray(f.models)) throw new Error("Registry file has no models array");
  for (const m of f.models) {
    if (!m || typeof m.id !== "string" || typeof m.name !== "string" || !Array.isArray(m.versions)) {
      throw new Error("Registry file has a malformed model entry (needs id, name, versions)");
    }
    m.tags = Array.isArray(m.tags) ? m.tags : [];
    for (const v of m.versions) {
      if (!v || typeof v.id !== "string" || typeof v.version !== "string" || !v.stats) {
        throw new Error(`Model "${m.name}" has a malformed version (needs id, version, stats)`);
      }
      v.modelId = m.id;
      v.layers ??= [];
      v.compat ??= [];
      v.estimates ??= [];
    }
  }
  return { schema: REGISTRY_SCHEMA, exportedAt: f.exportedAt ?? "", models: f.models };
}

export type ImportResult = { models: number; versions: number; skipped: number };

/**
 * Load a registry document into a store. "merge" (default) keeps existing data
 * and skips versions already present (same id, or same bytes for that model);
 * "replace" clears the store first.
 */
export async function importRegistry(store: RegistryStore, file: RegistryFile, mode: "merge" | "replace" = "merge"): Promise<ImportResult> {
  if (mode === "replace") await store.clear();
  const res: ImportResult = { models: 0, versions: 0, skipped: 0 };
  for (const { versions, ...entry } of file.models) {
    const existing = await store.getModel(entry.id);
    if (existing) {
      await store.putModel({ ...existing, tags: normalizeTags([...existing.tags, ...entry.tags]), description: existing.description ?? entry.description });
    } else {
      await store.putModel({ ...entry, tags: normalizeTags(entry.tags) });
      res.models++;
    }
    const have = await store.listVersions(entry.id);
    for (const v of versions) {
      if (have.some((h) => h.id === v.id || (v.sha256 && h.sha256 === v.sha256))) { res.skipped++; continue; }
      await store.putVersion({ ...v, modelId: entry.id, fileStored: false });
      have.push(v);
      res.versions++;
    }
  }
  return res;
}

/** In-memory store seeded from a registry document (the CLI's working copy). */
export async function storeFromFile(file: RegistryFile): Promise<MemoryRegistryStore> {
  const s = new MemoryRegistryStore();
  await importRegistry(s, file, "replace");
  return s;
}

// ── Search ──────────────────────────────────────────────────────────────────

/** Case-insensitive match on name, description and tags; optional exact tag filter. */
export function searchModels<T extends ModelEntry>(models: T[], query: string, tag?: string): T[] {
  const q = query.trim().toLowerCase();
  return models
    .filter((m) => !tag || m.tags.includes(tag))
    .filter((m) => !q || m.name.toLowerCase().includes(q) || (m.description ?? "").toLowerCase().includes(q) || m.tags.some((t) => t.includes(q)))
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

export function allTags(models: ModelEntry[]): string[] {
  return [...new Set(models.flatMap((m) => m.tags))].sort();
}
