import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type { Model } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { Badge } from "./Badge";
import { downloadFile } from "../utils/download";
import { fB, fmt } from "../utils/format";
import {
  allTags, checkStorageQuota, diffVersions, exportRegistry, importRegistry, normalizeTags, parseRegistryFile,
  registerModel, searchModels, serializeRegistry, sharedRegistryStore, updateModelEntry, nextVersion,
  type Delta, type ModelEntry, type ModelVersion, type RegistryStore,
} from "../registry";

const stripExt = (n: string) => n.replace(/\.[^.\\/]+$/, "");
const pct = (p: number | null) => (p == null ? "" : ` (${p >= 0 ? "+" : ""}${p.toFixed(1)}%)`);
const when = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};

/**
 * Model registry: a local-first history of analyzed models. Register the loaded
 * model as a new version (hash, stats, compat matrix, hardware estimates; the
 * file itself only on request), browse/search/tag models, walk a version
 * timeline, diff any two versions, and import/export the registry as JSON (the
 * same file the `modelvisio registry` CLI keeps in a repo).
 */
export function ModelRegistry({ model, modelBytes, store: storeProp }: {
  model: Model | null;
  modelBytes: ArrayBuffer | null;
  /** Inject a store (tests, or a shell-provided persistent store). Defaults to IndexedDB with a memory fallback. */
  store?: RegistryStore;
}) {
  const t = useT();
  const [store, setStore] = useState<RegistryStore | null>(storeProp ?? null);
  const [models, setModels] = useState<(ModelEntry & { count: number })[]>([]);
  const [selId, setSelId] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [pair, setPair] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [tagFilter, setTagFilter] = useState<string | null>(null);
  const [msg, setMsg] = useState<{ kind: "ok" | "err" | "info"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  // Registration form.
  const [regTarget, setRegTarget] = useState<string>("__new");
  const [regName, setRegName] = useState("");
  const [regLabel, setRegLabel] = useState("");
  const [regTags, setRegTags] = useState("");
  const [regNotes, setRegNotes] = useState("");
  const [storeFile, setStoreFile] = useState(false);
  const [tagEdit, setTagEdit] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (storeProp) { setStore(storeProp); return; }
    let alive = true;
    void sharedRegistryStore().then((s) => { if (alive) setStore(s); });
    return () => { alive = false; };
  }, [storeProp]);

  const refresh = useCallback(async (s: RegistryStore, keepSel?: string | null) => {
    const entries = await s.listModels();
    const withCounts = await Promise.all(entries.map(async (e) => ({ ...e, count: (await s.listVersions(e.id)).length })));
    setModels(withCounts);
    const id = keepSel !== undefined ? keepSel : selId;
    if (id && withCounts.some((m) => m.id === id)) {
      setVersions(await s.listVersions(id));
    } else {
      setSelId(null);
      setVersions([]);
    }
  }, [selId]);

  useEffect(() => { if (store) void refresh(store); }, [store]); // eslint-disable-line react-hooks/exhaustive-deps

  // Default the form from the loaded model; target an existing entry with the same name.
  useEffect(() => {
    if (!model) return;
    const base = stripExt(model.name);
    setRegName(base);
    const hit = models.find((m) => m.name.toLowerCase() === base.toLowerCase());
    setRegTarget(hit ? hit.id : "__new");
  }, [model?.name, models.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const selModel = models.find((m) => m.id === selId) ?? null;
  const tags = useMemo(() => allTags(models), [models]);
  const shown = useMemo(() => searchModels(models, query, tagFilter ?? undefined), [models, query, tagFilter]);
  // Diff runs oldest → newest regardless of tick order.
  const picked = versions.filter((v) => pair.includes(v.id));
  const [vA, vB]: (ModelVersion | undefined)[] = picked.length === 2 ? picked : [];
  const diff = useMemo(() => (vA && vB ? diffVersions(vA, vB) : null), [vA, vB]);
  const autoVersion = regTarget === "__new" ? "1.0.0" : selId === regTarget ? nextVersion(versions.map((v) => v.version)) : "auto";

  async function selectModel(id: string) {
    if (!store) return;
    setSelId(id);
    setPair([]);
    setVersions(await store.listVersions(id));
    setTagEdit((models.find((m) => m.id === id)?.tags ?? []).join(", "));
  }

  async function run(fn: () => Promise<void>) {
    setBusy(true);
    setMsg(null);
    try { await fn(); } catch (e) { setMsg({ kind: "err", text: e instanceof Error ? e.message : String(e) }); } finally { setBusy(false); }
  }

  const onRegister = () => run(async () => {
    if (!store || !model) return;
    if (storeFile && modelBytes) {
      const q = await checkStorageQuota(modelBytes.byteLength);
      if (!q.ok) throw new Error(`Not enough browser storage to keep the file. ${q.reason ?? ""}`);
    }
    const res = await registerModel(store, model, {
      modelId: regTarget === "__new" ? undefined : regTarget,
      name: regTarget === "__new" ? regName || stripExt(model.name) : undefined,
      fileName: model.name,
      bytes: modelBytes,
      label: regLabel || undefined,
      tags: normalizeTags(regTags),
      notes: regNotes || undefined,
      storeFile: storeFile && !!modelBytes,
    });
    setRegLabel(""); setRegNotes("");
    setMsg(res.duplicate
      ? { kind: "info", text: `Same file is already registered as ${res.entry.name} ${res.version.version}.` }
      : { kind: "ok", text: `Registered ${res.entry.name} ${res.version.version}.` });
    await refresh(store, res.entry.id);
    setSelId(res.entry.id);
    setTagEdit(res.entry.tags.join(", "));
  });

  const onExport = () => run(async () => {
    if (!store) return;
    const file = await exportRegistry(store);
    downloadFile(serializeRegistry(file), "modelvisio-registry.json", "application/json");
    setMsg({ kind: "ok", text: `Exported ${file.models.length} model(s).` });
  });

  const onImportFile = (f: File | undefined) => run(async () => {
    if (!store || !f) return;
    const res = await importRegistry(store, parseRegistryFile(await f.text()));
    setMsg({ kind: "ok", text: `Imported ${res.models} model(s), ${res.versions} version(s)${res.skipped ? `, skipped ${res.skipped} already present` : ""}.` });
    await refresh(store);
  });

  const onSaveTags = () => run(async () => {
    if (!store || !selId) return;
    await updateModelEntry(store, selId, { tags: normalizeTags(tagEdit) });
    await refresh(store, selId);
  });

  const onDeleteModel = () => run(async () => {
    if (!store || !selModel) return;
    if (typeof confirm === "function" && !confirm(`Delete ${selModel.name} and all ${selModel.count} version(s)?`)) return;
    await store.deleteModel(selModel.id);
    await refresh(store, null);
  });

  const onDeleteVersion = (v: ModelVersion) => run(async () => {
    if (!store) return;
    if (typeof confirm === "function" && !confirm(`Delete version ${v.version}?`)) return;
    await store.deleteVersion(v.id);
    setPair((p) => p.filter((id) => id !== v.id));
    if (versions.length <= 1) { await store.deleteModel(v.modelId); await refresh(store, null); } else await refresh(store, v.modelId);
  });

  const onDownloadFile = (v: ModelVersion) => run(async () => {
    const bytes = await store?.getBlob(v.id);
    if (!bytes) throw new Error("The file for this version isn't stored.");
    downloadFile(new Uint8Array(bytes), v.fileName, "application/octet-stream");
  });

  const togglePair = (id: string) => setPair((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-2)));

  // ── styles ────────────────────────────────────────────────────────────────
  const card: CSSProperties = { background: t.bg2, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: 10 };
  const input: CSSProperties = { width: "100%", boxSizing: "border-box", padding: "4px 6px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t0, fontSize: 11 };
  const btn = (c: string, solid = false): CSSProperties => ({ padding: "4px 10px", borderRadius: 4, border: `1px solid ${c}`, background: solid ? c + "22" : "transparent", color: c, fontSize: 10.5, fontWeight: 600, cursor: busy ? "wait" : "pointer" });
  const label: CSSProperties = { fontSize: 9.5, color: t.t3, marginBottom: 2, display: "block" };
  const h: CSSProperties = { fontSize: 11.5, fontWeight: 700, color: t.t0, marginBottom: 6 };
  const mono: CSSProperties = { fontFamily: "'JetBrains Mono',monospace" };
  const deltaColor = (d: Delta, upIsBad = true) => (d.delta === 0 ? t.t2 : (d.delta > 0) === upIsBad ? t.wrn : t.suc);

  const deltaRow = (name: string, d: Delta, f: (n: number) => string, upIsBad = true) => (
    <tr key={name}>
      <td style={{ color: t.t2, padding: "2px 4px" }}>{name}</td>
      <td style={{ ...mono, padding: "2px 4px", color: t.t1 }}>{f(d.before)}</td>
      <td style={{ ...mono, padding: "2px 4px", color: t.t1 }}>{f(d.after)}</td>
      <td style={{ ...mono, padding: "2px 4px", color: deltaColor(d, upIsBad) }}>{d.delta === 0 ? "=" : `${d.delta > 0 ? "+" : "-"}${f(Math.abs(d.delta))}${pct(d.pct)}`}</td>
    </tr>
  );

  return <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Model Registry</div>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        {store && <Badge c={store.persistent ? t.acc3 : t.wrn}>{store.persistent ? "saved locally" : "memory only"}</Badge>}
        <button type="button" style={btn(t.acc)} onClick={onExport} disabled={!store || busy}>Export JSON</button>
        <button type="button" style={btn(t.acc)} onClick={() => fileInput.current?.click()} disabled={!store || busy}>Import</button>
        <input ref={fileInput} type="file" accept=".json,application/json" style={{ display: "none" }}
          onChange={(e) => { void onImportFile(e.target.files?.[0]); e.target.value = ""; }} />
      </div>
    </div>

    {store && !store.persistent && <div style={{ fontSize: 9.5, color: t.wrn, lineHeight: 1.5 }}>
      Browser storage isn't available here, so the registry lasts until this page closes. Use Export JSON to keep it.
    </div>}
    {msg && <div style={{ fontSize: 10.5, color: msg.kind === "err" ? t.err : msg.kind === "ok" ? t.suc : t.t1, background: (msg.kind === "err" ? t.err : t.acc) + "10", borderRadius: 4, padding: "5px 8px" }}>{msg.text}</div>}

    {/* Register current model */}
    <div style={card}>
      <div style={h}>Register current model</div>
      {!model ? <div style={{ fontSize: 10.5, color: t.t3 }}>Load a model to register it.</div> : <>
        <div style={{ fontSize: 10, color: t.t2, marginBottom: 8, ...mono }}>{model.name} · {model.format} · {fB(modelBytes?.byteLength ?? model.sizeBytes)}{!modelBytes && " · no file bytes (hash unavailable)"}</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          <div>
            <span style={label}>Add to</span>
            <select style={input} value={regTarget} onChange={(e) => setRegTarget(e.target.value)}>
              <option value="__new">New model…</option>
              {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>
          {regTarget === "__new"
            ? <div><span style={label}>Name</span><input style={input} value={regName} onChange={(e) => setRegName(e.target.value)} /></div>
            : <div><span style={label}>Version label</span><input style={input} value={regLabel} placeholder={autoVersion} onChange={(e) => setRegLabel(e.target.value)} /></div>}
          {regTarget === "__new" && <div><span style={label}>Version label</span><input style={input} value={regLabel} placeholder="1.0.0" onChange={(e) => setRegLabel(e.target.value)} /></div>}
          <div><span style={label}>Tags (comma separated)</span><input style={input} value={regTags} placeholder="detector, int8" onChange={(e) => setRegTags(e.target.value)} /></div>
          <div style={{ gridColumn: "1 / -1" }}><span style={label}>Notes</span><input style={input} value={regNotes} placeholder="What changed in this version" onChange={(e) => setRegNotes(e.target.value)} /></div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 8, gap: 8 }}>
          <label style={{ fontSize: 10, color: modelBytes ? t.t2 : t.t3, display: "flex", gap: 4, alignItems: "center" }} title="Keeps a copy of the model file in browser storage. Off by default: only the analysis is saved.">
            <input type="checkbox" checked={storeFile} disabled={!modelBytes} onChange={(e) => setStoreFile(e.target.checked)} />
            Also store the file{modelBytes ? ` (${fB(modelBytes.byteLength)})` : ""}
          </label>
          <button type="button" style={btn(t.acc3, true)} onClick={onRegister} disabled={!store || busy}>Register</button>
        </div>
      </>}
    </div>

    {/* Models */}
    <div style={card}>
      <div style={h}>Models <span style={{ color: t.t3, fontWeight: 400 }}>· {models.length}</span></div>
      <input style={{ ...input, marginBottom: 6 }} value={query} placeholder="Search name, description, tag" onChange={(e) => setQuery(e.target.value)} />
      {tags.length > 0 && <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 6 }}>
        {tags.map((tg) => <button key={tg} type="button" onClick={() => setTagFilter(tagFilter === tg ? null : tg)}
          style={{ padding: "1px 7px", borderRadius: 10, fontSize: 9.5, cursor: "pointer", border: `1px solid ${tagFilter === tg ? t.acc : t.bdr}`, background: tagFilter === tg ? t.acc + "18" : "transparent", color: tagFilter === tg ? t.acc : t.t2 }}>#{tg}</button>)}
      </div>}
      {models.length === 0 && <div style={{ fontSize: 10.5, color: t.t3 }}>No models yet. Register the loaded model or import a registry JSON.</div>}
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {shown.map((m) => <button key={m.id} type="button" onClick={() => void selectModel(m.id)}
          style={{ textAlign: "left", padding: "5px 8px", borderRadius: 4, cursor: "pointer", border: `1px solid ${m.id === selId ? t.acc : "transparent"}`, background: m.id === selId ? t.acc + "12" : t.bg1, color: t.t0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, fontWeight: 600 }}><span>{m.name}</span><span style={{ color: t.t3, fontWeight: 400, fontSize: 10 }}>{m.count} version{m.count === 1 ? "" : "s"}</span></div>
          {m.tags.length > 0 && <div style={{ fontSize: 9.5, color: t.t3 }}>{m.tags.map((x) => "#" + x).join(" ")}</div>}
        </button>)}
      </div>
    </div>

    {/* Version timeline */}
    {selModel && <div style={card}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <div style={{ ...h, marginBottom: 0 }}>{selModel.name} · timeline</div>
        <button type="button" style={btn(t.err)} onClick={onDeleteModel} disabled={busy}>Delete model</button>
      </div>
      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        <input style={input} value={tagEdit} placeholder="tags" onChange={(e) => setTagEdit(e.target.value)} />
        <button type="button" style={btn(t.acc)} onClick={onSaveTags} disabled={busy}>Save tags</button>
      </div>
      <div style={{ fontSize: 9.5, color: t.t3, marginBottom: 6 }}>Tick two versions to compare them.</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 0, borderLeft: `2px solid ${t.bdr2}`, marginLeft: 5 }}>
        {[...versions].reverse().map((v) => <div key={v.id} style={{ position: "relative", padding: "4px 0 8px 12px" }}>
          <span style={{ position: "absolute", left: -6, top: 7, width: 10, height: 10, borderRadius: 5, background: pair.includes(v.id) ? t.acc : t.bg3, border: `1px solid ${t.bdr2}` }} />
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 }}>
            <label style={{ display: "flex", gap: 5, alignItems: "center", fontSize: 11.5, fontWeight: 700, color: t.t0, cursor: "pointer" }}>
              <input type="checkbox" checked={pair.includes(v.id)} onChange={() => togglePair(v.id)} />{v.version}
            </label>
            <span style={{ fontSize: 9.5, color: t.t3 }}>{when(v.createdAt)}</span>
          </div>
          <div style={{ fontSize: 10, color: t.t2, ...mono }}>{v.fileName} · {v.format} · {fB(v.sizeBytes)} · {fmt(v.stats.params)} params · {fmt(v.stats.flops)} FLOPs</div>
          <div style={{ fontSize: 9.5, color: t.t3, ...mono }}>{v.sha256 ? `sha256 ${v.sha256.slice(0, 12)}…` : "no hash"}{v.compat.some((c) => c.errors) ? ` · compat errors: ${v.compat.filter((c) => c.errors).map((c) => c.label).join(", ")}` : ""}</div>
          {v.notes && <div style={{ fontSize: 10, color: t.t1, marginTop: 2 }}>{v.notes}</div>}
          <div style={{ display: "flex", gap: 8, marginTop: 3 }}>
            {v.fileStored && <button type="button" onClick={() => void onDownloadFile(v)} style={{ ...btn(t.acc), padding: "1px 6px", fontSize: 9.5 }}>Download file</button>}
            <button type="button" onClick={() => void onDeleteVersion(v)} style={{ ...btn(t.t3), padding: "1px 6px", fontSize: 9.5, border: "none" }}>Delete</button>
          </div>
        </div>)}
      </div>
    </div>}

    {/* Diff */}
    {diff && vA && vB && <div style={card}>
      <div style={h}>Diff {vA.version} → {vB.version}{diff.identical && <span style={{ color: t.t3, fontWeight: 400 }}> · identical bytes</span>}</div>
      {diff.regressions.length > 0
        ? <div style={{ background: t.err + "12", border: `1px solid ${t.err}40`, borderRadius: 5, padding: "6px 8px", marginBottom: 8 }}>
          <div style={{ fontSize: 10.5, fontWeight: 700, color: t.err, marginBottom: 2 }}>{diff.regressions.length} regression{diff.regressions.length > 1 ? "s" : ""}</div>
          {diff.regressions.map((r, i) => <div key={i} style={{ fontSize: 10, color: r.severity === "error" ? t.err : t.wrn }}>{r.msg}</div>)}
        </div>
        : <div style={{ fontSize: 10.5, color: t.suc, marginBottom: 8 }}>No regressions.</div>}

      <table style={{ width: "100%", fontSize: 10, borderCollapse: "collapse", marginBottom: 8 }}>
        <thead><tr style={{ color: t.t3, textAlign: "left" }}><th style={{ padding: "2px 4px" }} /><th style={{ padding: "2px 4px" }}>{vA.version}</th><th style={{ padding: "2px 4px" }}>{vB.version}</th><th style={{ padding: "2px 4px" }}>Δ</th></tr></thead>
        <tbody>
          {deltaRow("Params", diff.stats.params, fmt)}
          {deltaRow("FLOPs", diff.stats.flops, fmt)}
          {deltaRow("Size", diff.stats.sizeBytes, fB)}
          {deltaRow("Layers", diff.stats.layers, (n) => String(n))}
          {deltaRow("Act. mem", diff.stats.memBytes, fB)}
        </tbody>
      </table>

      <div style={{ fontSize: 10.5, fontWeight: 600, color: t.t1, marginBottom: 3 }}>
        Layers: <span style={{ color: t.suc }}>+{diff.layers.added.length}</span> · <span style={{ color: t.err }}>-{diff.layers.removed.length}</span> · <span style={{ color: t.wrn }}>~{diff.layers.changed.length}</span> · {diff.layers.unchanged} unchanged
      </div>
      <div style={{ maxHeight: 140, overflow: "auto", fontSize: 9.5, ...mono, marginBottom: 8 }}>
        {diff.layers.added.slice(0, 50).map((l, i) => <div key={"a" + i} style={{ color: t.suc }}>+ {l.name} ({l.op}) {l.shape}</div>)}
        {diff.layers.removed.slice(0, 50).map((l, i) => <div key={"r" + i} style={{ color: t.err }}>- {l.name} ({l.op}) {l.shape}</div>)}
        {diff.layers.changed.slice(0, 50).map((c, i) => <div key={"c" + i} style={{ color: t.wrn }}>~ {c.name} ({c.op}): {c.fields.map((f) => f === "shape" ? `shape ${c.before.shape} → ${c.after.shape}` : `${f} ${fmt(c.before[f])} → ${fmt(c.after[f])}`).join(", ")}</div>)}
        {diff.layers.added.length + diff.layers.removed.length + diff.layers.changed.length > 150 && <div style={{ color: t.t3 }}>… list truncated</div>}
      </div>

      <div style={{ fontSize: 10.5, fontWeight: 600, color: t.t1, marginBottom: 3 }}>Compiler compatibility</div>
      <table style={{ width: "100%", fontSize: 10, borderCollapse: "collapse", marginBottom: 8 }}>
        <tbody>{diff.compat.map((c) => {
          const col = c.status === "regression" ? t.err : c.status === "improvement" ? t.suc : t.t2;
          return <tr key={c.target} style={{ background: c.status === "regression" ? t.err + "10" : "transparent" }}>
            <td style={{ padding: "2px 4px", color: t.t1 }}>{c.label}</td>
            <td style={{ padding: "2px 4px", ...mono, color: t.t2 }}>{c.before ? `${c.before.errors}E ${c.before.warnings}W` : "-"}</td>
            <td style={{ padding: "2px 4px", ...mono, color: t.t2 }}>{c.after ? `${c.after.errors}E ${c.after.warnings}W` : "-"}</td>
            <td style={{ padding: "2px 4px", color: col, fontWeight: 600 }} title={[...c.newIssues.map((x) => "new: " + x), ...c.resolvedIssues.map((x) => "fixed: " + x)].join("\n")}>{c.status}</td>
          </tr>;
        })}</tbody>
      </table>

      <div style={{ fontSize: 10.5, fontWeight: 600, color: t.t1, marginBottom: 3 }}>Estimated latency</div>
      <table style={{ width: "100%", fontSize: 10, borderCollapse: "collapse" }}>
        <tbody>{diff.latency.map((l) => <tr key={l.deviceId} style={{ background: l.status === "regression" || (l.fitsBefore && !l.fitsAfter) ? t.err + "10" : "transparent" }}>
          <td style={{ padding: "2px 4px", color: t.t1 }}>{l.deviceName}</td>
          <td style={{ padding: "2px 4px", ...mono, color: t.t2 }}>{l.beforeMs.toFixed(2)} ms</td>
          <td style={{ padding: "2px 4px", ...mono, color: t.t2 }}>{l.afterMs.toFixed(2)} ms{!l.fitsAfter && " (no fit)"}</td>
          <td style={{ padding: "2px 4px", ...mono, color: l.status === "regression" ? t.err : l.status === "improvement" ? t.suc : t.t2 }}>{pct(l.pct).trim() || "="}</td>
        </tr>)}</tbody>
      </table>
      <div style={{ fontSize: 9, color: t.t3, marginTop: 4 }}>Latency is the roofline estimate, not a measurement.</div>
    </div>}
  </div>;
}
