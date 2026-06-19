import { useState } from "react";
import type { Model } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { Badge } from "./Badge";
import { FIXES, applicableLayers, type FixId } from "../fixes/transforms";

/**
 * Compiler compatibility pre-flight + working auto-fix engine. The "Apply" button
 * runs a real transform over the in-memory model graph (see fixes/transforms.ts):
 * the graph, stats, and the issue list below all update live, and the fix is
 * reversible. Fixes edit the analysis model — export via Convert/Deploy to apply
 * to the original file.
 */
export function CompilerChecker({
  model, onApplyFix, onUndo, undoDepth = 0,
}: {
  model: Model;
  onApplyFix?: (id: FixId) => void;
  onUndo?: () => void;
  undoDepth?: number;
}) {
  const t = useT();
  const [target, setTarget] = useState("coral");

  const allIssues: { target: string; severity: "warn" | "error"; msg: string; layer: string; layerType: string }[] = [];
  model.layers.forEach((l) => {
    (l.compIssues || []).forEach((ci) => allIssues.push({ ...ci, layer: l.name, layerType: l.type }));
  });
  const filtered = target === "all" ? allIssues : allIssues.filter((ci) => ci.target.includes(target));
  const targets = [
    { id: "all", label: "All" }, { id: "coral", label: "Coral TPU" }, { id: "rk3588", label: "RK3588" },
    { id: "kneron", label: "Kneron" }, { id: "hailo", label: "Hailo" },
  ];
  const errors = filtered.filter((i) => i.severity === "error").length;
  const warns = filtered.filter((i) => i.severity === "warn").length;

  // Only surface fixes that actually apply to THIS model right now.
  const fixes = FIXES.map((f) => ({ def: f, layers: applicableLayers(model, f.id) })).filter((f) => f.layers.length > 0);

  return <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%" }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Compiler Compatibility Pre-Flight</div>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        {undoDepth > 0 && onUndo && (
          <button type="button" onClick={onUndo} title="Revert the last applied fix" style={{ padding: "2px 8px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 10, cursor: "pointer" }}>↶ Undo fix ({undoDepth})</button>
        )}
        <Badge c={t.err}>{errors} errors</Badge><Badge c={t.wrn}>{warns} warnings</Badge>
      </div>
    </div>

    {undoDepth > 0 && (
      <div style={{ fontSize: 9.5, color: t.acc3, background: t.acc3 + "12", border: `1px solid ${t.acc3}33`, borderRadius: 5, padding: "5px 8px", lineHeight: 1.5 }}>
        ✓ {undoDepth} fix{undoDepth > 1 ? "es" : ""} applied to the analysis model — graph, stats, compatibility & the Convert tab's Graph JSON / Layers CSV now reflect them. The original model file is unchanged (rewriting the binary is an export-pipeline step). Undo above.
      </div>
    )}

    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
      {targets.map((tg) => <button key={tg.id} type="button" onClick={() => setTarget(tg.id)} style={{ padding: "3px 8px", borderRadius: 4, border: `1px solid ${target === tg.id ? t.acc : t.bdr}`, background: target === tg.id ? t.acc + "18" : "transparent", color: target === tg.id ? t.acc : t.t2, fontSize: 10, cursor: "pointer" }}>{tg.label}</button>)}
    </div>

    <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
      {filtered.length === 0 && <div style={{ padding: 20, textAlign: "center", color: t.suc, fontSize: 12 }}>✓ No issues detected for this target</div>}
      {filtered.map((ci, i) => <div key={i} style={{ background: ci.severity === "error" ? t.err + "10" : t.wrn + "10", border: `1px solid ${ci.severity === "error" ? t.err : t.wrn}30`, borderRadius: 6, padding: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: t.t0 }}>{ci.layer} <span style={{ color: t.t3 }}>({ci.layerType})</span></span>
          <Badge c={ci.severity === "error" ? t.err : t.wrn}>{ci.severity}</Badge>
        </div>
        <div style={{ fontSize: 11, color: t.t1 }}>{ci.msg}</div>
      </div>)}

      <div style={{ fontSize: 12, fontWeight: 700, color: t.acc3, marginTop: 8 }}>✦ Auto-Fix Engine</div>
      {fixes.length === 0 && <div style={{ fontSize: 10, color: t.t3 }}>No applicable transforms for this model.</div>}
      {fixes.map(({ def, layers }) => <div key={def.id} style={{ background: t.acc3 + "08", border: `1px solid ${t.acc3}25`, borderRadius: 6, padding: 10 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: t.t0, marginBottom: 2 }}>{def.title} <span style={{ color: t.t3, fontWeight: 400 }}>· {layers.length} layer{layers.length > 1 ? "s" : ""}</span></div>
        <div style={{ fontSize: 10, color: t.acc3, marginBottom: 2 }}>→ {def.change}</div>
        <div style={{ fontSize: 9, color: t.t2, marginBottom: 4 }}>{def.impact}</div>
        <div style={{ fontSize: 9, color: t.t3, marginBottom: 6, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>Affected: {layers.join(", ")}</div>
        <button type="button" disabled={!onApplyFix} onClick={() => onApplyFix?.(def.id)} style={{ padding: "4px 12px", borderRadius: 4, border: `1px solid ${t.acc3}`, background: t.acc3 + "18", color: t.acc3, fontSize: 10, fontWeight: 600, cursor: onApplyFix ? "pointer" : "default", opacity: onApplyFix ? 1 : 0.5 }}>Apply Fix</button>
      </div>)}
    </div>
  </div>;
}
