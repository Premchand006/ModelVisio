import { useMemo, useState } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { useT } from "../../theme/ThemeContext";
import { GC } from "../../theme/theme";
import { fmt } from "../../utils/format";

export function LayersList({
  model, selected, onSelect, qHeatmap, query,
}: {
  model: Model;
  selected: ModelLayer | null;
  onSelect: (l: ModelLayer) => void;
  qHeatmap: boolean;
  query: string;
}) {
  const t = useT();
  const [local, setLocal] = useState("");
  const q = (query || local).trim().toLowerCase();
  const rows = useMemo(
    () => (q ? model.layers.filter((l) => (l.name + " " + l.op + " " + l.type).toLowerCase().includes(q)) : model.layers),
    [model, q],
  );

  // Keep the list in a constrained, centered column so it reads as a tidy stack of
  // cards instead of stretching edge-to-edge across the wide graph panel (which is
  // what pushed the param counts far away from the layer names).
  const COL = 560;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: t.bg }}>
      <div style={{ padding: 6, borderBottom: `1px solid ${t.bdr}` }}>
        <div style={{ maxWidth: COL, margin: "0 auto" }}>
          <input
            value={local}
            onChange={(e) => setLocal(e.target.value)}
            placeholder={`Filter ${model.layers.length} layers…`}
            style={{ width: "100%", padding: "5px 9px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t0, fontSize: 11, outline: "none" }}
          />
        </div>
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: "10px 12px" }}>
        <div style={{ maxWidth: COL, margin: "0 auto", display: "flex", flexDirection: "column", gap: 5 }}>
          {rows.length === 0 && <div style={{ padding: 24, textAlign: "center", color: t.t3, fontSize: 11 }}>No layers match “{q}”.</div>}
          {rows.map((l) => {
            const gc = t[GC[l.group]] || t.bdr;
            const qc = qHeatmap && l.qSens > 0 ? (l.qSens > 0.12 ? t.err : l.qSens > 0.06 ? t.wrn : t.suc) : null;
            const isSel = selected?.id === l.id;
            return (
              <div
                key={l.id}
                onClick={() => onSelect(l)}
                style={{
                  padding: "7px 11px", borderRadius: 8, cursor: "pointer",
                  background: isSel ? t.bg2 : t.bg1,
                  border: `1px solid ${isSel ? t.bdr2 : t.bdr}`,
                  borderLeft: `3px solid ${qc || gc}`,
                }}
              >
                {/* Op + name on one line, left-aligned — name truncates rather than
                    pushing anything to the far edge. */}
                <div style={{ display: "flex", alignItems: "baseline", gap: 8, minWidth: 0 }}>
                  <span style={{ fontSize: 11.5, fontWeight: 700, color: t.t0, fontFamily: "'JetBrains Mono'", flexShrink: 0 }}>{l.op}</span>
                  <span style={{ fontSize: 10, color: t.t3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>{l.name}</span>
                </div>
                {/* Shape + params on the meta line, sitting next to each other. */}
                <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginTop: 2 }}>
                  {l.shape && <span style={{ fontSize: 10, color: t.acc, fontFamily: "'JetBrains Mono'" }}>{l.shape}</span>}
                  {l.params > 0 && <span style={{ fontSize: 9.5, color: t.t3, fontFamily: "'JetBrains Mono'" }}>{fmt(l.params)} params</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
