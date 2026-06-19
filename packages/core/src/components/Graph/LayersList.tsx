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

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: 6, borderBottom: `1px solid ${t.bdr}` }}>
        <input
          value={local}
          onChange={(e) => setLocal(e.target.value)}
          placeholder={`Filter ${model.layers.length} layers…`}
          style={{ width: "100%", padding: "4px 8px", borderRadius: 4, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 10, outline: "none" }}
        />
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: 3 }}>
        {rows.map((l) => {
          const gc = t[GC[l.group]] || t.bdr;
          const qc = qHeatmap && l.qSens > 0 ? (l.qSens > 0.12 ? t.err : l.qSens > 0.06 ? t.wrn : t.suc) : null;
          return (
            <div
              key={l.id}
              onClick={() => onSelect(l)}
              style={{ padding: "4px 6px", borderRadius: 3, cursor: "pointer", marginBottom: 1, background: selected?.id === l.id ? t.bg3 : "transparent", borderLeft: `3px solid ${qc || gc}` }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 600, color: t.t1, fontFamily: "'JetBrains Mono'" }}>{l.op}</span>
                {l.params > 0 && <span style={{ fontSize: 9, color: t.t3, fontFamily: "'JetBrains Mono'" }}>{fmt(l.params)}</span>}
              </div>
              <div style={{ fontSize: 9, color: t.t3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{l.name}</div>
              <div style={{ fontSize: 9, color: t.acc, fontFamily: "'JetBrains Mono'" }}>{l.shape}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
