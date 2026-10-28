import { useT } from "../theme/ThemeContext";
import { TOOL_GROUPS, groupOf } from "../data/tools";
import { Icon } from "./Icon";

/**
 * Two-level navigation for the analysis sidebar: workflow stages (Analyze,
 * Optimize, Ship, Copilot) and the tools of the active stage. Picking a stage
 * opens its first tool. The stage row renders separately so a narrow layout
 * can dock it at the bottom of the screen.
 */
export function ToolStages({ active, onChange, docked = false }: {
  active: string;
  onChange: (toolId: string) => void;
  /** Bottom-docked bar (narrow layouts): no fill on the active stage, accent text instead. */
  docked?: boolean;
}) {
  const t = useT();
  const cur = groupOf(active);
  return (
    <nav aria-label="Workflow stage" style={{ display: "grid", gridTemplateColumns: `repeat(${TOOL_GROUPS.length}, minmax(0, 1fr))`, gap: docked ? 0 : 4, padding: docked ? 0 : 8, borderBottom: docked ? "none" : `1px solid ${t.bdr}`, borderTop: docked ? `1px solid ${t.bdr}` : "none", background: t.bg1, flexShrink: 0 }}>
      {TOOL_GROUPS.map((g) => {
        const on = g.id === cur.id;
        return (
          <button key={g.id} type="button" aria-pressed={on} onClick={() => { if (!on) onChange(g.tools[0].id); }}
            style={{
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 3,
              minHeight: docked ? 56 : 50, padding: "6px 4px", borderRadius: docked ? 0 : 8, cursor: "pointer",
              border: `1px solid ${on && !docked ? t.acc + "66" : "transparent"}`,
              background: on && !docked ? t.acc + "1A" : "transparent",
              color: on ? (docked ? t.acc : t.t0) : t.t2, fontSize: 11.5, fontWeight: 600,
            }}>
            <Icon name={g.id} size={docked ? 19 : 17} />
            {g.label}
          </button>
        );
      })}
    </nav>
  );
}

export function ToolTabs({ active, onChange }: { active: string; onChange: (toolId: string) => void }) {
  const t = useT();
  const g = groupOf(active);
  return (
    <div role="tablist" aria-label={`${g.label} tools`} style={{ display: "flex", gap: 2, padding: "0 8px", borderBottom: `1px solid ${t.bdr}`, overflowX: "auto", flexShrink: 0 }}>
      {g.tools.map((tl) => {
        const on = tl.id === active;
        return (
          <button key={tl.id} type="button" role="tab" aria-selected={on} title={tl.desc} onClick={() => onChange(tl.id)}
            style={{
              flexShrink: 0, minHeight: 38, padding: "0 11px", border: "none", background: "transparent", cursor: "pointer",
              fontSize: 12.5, fontWeight: on ? 600 : 500, color: on ? t.t0 : t.t2, whiteSpace: "nowrap",
              boxShadow: `inset 0 -2px 0 ${on ? t.acc : "transparent"}`,
            }}>
            {tl.label}
          </button>
        );
      })}
    </div>
  );
}
