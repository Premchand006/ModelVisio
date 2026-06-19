import { useT } from "../theme/ThemeContext";

export type Tab = { id: string; label: string; icon?: string };

export function Tabs({
  tabs, active, onChange,
}: {
  tabs: Tab[];
  active: string;
  onChange: (id: string) => void;
}) {
  const t = useT();
  return (
    <div style={{ display: "flex", gap: 1, padding: 3, background: t.bg + "88", borderRadius: 6, backdropFilter: "blur(8px)", maxWidth: "100%", overflowX: "auto" }}>
      {tabs.map((tb) => (
        <button
          key={tb.id}
          type="button"
          onClick={() => onChange(tb.id)}
          style={{
            padding: "5px 11px", borderRadius: 4, border: "none", fontSize: 11.5,
            fontWeight: active === tb.id ? 600 : 500,
            cursor: "pointer", transition: "all .15s", flexShrink: 0, whiteSpace: "nowrap",
            background: active === tb.id ? t.bg2 : "transparent",
            color: active === tb.id ? t.t0 : t.t2,
          }}
        >
          {tb.icon && <span style={{ marginRight: 3 }}>{tb.icon}</span>}
          {tb.label}
        </button>
      ))}
    </div>
  );
}
