import { useEffect, useMemo, useRef, useState } from "react";
import { useT } from "../theme/ThemeContext";
import { Icon, type IconName } from "./Icon";

export type Command = {
  id: string;
  section: "Tools" | "Layers" | "Actions";
  label: string;
  /** Right-aligned context, e.g. the tool's stage or a layer's op. Also searched. */
  hint?: string;
  icon: IconName;
  run: () => void;
};

const SECTIONS: Command["section"][] = ["Tools", "Actions", "Layers"];
const MAX_PER_SECTION = 40;

/**
 * Commands matching `query`, grouped by section (Tools, Actions, Layers) and
 * ranked within a section: label prefix, then a word in the label starting
 * with the query, then a substring of the label or hint. An empty query keeps
 * every command in its original order. Each section is capped so a model with
 * thousands of layers stays fast to render.
 */
export function filterCommands(cmds: Command[], query: string): Command[] {
  const q = query.trim().toLowerCase();
  const rank = (c: Command): number => {
    if (!q) return 0;
    const l = c.label.toLowerCase();
    if (l.startsWith(q)) return 0;
    if (l.split(/[\s._/:-]+/).some((w) => w.startsWith(q))) return 1;
    if (l.includes(q) || (c.hint ?? "").toLowerCase().includes(q)) return 2;
    return -1;
  };
  return SECTIONS.flatMap((s) =>
    cmds.map((c, i) => ({ c, i, r: c.section === s ? rank(c) : -1 }))
      .filter((x) => x.r >= 0)
      .sort((a, b) => a.r - b.r || a.i - b.i)
      .slice(0, MAX_PER_SECTION)
      .map((x) => x.c));
}

/**
 * Ctrl/Cmd+K command palette: jump to any tool or layer, or run an app action.
 * Arrow keys move, Enter runs, Escape or a backdrop click closes. Focus stays
 * in the search box and returns to the opener on close.
 */
export function CommandPalette({ commands, onClose }: { commands: Command[]; onClose: () => void }) {
  const t = useT();
  const [query, setQuery] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const hits = useMemo(() => filterCommands(commands, query), [commands, query]);
  const cur = Math.min(sel, Math.max(0, hits.length - 1));

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    inputRef.current?.focus();
    return () => { opener?.focus?.(); };
  }, []);

  useEffect(() => {
    listRef.current?.querySelector(`[data-idx="${cur}"]`)?.scrollIntoView({ block: "nearest" });
  }, [cur]);

  const run = (c: Command | undefined) => { if (!c) return; onClose(); c.run(); };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setSel((cur + 1) % Math.max(1, hits.length)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setSel((cur - 1 + hits.length) % Math.max(1, hits.length)); }
    else if (e.key === "Enter") { e.preventDefault(); run(hits[cur]); }
    else if (e.key === "Tab") e.preventDefault(); // the search box is the only stop
  };

  const kbd = { fontFamily: "'JetBrains Mono'", fontSize: 10.5, color: t.t1, border: `1px solid ${t.bdr2}`, borderRadius: 4, padding: "1px 5px" };
  let lastSection = "";
  return (
    <div onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
      style={{ position: "fixed", inset: 0, zIndex: 1200, background: "rgba(0,0,0,0.5)", display: "flex", justifyContent: "center", alignItems: "flex-start", padding: "min(14vh, 120px) 16px 16px" }}>
      <div role="dialog" aria-modal="true" aria-label="Command palette" onKeyDown={onKey}
        style={{ width: "min(600px, 100%)", maxHeight: "100%", display: "flex", flexDirection: "column", background: t.bg1, color: t.t0, border: `1px solid ${t.bdr2}`, borderRadius: 14, boxShadow: "0 24px 64px rgba(0,0,0,0.45)", overflow: "hidden", fontFamily: "'Inter',-apple-system,sans-serif" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, minHeight: 54, padding: "0 16px", borderBottom: `1px solid ${t.bdr}`, color: t.t2 }}>
          <Icon name="analyze" size={17} stroke={2} />
          <input ref={inputRef} value={query} onChange={(e) => { setQuery(e.target.value); setSel(0); }}
            placeholder="Search tools, layers, actions…" aria-label="Search tools, layers and actions"
            role="combobox" aria-expanded="true" aria-controls="mv-cmd-list" aria-activedescendant={hits.length ? `mv-cmd-${cur}` : undefined}
            style={{ flex: 1, minWidth: 0, height: 52, border: "none", outline: "none", background: "transparent", color: t.t0, fontSize: 15, fontFamily: "inherit" }} />
          <span style={kbd}>Esc</span>
        </div>
        <div ref={listRef} id="mv-cmd-list" role="listbox" aria-label="Results" style={{ overflowY: "auto", padding: 6, maxHeight: 420 }}>
          {hits.length === 0 && <div style={{ padding: 24, textAlign: "center", fontSize: 13, color: t.t2 }}>Nothing matches “{query}”.</div>}
          {hits.map((c, i) => {
            const head = c.section !== lastSection;
            lastSection = c.section;
            const on = i === cur;
            return <div key={c.id}>
              {head && <div role="presentation" style={{ padding: "10px 10px 4px", fontSize: 10.5, letterSpacing: .6, textTransform: "uppercase", color: t.t2 }}>{c.section}</div>}
              <div id={`mv-cmd-${i}`} data-idx={i} role="option" aria-selected={on}
                onMouseMove={() => { if (!on) setSel(i); }} onClick={() => run(c)}
                style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 38, padding: "0 10px", borderRadius: 8, cursor: "pointer", fontSize: 13.5, background: on ? t.acc + "26" : "transparent", color: t.t0 }}>
                <span style={{ color: on ? t.acc : t.t2, display: "flex" }}><Icon name={c.icon} size={15} /></span>
                <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: c.section === "Layers" ? "'JetBrains Mono'" : "inherit", fontSize: c.section === "Layers" ? 12.5 : 13.5 }}>{c.label}</span>
                {c.hint && <span style={{ fontSize: 11.5, color: t.t2, whiteSpace: "nowrap" }}>{c.hint}</span>}
              </div>
            </div>;
          })}
        </div>
        <div style={{ display: "flex", gap: 16, padding: "9px 16px", borderTop: `1px solid ${t.bdr}`, fontSize: 11.5, color: t.t2 }}>
          <span><span style={kbd}>↑↓</span> move</span>
          <span><span style={kbd}>Enter</span> open</span>
          <span><span style={kbd}>Esc</span> close</span>
        </div>
      </div>
    </div>
  );
}
