import { useMemo, useState } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { useT } from "../../theme/ThemeContext";

// Module hierarchy derived from layer names (split on "." and "/"), mirroring
// Netron's PyTorch module tree (DetectionModel → Sequential → Conv → Conv2d…).
type TNode = { key: string; children: Map<string, TNode>; layer?: ModelLayer };

function buildTree(model: Model): TNode {
  const root: TNode = { key: "", children: new Map() };
  for (const l of model.layers) {
    const parts = l.name.split(/[./]/).filter(Boolean);
    if (parts.length === 0) parts.push(l.op);
    let node = root;
    for (const p of parts) {
      if (!node.children.has(p)) node.children.set(p, { key: p, children: new Map() });
      node = node.children.get(p)!;
    }
    node.layer = l;
  }
  return root;
}

function shapeStr(w: ModelLayer["w"]): string {
  if (!w) return "";
  return typeof w.shape === "string" ? w.shape : w.shape.join("×");
}

function Row({ node, depth, selected, onSelect }: { node: TNode; depth: number; selected: ModelLayer | null; onSelect: (l: ModelLayer) => void }) {
  const t = useT();
  const [open, setOpen] = useState(depth < 2);
  const kids = [...node.children.values()];
  const isLeaf = kids.length === 0;
  const sel = node.layer && selected?.id === node.layer.id;

  return (
    <div>
      <div
        onClick={() => (isLeaf && node.layer ? onSelect(node.layer) : setOpen((o) => !o))}
        style={{
          display: "flex", alignItems: "center", gap: 6, padding: "2px 6px", paddingLeft: 6 + depth * 12,
          cursor: "pointer", borderRadius: 3, background: sel ? t.bg3 : "transparent", fontSize: 10,
        }}
      >
        {!isLeaf && <span style={{ color: t.t3, width: 8 }}>{open ? "▾" : "▸"}</span>}
        {isLeaf && <span style={{ width: 8 }} />}
        <span style={{ color: isLeaf ? t.t1 : t.t0, fontWeight: isLeaf ? 400 : 600, fontFamily: "'JetBrains Mono'" }}>{node.key}</span>
        {node.layer && <span style={{ color: t.acc2, fontSize: 9 }}>{node.layer.op}</span>}
        {node.layer?.w && <span style={{ color: t.t3, fontSize: 9, fontFamily: "'JetBrains Mono'" }}>⟨{shapeStr(node.layer.w)}⟩</span>}
      </div>
      {open && kids.map((c) => <Row key={c.key} node={c} depth={depth + 1} selected={selected} onSelect={onSelect} />)}
    </div>
  );
}

export function Tree({ model, selected, onSelect }: { model: Model; selected: ModelLayer | null; onSelect: (l: ModelLayer) => void }) {
  const t = useT();
  const root = useMemo(() => buildTree(model), [model]);
  return (
    <div style={{ flex: 1, overflow: "auto", padding: 6, background: t.bg }}>
      {[...root.children.values()].map((c) => <Row key={c.key} node={c} depth={0} selected={selected} onSelect={onSelect} />)}
    </div>
  );
}
