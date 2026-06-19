import { useState, type ReactNode } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { GC } from "../theme/theme";
import { fmt, fB } from "../utils/format";
import { Badge } from "./Badge";
import { Tabs } from "./Tabs";

export function Inspector({ layer, model }: { layer: ModelLayer | null; model: Model }) {
  const t = useT();
  const [tab, setTab] = useState("props");
  if (!layer) return <div style={{ padding: 24, textAlign: "center", color: t.t3, fontSize: 11 }}>Select a layer node to inspect</div>;
  const gc = t[GC[layer.group]] || t.t3;
  const totalP = model.layers.reduce((s, l) => s + l.params, 0);
  const pct = totalP > 0 ? ((layer.params / totalP) * 100).toFixed(1) : "0";

  const R = ({ l, v, c }: { l: string; v: ReactNode; c?: string }) => (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: `1px solid ${t.bdr}22` }}>
      <span style={{ fontSize: 10, color: t.t2 }}>{l}</span>
      <span style={{ fontSize: 10, color: c || t.t1, fontFamily: "'JetBrains Mono',monospace", textAlign: "right", maxWidth: "60%", wordBreak: "break-all" }}>{v}</span>
    </div>
  );

  const TABS = [
    { id: "props", label: "Properties" }, { id: "tensors", label: "Tensors" },
    { id: "math", label: "Math" }, { id: "metrics", label: "Metrics" }, { id: "ai", label: "AI" },
  ];

  return <div style={{ display: "flex", flexDirection: "column", gap: 6, height: "100%" }}>
    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      <Badge c={gc}>{layer.group}</Badge>
      <span style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>{layer.name}</span>
      <Badge c={t.t3}>{layer.op}</Badge>
    </div>
    <Tabs tabs={TABS} active={tab} onChange={setTab} />
    <div style={{ flex: 1, overflow: "auto", background: t.glass, backdropFilter: "blur(12px)", borderRadius: 8, border: `1px solid ${t.glassBdr}`, padding: 10 }}>
      {tab === "props" && <div>
        <R l="Name" v={layer.name} /><R l="Type" v={layer.type} /><R l="Op" v={layer.op} />
        <R l="Shape" v={layer.shape} c={t.acc} /><R l="Dtype" v={layer.dt || "float32"} /><R l="Group" v={layer.group} />
        {Object.keys(layer.attr || {}).length > 0 && <>
          <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1, marginTop: 8, marginBottom: 4 }}>Attributes</div>
          {Object.entries(layer.attr).map(([k, v]) => <R key={k} l={k} v={JSON.stringify(v)} c={t.acc} />)}
        </>}
      </div>}

      {tab === "tensors" && <div>
        <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>Inputs ({layer.ins?.length || 0})</div>
        {(layer.ins || []).map((inp, i) => <div key={i} style={{ background: t.bg2, borderRadius: 4, padding: 6, marginBottom: 4, border: `1px solid ${t.bdr}` }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: t.t1, fontFamily: "'JetBrains Mono',monospace" }}>{inp.n}</div>
          {inp.s && <div style={{ fontSize: 9, color: t.acc }}>Shape: {JSON.stringify(inp.s)}</div>}
        </div>)}
        <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1, marginTop: 8, marginBottom: 4 }}>Outputs ({layer.outs?.length || 0})</div>
        {(layer.outs || []).map((o, i) => <div key={i} style={{ background: t.bg2, borderRadius: 4, padding: 6, marginBottom: 4, border: `1px solid ${t.bdr}` }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: t.t1, fontFamily: "'JetBrains Mono',monospace" }}>{o.n}</div>
          {o.s && <div style={{ fontSize: 9, color: t.acc }}>Shape: {JSON.stringify(o.s)}</div>}
        </div>)}
        {layer.w && <>
          <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1, marginTop: 8, marginBottom: 4 }}>Weights</div>
          <div style={{ background: t.bg2, borderRadius: 4, padding: 6, border: `1px solid ${t.bdr}` }}>
            <R l="Shape" v={typeof layer.w.shape === "object" ? JSON.stringify(layer.w.shape) : layer.w.shape} />
            {layer.w.size != null && <R l="Elements" v={fmt(layer.w.size)} />}
            {layer.w.min != null && <R l="Range" v={`[${layer.w.min}, ${layer.w.max}]`} c={t.acc} />}
            {layer.w.mean != null && <R l="Mean±Std" v={`${layer.w.mean}±${layer.w.std}`} />}
            {layer.w.sparse != null && <R l="Sparsity" v={`${(layer.w.sparse * 100).toFixed(1)}%`} />}
          </div>
        </>}
      </div>}

      {tab === "math" && <pre style={{ background: t.bg, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: 10, fontSize: 10.5, color: t.acc, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.6, whiteSpace: "pre-wrap", margin: 0 }}>{layer.math || "No math info."}</pre>}

      {tab === "metrics" && <div>
        <R l="Parameters" v={`${fmt(layer.params)} (${pct}%)`} /><R l="FLOPs" v={fmt(layer.flops)} c={t.acc} />
        <R l="MACs" v={fmt(layer.macs || 0)} /><R l="Activation memory" v={fB(layer.mem || 0)} />
        {layer.params > 0 && <><R l="Weight mem (FP32)" v={fB(layer.params * 4)} /><R l="Weight mem (INT8)" v={fB(layer.params)} c={t.suc} /></>}
        <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1, marginTop: 8, marginBottom: 4 }}>Est. Layer Latency</div>
        <R l="Jetson Orin Nano (FP16)" v={layer.flops > 0 ? `~${(layer.flops / 40e9 * 1000).toFixed(2)}ms` : "—"} />
        <R l="Hailo-8 (INT8)" v={layer.flops > 0 ? `~${(layer.flops / 26e9 * 1000).toFixed(2)}ms` : "—"} />
        <R l="Coral TPU (INT8)" v={layer.flops > 0 ? `~${(layer.flops / 4e9 * 1000).toFixed(2)}ms` : "—"} />
        <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1, marginTop: 8, marginBottom: 4 }}>Quantization Sensitivity</div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ flex: 1, height: 6, background: t.bg3, borderRadius: 3, overflow: "hidden" }}>
            <div style={{ width: `${Math.min(100, (layer.qSens || 0) * 100 / 0.2 * 100)}%`, height: "100%", background: layer.qSens > 0.12 ? t.err : layer.qSens > 0.06 ? t.wrn : t.suc, borderRadius: 3 }} />
          </div>
          <span style={{ fontSize: 10, fontWeight: 700, color: layer.qSens > 0.12 ? t.err : layer.qSens > 0.06 ? t.wrn : t.suc, fontFamily: "'JetBrains Mono',monospace" }}>{((layer.qSens || 0) * 100).toFixed(0)}%</span>
        </div>
      </div>}

      {tab === "ai" && <div style={{ background: t.acc2 + "12", border: `1px solid ${t.acc2}33`, borderRadius: 6, padding: 10 }}>
        <div style={{ fontSize: 9, fontWeight: 700, color: t.acc2, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>✦ AI Analysis</div>
        <div style={{ fontSize: 11, color: t.t1, lineHeight: 1.6 }}>{layer.insight || "No insight."}</div>
        {layer.compIssues?.length > 0 && <>
          <div style={{ fontSize: 9, fontWeight: 700, color: t.err, textTransform: "uppercase", letterSpacing: 1, marginTop: 10, marginBottom: 4 }}>⚠ Compiler Issues</div>
          {layer.compIssues.map((ci, i) => <div key={i} style={{ background: ci.severity === "error" ? t.err + "15" : t.wrn + "15", border: `1px solid ${ci.severity === "error" ? t.err : t.wrn}33`, borderRadius: 4, padding: 6, marginBottom: 4 }}>
            <div style={{ fontSize: 9, fontWeight: 700, color: ci.severity === "error" ? t.err : t.wrn }}>{ci.target}</div>
            <div style={{ fontSize: 10, color: t.t1 }}>{ci.msg}</div>
          </div>)}
        </>}
      </div>}
    </div>
  </div>;
}
