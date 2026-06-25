import { useEffect, useRef, useState } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import { sendChat, type ChatMessage, type Source } from "@modelvisio/ai";
import { useT } from "../theme/ThemeContext";
import { fmt, fB } from "../utils/format";
import { scoreAll } from "../scoring";
import { HW } from "../data/hardware";

type UiMsg = { role: "user" | "assistant"; content: string; sources?: Source[] };

export function Chat({ model, sel }: { model: Model | null; sel: ModelLayer | null }) {
  const t = useT();
  const [msgs, setMsgs] = useState<UiMsg[]>([]);
  const [inp, setInp] = useState("");
  const [ld, setLd] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  const QP = model ? ["Explain this architecture", "TensorRT conversion + reference docs", "Deploy on Jetson Orin", "Quantization plan", "INT8 vs FP16 tradeoffs", "Best runtime for Hailo-8"] : [];

  const send = async (text: string) => {
    if (!text.trim() || ld || !model) return;
    const next: UiMsg[] = [...msgs, { role: "user", content: text }];
    setMsgs(next);
    setInp("");
    setLd(true);
    const totalP = model.layers.reduce((s, l) => s + l.params, 0);
    const totalF = model.layers.reduce((s, l) => s + l.flops, 0);

    // Computed roofline/hardware analysis for THIS model, so the copilot can give
    // dedicated, to-the-point optimization + hardware answers grounded in numbers.
    const ranked = scoreAll(model, HW);
    const topHw = ranked.slice(0, 6).map((r) =>
      `  ${r.device.n} — ${r.overall}/100 · ~${Math.round(r.fps)} FPS · ${r.fpsPerW.toFixed(1)} FPS/W · ${r.roofline.bound}-bound`
      + `${r.memory.hardFail ? " · WON'T FIT" : ""}${r.opSupport.compileFail ? " · WON'T COMPILE" : r.opSupport.conversionNeeded ? " · needs conversion" : ""}`,
    ).join("\n");
    const wontFit = ranked.filter((r) => r.memory.hardFail).map((r) => r.device.n);
    const qSensitive = model.layers.filter((l) => l.qSens > 0.1).sort((a, b) => b.qSens - a.qSens)
      .slice(0, 8).map((l) => `${l.name} (${(l.qSens * 100).toFixed(0)}%)`);
    const opIssues = model.layers.flatMap((l) => l.compIssues.map((c) => `${l.name} [${c.target}/${c.severity}]: ${c.msg}`)).slice(0, 10);
    const wl = ranked[0]?.roofline.workload;

    const modelSummary = [
      `Model: ${model.name} | ${model.format} | ${model.framework} | opset ${model.opset}`,
      `Size: ${fB(model.sizeBytes)} | Input: ${model.inputShape.join("×")} | Params: ${fmt(totalP)} | FLOPs: ${fmt(totalF)}`,
      wl ? `Workload class: ${wl} | deployment precision assumed: ${ranked[0]?.precision?.toUpperCase()}` : "",
      `Layers (${model.layers.length}): ${model.layers.map((l) => l.op).join(", ")}`,
      `Hardware fit — roofline scoring, best first (score · est. FPS · efficiency · bottleneck regime):\n${topHw}`,
      wontFit.length ? `Exceeds device memory (cannot deploy without shrinking): ${wontFit.join(", ")}` : "",
      qSensitive.length ? `Most quantization-sensitive layers (INT8 accuracy risk): ${qSensitive.join(", ")}` : "",
      opIssues.length ? `Compiler / op-support issues:\n  ${opIssues.join("\n  ")}` : "",
      sel ? `Currently selected layer: ${sel.name} (${sel.type}), shape ${sel.shape}, ${fmt(sel.params)} params, qSens ${(sel.qSens * 100).toFixed(0)}%` : "",
    ].filter(Boolean).join("\n");
    const wire: ChatMessage[] = next.map((m) => ({ role: m.role, content: m.content }));
    try {
      // Routes through the server-side proxy — the API key never reaches the browser.
      const { text: reply, sources } = await sendChat({ messages: wire, modelSummary });
      setMsgs((p) => [...p, { role: "assistant", content: reply, sources }]);
    } catch (e) {
      setMsgs((p) => [...p, { role: "assistant", content: e instanceof Error ? e.message : "Connection error. Is the /api/chat proxy running?" }]);
    }
    setLd(false);
  };

  return <div style={{ display: "flex", flexDirection: "column", height: "100%", background: t.bg, borderRadius: 6, border: `1px solid ${t.bdr}`, overflow: "hidden" }}>
    <div style={{ padding: "6px 10px", borderBottom: `1px solid ${t.bdr}`, fontSize: 11, fontWeight: 600, color: t.t0, display: "flex", alignItems: "center", gap: 5 }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: t.suc, display: "inline-block" }} />AI Copilot
      <span style={{ marginLeft: "auto", fontSize: 8, color: t.t3, fontWeight: 400 }}>edge-ML expert · cites sources</span>
    </div>
    <div style={{ flex: 1, overflow: "auto", padding: 8, display: "flex", flexDirection: "column", gap: 5 }}>
      {msgs.length === 0 && <div style={{ color: t.t3, fontSize: 10, padding: 10, textAlign: "center" }}>{model ? "Ask about architectures, formats, layers, hardware, quantization, deployment — ask for reference links and it will search + cite them." : "Load a model."}</div>}
      {msgs.map((m, i) => <div key={i} style={{ alignSelf: m.role === "user" ? "flex-end" : "flex-start", maxWidth: "90%", display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ padding: "6px 10px", borderRadius: m.role === "user" ? "10px 10px 3px 10px" : "10px 10px 10px 3px", background: m.role === "user" ? t.acc : t.bg1, border: m.role === "user" ? "none" : `1px solid ${t.bdr}`, fontSize: 11, color: m.role === "user" ? "#fff" : t.t0, lineHeight: 1.5, whiteSpace: "pre-wrap", fontFamily: m.role === "assistant" ? "'JetBrains Mono',monospace" : "inherit" }}>{m.content}</div>
        {m.sources && m.sources.length > 0 && <div style={{ padding: "5px 8px", borderRadius: 6, background: t.acc + "0E", border: `1px solid ${t.acc}33` }}>
          <div style={{ fontSize: 8, fontWeight: 700, color: t.acc, textTransform: "uppercase", letterSpacing: 1, marginBottom: 3 }}>Sources</div>
          {m.sources.map((s, j) => <a key={j} href={s.url} target="_blank" rel="noreferrer noopener" style={{ display: "block", fontSize: 10, color: t.acc, textDecoration: "none", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: 1 }}>{j + 1}. {s.title}</a>)}
        </div>}
      </div>)}
      {ld && <div style={{ alignSelf: "flex-start", padding: "6px 10px", borderRadius: 10, background: t.bg1, border: `1px solid ${t.bdr}`, fontSize: 10, color: t.t3 }}><span style={{ animation: "pulse 1.2s infinite", color: t.acc }}>●</span> Researching…</div>}
      <div ref={end} />
    </div>
    {model && msgs.length === 0 && <div style={{ padding: "0 8px 4px", display: "flex", flexWrap: "wrap", gap: 2 }}>{QP.map((p, i) => <button key={i} type="button" onClick={() => send(p)} style={{ padding: "2px 7px", borderRadius: 8, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 9, cursor: "pointer" }}>{p}</button>)}</div>}
    <div style={{ padding: 5, borderTop: `1px solid ${t.bdr}`, display: "flex", gap: 4 }}>
      <input value={inp} onChange={(e) => setInp(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send(inp)} placeholder="Ask anything…" disabled={!model} style={{ flex: 1, padding: "6px 8px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t0, fontSize: 10, outline: "none" }} />
      <button type="button" onClick={() => send(inp)} disabled={ld || !model} style={{ padding: "6px 12px", borderRadius: 5, border: "none", background: ld ? t.bg3 : t.acc, color: "#fff", fontSize: 10, fontWeight: 600, cursor: ld ? "default" : "pointer" }}>Send</button>
    </div>
  </div>;
}
