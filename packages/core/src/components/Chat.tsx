import { useEffect, useRef, useState } from "react";
import type { Model, ModelLayer } from "@modelvisio/parsers";
import {
  sendChat, extractUrls, scrapeMessageUrls, formatScrapedForPrompt,
  type ChatMessage, type Source, type ScrapedPage, type ScrapeFailure,
} from "@modelvisio/ai";
import { useT } from "../theme/ThemeContext";
import { fmt, fB } from "../utils/format";
import { scoreAll } from "../scoring";
import { HW } from "../data/hardware";
import { isDesktop, getAiProvider, setAiProvider, getUserApiKey, setUserApiKey, type AiProvider } from "../utils/apiKey";

type UiMsg = { role: "user" | "assistant"; content: string; sources?: Source[]; unread?: ScrapeFailure[] };

/** Desktop key-panel copy per provider (bring-your-own-key). */
const PROVIDERS: Record<AiProvider, { name: string; placeholder: string; keyUrl: string; keyLabel: string; free: boolean }> = {
  gemini: { name: "Gemini", placeholder: "AIza…", keyUrl: "https://aistudio.google.com/apikey", keyLabel: "aistudio.google.com/apikey", free: true },
  grok: { name: "Grok", placeholder: "xai-…", keyUrl: "https://console.x.ai", keyLabel: "console.x.ai", free: false },
};

/** Linked pages kept as grounding across follow-up turns (most recent wins). */
const MAX_LINKED_PAGES = 4;

export function Chat({ model, sel }: { model: Model | null; sel: ModelLayer | null }) {
  const t = useT();
  const [msgs, setMsgs] = useState<UiMsg[]>([]);
  const [inp, setInp] = useState("");
  const [ld, setLd] = useState(false);
  const [phase, setPhase] = useState("Researching…");
  // Pages the user linked earlier in this conversation, so a follow-up like
  // "what about INT8 on that page?" stays grounded without re-pasting the URL.
  const linked = useRef<ScrapedPage[]>([]);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  // Bring-your-own-key (desktop only): the user picks Gemini or Grok and pastes
  // their own key for it, stored locally; the desktop proxy passes both to the
  // Rust chat command. On the web build the hosted proxy holds the keys, so
  // this UI stays hidden.
  const desktop = isDesktop();
  const [prov, setProv] = useState<AiProvider>(getAiProvider);
  const P = PROVIDERS[prov];
  const [showKey, setShowKey] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [hasKey, setHasKey] = useState(() => !!getUserApiKey(prov));
  const openKeyPanel = () => { setKeyInput(getUserApiKey(prov)); setShowKey((s) => !s); };
  const pickProvider = (p: AiProvider) => { setAiProvider(p); setProv(p); setKeyInput(getUserApiKey(p)); setHasKey(!!getUserApiKey(p)); };
  const saveKey = () => { setUserApiKey(keyInput, prov); setHasKey(!!keyInput.trim()); setShowKey(false); };
  const clearKey = () => { setUserApiKey("", prov); setHasKey(false); setKeyInput(""); };

  const QP = model ? ["Explain this architecture", "TensorRT conversion + reference docs", "Deploy on Jetson Orin", "Quantization plan", "INT8 vs FP16 tradeoffs", "Best runtime for Hailo-8"] : [];

  const send = async (text: string) => {
    if (!text.trim() || ld || !model) return;
    // Desktop with no key yet: don't fire a doomed request — prompt for the key.
    if (desktop && !getUserApiKey(prov)) {
      setMsgs((m) => [...m,
        { role: "user", content: text },
        { role: "assistant", content: `Add your ${P.name} API key first — click the key icon above. Get ${P.free ? "a free one" : "one"} at ${P.keyUrl} (stored only on this device).` },
      ]);
      setInp("");
      setShowKey(true);
      return;
    }
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
      // Read any URLs the user pasted via the SSRF-guarded /api/scrape proxy and
      // ground this turn on them. Failures don't block the chat — the model is
      // told which links it couldn't read.
      let failures: ScrapeFailure[] = [];
      let turnPages: ScrapedPage[] = [];
      const urlCount = extractUrls(text).length;
      if (urlCount > 0) {
        setPhase(`Reading ${urlCount === 1 ? "linked page" : `${Math.min(urlCount, 3)} linked pages`}…`);
        ({ pages: turnPages, failures } = await scrapeMessageUrls(text));
        const fresh = new Set(turnPages.map((p) => p.finalUrl));
        linked.current = [...linked.current.filter((p) => !fresh.has(p.finalUrl)), ...turnPages].slice(-MAX_LINKED_PAGES);
        setPhase("Researching…");
      }
      const grounding = formatScrapedForPrompt(linked.current, failures);
      // Routes through the server-side proxy — the API key never reaches the browser.
      const { text: reply, sources } = await sendChat({
        messages: wire,
        modelSummary: grounding ? `${modelSummary}\n\n${grounding}` : modelSummary,
      });
      // This turn's linked pages first (the user asked about them), then search citations.
      const seen = new Set<string>();
      const allSources = [...turnPages.map((p) => ({ title: p.title || p.finalUrl, url: p.finalUrl })), ...sources]
        .filter((s) => !seen.has(s.url) && !!seen.add(s.url));
      setMsgs((p) => [...p, { role: "assistant", content: reply, sources: allSources, unread: failures }]);
    } catch (e) {
      setMsgs((p) => [...p, { role: "assistant", content: e instanceof Error ? e.message : "Connection error. Is the /api/chat proxy running?" }]);
    }
    setLd(false);
    setPhase("Researching…");
  };

  return <div style={{ display: "flex", flexDirection: "column", height: "100%", background: t.bg, borderRadius: 6, border: `1px solid ${t.bdr}`, overflow: "hidden" }}>
    <div style={{ padding: "6px 10px", borderBottom: `1px solid ${t.bdr}`, fontSize: 11, fontWeight: 600, color: t.t0, display: "flex", alignItems: "center", gap: 5 }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: t.suc, display: "inline-block" }} />AI Copilot
      <span style={{ marginLeft: "auto", fontSize: 8, color: t.t3, fontWeight: 400 }}>edge-ML expert · cites sources</span>
      {desktop && <button type="button" onClick={openKeyPanel} title={hasKey ? `${P.name} API key set — click to change` : `Set your ${P.name} API key`} style={{ marginLeft: 4, padding: "2px 6px", borderRadius: 5, border: `1px solid ${hasKey ? t.bdr : t.acc}`, background: "transparent", color: hasKey ? t.t2 : t.acc, fontSize: 9, fontWeight: 600, cursor: "pointer" }}>🔑{hasKey ? "" : " Set key"}</button>}
    </div>
    {desktop && showKey && <div style={{ padding: 8, borderBottom: `1px solid ${t.bdr}`, background: t.bg1, display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <span style={{ fontSize: 9, color: t.t3 }}>Provider</span>
        {(["gemini", "grok"] as const).map((p) => <button key={p} type="button" onClick={() => pickProvider(p)} aria-pressed={prov === p} style={{ padding: "2px 8px", borderRadius: 5, border: `1px solid ${prov === p ? t.acc : t.bdr}`, background: prov === p ? t.acc : "transparent", color: prov === p ? "#fff" : t.t2, fontSize: 9, fontWeight: 600, cursor: "pointer" }}>{PROVIDERS[p].name}</button>)}
      </div>
      <div style={{ fontSize: 9, color: t.t2, lineHeight: 1.5 }}>Paste your own <b>{P.name} API key</b> — stored only on this device, used for your chats. {P.free ? "Free key" : "Get a key"} at <a href={P.keyUrl} target="_blank" rel="noreferrer noopener" style={{ color: t.acc }}>{P.keyLabel}</a>.</div>
      <div style={{ display: "flex", gap: 4 }}>
        <input type="password" value={keyInput} onChange={(e) => setKeyInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && saveKey()} placeholder={P.placeholder} style={{ flex: 1, padding: "6px 8px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 10, outline: "none" }} />
        <button type="button" onClick={saveKey} style={{ padding: "6px 12px", borderRadius: 5, border: "none", background: t.acc, color: "#fff", fontSize: 10, fontWeight: 600, cursor: "pointer" }}>Save</button>
        {hasKey && <button type="button" onClick={clearKey} style={{ padding: "6px 10px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 10, cursor: "pointer" }}>Clear</button>}
      </div>
    </div>}
    <div style={{ flex: 1, overflow: "auto", padding: 8, display: "flex", flexDirection: "column", gap: 5 }}>
      {msgs.length === 0 && <div style={{ color: t.t3, fontSize: 10, padding: 10, textAlign: "center" }}>{desktop && !hasKey ? `Add your ${P.name} API key (key icon, top-right) to start chatting.` : model ? "Ask about architectures, formats, layers, hardware, quantization, deployment — ask for reference links and it will search + cite them, or paste a URL (spec sheet, op-support page, GitHub issue) and it will read it." : "Load a model."}</div>}
      {msgs.map((m, i) => <div key={i} style={{ alignSelf: m.role === "user" ? "flex-end" : "flex-start", maxWidth: "90%", display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ padding: "6px 10px", borderRadius: m.role === "user" ? "10px 10px 3px 10px" : "10px 10px 10px 3px", background: m.role === "user" ? t.acc : t.bg1, border: m.role === "user" ? "none" : `1px solid ${t.bdr}`, fontSize: 11, color: m.role === "user" ? "#fff" : t.t0, lineHeight: 1.5, whiteSpace: "pre-wrap", fontFamily: m.role === "assistant" ? "'JetBrains Mono',monospace" : "inherit" }}>{m.content}</div>
        {m.sources && m.sources.length > 0 && <div style={{ padding: "5px 8px", borderRadius: 6, background: t.acc + "0E", border: `1px solid ${t.acc}33` }}>
          <div style={{ fontSize: 8, fontWeight: 700, color: t.acc, textTransform: "uppercase", letterSpacing: 1, marginBottom: 3 }}>Sources</div>
          {m.sources.map((s, j) => <a key={j} href={s.url} target="_blank" rel="noreferrer noopener" style={{ display: "block", fontSize: 10, color: t.acc, textDecoration: "none", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: 1 }}>{j + 1}. {s.title}</a>)}
        </div>}
        {m.unread && m.unread.length > 0 && <div style={{ padding: "4px 8px", borderRadius: 6, background: t.wrn + "12", border: `1px solid ${t.wrn}44`, fontSize: 9, color: t.t2 }}>
          {m.unread.map((f, j) => <div key={j} title={f.error} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}><span style={{ color: t.wrn, fontWeight: 700 }}>Couldn't read</span> {f.url} — {f.error}</div>)}
        </div>}
      </div>)}
      {ld && <div style={{ alignSelf: "flex-start", padding: "6px 10px", borderRadius: 10, background: t.bg1, border: `1px solid ${t.bdr}`, fontSize: 10, color: t.t3 }}><span style={{ animation: "pulse 1.2s infinite", color: t.acc }}>●</span> {phase}</div>}
      <div ref={end} />
    </div>
    {model && msgs.length === 0 && <div style={{ padding: "0 8px 4px", display: "flex", flexWrap: "wrap", gap: 2 }}>{QP.map((p, i) => <button key={i} type="button" onClick={() => send(p)} style={{ padding: "2px 7px", borderRadius: 8, border: `1px solid ${t.bdr}`, background: "transparent", color: t.t2, fontSize: 9, cursor: "pointer" }}>{p}</button>)}</div>}
    <div style={{ padding: 5, borderTop: `1px solid ${t.bdr}`, display: "flex", gap: 4 }}>
      <input value={inp} onChange={(e) => setInp(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send(inp)} placeholder="Ask anything…" disabled={!model} style={{ flex: 1, padding: "6px 8px", borderRadius: 5, border: `1px solid ${t.bdr}`, background: t.bg1, color: t.t0, fontSize: 10, outline: "none" }} />
      <button type="button" onClick={() => send(inp)} disabled={ld || !model} style={{ padding: "6px 12px", borderRadius: 5, border: "none", background: ld ? t.bg3 : t.acc, color: "#fff", fontSize: 10, fontWeight: 600, cursor: ld ? "default" : "pointer" }}>Send</button>
    </div>
  </div>;
}
