import { useEffect, useState } from "react";
import { ggufMetadataJson, toGraphJson, toLayerCsv, type ConvTarget, type Model, type Precision } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { CONV_SCRIPTS, type ConvOpts } from "../data/convScripts";
import { buildKit } from "../data/convKits";
import { convertWeightsInWorker } from "../workers/parseClient";
import { downloadFile } from "../utils/download";

const base = (n: string) => n.replace(/\.[^.]+$/, "");

export function Converter({ model, modelBytes }: { model: Model; modelBytes: ArrayBuffer | null }) {
  const t = useT();
  const convs = Object.keys(CONV_SCRIPTS);
  const fmt = model.format;
  const isOnnx = /onnx/i.test(fmt);
  const isSafetensors = /safetensors/i.test(fmt);
  const isNumpy = /numpy/i.test(fmt);
  const isGguf = /gguf/i.test(fmt);
  const isPt = /pytorch|torchscript|\.?pt$|\.?pth$|torch/i.test(fmt);
  // Only offer conversion targets whose SOURCE matches the loaded model.
  // ONNX is the standard interchange format — no format conversion is offered.
  const srcKey = isPt ? "pt" : null;
  const eligible = srcKey ? convs.filter((c) => c.split("→")[0] === srcKey) : [];

  const [sel, setSel] = useState(eligible[0] ?? convs[0]);
  const [opts, setOpts] = useState<ConvOpts>({
    opset: 18, sz: 640, dyn: false, i8: false, fp16: true,
    inp: model?.name || "model.onnx", out: "", prec: "fp16", plat: "rk3588",
  });
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  // Keep the selected target valid when a different model is loaded.
  useEffect(() => {
    setSel((s) => (eligible.includes(s) ? s : eligible[0] ?? convs[0]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fmt]);

  const script = CONV_SCRIPTS[sel]?.(opts) || "";
  const hasTargets = eligible.length > 0;

  const numFields: [string, keyof ConvOpts, number, number][] = [["Opset", "opset", 9, 21], ["ImgSz", "sz", 64, 1280]];
  const boolFields: [string, keyof ConvOpts][] = [["Dynamic", "dyn"], ["INT8", "i8"], ["FP16", "fp16"]];

  async function run(kind: string, label: string) {
    setErr(null); setDone(null); setBusy(kind);
    try {
      if (kind === "json") {
        downloadFile(toGraphJson(model), `${base(model.name)}.graph.json`, "application/json");
      } else if (kind === "csv") {
        downloadFile(toLayerCsv(model), `${base(model.name)}.layers.csv`, "text/csv");
      } else if (kind === "gguf-json") {
        if (!modelBytes) throw new Error("Original file isn't available (demo model).");
        downloadFile(ggufMetadataJson(modelBytes), `${base(model.name)}.gguf.json`, "application/json");
      } else if (kind.startsWith("st:") || kind.startsWith("npz:")) {
        if (!modelBytes) throw new Error("Original file isn't available (demo model). Load a real file to export weights.");
        const [tg, pr] = kind.split(":");
        const target: ConvTarget = tg === "npz" ? "npz" : "safetensors";
        const r = await convertWeightsInWorker(modelBytes, model.name, fmt, target, pr as Precision);
        downloadFile(r.data, r.filename, r.mime);
      } else if (kind === "kit") {
        if (!modelBytes) throw new Error("Original file isn't available (demo model). Load a real file to build a runnable kit.");
        const scriptForKit = CONV_SCRIPTS[sel]?.({ ...opts, inp: model.name }) || script;
        const { filename, data } = buildKit(sel, scriptForKit, modelBytes, model.name);
        downloadFile(data, filename, "application/zip");
      }
      setDone(label);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  const dlBtn = (kind: string, label: string, accent = t.acc3) => (
    <button
      type="button"
      disabled={busy !== null}
      onClick={() => run(kind, label)}
      style={{ padding: "5px 10px", borderRadius: 5, border: `1px solid ${accent}55`, background: accent + "14", color: accent, fontSize: 10, fontWeight: 600, cursor: busy ? "default" : "pointer", opacity: busy && busy !== kind ? 0.5 : 1 }}
    >
      {busy === kind ? "…" : "⬇"} {label}
    </button>
  );

  return <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%" }}>
    <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Format Converter</div>

    {/* Convert & Download */}
    <div style={{ background: t.glass, backdropFilter: "blur(8px)", borderRadius: 6, border: `1px solid ${t.glassBdr}`, padding: 8 }}>
      <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>Convert &amp; Download</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {dlBtn("json", "Graph JSON", t.acc)}
        {dlBtn("csv", "Layers CSV", t.acc)}
        {isGguf && dlBtn("gguf-json", "GGUF metadata JSON", t.acc)}
        {isSafetensors && dlBtn("st:f16", "Recast → FP16")}
        {isSafetensors && dlBtn("st:f32", "Recast → FP32")}
        {isSafetensors && dlBtn("npz:keep", "→ NumPy .npz")}
        {isNumpy && dlBtn("st:keep", "→ Safetensors")}
        {isNumpy && dlBtn("st:f16", "→ Safetensors (FP16)")}
        {hasTargets && dlBtn("kit", `${sel} kit (.zip)`, t.acc2)}
      </div>
      {err && <div style={{ marginTop: 6, fontSize: 10, color: t.err }}>{err}</div>}
      {done && !err && <div style={{ marginTop: 6, fontSize: 10, color: t.suc }}>✓ Downloaded {done}</div>}
      <div style={{ marginTop: 6, fontSize: 9, color: t.t3, lineHeight: 1.5 }}>
        {isOnnx
          ? "ONNX is already the standard interchange format — no model-format conversion is offered. Graph JSON / Layers CSV export the analysis."
          : "In-browser exports (JSON / CSV / Safetensors / NumPy .npz, FP16↔FP32 recast) produce a real file instantly. Native targets (TensorRT, TFLite, CoreML, OpenVINO, RKNN, …) need their toolchain — the kit bundles your model + a one-command script (Docker or pip)."}
      </div>
    </div>

    {/* Target + options + script — only for sources with native conversion targets (ONNX, PyTorch) */}
    {hasTargets ? <>
      <div style={{ fontSize: 9, fontWeight: 700, color: t.t3, textTransform: "uppercase", letterSpacing: 1 }}>Conversion kit / script · {srcKey} source</div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>{eligible.map((c) => <button key={c} type="button" onClick={() => setSel(c)} style={{ padding: "4px 8px", borderRadius: 4, border: `1px solid ${sel === c ? t.acc : t.bdr}`, background: sel === c ? t.acc + "18" : t.bg1, color: sel === c ? t.acc : t.t2, fontSize: 10, fontWeight: 600, cursor: "pointer", fontFamily: "'JetBrains Mono',monospace" }}>{c}</button>)}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, padding: 8, background: t.glass, backdropFilter: "blur(8px)", borderRadius: 6, border: `1px solid ${t.glassBdr}` }}>
        {numFields.map(([l, k, mn, mx]) => <label key={k} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: t.t1 }}>{l}<input type="number" value={opts[k] as number} min={mn} max={mx} onChange={(e) => setOpts((p) => ({ ...p, [k]: +e.target.value || 0 }))} style={{ width: 50, padding: "2px 4px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: t.bg, color: t.t0, fontSize: 10, fontFamily: "'JetBrains Mono',monospace" }} /></label>)}
        {boolFields.map(([l, k]) => <label key={k} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: t.t1, cursor: "pointer" }}><input type="checkbox" checked={opts[k] as boolean} onChange={(e) => setOpts((p) => ({ ...p, [k]: e.target.checked }))} style={{ accentColor: t.acc }} />{l}</label>)}
      </div>
      <div style={{ flex: 1, overflow: "auto", position: "relative", minHeight: 120 }}>
        <button type="button" onClick={() => navigator.clipboard?.writeText(script)} style={{ position: "absolute", top: 6, right: 6, padding: "3px 8px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t2, fontSize: 9, cursor: "pointer", zIndex: 2 }}>Copy</button>
        <pre style={{ background: t.bg, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: 12, fontSize: 10.5, color: t.t1, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.6, whiteSpace: "pre-wrap", height: "100%", margin: 0, overflow: "auto" }}>{script}</pre>
      </div>
    </> : (
      <div style={{ fontSize: 10, color: t.t3, lineHeight: 1.5 }}>
        {isOnnx
          ? "This is an ONNX model — the standard interchange format. No conversion needed; it runs directly in any ONNX runtime (ONNX Runtime, TensorRT, OpenVINO, etc.)."
          : `No native conversion targets for a ${fmt} source yet — use the in-browser exports above. (Conversion kits are available for PyTorch sources.)`}
      </div>
    )}
  </div>;
}
