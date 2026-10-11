// Deploy recipe for an optimization-search config. Reuses the Converter's
// script templates (data/convScripts.ts) with options derived from the config,
// instead of duplicating conversion commands here.
//
// Pure module: no React.

import type { Model } from "@modelvisio/parsers";
import { CONV_SCRIPTS, type ConvOpts } from "../data/convScripts";
import { fixById } from "../fixes/transforms";
import type { ConfigResult } from "./search";

export type RecipeStep = { title: string; scriptKey?: string; body: string };
export type ConfigRecipe = { steps: RecipeStep[]; text: string };

const isPt = (fmt: string) => /pytorch|torchscript|torch|\.pt$/i.test(fmt);
const isOnnx = (fmt: string) => /onnx/i.test(fmt);

/** Converter options matching a config's precision, resolution and target. */
export function convOptsFor(model: Model, c: ConfigResult): ConvOpts {
  const h = model.inputShape[2] > 0 ? model.inputShape[2] : 640;
  return {
    opset: model.opset > 0 ? Math.max(model.opset, 13) : 18,
    sz: Math.max(32, Math.round((h * c.scale) / 32) * 32),
    dyn: false,
    i8: c.precision === "int8" || c.precision === "int4",
    fp16: c.precision === "fp16",
    inp: model.name || "model.onnx",
    out: "",
    // trtexec: --int8 / --fp16; strict FP32 is --noTF32 (FP32 is the default).
    prec: c.precision === "int8" || c.precision === "int4" ? "int8" : c.precision === "fp16" ? "fp16" : "noTF32",
    // The only RKNN target in the device catalog is the RK3588.
    plat: "rk3588",
  };
}

export function buildRecipe(model: Model, c: ConfigResult): ConfigRecipe {
  const steps: RecipeStep[] = [];
  const o = convOptsFor(model, c);

  if (c.fixes.length) {
    steps.push({
      title: `Apply ${c.fixes.length} graph fix${c.fixes.length > 1 ? "es" : ""} in your export pipeline`,
      body: c.fixes.map((id) => { const f = fixById(id); return f ? `# ${f.title}: ${f.change}\n#   ${f.impact}` : `# ${id}`; }).join("\n"),
    });
  }

  const key = c.scriptKey;
  if (key && CONV_SCRIPTS[key]) {
    const src = key.split("→")[0];
    if (src === "onnx" && isPt(model.format) && CONV_SCRIPTS["pt→onnx"]) {
      steps.push({ title: "Export to ONNX", scriptKey: "pt→onnx", body: CONV_SCRIPTS["pt→onnx"]({ ...o, inp: model.name }) });
      o.inp = model.name.replace(/\.[^.]+$/, "") + ".onnx";
    }
    const pre = src === "pt" && isOnnx(model.format)
      ? `# This step starts from the PyTorch checkpoint; ${model.name} is ONNX, so run it on the source .pt.\n`
      : "";
    steps.push({ title: `Convert for ${c.runtime} (${c.precision.toUpperCase()})`, scriptKey: key, body: pre + CONV_SCRIPTS[key](o) });
  } else {
    steps.push({
      title: `Compile with ${c.runtime}`,
      body: `# No generated script for ${c.runtime} yet. Use the vendor toolchain (${c.device.toolchain.join(", ")})\n`
        + `# to compile ${model.name} at ${c.precision.toUpperCase()}${c.scale !== 1 ? ` with input ${o.sz}×${o.sz}` : ""}`
        + `${c.precision === "int8" ? ", with a calibration set of ~200 representative inputs" : ""}.`,
    });
  }

  const head = [
    `# ModelVisio optimization search: ${c.device.n} · ${c.runtime} · ${c.precision.toUpperCase()}`
      + `${c.fixes.length ? ` · fixes: ${c.fixes.join(", ")}` : ""}${c.scale !== 1 ? ` · ${o.sz}px input` : ""}`,
    `# Est. ${c.latencyMs.toFixed(2)} ms (~${Math.round(c.fps)} FPS), ${c.sizeMB.toFixed(1)} MB weights, accuracy risk ${(c.risk * 100).toFixed(0)}% (estimates)`,
  ].join("\n");
  const text = [head, ...steps.map((s, i) => `\n# ── Step ${i + 1}: ${s.title} ──\n${s.body}`)].join("\n");
  return { steps, text };
}
