import { useState } from "react";
import type { Model } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";
import { deployRecipes, DEPLOY_RECIPE_TARGETS } from "../deploy/recipeTemplates";

export function DeployRecipe({ model }: { model: Model }) {
  const t = useT();
  const [hw, setHw] = useState("orin_nano");
  const modelFile = model?.name || "model.onnx";

  const recipes = deployRecipes(modelFile);
  const targets = DEPLOY_RECIPE_TARGETS;

  return <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%" }}>
    <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Deployment Recipe Generator</div>
    <div style={{ display: "flex", gap: 4 }}>{targets.map((tg) => <button key={tg.id} type="button" onClick={() => setHw(tg.id)} style={{ padding: "4px 10px", borderRadius: 4, border: `1px solid ${hw === tg.id ? t.acc3 : t.bdr}`, background: hw === tg.id ? t.acc3 + "18" : "transparent", color: hw === tg.id ? t.acc3 : t.t2, fontSize: 10, cursor: "pointer", fontWeight: 500 }}>{tg.n}</button>)}</div>
    <div style={{ flex: 1, overflow: "auto", position: "relative" }}>
      <button type="button" onClick={() => navigator.clipboard?.writeText(recipes[hw] || "")} style={{ position: "absolute", top: 6, right: 6, padding: "3px 8px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t2, fontSize: 9, cursor: "pointer", zIndex: 2 }}>Copy All</button>
      <pre style={{ background: t.bg, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: 12, fontSize: 10.5, color: t.suc, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.5, whiteSpace: "pre-wrap", height: "100%", margin: 0, overflow: "auto" }}>{recipes[hw] || "Coming soon."}</pre>
    </div>
  </div>;
}
