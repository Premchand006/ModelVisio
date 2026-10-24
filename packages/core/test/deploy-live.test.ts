import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { LlmFn } from "@modelvisio/ai";
import { runChatProxy } from "@modelvisio/ai/proxy";
import { demoModel } from "../src/demo/demoModel";
import { runDeployment } from "../src/deploy/agent";

// Live check of the deployment agent's AI path: the real tool loop driven by a
// real Gemini call through the same proxy the app uses. Skipped unless a key
// is available (GEMINI_API_KEY / GOOGLE_API_KEY in the environment or the
// repo-root .env), so CI and offline runs are unaffected. Run it with:
//   pnpm --filter @modelvisio/core test:live-ai
// It makes a handful of model calls (≤ 8 agent steps) on the configured model.

function keyFromEnvFile(): string | undefined {
  const f = new URL("../../../.env", import.meta.url);
  if (!existsSync(f)) return undefined;
  const m = /^(?:GEMINI_API_KEY|GOOGLE_API_KEY)=(.+)$/m.exec(readFileSync(f, "utf8"));
  return m?.[1].trim().replace(/^["']|["']$/g, "") || undefined;
}

const KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || keyFromEnvFile();
const MODEL = process.env.MODELVISIO_MODEL || "gemini-2.5-flash";

describe.skipIf(!KEY)("deployment agent against live Gemini", () => {
  it("drives real tools and returns a valid package without falling back", async () => {
    const llm: LlmFn = async ({ system, messages }) =>
      (await runChatProxy({ key: KEY!, system, messages, model: MODEL, webSearch: false })).text;
    const { run, pkg } = await runDeployment({
      model: demoModel,
      goal: "Deploy this detector to a Jetson Orin Nano at 30 FPS or better; INT8 is fine.",
      llm,
    });
    // "fallback" would mean the model's replies were unusable and the
    // deterministic planner took over: exactly what this test exists to catch.
    expect(run.mode, run.fallbackReason).toBe("ai");
    expect(run.steps.length).toBeGreaterThan(0);
    expect(run.steps.every((s) => s.source === "ai")).toBe(true);
    expect(pkg.decision.device.id).toBe("orin_nano");
    expect(["tensorrt", "onnxruntime"]).toContain(pkg.decision.runtime);
    expect(pkg.files.length).toBeGreaterThan(5);
    console.log(`live agent: ${run.steps.length} tool step(s): ${run.steps.map((s) => s.tool).join(" → ")}; ${pkg.decision.device.n} · ${pkg.decision.runtime} · ${pkg.decision.precision}`);
  }, 180_000);
});
