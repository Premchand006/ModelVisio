import { describe, expect, it } from "vitest";
import type { LlmFn } from "@modelvisio/ai";
import { demoModel } from "../src/demo/demoModel";
import { HW } from "../src/data/hardware";
import { parseGoal, resolveDevice } from "../src/deploy/goal";
import { generateRecipe } from "../src/deploy/recipes";
import { deployRecipes, deployRecipeText, DEPLOY_RECIPE_TARGETS } from "../src/deploy/recipeTemplates";
import { CONV_SCRIPTS } from "../src/data/convScripts";
import { runtimesFor, precisionsFor } from "../src/deploy/runtimes";
import { createDeploySession, createDeployTools } from "../src/deploy/tools";
import { runDeployment } from "../src/deploy/agent";
import { packageToZip } from "../src/deploy/package";

const dev = (id: string) => HW.find((d) => d.id === id)!;

describe("parseGoal", () => {
  it("extracts device, FPS target and INT8 permission", () => {
    const g = parseGoal("deploy this to a Jetson Orin Nano at ≥30 FPS, INT8 OK");
    expect(g.deviceIds).toEqual(["orin_nano"]);
    expect(g.minFps).toBe(30);
    expect(g.int8Ok).toBe(true);
  });

  it("maps platforms and runtimes", () => {
    const g = parseGoal("ship to Android phones via TFLite");
    expect(g.platform).toBe("Android phones");
    expect(g.platformDeviceIds).toEqual(["rb5", "rb6", "mtk_genio"]);
    expect(g.runtime).toBe("tflite");
    expect(parseGoal("Raspberry Pi, lowest power, under 5 W").maxPowerW).toBe(5);
    expect(parseGoal("Raspberry Pi, lowest power").optimize).toBe("power");
    expect(parseGoal("no INT8, fp16 only on Intel NPU").int8Ok).toBe(false);
    expect(parseGoal("Hailo-8L under 20 ms").deviceIds).toEqual(["hailo8l"]);
    expect(parseGoal("Hailo-8L under 20 ms").maxLatencyMs).toBe(20);
  });

  it("resolveDevice accepts ids, names and aliases", () => {
    expect(resolveDevice("orin_nano")?.id).toBe("orin_nano");
    expect(resolveDevice("Hailo-8")?.id).toBe("hailo8");
    expect(resolveDevice("Jetson AGX Orin")?.id).toBe("agx_orin");
    expect(resolveDevice("mystery box")).toBeNull();
  });
});

describe("recipe templates (extracted from DeployRecipe)", () => {
  it("keeps the three quick-start targets and substitutes the model file", () => {
    expect(DEPLOY_RECIPE_TARGETS.map((t) => t.id)).toEqual(["orin_nano", "hailo8", "rk3588"]);
    const r = deployRecipes("foo.onnx");
    expect(r.orin_nano).toContain("trtexec --onnx=foo.onnx --saveEngine=model.engine --fp16");
    expect(r.rk3588).toContain('rknn.load_onnx(model="foo.onnx")');
    expect(deployRecipeText("coral", "x")).toBeUndefined();
  });
});

describe("generateRecipe", () => {
  it("produces conversion + validation files for every device × runtime × precision", () => {
    for (const d of HW) for (const rt of runtimesFor(d)) for (const p of precisionsFor(d)) {
      const r = generateRecipe({ model: demoModel, device: d, precision: p, runtime: rt });
      const paths = r.files.map((f) => f.path);
      expect(paths.some((x) => x.startsWith("convert/")), `${d.id}/${rt}/${p}`).toBe(true);
      expect(paths).toContain("validate/accuracy_check.py");
      expect(paths).toContain("validate/reference.py");
      if (p === "int8") expect(paths).toContain("calib/make_calib.py");
      expect(r.files.every((f) => f.content.length > 0)).toBe(true);
      expect(r.benchmarkCmd.length).toBeGreaterThan(0);
    }
  });

  it("reuses the Converter's RKNN template and the quick-start recipe", () => {
    const r = generateRecipe({ model: demoModel, device: dev("rk3588"), precision: "int8" });
    const conv = r.files.find((f) => f.path === "convert/convert.py")!.content;
    expect(conv).toContain(CONV_SCRIPTS["onnx→rknn"]({ opset: 17, sz: 640, dyn: false, i8: true, fp16: true, inp: "yolo26n.onnx", out: "model.rknn", prec: "fp16", plat: "rk3588" }).split("\n")[0]);
    expect(conv).toContain('dataset="calib/calib_list.txt"');
    expect(r.files.some((f) => f.path === "reference/quickstart.sh")).toBe(true);
  });

  it("TensorRT INT8 warns about calibration and sets the target FPS gate", () => {
    const r = generateRecipe({ model: demoModel, device: dev("orin_nano"), precision: "int8", targetFps: 30 });
    expect(r.runtime).toBe("tensorrt");
    expect(r.files.find((f) => f.path === "convert/convert.sh")!.content).toMatch(/--int8 --fp16[\s\S]*PLACEHOLDER|PLACEHOLDER[\s\S]*--int8 --fp16/);
    expect(r.files.find((f) => f.path === "runtime/infer.py")!.content).toContain('"TARGET_FPS", "30"');
  });

  it("adds an ONNX export step for PyTorch sources and an Android snippet for TFLite", () => {
    const pt = { ...demoModel, name: "yolo26n.pt", format: "PyTorch" };
    const r = generateRecipe({ model: pt, device: dev("rb6"), precision: "int8", runtime: "tflite" });
    expect(r.files[0].path).toBe("convert/export_onnx.py");
    expect(r.files.some((f) => f.path === "runtime/android/Detector.kt")).toBe(true);
  });

  it("falls back to the native runtime when given one the device can't use", () => {
    expect(generateRecipe({ model: demoModel, device: dev("hailo8"), precision: "int8", runtime: "tensorrt" }).runtime).toBe("hailort");
  });
});

describe("deploy tools", () => {
  it("apply_fix edits the working copy only", async () => {
    const s = createDeploySession(demoModel);
    const tools = Object.fromEntries(createDeployTools(s).map((t) => [t.name, t]));
    const fixes = (await tools.list_fixes.run({ device: "coral" })) as { id: string; issuesBefore: number; issuesAfter: number }[];
    const silu = fixes.find((f) => f.id === "silu_hardswish")!;
    expect(silu.issuesAfter).toBeLessThan(silu.issuesBefore);
    await tools.apply_fix.run({ id: "silu_hardswish" });
    expect(s.applied).toEqual(["silu_hardswish"]);
    expect(s.working).not.toBe(demoModel);
    expect(demoModel.layers.some((l) => /silu/i.test(l.type))).toBe(true); // base untouched
    await expect(Promise.resolve().then(() => tools.apply_fix.run({ id: "bogus" }))).rejects.toThrow(/Unknown fix/);
    await expect(Promise.resolve().then(() => tools.estimate_latency.run({ device: "nope" }))).rejects.toThrow(/Unknown device/);
  });

  it("tool digests are short one-liners", async () => {
    const s = createDeploySession(demoModel);
    for (const t of createDeployTools(s)) {
      const args = t.name === "apply_fix" ? { id: "silu_hardswish" } : { device: "orin_nano", precision: "int8" };
      const res = await t.run(args);
      expect(t.digest!(res).length).toBeGreaterThan(0);
    }
  });
});

describe("runDeployment (end to end)", () => {
  it("deterministic planner: Jetson Orin Nano ≥30 FPS INT8 → full package", async () => {
    const steps: string[] = [];
    const { run, pkg } = await runDeployment({
      model: demoModel, goal: "deploy this to a Jetson Orin Nano at ≥30 FPS, INT8 OK", llm: null,
      noAiReason: "No Gemini key set", onStep: (s) => steps.push(s.tool),
    });
    expect(run.mode).toBe("fallback");
    expect(run.fallbackReason).toBe("No Gemini key set");
    expect(steps.slice(0, 3)).toEqual(["get_model_summary", "list_devices", "estimate_latency"]);
    expect(steps).toContain("generate_recipe");
    expect(steps).toContain("fleet_simulate");
    expect(pkg.decision.device.id).toBe("orin_nano");
    expect(pkg.decision.precision).toBe("int8");
    expect(pkg.decision.runtime).toBe("tensorrt");
    expect(pkg.checklist.some((c) => c.kind === "accuracy")).toBe(true);
    expect(pkg.checklist.some((c) => c.kind === "latency")).toBe(true);
    const paths = pkg.files.map((f) => f.path);
    for (const p of ["README.md", "PLAN.md", "VALIDATION.md", "RISKS.md", "deploy.json", "convert/convert.sh", "runtime/infer.py"]) expect(paths).toContain(p);
    const zip = packageToZip(pkg);
    expect(zip.filename).toBe("yolo26n-orin_nano-tensorrt-int8.zip");
    expect(Array.from(zip.data.slice(0, 4))).toEqual([0x50, 0x4b, 0x03, 0x04]);
  });

  it("planner fixes Coral compat issues before generating the recipe", async () => {
    const { pkg, session } = await runDeployment({ model: demoModel, goal: "Coral Edge TPU", llm: null });
    expect(pkg.decision.device.id).toBe("coral");
    expect(session.applied).toContain("silu_hardswish");
    expect(pkg.risks.some((r) => /analysis model only/.test(r))).toBe(true);
  });

  it("Android via TFLite picks a phone-class SoC on TFLite", async () => {
    const { pkg } = await runDeployment({ model: demoModel, goal: "ship to Android phones via TFLite", llm: null });
    expect(["rb5", "rb6", "mtk_genio"]).toContain(pkg.decision.device.id);
    expect(pkg.decision.runtime).toBe("tflite");
  });

  it("AI path: scripted LLM drives tools; invalid choices are corrected", async () => {
    const replies = [
      '{"thought":"size","action":"get_model_summary","args":{}}',
      '{"thought":"est","action":"estimate_latency","args":{"device":"hailo8","precision":"int8"}}',
      '{"thought":"recipe","action":"generate_recipe","args":{"device":"hailo8","precision":"int8"}}',
      '{"final":{"target":"Hailo-8","runtime":"TensorRT","precision":"fp16","justification":"fast","risks":["AI risk"]}}',
    ];
    let i = 0;
    const llm: LlmFn = async () => replies[i++];
    const { run, pkg } = await runDeployment({ model: demoModel, goal: "Hailo-8 please", llm });
    expect(run.mode).toBe("ai");
    expect(pkg.decision.device.id).toBe("hailo8");
    expect(pkg.decision.precision).toBe("int8");
    expect(pkg.decision.runtime).toBe("hailort");
    expect(pkg.decision.adjustments.length).toBe(2);
    expect(pkg.decision.justification).toBe("fast");
    expect(pkg.risks[0]).toBe("AI risk");
    expect(pkg.steps.map((s) => s.tool)).toEqual(["get_model_summary", "estimate_latency", "generate_recipe"]);
  });

  it("proxy failure falls back to the planner and still yields a package", async () => {
    const llm: LlmFn = async () => { throw new Error("Chat proxy returned 502."); };
    const { run, pkg } = await runDeployment({ model: demoModel, goal: "cheapest device at 15 FPS", llm });
    expect(run.mode).toBe("fallback");
    expect(run.fallbackReason).toMatch(/502/);
    expect(pkg.files.length).toBeGreaterThan(5);
  });
});
