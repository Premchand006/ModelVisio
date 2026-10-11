import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
// The parsers' own protobuf schema, so test models use the real ONNX wire format.
import { modelProtoType } from "../../parsers/src/onnx/proto";
import { run } from "../src/cli";

const PKG = fileURLToPath(new URL("..", import.meta.url));
const YOLO_PT = fileURLToPath(new URL("../../../yolov8n.pt", import.meta.url));

// X[1,3,H,W] → Conv(C×3×3×3) → Relu → GlobalAveragePool → Flatten → Gemm(10×C) → out[1,10]
function onnx({ hw = 32, ch = 8, extra = false }: { hw?: number; ch?: number; extra?: boolean } = {}): Uint8Array {
  const ModelProto = modelProtoType();
  const tensor = (name: string, dims: number[]) => ({ name, data_type: 1, dims, float_data: new Array(dims.reduce((a, b) => a * b, 1)).fill(0.1) });
  const vi = (name: string, dims: number[]) => ({ name, type: { tensor_type: { elem_type: 1, shape: { dim: dims.map((d) => ({ dim_value: d })) } } } });
  const nodes: unknown[] = [
    { op_type: "Conv", name: "conv1", input: ["X", "convW"], output: ["c1"], attribute: [
      { name: "kernel_shape", type: 7, ints: [3, 3] }, { name: "strides", type: 7, ints: [1, 1] },
      { name: "pads", type: 7, ints: [1, 1, 1, 1] }, { name: "group", type: 2, i: 1 }] },
    { op_type: "Relu", name: "relu1", input: ["c1"], output: ["r1"] },
    { op_type: "GlobalAveragePool", name: "gap", input: [extra ? "r2" : "r1"], output: ["g1"] },
    { op_type: "Flatten", name: "flatten", input: ["g1"], output: ["f1"], attribute: [{ name: "axis", type: 2, i: 1 }] },
    { op_type: "Gemm", name: "fc", input: ["f1", "fcW", "fcB"], output: ["out"], attribute: [{ name: "transB", type: 2, i: 1 }] },
  ];
  if (extra) nodes.splice(2, 0, { op_type: "Sigmoid", name: "sig", input: ["r1"], output: ["r2"] });
  const obj = {
    ir_version: 8, producer_name: "cli-test", opset_import: [{ domain: "", version: 17 }],
    graph: {
      name: "tiny", input: [vi("X", [1, 3, hw, hw])], output: [vi("out", [1, 10])],
      initializer: [tensor("convW", [ch, 3, 3, 3]), tensor("fcW", [10, ch]), tensor("fcB", [10])],
      node: nodes,
    },
  };
  return ModelProto.encode(ModelProto.create(obj)).finish();
}

let dir = "";
async function cli(...argv: string[]) {
  let stdout = "", stderr = "";
  const code = await run(argv, { stdout: (s) => { stdout += s; }, stderr: (s) => { stderr += s; }, cwd: dir, env: {} });
  return { code, stdout, stderr };
}

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "modelvisio-cli-"));
  writeFileSync(join(dir, "a.onnx"), onnx());
  writeFileSync(join(dir, "a-copy.onnx"), onnx());
  writeFileSync(join(dir, "b.onnx"), onnx({ hw: 128, ch: 64, extra: true }));
  writeFileSync(join(dir, "junk.bin2"), new Uint8Array([1, 2, 3, 4]));
});
afterAll(() => { rmSync(dir, { recursive: true, force: true }); });

describe("inspect", () => {
  it("prints a text summary", async () => {
    const r = await cli("inspect", "a.onnx");
    expect(r.code).toBe(0);
    expect(r.stdout).toMatch(/Format\s+ONNX/);
    expect(r.stdout).toMatch(/Opset\s+17/);
    expect(r.stdout).toContain("X  [1, 3, 32, 32]");
    expect(r.stdout).toContain("out  [1, 10]");
  });

  it("--json gives machine-readable stats", async () => {
    const r = await cli("inspect", "a.onnx", "--json");
    const s = JSON.parse(r.stdout);
    expect(s).toMatchObject({ format: "ONNX", opset: 17, framework: "cli-test", graphParsed: true });
    expect(s.params).toBe(8 * 3 * 3 * 3 + 10 * 8 + 10);
    expect(s.flops).toBeGreaterThan(0);
    expect(s.inputs).toEqual([{ name: "X", shape: [1, 3, 32, 32] }]);
  });

  it("errors with exit 2 on a missing file", async () => {
    const r = await cli("inspect", "nope.onnx");
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("File not found");
  });
});

describe("check", () => {
  it("passes a small model on generous budgets", async () => {
    const r = await cli("check", "a.onnx", "--target", "coral,hailo", "--max-size-mb", "1", "--max-latency-ms", "100");
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("PASSED");
    expect(r.stdout).toContain("Latency on Google Coral Edge TPU");
  });

  it("fails (exit 1) when a budget is exceeded", async () => {
    const r = await cli("check", "b.onnx", "--target", "rk3588", "--max-size-mb", "0.001");
    expect(r.code).toBe(1);
    expect(r.stdout).toContain("EXCEEDED");
  });

  it("--device overrides the latency device", async () => {
    const r = await cli("check", "a.onnx", "--target", "coral", "--device", "agx_orin", "--max-latency-ms", "0.000001", "--format", "json");
    expect(r.code).toBe(1);
    const rep = JSON.parse(r.stdout);
    expect(rep.models[0].budgets).toHaveLength(1);
    expect(rep.models[0].budgets[0]).toMatchObject({ subject: "agx_orin", status: "exceeded" });
  });

  it("metadata-only formats warn, and fail only with --fail-on warn", async () => {
    writeFileSync(join(dir, "m.engine"), new Uint8Array(64));
    const ok = await cli("check", "m.engine", "--target", "coral");
    expect(ok.code).toBe(0);
    expect(ok.stdout).toContain("graph not parsed");
    const strict = await cli("check", "m.engine", "--target", "coral", "--fail-on", "warn");
    expect(strict.code).toBe(1);
  });

  it("junit output is well-formed and counts failures", async () => {
    const r = await cli("check", "b.onnx", "--target", "coral", "--max-size-mb", "0.001", "--format", "junit");
    expect(r.stdout.startsWith('<?xml version="1.0" encoding="UTF-8"?>')).toBe(true);
    expect(r.stdout).toMatch(/<testsuites name="modelvisio check" tests="\d+" failures="1">/);
    expect(r.stdout).toContain("<failure message=");
    expect((r.stdout.match(/<testsuite /g) ?? []).length).toBe((r.stdout.match(/<\/testsuite>/g) ?? []).length);
  });

  it("markdown to --output and --summary (GITHUB_STEP_SUMMARY style append)", async () => {
    const summary = join(dir, "summary.md");
    writeFileSync(summary, "# existing\n");
    const r = await cli("check", "a.onnx", "b.onnx", "--format", "markdown", "--output", "report.md", "--summary", summary);
    expect(r.code).toBe(0);
    expect(r.stdout).toBe("");
    const md = readFileSync(join(dir, "report.md"), "utf8");
    expect(md).toContain("## ModelVisio check: passed");
    expect(md).toContain("| `a.onnx` | ONNX |");
    expect(md).toContain("| `b.onnx` | ONNX |");
    expect(readFileSync(summary, "utf8")).toMatch(/^# existing\n## ModelVisio check/);
  });

  it("rejects unknown targets and bad flags with exit 2", async () => {
    expect((await cli("check", "a.onnx", "--target", "warp-drive")).code).toBe(2);
    expect((await cli("check", "a.onnx", "--fail-on", "maybe")).code).toBe(2);
    expect((await cli("check", "a.onnx", "--bogus")).code).toBe(2);
    expect((await cli("check", "a.onnx", "--max-size-mb", "-1")).code).toBe(2);
  });
});

describe("diff", () => {
  it("identical files: no regressions, exit 0 even with --fail-on-regression", async () => {
    const r = await cli("diff", "a.onnx", "a-copy.onnx", "--fail-on-regression");
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("identical bytes");
    expect(r.stdout).toContain("No regressions.");
  });

  it("bigger model: structural + latency changes; exit 1 with --fail-on-regression", async () => {
    const plain = await cli("diff", "a.onnx", "b.onnx");
    expect(plain.code).toBe(0);
    const r = await cli("diff", "a.onnx", "b.onnx", "--format", "json", "--fail-on-regression");
    expect(r.code).toBe(1);
    const d = JSON.parse(r.stdout);
    expect(d.layers.added.map((l: { name: string }) => l.name)).toEqual(["sig"]);
    expect(d.layers.changed.map((c: { name: string }) => c.name)).toContain("conv1");
    expect(d.stats.params.delta).toBeGreaterThan(0);
    expect(d.regressions.some((x: { kind: string }) => x.kind === "latency")).toBe(true);
    // A huge tolerance hides latency regressions.
    expect((await cli("diff", "a.onnx", "b.onnx", "--fail-on-regression", "--latency-tolerance", "1000000000")).code).toBe(0);
  });

  it("markdown diff", async () => {
    const r = await cli("diff", "a.onnx", "b.onnx", "--format", "markdown");
    expect(r.stdout).toContain("## ModelVisio diff: `a.onnx` → `b.onnx`");
    expect(r.stdout).toContain("| Params |");
  });
});

describe("registry", () => {
  it("add / list / export round-trip with stable output", async () => {
    const reg = "reg.json";
    expect((await cli("registry", "add", "a.onnx", "--registry", reg, "--name", "tiny", "--tags", "Test,cnn", "--notes", "first")).code).toBe(0);
    const added = await cli("registry", "add", "b.onnx", "--registry", reg, "--name", "tiny", "--label", "big");
    expect(added.stderr).toContain("registered tiny big");
    const dup = await cli("registry", "add", "a-copy.onnx", "--registry", reg, "--name", "tiny");
    expect(dup.code).toBe(0);
    expect(dup.stderr).toContain("already registered as tiny 1.0.0");

    const file = JSON.parse(readFileSync(join(dir, reg), "utf8"));
    expect(file.schema).toBe("modelvisio.registry/v1");
    expect(file.models).toHaveLength(1);
    expect(file.models[0]).toMatchObject({ name: "tiny", tags: ["cnn", "test"] });
    expect(file.models[0].versions.map((v: { version: string }) => v.version)).toEqual(["1.0.0", "big"]);
    expect(file.models[0].versions[0].sha256).toMatch(/^[0-9a-f]{64}$/);

    const list = await cli("registry", "list", "--registry", reg);
    expect(list.stdout).toMatch(/tiny .*2 version\(s\), latest big/);

    // Export is the same document the file holds, so re-writing is a no-op diff.
    const exp = await cli("registry", "export", "--registry", reg);
    expect(exp.stdout).toBe(readFileSync(join(dir, reg), "utf8"));
    const md = await cli("registry", "export", "--registry", reg, "--format", "markdown");
    expect(md.stdout).toContain("### tiny");
    expect(md.stdout).toContain("| big |");
  });

  it("rejects a corrupt registry file with exit 2", async () => {
    writeFileSync(join(dir, "bad.json"), "{ nope");
    const r = await cli("registry", "list", "--registry", "bad.json");
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("not valid JSON");
  });

  it("SOURCE_DATE_EPOCH pins timestamps", async () => {
    let out = "";
    const code = await run(["registry", "add", "a.onnx", "--registry", "epoch.json"], { stdout: () => {}, stderr: (s) => { out += s; }, cwd: dir, env: { SOURCE_DATE_EPOCH: "1700000000" } });
    expect(code).toBe(0);
    const f = JSON.parse(readFileSync(join(dir, "epoch.json"), "utf8"));
    expect(f.models[0].versions[0].createdAt).toBe("2023-11-14T22:13:20.000Z");
    expect(f.exportedAt).toBe("2023-11-14T22:13:20.000Z");
    expect(out).toContain("registered a 1.0.0");
  });
});

describe.skipIf(!existsSync(YOLO_PT))("real model (yolov8n.pt at repo root)", () => {
  it("inspect + check", async () => {
    const i = await cli("inspect", YOLO_PT, "--json");
    expect(i.code).toBe(0);
    const s = JSON.parse(i.stdout);
    expect(s.format).toBe("PyTorch");
    expect(s.params).toBeGreaterThan(3_000_000);
    const r = await cli("check", YOLO_PT, "--target", "coral,rk3588", "--max-latency-ms", "50", "--format", "json");
    const rep = JSON.parse(r.stdout);
    // The PyTorch parser has no FLOP counts, so the latency budget is honestly "unknown" (a warning).
    expect(rep.models[0].budgets.every((b: { status: string }) => b.status === "unknown")).toBe(true);
    expect(r.code).toBe(0);
  });
});

describe("bundled binary", () => {
  it("builds with esbuild and runs under plain node", () => {
    execFileSync(process.execPath, ["build.mjs"], { cwd: PKG, stdio: "pipe" });
    const bin = join(PKG, "dist", "modelvisio.cjs");
    expect(readFileSync(bin, "utf8").startsWith("#!/usr/bin/env node")).toBe(true);
    const v = spawnSync(process.execPath, [bin, "--version"], { encoding: "utf8" });
    expect(v.status).toBe(0);
    expect(v.stdout.trim()).toMatch(/^\d+\.\d+\.\d+$/);
    const c = spawnSync(process.execPath, [bin, "check", join(dir, "b.onnx"), "--max-size-mb", "0.001"], { encoding: "utf8" });
    expect(c.status).toBe(1);
    expect(c.stdout).toContain("FAILED");
  });
});
