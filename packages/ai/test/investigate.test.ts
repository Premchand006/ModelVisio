import { describe, it, expect, afterEach, vi } from "vitest";
import {
  buildInvestigationPrompt, extractJsonObject, validateInvestigation, parseInvestigation,
  ruleBasedReport, investigate, knownLayers,
  type EvidencePack, type SendChatArgs, type ChatResponse,
} from "../src/index";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; vi.restoreAllMocks(); });

/** Hand-built evidence pack fixture (shape produced by core's buildEvidence). */
function pack(over: Partial<EvidencePack["totals"]> = {}): EvidencePack {
  return {
    version: 1,
    model: { name: "net.onnx", format: "ONNX", framework: "PyTorch", params: 3e6, flops: 8e9, layers: 40, inputShape: [1, 3, 640, 640], workload: "cnnDet" },
    device: { id: "coral", name: "Google Coral Edge TPU", vendor: "Google", toolchain: ["EdgeTPU Compiler"], precision: "int8", int8Only: true, cpuFallback: "single-partition", peakTops: 4, bandwidthGBs: 64, ridgeAI: 62.5, memType: "onchip" },
    totals: { latencyMs: 40, fps: 25, computeMs: 39, dispatchMs: 1, util: 0.14, calibrated: false, bound: "memory", computeBoundShare: 0.3, memoryBoundShare: 0.68, footprintMB: 9, budgetMB: 6.9, memFits: true, opCoverage: 0.6, ...over },
    bottlenecks: [
      { id: "E2", layer: "stem.conv", op: "Conv", timeMs: 12, share: 0.3, bound: "memory", ai: 8 },
      { id: "E3", layer: "neck.sppf", op: "SPPF", timeMs: 6, share: 0.15, bound: "compute", ai: 80 },
    ],
    opBreakdown: [{ op: "Conv", count: 20, timeMs: 30, share: 0.75 }],
    evidence: [
      { id: "E1", kind: "summary", severity: "info", title: "Roofline summary", detail: "Est. 40 ms", layers: [] },
      { id: "E2", kind: "bottleneck", severity: "warn", tag: "memory", title: "stem.conv (Conv)", detail: "12 ms = 30%", layers: ["stem.conv"], share: 0.3, fixIds: [] },
      { id: "E3", kind: "bottleneck", severity: "info", tag: "compute", title: "neck.sppf (SPPF)", detail: "6 ms = 15%", layers: ["neck.sppf"], share: 0.15, fixIds: ["sppf_parallel"] },
      { id: "E4", kind: "compiler", severity: "error", title: "neck.sppf: unsupported", detail: "Cascaded MaxPool not supported", layers: ["neck.sppf"], share: 0.15, fixIds: ["sppf_parallel"] },
      { id: "E5", kind: "memory", severity: "error", title: "Memory fit", detail: "9 MB vs 6.9 MB", layers: [] },
      { id: "E6", kind: "pattern", tag: "layout", severity: "warn", title: "Transpose/Reshape churn", detail: "3 chains", layers: ["head.reshape"], share: 0.12 },
    ],
  };
}

const aiAnswer = (hyps: unknown[], summary = "SPPF falls back to CPU.") => JSON.stringify({ summary, hypotheses: hyps });

describe("buildInvestigationPrompt", () => {
  it("embeds the evidence pack, the user's question, and the JSON schema", () => {
    const { modelSummary, userMessage } = buildInvestigationPrompt(pack(), "why so slow?");
    expect(modelSummary).toContain("Google Coral Edge TPU");
    expect(modelSummary).toContain("\"E4\"");
    expect(modelSummary).toContain("Cascaded MaxPool");
    expect(userMessage).toContain("QUESTION: why so slow?");
    expect(userMessage).toMatch(/cite/i);
    expect(userMessage).toContain("\"hypotheses\"");
  });
  it("defaults the question to the device", () => {
    expect(buildInvestigationPrompt(pack()).userMessage).toContain("Why is this model slow on Google Coral Edge TPU");
  });
});

describe("extractJsonObject", () => {
  it("reads fenced JSON", () => {
    expect(extractJsonObject("Here:\n```json\n{\"a\":1}\n```\nthanks")).toEqual({ a: 1 });
  });
  it("reads JSON surrounded by prose, with braces inside strings", () => {
    expect(extractJsonObject("Sure! {\"s\":\"a } b {\",\"n\":{\"x\":2}} done")).toEqual({ s: "a } b {", n: { x: 2 } });
  });
  it("returns null when there is no JSON", () => {
    expect(extractJsonObject("no json here")).toBeNull();
    expect(extractJsonObject("{broken")).toBeNull();
  });
});

describe("validateInvestigation", () => {
  it("keeps cited hypotheses and strips hallucinated layers and ids", () => {
    const r = validateInvestigation(JSON.parse(aiAnswer([
      { title: "SPPF CPU fallback", confidence: "HIGH", evidence: ["E4", "e3", "E99"], layers: ["neck.sppf", "ghost.layer"], explanation: "x", experiments: [{ action: "Apply SPPF fix", expectedGain: "2x" }, { nope: 1 }] },
      { title: "Invented", evidence: ["E42"], layers: ["neck.sppf"] },
      { title: "", evidence: ["E1"] },
    ])), pack());
    expect(r).not.toBeNull();
    expect(r!.hypotheses).toHaveLength(1);
    const h = r!.hypotheses[0];
    expect(h.rank).toBe(1);
    expect(h.confidence).toBe("high");
    expect(h.evidence).toEqual(["E4", "E3"]);
    expect(h.layers).toEqual(["neck.sppf"]);
    expect(h.experiments).toEqual([{ action: "Apply SPPF fix", expectedGain: "2x" }]);
    expect(h.fixIds).toEqual(["sppf_parallel"]);
    expect(r!.rejected.layers).toEqual(["ghost.layer"]);
    expect(r!.rejected.evidence).toEqual(expect.arrayContaining(["E99", "E42"]));
    expect(r!.rejected.hypotheses).toBe(2);
  });
  it("returns null when nothing is grounded", () => {
    expect(validateInvestigation({ hypotheses: [{ title: "x", evidence: ["E77"] }] }, pack())).toBeNull();
    expect(validateInvestigation({ summary: "x" }, pack())).toBeNull();
    expect(validateInvestigation("text", pack())).toBeNull();
  });
  it("caps the number of hypotheses", () => {
    const many = Array.from({ length: 12 }, (_, i) => ({ title: `h${i}`, evidence: ["E1"] }));
    expect(parseInvestigation(aiAnswer(many), pack())!.hypotheses.length).toBe(6);
  });
  it("knownLayers covers bottlenecks and evidence layers only", () => {
    expect([...knownLayers(pack())].sort()).toEqual(["head.reshape", "neck.sppf", "stem.conv"]);
  });
});

describe("ruleBasedReport", () => {
  it("ranks blocking issues first and cites only existing evidence", () => {
    const p = pack();
    const r = ruleBasedReport(p);
    const ids = new Set(p.evidence.map((e) => e.id));
    const layers = knownLayers(p);
    expect(r.source).toBe("rules");
    expect(r.hypotheses[0].title).toMatch(/CPU fallback/);
    expect(r.hypotheses[0].fixIds).toContain("sppf_parallel");
    for (const h of r.hypotheses) {
      expect(h.evidence.length).toBeGreaterThan(0);
      for (const e of h.evidence) expect(ids.has(e)).toBe(true);
      for (const l of h.layers) expect(layers.has(l)).toBe(true);
    }
    expect(r.hypotheses.some((h) => /Bandwidth-bound/.test(h.title))).toBe(true);
    expect(r.hypotheses.some((h) => /Transpose\/Reshape/.test(h.title))).toBe(true);
    expect(r.hypotheses.map((h) => h.rank)).toEqual(r.hypotheses.map((_, i) => i + 1));
  });
  it("still produces a hypothesis when nothing is wrong", () => {
    const p = pack({ memoryBoundShare: 0.1, computeBoundShare: 0.9, bound: "compute" });
    p.evidence = p.evidence.filter((e) => e.kind === "summary");
    p.bottlenecks = [];
    const r = ruleBasedReport(p);
    expect(r.hypotheses).toHaveLength(1);
    expect(r.hypotheses[0].evidence).toEqual(["E1"]);
  });
});

describe("investigate", () => {
  const ok = (text: string) => vi.fn(async (_a: SendChatArgs): Promise<ChatResponse> => ({ text, sources: [{ title: "TRT docs", url: "https://docs.nvidia.com/x" }] }));

  it("returns the validated AI report with sources", async () => {
    const send = ok("```json\n" + aiAnswer([{ title: "SPPF fallback", confidence: "high", evidence: ["E4"], layers: ["neck.sppf"], experiments: [] }]) + "\n```");
    const r = await investigate({ evidence: pack(), question: "why?", send });
    expect(r.source).toBe("ai");
    expect(r.hypotheses[0].title).toBe("SPPF fallback");
    expect(r.sources).toHaveLength(1);
    const args = send.mock.calls[0][0];
    expect(args.messages[0].content).toContain("QUESTION: why?");
    expect(args.modelSummary).toContain("EVIDENCE PACK");
  });
  it("falls back to rules when the proxy fails", async () => {
    const send = vi.fn(async () => { throw new Error("Chat proxy returned 500"); });
    const r = await investigate({ evidence: pack(), send });
    expect(r.source).toBe("rules");
    expect(r.note).toMatch(/Chat proxy returned 500/);
    expect(r.hypotheses.length).toBeGreaterThan(0);
  });
  it("falls back to rules when the answer is ungrounded or not JSON", async () => {
    expect((await investigate({ evidence: pack(), send: ok("I think it's slow.") })).source).toBe("rules");
    const r = await investigate({ evidence: pack(), send: ok(aiAnswer([{ title: "x", evidence: ["E404"] }])) });
    expect(r.source).toBe("rules");
    expect(r.note).toMatch(/could not be validated/);
  });
  it("skips the AI entirely when disabled (desktop without a key)", async () => {
    const send = ok("{}");
    const r = await investigate({ evidence: pack(), aiEnabled: false, send });
    expect(send).not.toHaveBeenCalled();
    expect(r.source).toBe("rules");
  });
  it("goes through the /api/chat proxy contract by default", async () => {
    const fn = vi.fn(async (_i: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify({ text: aiAnswer([{ title: "Mem", evidence: ["E5"] }]), sources: [] }), { status: 200, headers: { "content-type": "application/json" } }));
    globalThis.fetch = fn as unknown as typeof fetch;
    const r = await investigate({ evidence: pack() });
    expect(r.source).toBe("ai");
    expect(String(fn.mock.calls[0][0])).toBe("/api/chat");
    const body = JSON.parse(String(fn.mock.calls[0][1]?.body)) as { system: string; messages: { role: string; content: string }[] };
    expect(body.system).toContain("EVIDENCE PACK");
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0].role).toBe("user");
  });
});
