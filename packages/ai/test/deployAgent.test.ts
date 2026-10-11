import { describe, it, expect, vi, afterEach } from "vitest";
import {
  parseAgentReply, runDeployAgent, buildAgentSystemPrompt, proxyLlm,
  type AgentTool, type LlmFn, type PlannerFn,
} from "../src/deployAgent";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; vi.restoreAllMocks(); });

describe("parseAgentReply (action protocol)", () => {
  it("parses a bare tool action", () => {
    const a = parseAgentReply('{"thought":"look","action":"list_devices","args":{"vendor":"NVIDIA"}}');
    expect(a).toEqual({ kind: "tool", tool: "list_devices", args: { vendor: "NVIDIA" }, thought: "look" });
  });

  it("parses an action inside a ```json fence surrounded by prose", () => {
    const a = parseAgentReply('Sure! Here is my next step:\n```json\n{"action": "estimate_latency", "args": {"device": "orin_nano", "precision": "int8"}}\n```\nThanks');
    expect(a.kind).toBe("tool");
    if (a.kind === "tool") expect(a.args).toEqual({ device: "orin_nano", precision: "int8" });
  });

  it("parses a final decision", () => {
    const a = parseAgentReply('{"thought":"done","final":{"target":"orin_nano","precision":"int8","risks":["x"]}}');
    expect(a.kind).toBe("final");
    if (a.kind === "final") expect(a.final.target).toBe("orin_nano");
  });

  it("accepts the {action: final, args} variant and tool/arguments aliases", () => {
    const f = parseAgentReply('{"action":"final","args":{"target":"hailo8"}}');
    expect(f.kind === "final" && f.final.target).toBe("hailo8");
    const t = parseAgentReply('{"tool":"check_compat","arguments":"{\\"target\\":\\"coral\\"}"}');
    expect(t.kind === "tool" && t.args).toEqual({ target: "coral" });
  });

  it("repairs trailing commas and ignores braces inside strings", () => {
    const a = parseAgentReply('{"thought":"use {braces} here","action":"list_fixes","args":{},}');
    expect(a.kind).toBe("tool");
    if (a.kind === "tool") expect(a.thought).toBe("use {braces} here");
  });

  it("defaults missing args to {}", () => {
    const a = parseAgentReply('{"action":"get_model_summary"}');
    expect(a.kind === "tool" && a.args).toEqual({});
  });

  it("reports invalid replies", () => {
    expect(parseAgentReply("I think we should use TensorRT.").kind).toBe("invalid");
    expect(parseAgentReply('{"foo": 1}').kind).toBe("invalid");
    expect(parseAgentReply("").kind).toBe("invalid");
  });
});

function fakeTools(log: string[] = []): AgentTool[] {
  return [
    { name: "get_model_summary", description: "summary", run: () => { log.push("summary"); return { params: 1 }; } },
    { name: "list_devices", description: "devices", run: () => { log.push("devices"); return [{ id: "orin_nano" }]; }, digest: () => "1 device" },
    { name: "boom", description: "throws", run: () => { throw new Error("kaput"); } },
  ];
}

const planner: PlannerFn = async ({ call }) => {
  await call("get_model_summary", {}, "planner summary");
  await call("list_devices", {}, "planner devices");
  return { target: "orin_nano", precision: "int8", summary: "planned" };
};

/** Scripted LLM that returns the given replies in order. */
function scripted(replies: string[]): LlmFn & { calls: number; lastMessages: () => unknown[] } {
  let i = 0;
  let last: unknown[] = [];
  const fn = (async ({ messages }) => {
    last = [...messages];
    fn.calls++;
    return replies[Math.min(i++, replies.length - 1)];
  }) as LlmFn & { calls: number; lastMessages: () => unknown[] };
  fn.calls = 0;
  fn.lastMessages = () => last;
  return fn;
}

describe("runDeployAgent", () => {
  it("runs injected tools, records the transcript, and stops at final", async () => {
    const log: string[] = [];
    const llm = scripted([
      '{"thought":"need model","action":"get_model_summary","args":{}}',
      '{"thought":"devices","action":"list_devices","args":{}}',
      '{"thought":"ok","final":{"target":"orin_nano","precision":"int8"}}',
    ]);
    const seen: number[] = [];
    const run = await runDeployAgent({ goal: "Jetson", tools: fakeTools(log), llm, planner, onStep: (s) => seen.push(s.index) });
    expect(run.mode).toBe("ai");
    expect(run.stoppedBy).toBe("final");
    expect(run.final.target).toBe("orin_nano");
    expect(log).toEqual(["summary", "devices"]);
    expect(run.steps.map((s) => s.tool)).toEqual(["get_model_summary", "list_devices"]);
    expect(run.steps[1].digest).toBe("1 device");
    expect(run.steps.every((s) => s.source === "ai")).toBe(true);
    expect(seen).toEqual([1, 2]);
    // Observations are fed back as user turns, alternating roles.
    const msgs = llm.lastMessages() as { role: string; content: string }[];
    expect(msgs.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant", "user"]);
    expect(msgs[2].content).toMatch(/^OBSERVATION \(get_model_summary\)/);
  });

  it("records tool errors and unknown tools without crashing, and feeds them back", async () => {
    const llm = scripted([
      '{"action":"boom"}',
      '{"action":"nope"}',
      '{"final":{"target":"x"}}',
    ]);
    const run = await runDeployAgent({ goal: "g", tools: fakeTools(), llm, planner });
    expect(run.steps[0]).toMatchObject({ tool: "boom", ok: false, error: "kaput" });
    expect(run.steps[1].ok).toBe(false);
    expect(run.steps[1].error).toMatch(/Unknown tool/);
    expect(run.mode).toBe("ai");
  });

  it("caps the number of steps, then falls back to the planner", async () => {
    const llm = scripted(['{"action":"get_model_summary"}']); // never finalizes
    const run = await runDeployAgent({ goal: "g", tools: fakeTools(), llm, planner, maxSteps: 3 });
    const aiSteps = run.steps.filter((s) => s.source === "ai");
    expect(aiSteps).toHaveLength(3);
    expect(llm.calls).toBe(4); // 3 capped turns + 1 finalize attempt
    expect(run.mode).toBe("fallback");
    expect(run.fallbackReason).toMatch(/3-step cap/);
    expect(run.final.summary).toBe("planned");
    expect(run.steps.filter((s) => s.source === "planner").map((s) => s.tool)).toEqual(["get_model_summary", "list_devices"]);
  });

  it("accepts a final emitted on the finalize turn after the cap", async () => {
    const llm = scripted(['{"action":"get_model_summary"}', '{"action":"get_model_summary"}', '{"final":{"target":"hailo8"}}']);
    const run = await runDeployAgent({ goal: "g", tools: fakeTools(), llm, planner, maxSteps: 2 });
    expect(run.mode).toBe("ai");
    expect(run.stoppedBy).toBe("step-cap");
    expect(run.final.target).toBe("hailo8");
  });

  it("uses the deterministic planner when no LLM is given", async () => {
    const run = await runDeployAgent({ goal: "g", tools: fakeTools(), llm: null, planner });
    expect(run.mode).toBe("fallback");
    expect(run.steps.every((s) => s.source === "planner")).toBe(true);
    expect(run.final.target).toBe("orin_nano");
  });

  it("falls back when the proxy errors, keeping the reason", async () => {
    const llm: LlmFn = async () => { throw new Error("Gemini rate limit (429)"); };
    const run = await runDeployAgent({ goal: "g", tools: fakeTools(), llm, planner });
    expect(run.mode).toBe("fallback");
    expect(run.fallbackReason).toMatch(/429/);
    expect(run.final.summary).toBe("planned");
  });

  it("falls back after two consecutive protocol violations", async () => {
    const llm = scripted(["I recommend TensorRT.", "Still prose."]);
    const run = await runDeployAgent({ goal: "g", tools: fakeTools(), llm, planner });
    expect(run.mode).toBe("fallback");
    expect(run.steps.filter((s) => s.tool === "(invalid reply)")).toHaveLength(2);
  });

  it("rethrows an abort instead of falling back", async () => {
    const ctrl = new AbortController();
    const llm: LlmFn = async () => { ctrl.abort(); const e = new Error("aborted"); e.name = "AbortError"; throw e; };
    await expect(runDeployAgent({ goal: "g", tools: fakeTools(), llm, planner, signal: ctrl.signal })).rejects.toThrow(/Abort/);
  });
});

describe("buildAgentSystemPrompt / proxyLlm", () => {
  it("lists the injected tools and the step cap", () => {
    const p = buildAgentSystemPrompt(fakeTools(), 5, "Model: yolo");
    expect(p).toContain("- list_devices {}: devices");
    expect(p).toContain("at most 5 tool calls");
    expect(p).toContain("MODEL CONTEXT:\nModel: yolo");
  });

  it("posts the unchanged {system, messages} contract to the proxy", async () => {
    const fn = vi.fn(async () => new Response(JSON.stringify({ text: '{"final":{}}', sources: [] }), { status: 200 }));
    globalThis.fetch = fn as unknown as typeof fetch;
    const text = await proxyLlm()({ system: "S", messages: [{ role: "user", content: "hi" }] });
    expect(text).toBe('{"final":{}}');
    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/chat");
    expect(JSON.parse(String(init.body))).toEqual({ system: "S", messages: [{ role: "user", content: "hi" }] });
  });

  it("surfaces proxy errors", async () => {
    globalThis.fetch = (async () => new Response(JSON.stringify({ error: "no key" }), { status: 500 })) as unknown as typeof fetch;
    await expect(proxyLlm()({ system: "", messages: [] })).rejects.toThrow("no key");
  });
});
