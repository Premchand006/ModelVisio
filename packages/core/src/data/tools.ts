// The analysis sidebar's tools, grouped by workflow stage. The ids are the
// App's panel ids; ToolNav and the command palette both read this list.

export type ToolGroupId = "analyze" | "optimize" | "ship" | "copilot";
export type ToolDef = { id: string; label: string; desc: string };
export type ToolGroup = { id: ToolGroupId; label: string; tools: ToolDef[] };

export const TOOL_GROUPS: ToolGroup[] = [
  { id: "analyze", label: "Analyze", tools: [
    { id: "inspector", label: "Inspector", desc: "Shapes, weights, FLOPs/MACs and quantization sensitivity of the selected layer" },
    { id: "compiler", label: "Compiler", desc: "Op-support checks per target, with auto-fixes you can undo" },
    { id: "hardware", label: "Hardware", desc: "Roofline fit for 21 edge accelerators: FPS, FPS/W, memory fit" },
    { id: "quant", label: "Quant", desc: "Per-layer INT8/INT4 sensitivity and a mixed-precision plan" },
  ] },
  { id: "optimize", label: "Optimize", tools: [
    { id: "advisor", label: "Advisor", desc: "Ranked, device-specific recommendations with latency and size impact" },
    { id: "investigate", label: "Investigate", desc: "Bottleneck layers and patterns for a target device" },
    { id: "optimize", label: "Search", desc: "Pareto search over device, runtime, precision, fixes and input size" },
    { id: "bench", label: "Bench", desc: "Run the ONNX model on this device with onnxruntime-web" },
  ] },
  { id: "ship", label: "Ship", tools: [
    { id: "convert", label: "Convert", desc: "Graph JSON, Layers CSV, Safetensors and NumPy exports, conversion kits" },
    { id: "deploy", label: "Deploy", desc: "TensorRT, HailoRT and RKNN deployment scripts" },
    { id: "agent", label: "Agent", desc: "Plan a deployment from a goal and download the scripts" },
    { id: "fleet", label: "Fleet", desc: "Latency, SLO attainment and energy across a mixed fleet" },
    { id: "registry", label: "Registry", desc: "Local history of model versions with diffs" },
  ] },
  { id: "copilot", label: "Copilot", tools: [
    { id: "chat", label: "Chat", desc: "AI copilot that sees your model's computed analysis" },
  ] },
];

export const ALL_TOOLS: (ToolDef & { group: ToolGroup })[] =
  TOOL_GROUPS.flatMap((g) => g.tools.map((tl) => ({ ...tl, group: g })));

/** The group a tool id belongs to (the first group for an unknown id). */
export function groupOf(toolId: string): ToolGroup {
  return TOOL_GROUPS.find((g) => g.tools.some((tl) => tl.id === toolId)) ?? TOOL_GROUPS[0];
}
