// Parse a free-text deployment goal ("Jetson Orin Nano at >=30 FPS, INT8 OK",
// "ship to Android phones via TFLite") into structured constraints. Used by the
// deterministic planner and to validate/complete the AI agent's decision.

import { HW, type DeviceSpec } from "../data/hardware";
import { parseRuntime, type DeployPrecision, type RuntimeId } from "./runtimes";

export type DeployGoal = {
  text: string;
  /** Devices named explicitly in the goal (best match first). */
  deviceIds: string[];
  /** Devices implied by a platform/vendor mention (Android, Raspberry Pi, Jetson…). */
  platformDeviceIds: string[];
  platform?: string;
  runtime?: RuntimeId;
  minFps?: number;
  maxLatencyMs?: number;
  maxPowerW?: number;
  maxPriceUSD?: number;
  /** INT8 explicitly allowed (true), forbidden (false) or unstated (undefined). */
  int8Ok?: boolean;
  precision?: DeployPrecision;
  /** "cheapest", "lowest power", "fastest". */
  optimize?: "cost" | "power" | "speed";
};

const ALIASES: [RegExp, string][] = [
  [/agx\s*orin/i, "agx_orin"],
  [/orin\s*nx/i, "orin_nx"],
  [/orin\s*nano|jetson\s*nano/i, "orin_nano"],
  [/coral|edge\s*tpu/i, "coral"],
  [/\brb5\b|qrb5165/i, "rb5"],
  [/\brb6\b|qcs6490/i, "rb6"],
  [/core\s*ultra|intel\s*npu|meteor\s*lake|\bnuc\b/i, "intel_npu"],
  [/myriad|movidius|ncs2/i, "movidius"],
  [/kv260|kria/i, "kv260"],
  [/hailo-?8l|ai\s*kit/i, "hailo8l"],
  [/hailo-?8(?!l)/i, "hailo8"],
  [/rk3588|rockchip|orange\s*pi\s*5|rock\s*5/i, "rk3588"],
  [/sima|mlsoc/i, "sima"],
  [/deepx|dx-?m1/i, "deepx"],
  [/mobilint|ma35/i, "mobilint"],
  [/kneron|kl720/i, "kneron"],
  [/blaize/i, "blaize"],
  [/i\.?mx\s*93|nxp/i, "nxp"],
  [/amlogic|a311d/i, "amlogic"],
  [/tda4|jacinto/i, "ti_tda4"],
  [/genio|mt8395/i, "mtk_genio"],
];

const PLATFORMS: { re: RegExp; name: string; ids: string[]; runtime?: RuntimeId }[] = [
  { re: /android|phone|mobile|smartphone/i, name: "Android phones", ids: ["rb5", "rb6", "mtk_genio"], runtime: "tflite" },
  { re: /raspberry|\bpi\s*5?\b|\brpi\b/i, name: "Raspberry Pi", ids: ["hailo8l", "coral"] },
  { re: /jetson|nvidia/i, name: "NVIDIA Jetson", ids: ["orin_nano", "orin_nx", "agx_orin"], runtime: "tensorrt" },
  { re: /intel|laptop|\bpc\b|x86/i, name: "Intel PC", ids: ["intel_npu"], runtime: "openvino" },
  { re: /qualcomm|snapdragon/i, name: "Qualcomm", ids: ["rb5", "rb6"], runtime: "qnn" },
  { re: /automotive|\bcar\b|adas/i, name: "Automotive", ids: ["ti_tda4"] },
];

const num = (s: string | undefined) => (s != null ? parseFloat(s) : undefined);

export function parseGoal(text: string, devices: DeviceSpec[] = HW): DeployGoal {
  const t = text || "";
  const known = new Set(devices.map((d) => d.id));
  const deviceIds: string[] = [];
  const add = (id: string) => { if (known.has(id) && !deviceIds.includes(id)) deviceIds.push(id); };
  for (const [re, id] of ALIASES) if (re.test(t)) add(id);
  for (const d of devices) if (new RegExp(`\\b${d.id}\\b`, "i").test(t)) add(d.id);

  const platformDeviceIds: string[] = [];
  let platform: string | undefined;
  let platformRuntime: RuntimeId | undefined;
  for (const p of PLATFORMS) {
    if (p.re.test(t)) {
      platform ??= p.name;
      platformRuntime ??= p.runtime;
      for (const id of p.ids) if (known.has(id) && !platformDeviceIds.includes(id)) platformDeviceIds.push(id);
    }
  }

  // "via TFLite", "using TensorRT", "with OpenVINO", or just the name.
  const rtMatch = t.match(/(?:via|using|with|on|in)\s+(tensor\s?rt|trt|openvino|rknn|hailo\s?rt|tflite|lite\s?rt|tensorflow lite|onnx\s?runtime|qnn|snpe|edge\s?tpu)/i)
    ?? t.match(/(tensor\s?rt|openvino|rknn|hailort|tflite|lite\s?rt|onnx\s?runtime|qnn|snpe)/i);
  const runtime = (rtMatch ? parseRuntime(rtMatch[1]) : null) ?? (deviceIds.length === 0 ? platformRuntime : undefined);

  const fps = t.match(/(?:≥|>=|>|at least|min(?:imum)?|@|at)?\s*(\d+(?:\.\d+)?)\s*(?:fps|frames?\s*(?:per|\/)\s*s)/i);
  const lat = t.match(/(?:≤|<=|<|under|below|within|max(?:imum)?)?\s*(\d+(?:\.\d+)?)\s*ms\b/i);
  const pw = t.match(/(?:≤|<=|<|under|below|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*w(?:att)?s?\b/i);
  const price = t.match(/(?:≤|<=|<|under|below|budget(?: of)?)\s*\$\s*(\d+(?:\.\d+)?)/i);

  let int8Ok: boolean | undefined;
  if (/no\s*int8|int8\s*(?:not ok|forbidden|not allowed)|avoid\s*(?:int8|quantiz)|no\s*quantiz|fp32 only/i.test(t)) int8Ok = false;
  else if (/int8|quantiz|8-?bit/i.test(t)) int8Ok = true;

  let precision: DeployPrecision | undefined;
  if (/fp32\s*only|full precision/i.test(t)) precision = "fp32";
  else if (/\bfp16\b|half precision/i.test(t) && int8Ok !== true) precision = "fp16";
  else if (int8Ok === true && /int8\s*only|must\s*(?:be\s*)?int8/i.test(t)) precision = "int8";

  let optimize: DeployGoal["optimize"];
  if (/cheap|lowest cost|budget|cost/i.test(t)) optimize = "cost";
  else if (/low(?:est)?\s*power|battery|energy|efficien|watt/i.test(t)) optimize = "power";
  else if (/fastest|max(?:imum)?\s*(?:fps|throughput|speed)|lowest latency/i.test(t)) optimize = "speed";

  return {
    text: t, deviceIds, platformDeviceIds, platform, runtime: runtime ?? undefined,
    minFps: num(fps?.[1]), maxLatencyMs: num(lat?.[1]), maxPowerW: num(pw?.[1]), maxPriceUSD: num(price?.[1]),
    int8Ok, precision, optimize,
  };
}

/** Resolve a device reference (id, exact or partial name, alias) to a device. */
export function resolveDevice(ref: unknown, devices: DeviceSpec[] = HW): DeviceSpec | null {
  if (typeof ref !== "string" || !ref.trim()) return null;
  const r = ref.trim().toLowerCase();
  const byId = devices.find((d) => d.id.toLowerCase() === r);
  if (byId) return byId;
  const byName = devices.find((d) => d.n.toLowerCase() === r) ?? devices.find((d) => d.n.toLowerCase().includes(r) || r.includes(d.n.toLowerCase()));
  if (byName) return byName;
  const alias = parseGoal(ref, devices).deviceIds[0];
  return alias ? devices.find((d) => d.id === alias) ?? null : null;
}
