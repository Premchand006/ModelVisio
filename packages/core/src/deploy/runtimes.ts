// Deployment runtimes and which devices/precisions they apply to. Pure data +
// lookups, shared by the deployment agent, the recipe generator and the fleet
// simulator. Derived from each device's toolchain in data/hardware.ts, so a new
// device picks up runtimes automatically.

import type { DeviceSpec } from "../data/hardware";

export type RuntimeId =
  | "tensorrt" | "openvino" | "rknn" | "hailort" | "tflite-edgetpu"
  | "qnn" | "tflite" | "onnxruntime" | "vendor";

/** Precisions the deployment tools reason about (INT4 is out of scope here). */
export type DeployPrecision = "int8" | "fp16" | "fp32";
export const DEPLOY_PRECISIONS: DeployPrecision[] = ["int8", "fp16", "fp32"];

export type RuntimeInfo = {
  id: RuntimeId;
  label: string;
  /** Throughput factor when this runtime is NOT the device's native one
   *  (e.g. TFLite through a delegate, ONNX Runtime with a generic EP). The
   *  native/primary runtime of a device always gets 1.0. Heuristic. */
  eff: number;
  /** Substrings of ModelLayer.compIssues[].target this runtime is affected by. */
  compatTargets: string[];
};

export const RUNTIMES: Record<RuntimeId, RuntimeInfo> = {
  tensorrt: { id: "tensorrt", label: "TensorRT", eff: 1, compatTargets: ["tensorrt"] },
  openvino: { id: "openvino", label: "OpenVINO", eff: 1, compatTargets: ["openvino"] },
  rknn: { id: "rknn", label: "RKNN", eff: 1, compatTargets: ["rk3588", "rknn"] },
  hailort: { id: "hailort", label: "HailoRT", eff: 1, compatTargets: ["hailo"] },
  "tflite-edgetpu": { id: "tflite-edgetpu", label: "TFLite + Edge TPU", eff: 1, compatTargets: ["coral", "edgetpu"] },
  qnn: { id: "qnn", label: "Qualcomm QNN / SNPE", eff: 1, compatTargets: ["qnn", "snpe", "qualcomm"] },
  tflite: { id: "tflite", label: "TFLite / LiteRT (delegate)", eff: 0.8, compatTargets: ["tflite"] },
  onnxruntime: { id: "onnxruntime", label: "ONNX Runtime", eff: 0.6, compatTargets: ["onnxruntime"] },
  vendor: { id: "vendor", label: "Vendor SDK", eff: 1, compatTargets: [] },
};

const has = (dev: DeviceSpec, re: RegExp) => dev.toolchain.some((t) => re.test(t));

/** Runtimes a device can run, native/primary first. */
export function runtimesFor(dev: DeviceSpec): RuntimeId[] {
  const tflite = dev.fmts.includes("TFLite");
  if (has(dev, /tensorrt/i)) return ["tensorrt", "onnxruntime"];
  if (has(dev, /openvino/i)) return ["openvino", "onnxruntime"];
  if (has(dev, /rknn/i)) return ["rknn", "onnxruntime"];
  if (has(dev, /hailo/i)) return ["hailort"];
  if (has(dev, /edgetpu/i)) return ["tflite-edgetpu"];
  if (has(dev, /snpe|qnn/i)) return ["qnn", "tflite"];
  if (has(dev, /neuropilot|eiq/i)) return ["tflite", "vendor"];
  return tflite ? ["vendor", "tflite"] : ["vendor"];
}

/** Precisions a device can run (INT8-only parts → INT8). */
export function precisionsFor(dev: DeviceSpec): DeployPrecision[] {
  return dev.quant.int8Only ? ["int8"] : ["int8", "fp16", "fp32"];
}

/** Throughput factor of `rt` on `dev` (1 for the native runtime). */
export function runtimeEff(dev: DeviceSpec, rt: RuntimeId): number {
  return runtimesFor(dev)[0] === rt ? 1 : RUNTIMES[rt]?.eff ?? 0.6;
}

/** Normalize free text ("TensorRT", "tflite", "Edge TPU", "ORT") to a runtime id. */
export function parseRuntime(s: unknown): RuntimeId | null {
  if (typeof s !== "string") return null;
  const v = s.toLowerCase();
  if (/edge ?tpu|coral/.test(v)) return "tflite-edgetpu";
  if (/tensor ?rt|trt/.test(v)) return "tensorrt";
  if (/openvino/.test(v)) return "openvino";
  if (/rknn/.test(v)) return "rknn";
  if (/hailo/.test(v)) return "hailort";
  if (/qnn|snpe|qualcomm/.test(v)) return "qnn";
  if (/tflite|lite ?rt|tensorflow lite/.test(v)) return "tflite";
  if (/onnx ?runtime|\bort\b/.test(v)) return "onnxruntime";
  if (/vendor|sdk/.test(v)) return "vendor";
  return null;
}

/** Normalize "INT8"/"fp16"/"half" to a deploy precision. */
export function parsePrecision(s: unknown): DeployPrecision | null {
  if (typeof s !== "string") return null;
  const v = s.toLowerCase();
  if (/int8|i8|8-?bit|quant/.test(v)) return "int8";
  if (/fp16|f16|half|float16/.test(v)) return "fp16";
  if (/fp32|f32|float32|full/.test(v)) return "fp32";
  return null;
}
