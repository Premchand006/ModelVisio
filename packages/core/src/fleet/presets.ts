// Fleet presets, built ONLY from devices in data/hardware.ts (a test enforces
// it). Phone SoCs aren't in the DB, so the Android preset uses the Qualcomm
// and MediaTek dev-kit SoCs as proxies for flagship / mid-range phones.

import type { FleetEntry } from "./simulate";

export type FleetPreset = { id: string; label: string; note: string; entries: Omit<FleetEntry, "id">[] };

export const FLEET_PRESETS: FleetPreset[] = [
  {
    id: "android",
    label: "Mixed Android phones",
    note: "Qualcomm RB5 (Snapdragon 865-class) and RB6 (QCS6490) plus MediaTek Genio 1200 stand in for flagship, mid-range and MediaTek phones. TFLite delegates, INT8.",
    entries: [
      { deviceId: "rb5", count: 400, runtime: "tflite", precision: "int8", thermal: "passive", targetFps: 15, label: "Flagship (Snapdragon 865-class)" },
      { deviceId: "rb6", count: 600, runtime: "tflite", precision: "int8", thermal: "hot", targetFps: 15, label: "Mid-range (QCS6490-class)" },
      { deviceId: "mtk_genio", count: 500, runtime: "tflite", precision: "int8", thermal: "hot", targetFps: 15, label: "MediaTek (Genio 1200-class)" },
    ],
  },
  {
    id: "retail",
    label: "Retail edge boxes (Jetson + NUC)",
    note: "In-store camera analytics: Jetson Orin Nano / NX boxes on TensorRT and Core Ultra NUCs on OpenVINO, 30 FPS per camera stream.",
    entries: [
      { deviceId: "orin_nano", count: 120, runtime: "tensorrt", precision: "int8", thermal: "passive", targetFps: 30 },
      { deviceId: "orin_nx", count: 30, runtime: "tensorrt", precision: "fp16", thermal: "cooled", targetFps: 30 },
      { deviceId: "intel_npu", count: 80, runtime: "openvino", precision: "int8", thermal: "cooled", targetFps: 30, label: "Intel NUC (Core Ultra NPU)" },
    ],
  },
  {
    id: "rpi",
    label: "Raspberry Pi fleet",
    note: "Raspberry Pi 5 nodes with the Hailo-8L AI Kit, plus older Pis with a Coral USB Edge TPU. INT8 only.",
    entries: [
      { deviceId: "hailo8l", count: 200, runtime: "hailort", precision: "int8", thermal: "passive", targetFps: 15, label: "Pi 5 + Hailo-8L AI Kit" },
      { deviceId: "coral", count: 150, runtime: "tflite-edgetpu", precision: "int8", thermal: "passive", targetFps: 15, label: "Pi + Coral USB" },
    ],
  },
  {
    id: "industrial",
    label: "Industrial SBCs (Rockchip + TI)",
    note: "Factory-floor inspection on RK3588 boards (RKNN) and TI TDA4VM (TIDL), fanless enclosures.",
    entries: [
      { deviceId: "rk3588", count: 250, runtime: "rknn", precision: "int8", thermal: "hot", targetFps: 10 },
      { deviceId: "ti_tda4", count: 60, runtime: "vendor", precision: "int8", thermal: "passive", targetFps: 10 },
    ],
  },
];

let seq = 0;
/** Unique-ish row id for a new fleet entry. */
export function newEntryId(): string {
  seq += 1;
  return `c${Date.now().toString(36)}${seq}`;
}

/** Materialize a preset's entries with fresh ids. */
export function presetEntries(p: FleetPreset): FleetEntry[] {
  return p.entries.map((e, i) => ({ ...e, id: `${p.id}-${i}` }));
}
