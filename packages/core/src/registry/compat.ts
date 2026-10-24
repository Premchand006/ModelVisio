// Compiler-compat matrix for the registry and the CLI. Pure, no React.
//
// Two sources of findings per target:
//   • op-level: the `compIssues` parsers/transforms attach to layers, matched the
//     same way the Compiler tab filters them (`issue.target` contains the id);
//   • device-level: hard deployment blockers the roofline scorer computes for the
//     target's reference device (memory does not fit, no-fallback compile failure,
//     on-chip spill, unsupported source format).

import type { Model } from "@modelvisio/parsers";
import { HW, type DeviceSpec } from "../data/hardware";
import { scoreDevice } from "../scoring/scoreDevice";
import type { CompatIssueRecord, CompatResult } from "./types";

export type CompatTarget = {
  id: string;
  label: string;
  /** Substring matched against ModelLayer.compIssues[].target. */
  match?: string;
  /** Reference device in HW used for device-level checks and latency. */
  deviceId?: string;
};

/** The runtime targets the Compiler tab offers, plus their reference device. */
export const COMPAT_TARGETS: CompatTarget[] = [
  { id: "coral", label: "Coral TPU", match: "coral", deviceId: "coral" },
  { id: "rk3588", label: "RK3588", match: "rk3588", deviceId: "rk3588" },
  { id: "kneron", label: "Kneron", match: "kneron", deviceId: "kneron" },
  { id: "hailo", label: "Hailo", match: "hailo", deviceId: "hailo8" },
];

/** Reference devices whose estimates are stored with every registry version. */
export const REFERENCE_DEVICES = ["orin_nano", "hailo8", "rk3588", "coral", "intel_npu"];

export function findDevice(id: string): DeviceSpec | undefined {
  const k = id.toLowerCase();
  return HW.find((d) => d.id === k);
}

/**
 * Resolve a user-supplied target id: a COMPAT_TARGETS id ("coral"), or any
 * hardware device id from the catalog ("agx_orin", "hailo8l"). Returns
 * undefined for unknown ids.
 */
export function resolveTarget(id: string): CompatTarget | undefined {
  const k = id.trim().toLowerCase();
  const known = COMPAT_TARGETS.find((t) => t.id === k);
  if (known) return known;
  const dev = findDevice(k);
  if (dev) return { id: dev.id, label: dev.n, match: dev.compatId, deviceId: dev.id };
  return undefined;
}

/** All ids accepted by resolveTarget (for help text / error messages). */
export function knownTargetIds(): string[] {
  return [...COMPAT_TARGETS.map((t) => t.id), ...HW.map((d) => d.id).filter((id) => !COMPAT_TARGETS.some((t) => t.id === id))];
}

/** Compat findings for one target. */
export function checkCompat(model: Model, target: CompatTarget): CompatResult {
  const issues: CompatIssueRecord[] = [];
  if (target.match) {
    const m = target.match;
    for (const l of model.layers) {
      for (const ci of l.compIssues ?? []) {
        if (ci.target.includes(m)) issues.push({ severity: ci.severity, msg: ci.msg, layer: l.name, source: "op" });
      }
    }
  }
  const dev = target.deviceId ? findDevice(target.deviceId) : undefined;
  if (dev && model.layers.length > 0) {
    const s = scoreDevice(model, dev);
    for (const b of s.banners) {
      if (b.level === "info") continue;
      issues.push({ severity: b.level === "error" ? "error" : "warn", msg: b.msg, source: "device" });
    }
  }
  return {
    target: target.id,
    label: target.label,
    deviceId: dev?.id,
    errors: issues.filter((i) => i.severity === "error").length,
    warnings: issues.filter((i) => i.severity === "warn").length,
    issues,
  };
}

/** Compat matrix across targets (defaults to the Compiler tab's targets). */
export function compatMatrix(model: Model, targets: CompatTarget[] = COMPAT_TARGETS): CompatResult[] {
  return targets.map((t) => checkCompat(model, t));
}
