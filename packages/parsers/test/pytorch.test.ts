import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parsePytorch } from "../src/index";

// Integration test against the real yolov8n.pt at the repo root (if present).
const PT = fileURLToPath(new URL("../../../yolov8n.pt", import.meta.url));

describe("pytorch (real yolov8n.pt)", () => {
  const has = existsSync(PT);
  (has ? it : it.skip)("recovers the module hierarchy and weight shapes", () => {
    const buf = readFileSync(PT);
    const model = parsePytorch(buf, "yolov8n.pt");
    expect(model.format).toBe("PyTorch");

    const params = model.layers.reduce((s, l) => s + l.params, 0);
    expect(params).toBeGreaterThan(3_000_000); // yolov8n ≈ 3.15M

    const ops = new Set(model.layers.map((l) => l.op));
    expect(ops.has("Conv2d")).toBe(true);
    expect(ops.has("BatchNorm2d")).toBe(true);

    // The stem conv weight is 16×3×3×3.
    const stem = model.layers.find((l) => l.op === "Conv2d" && l.shape === "16×3×3×3");
    expect(stem).toBeTruthy();

    // Hierarchy edges (parent → child module tree).
    expect(model.edges.length).toBeGreaterThan(50);
  });
});
