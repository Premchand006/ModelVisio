import type { Model, ModelLayer } from "../types";

// Darknet .cfg is a human-readable INI-like config (YOLOv3/v4 etc.). We parse it
// into a real graph: per-section layers, sequential + route/shortcut edges, and
// best-effort shape/param inference from the [net] input through strides.

type Block = { type: string; props: Record<string, string> };

function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let cur: Block | null = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.replace(/[#;].*$/, "").trim();
    if (!line) continue;
    const m = line.match(/^\[(.+)\]$/);
    if (m) {
      cur = { type: m[1].trim(), props: {} };
      blocks.push(cur);
    } else if (cur) {
      const eq = line.indexOf("=");
      if (eq > 0) cur.props[line.slice(0, eq).trim()] = line.slice(eq + 1).trim();
    }
  }
  return blocks;
}

const num = (p: Record<string, string>, k: string, d: number) =>
  p[k] != null && !isNaN(Number(p[k])) ? Number(p[k]) : d;

function groupFor(type: string): ModelLayer["group"] {
  if (type === "net" || type === "network") return "input";
  if (type === "yolo" || type === "region" || type === "detection" || type === "softmax") return "output";
  if (type === "route" || type === "upsample" || type === "reorg") return "neck";
  if (type === "connected") return "head";
  return "backbone";
}

export function parseDarknet(text: string, name = "model.cfg"): Model {
  const blocks = parseBlocks(text);
  if (!blocks.length) throw new Error("Empty or invalid Darknet .cfg.");
  const net = blocks[0];
  let w = num(net.props, "width", 0);
  let h = num(net.props, "height", 0);
  let c = num(net.props, "channels", 3);

  const layers: ModelLayer[] = [];
  const layerIds: number[] = []; // darknet layer index → ModelLayer id
  const edges: [number, number][] = [];
  const dims: { c: number; w: number; h: number }[] = []; // per darknet layer

  // Input node.
  layers.push({
    id: 0, name: "input", type: "Input", op: "Placeholder",
    shape: `1×${c}×${h}×${w}`, dt: "float32", params: 0, flops: 0, macs: 0,
    mem: c * w * h * 4, group: "input", ins: [], outs: [{ n: "input", s: [1, c, h, w] }],
    attr: { width: w, height: h, channels: c }, w: null,
    math: `Input 1×${c}×${h}×${w}`, insight: "Darknet network input.", qSens: 0, compIssues: [],
  });

  const body = blocks.slice(1);
  let nextId = 1;
  body.forEach((b, k) => {
    const p = b.props;
    const id = nextId++;
    layerIds[k] = id;
    const prevDims = k > 0 ? dims[k - 1] : { c, w, h };
    let cur = { ...prevDims };
    let params = 0;
    const ins: number[] = [];

    if (b.type === "convolutional") {
      const filters = num(p, "filters", prevDims.c);
      const size = num(p, "size", 1);
      const stride = num(p, "stride", 1);
      const pad = num(p, "pad", 0) ? Math.floor(size / 2) : num(p, "padding", 0);
      const bn = num(p, "batch_normalize", 0);
      cur = {
        c: filters,
        w: Math.floor((prevDims.w + 2 * pad - size) / stride) + 1,
        h: Math.floor((prevDims.h + 2 * pad - size) / stride) + 1,
      };
      params = filters * prevDims.c * size * size + filters + (bn ? 2 * filters : 0);
      ins.push(k > 0 ? layerIds[k - 1] : 0);
    } else if (b.type === "maxpool" || b.type === "avgpool") {
      const size = num(p, "size", b.type === "avgpool" ? prevDims.w : 2);
      const stride = num(p, "stride", b.type === "avgpool" ? 1 : size);
      cur = b.type === "avgpool"
        ? { c: prevDims.c, w: 1, h: 1 }
        : { c: prevDims.c, w: Math.floor(prevDims.w / stride), h: Math.floor(prevDims.h / stride) };
      ins.push(k > 0 ? layerIds[k - 1] : 0);
    } else if (b.type === "upsample") {
      const stride = num(p, "stride", 2);
      cur = { c: prevDims.c, w: prevDims.w * stride, h: prevDims.h * stride };
      ins.push(k > 0 ? layerIds[k - 1] : 0);
    } else if (b.type === "route") {
      const refs = (p["layers"] || "").split(",").map((s) => Number(s.trim())).filter((n) => !isNaN(n));
      let sumC = 0; let rd = prevDims;
      refs.forEach((ref, idx) => {
        const di = ref < 0 ? k + ref : ref;
        if (di >= 0 && di < dims.length) {
          sumC += dims[di].c;
          if (idx === 0) rd = dims[di];
          ins.push(layerIds[di] ?? 0);
        }
      });
      cur = { c: sumC || prevDims.c, w: rd.w, h: rd.h };
    } else if (b.type === "shortcut") {
      const from = num(p, "from", -3);
      const di = from < 0 ? k + from : from;
      ins.push(k > 0 ? layerIds[k - 1] : 0);
      if (di >= 0 && di < dims.length) ins.push(layerIds[di]);
      cur = { ...prevDims };
    } else if (b.type === "connected") {
      const out = num(p, "output", prevDims.c);
      params = prevDims.c * prevDims.w * prevDims.h * out + out;
      cur = { c: out, w: 1, h: 1 };
      ins.push(k > 0 ? layerIds[k - 1] : 0);
    } else {
      // yolo / region / softmax / dropout / cost / reorg etc. — pass through.
      ins.push(k > 0 ? layerIds[k - 1] : 0);
    }

    dims[k] = cur;
    for (const from of ins) edges.push([from, id]);

    layers.push({
      id, name: `${b.type}_${k}`, type: cap(b.type), op: b.type,
      shape: `1×${cur.c}×${cur.h}×${cur.w}`, dt: "float32", params,
      flops: 0, macs: 0, mem: cur.c * cur.w * cur.h * 4, group: groupFor(b.type),
      ins: ins.map((fid) => ({ n: layers[fid]?.name ?? `#${fid}` })),
      outs: [{ n: `${b.type}_${k}`, s: [1, cur.c, cur.h, cur.w] }],
      attr: { ...p }, w: params > 0 ? { shape: "[multi]", size: params } : null,
      math: describe(b),
      insight: insightFor(b.type),
      qSens: b.type === "convolutional" ? 0.5 : 0.1, compIssues: [],
    });
  });

  const out = dims[dims.length - 1] ?? { c, w, h };
  return {
    name, format: "Darknet", framework: "Darknet (YOLO)", opset: 0,
    sizeBytes: text.length, inputShape: [1, c, h, w], outputShape: [1, out.c, out.h, out.w],
    layers, edges,
  };
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
function describe(b: Block): string {
  if (b.type === "convolutional") return `Conv ${b.props.filters ?? "?"}×${b.props.size ?? "?"}, stride ${b.props.stride ?? 1}, act ${b.props.activation ?? "linear"}`;
  if (b.type === "route") return `Route layers=${b.props.layers ?? ""}`;
  if (b.type === "shortcut") return `Shortcut from=${b.props.from ?? ""}`;
  return cap(b.type);
}
function insightFor(type: string): string {
  switch (type) {
    case "convolutional": return "Darknet convolution (+ optional BN + activation).";
    case "route": return "Concatenates/selects earlier feature maps (FPN-style).";
    case "shortcut": return "Residual add from an earlier layer.";
    case "yolo": return "YOLO detection head — decodes boxes at this scale.";
    case "upsample": return "Nearest upsample for multi-scale fusion.";
    default: return `Darknet ${type} layer.`;
  }
}
