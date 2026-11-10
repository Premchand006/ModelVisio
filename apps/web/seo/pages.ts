// Static, crawlable reference pages for the website, generated at build time
// from the same data the app uses (device catalog, compiler rules, format
// table, auto-fixes), so the pages can never drift from what the tool does.
// The Vite plugin in vite.config.ts writes them to dist/ with the sitemap.
// Pure functions: no Node or DOM APIs.

import { HW, type DeviceSpec } from "@modelvisio/core/hardware";
import { FIXES } from "@modelvisio/core/fixes";
import { FORMAT_SUPPORT } from "@modelvisio/parsers/formats";
import { COMPAT_RULE_OPS } from "@modelvisio/parsers/compat";

export const SITE = "https://modelvisio.netlify.app";
const REPO = "https://github.com/Premchand006/ModelVisio";

export type SeoPage = { path: string; title: string; description: string; html: string };

const esc = (s: string | number) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fmtNum = (n: number, digits = 1) => (Number.isInteger(n) ? String(n) : n.toFixed(digits));

const NAV = [
  { path: "/edge-ai-accelerators/", label: "Accelerators" },
  { path: "/edge-npu-unsupported-ops/", label: "Operator support" },
  { path: "/model-formats/", label: "Formats" },
];

const CSS = `
:root{color-scheme:dark;--bg:#0A0A0B;--bg1:#141417;--bg2:#1C1C20;--bdr:#2A2A30;--t0:#F4F4F5;--t1:#D2D2D8;--t2:#9A9AA4;--acc:#4D8DFF;--ok:#34D399;--wrn:#FBBF24;--err:#F87171}
@media (prefers-color-scheme:light){:root{color-scheme:light;--bg:#F1F5F9;--bg1:#FFFFFF;--bg2:#F8FAFC;--bdr:#E2E8F0;--t0:#0F172A;--t1:#1E293B;--t2:#475569;--acc:#0284C7;--ok:#059669;--wrn:#B45309;--err:#DC2626}}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--t1);font:15px/1.65 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--acc)}
header.top{border-bottom:1px solid var(--bdr);background:var(--bg1)}
header.top .in,main,footer .in{max-width:1120px;margin:0 auto;padding-left:16px;padding-right:16px}
header.top .in{display:flex;align-items:center;gap:16px;flex-wrap:wrap;min-height:56px}
header.top .brand{display:flex;align-items:center;gap:9px;color:var(--t0);font-weight:700;text-decoration:none;font-size:17px}
header.top .brand img{width:28px;height:28px;border-radius:7px}
header.top nav{display:flex;gap:14px;flex-wrap:wrap;flex:1}
header.top nav a{color:var(--t2);text-decoration:none;font-size:14px}
header.top nav a[aria-current]{color:var(--t0);font-weight:600}
.cta{display:inline-block;padding:8px 16px;border-radius:9px;background:var(--acc);color:#fff!important;font-weight:600;text-decoration:none;font-size:14px}
main{padding-top:36px;padding-bottom:48px}
.crumbs{font-size:13px;color:var(--t2);margin-bottom:12px}
.crumbs a{color:var(--t2)}
h1{color:var(--t0);font-size:clamp(26px,4.5vw,40px);line-height:1.15;letter-spacing:-.02em;margin-bottom:14px}
h2{color:var(--t0);font-size:21px;margin:36px 0 10px}
h3{color:var(--t0);font-size:16px;margin:22px 0 6px}
p,ul{margin-bottom:12px;max-width:78ch}
ul{padding-left:20px}
.lede{font-size:16.5px;color:var(--t1)}
.muted{color:var(--t2);font-size:13px}
.box{border:1px solid var(--bdr);background:var(--bg1);border-radius:12px;padding:16px 18px;margin:18px 0}
.scroll{overflow-x:auto;border:1px solid var(--bdr);border-radius:12px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--bg1)}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--bdr);vertical-align:top}
th{color:var(--t0);font-weight:600;background:var(--bg2);white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
code{font-family:"JetBrains Mono",ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--bg2);border:1px solid var(--bdr);border-radius:5px;padding:1px 5px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 14px}
.tag{display:inline-block;font-size:11.5px;font-weight:600;border-radius:5px;padding:1px 7px;border:1px solid var(--bdr)}
.err{color:var(--err)}.wrn{color:var(--wrn)}.ok{color:var(--ok)}
footer{border-top:1px solid var(--bdr);padding:22px 0;font-size:13px;color:var(--t2)}
footer .in{display:flex;gap:16px;flex-wrap:wrap;justify-content:space-between}
@media (max-width:640px){header.top .in{padding-top:10px;padding-bottom:10px;row-gap:8px}header.top .brand{flex:1}header.top nav{order:3;flex-basis:100%;gap:16px}main{padding-top:22px}}
`;

function layout(p: { path: string; title: string; description: string; crumb: string; body: string; jsonLd?: object[] }): SeoPage {
  const url = SITE + p.path;
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage", "@id": url + "#page", url, name: p.title, description: p.description, inLanguage: "en",
        isPartOf: { "@id": SITE + "/#website" }, about: { "@id": SITE + "/#app" },
        breadcrumb: { "@id": url + "#breadcrumb" },
      },
      {
        "@type": "BreadcrumbList", "@id": url + "#breadcrumb",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "ModelVisio", item: SITE + "/" },
          { "@type": "ListItem", position: 2, name: p.crumb, item: url },
        ],
      },
      ...(p.jsonLd ?? []),
    ],
  };
  const nav = NAV.map((n) => `<a href="${n.path}"${n.path === p.path ? ' aria-current="page"' : ""}>${esc(n.label)}</a>`).join("");
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${esc(p.title)}</title>
<meta name="description" content="${esc(p.description)}" />
<meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1" />
<link rel="canonical" href="${url}" />
<meta name="theme-color" content="#0A0A0B" />
<meta property="og:type" content="article" />
<meta property="og:site_name" content="ModelVisio" />
<meta property="og:url" content="${url}" />
<meta property="og:title" content="${esc(p.title)}" />
<meta property="og:description" content="${esc(p.description)}" />
<meta property="og:image" content="${SITE}/og-image.png" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="630" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:image" content="${SITE}/og-image.png" />
<link rel="icon" href="/favicon.ico" sizes="any" />
<link rel="apple-touch-icon" href="/apple-touch-icon.png" />
<link rel="manifest" href="/manifest.webmanifest" />
<script type="application/ld+json">${JSON.stringify(ld).replace(/</g, "\\u003c")}</script>
<style>${CSS.trim()}</style>
</head>
<body>
<header class="top"><div class="in">
<a class="brand" href="/"><img src="/icon-192.png" alt="" width="28" height="28" />ModelVisio</a>
<nav aria-label="Reference">${nav}</nav>
<a class="cta" href="/">Open the app</a>
</div></header>
<main>
<div class="crumbs"><a href="/">ModelVisio</a> / ${esc(p.crumb)}</div>
${p.body.trim()}
</main>
<footer><div class="in">
<span>ModelVisio is free, open-source (MIT) software by Premchand. Generated from the app's own data.</span>
<span><a href="${REPO}">GitHub</a> · <a href="${REPO}/releases/latest">Desktop app</a> · <a href="${REPO}/blob/main/docs/roofline-scoring-research.md">Scoring research</a></span>
</div></footer>
</body>
</html>
`;
  return { path: p.path, title: p.title, description: p.description, html };
}

// ── Edge AI accelerators ────────────────────────────────────────────────────

const CONF: Record<DeviceSpec["conf"], string> = { V: "vendor datasheet", B: "independent benchmark", M: "MLPerf result", "?": "estimated" };
const FALLBACK: Record<DeviceSpec["cpuFallback"], string> = {
  graceful: "Per-op CPU fallback",
  "single-partition": "One partition: everything after the first unsupported op runs on CPU",
  none: "No CPU fallback: unsupported ops must be cut from the graph",
};

function acceleratorsPage(): SeoPage {
  const rows = [...HW].sort((a, b) => (b.peak.int8 ?? 0) - (a.peak.int8 ?? 0));
  const n = rows.length;
  const pct = (x: number) => `${Math.round(x * 100)}%`;
  const tableRows = rows.map((d) => {
    const int8 = d.peak.int8 ?? 0;
    const ridge = d.bandwidthGBs > 0 ? (int8 * 1000) / d.bandwidthGBs : 0;
    const mem = d.ramGB > 0 ? `${fmtNum(d.ramGB)} GB ${d.memType}` : d.sramMB ? `On-chip (${fmtNum(d.sramMB, 2)} MB)` : "On-chip";
    const flags = [d.measured ? '<span class="tag ok">calibrated</span>' : "", d.eol ? '<span class="tag wrn">end of life</span>' : ""].filter(Boolean).join(" ");
    return `<tr><td><strong>${esc(d.n)}</strong> ${flags}<br /><span class="muted">${esc(d.formFactor)}</span></td>`
      + `<td class="num">${fmtNum(int8)}${d.peak.sparseInt8 && d.peak.sparseInt8 !== int8 ? `<br /><span class="muted">${fmtNum(d.peak.sparseInt8)} sparse</span>` : ""}</td>`
      + `<td>${esc(mem)}</td><td class="num">${fmtNum(d.bandwidthGBs)}${d.bwConf === "?" ? " <span class=\"muted\">est.</span>" : ""}</td>`
      + `<td class="num">${fmtNum(d.power.typ)}</td><td class="num">${fmtNum(int8 / d.power.typ)}</td>`
      + `<td class="num">${ridge ? Math.round(ridge) : "–"}</td>`
      + `<td>${esc(d.toolchain.join(", "))}<br /><span class="muted">${esc(d.fmts.join(", "))}</span></td>`
      + `<td class="muted">${esc(CONF[d.conf])}</td></tr>`;
  }).join("\n");
  const calibrated = rows.filter((d) => d.measured);
  const sources = rows.map((d) => `<li><strong>${esc(d.n)}:</strong> ${esc(d.src.join("; "))}${d.notes ? ` <span class="muted">${esc(d.notes)}</span>` : ""}</li>`).join("\n");
  const fallbackRows = (Object.keys(FALLBACK) as DeviceSpec["cpuFallback"][]).map((k) => {
    const names = rows.filter((d) => d.cpuFallback === k).map((d) => d.n);
    return names.length ? `<li><strong>${esc(FALLBACK[k])}:</strong> ${esc(names.join(", "))}.</li>` : "";
  }).join("\n");
  const ex = rows.find((d) => d.id === "orin_nano");
  const exRidge = ex ? Math.round(((ex.peak.int8 ?? 0) * 1000) / ex.bandwidthGBs) : 0;

  return layout({
    path: "/edge-ai-accelerators/",
    title: `Edge AI Accelerator Comparison: ${n} NPUs | ModelVisio`,
    description: `Jetson Orin, Hailo-8, Coral Edge TPU, RK3588 and ${n - 4} more edge AI accelerators compared: INT8 TOPS, memory bandwidth, power, TOPS/W and toolchains.`,
    crumb: "Edge AI accelerators",
    body: `
<h1>Edge AI accelerator comparison: ${n} NPUs, GPUs and TPUs</h1>
<p class="lede">The ${n} edge accelerators ModelVisio scores models against, with the numbers that decide how fast a model runs: dense INT8 TOPS, memory bandwidth, typical power and the toolchain you compile with. Each figure carries its source.</p>
<p><a class="cta" href="/">Score your model on all ${n}</a> <span class="muted">&nbsp;Drop a model file in the browser. It is never uploaded.</span></p>

<h2>Specifications</h2>
<p class="muted">Sorted by dense INT8 TOPS. Ridge point = TOPS ÷ bandwidth, in operations per byte. "est." marks an estimated bandwidth.</p>
<div class="scroll"><table>
<thead><tr><th>Device</th><th class="num">INT8 TOPS</th><th>Memory</th><th class="num">Bandwidth GB/s</th><th class="num">Power W</th><th class="num">TOPS/W</th><th class="num">Ridge ops/byte</th><th>Toolchain and formats</th><th>Data</th></tr></thead>
<tbody>
${tableRows}
</tbody></table></div>

<h2>Why TOPS doesn't tell you FPS</h2>
<p>Peak TOPS is an upper bound that real networks rarely reach. Two limits sit below it:</p>
<ul>
<li><strong>Memory bandwidth.</strong> In the roofline model (Williams, Waterman and Patterson, 2009), a layer that does fewer operations per byte it moves than the device's ridge point is memory-bound. It runs at bandwidth × intensity, however many TOPS the chip has.${ex ? ` On the ${esc(ex.n)} the ridge point is about ${exRidge} ops/byte, so a layer needs that much arithmetic intensity to use the full ${fmtNum(ex.peak.int8 ?? 0)} TOPS.` : ""}</li>
<li><strong>Utilization.</strong> Compilers rarely keep every MAC busy. ModelVisio assumes typical utilization per workload, for example ${pct(rows[0].util.cnnCls)} for CNN classification, ${pct(rows[0].util.cnnDet)} for detection and ${pct(rows[0].util.transformer)} for transformers on a mature toolchain.</li>
</ul>
<p>${calibrated.length ? `Where a published benchmark exists, utilization is calibrated from it instead: ${calibrated.map((d) => `${esc(d.n)} (${esc(d.measured!.note)})`).join("; ")}. Results for these devices are marked calibrated in the app.` : ""} All other devices use estimates and are labelled that way. A model whose memory footprint exceeds the device's memory scores at most 15 out of 100.</p>

<h2>What happens to unsupported ops</h2>
<p>When the compiler meets an op the accelerator can't run, the device's fallback behaviour decides the cost:</p>
<ul>
${fallbackRows}
</ul>
<p>See <a href="/edge-npu-unsupported-ops/">unsupported operators on Coral, RKNN, Hailo and Kneron</a> for the op lists ModelVisio checks.</p>

<h2>Sources</h2>
<ul class="muted">
${sources}
</ul>
<p class="muted">Full methodology and citations: <a href="${REPO}/blob/main/docs/roofline-scoring-research.md">roofline scoring research</a>. Spot an outdated number? <a href="${REPO}/issues/new?template=feature_request.yml">Open an issue</a> with the datasheet.</p>
`,
    jsonLd: [{
      "@type": "ItemList", name: "Edge AI accelerators scored by ModelVisio", numberOfItems: n,
      itemListElement: rows.map((d, i) => ({ "@type": "ListItem", position: i + 1, name: d.n })),
    }],
  });
}

// ── Unsupported ops on edge NPUs ────────────────────────────────────────────

function opsPage(): SeoPage {
  const code = (xs: readonly string[]) => `<div class="chips">${xs.map((x) => `<code>${esc(x)}</code>`).join("")}</div>`;
  const E = '<span class="err">✕ error</span>', W = '<span class="wrn">! warning</span>', O = '<span class="muted">not flagged</span>';
  const matrix: [string, string, string, string, string][] = [
    ["Dynamic input shape (any non-batch dim unknown)", E, W, E, E],
    ["Data-dependent ops (NMS, NonZero, TopK, Loop/If, Scatter…)", E, W, E, E],
    [`Ops outside the Edge TPU op list (${COMPAT_RULE_OPS.coralUnsupported.length} ops)`, E, O, O, O],
    ["MatMul of two activations (attention-style)", E, O, O, O],
    ["Softmax not on a 1-D tensor of ≤ 16,000 elements", E, O, O, O],
    ["Output with more than 3 non-unit dimensions", E, O, O, O],
    ["Resize / Upsample", W, O, O, O],
  ];
  const fixes = FIXES.map((f) => `<li><strong>${esc(f.title)}.</strong> ${esc(f.change)} <span class="muted">App's estimate: ${esc(f.impact)}</span></li>`).join("\n");
  return layout({
    path: "/edge-npu-unsupported-ops/",
    title: "Edge NPU Unsupported Ops: Coral, RKNN, Hailo | ModelVisio",
    description: "Which ONNX and TFLite ops fail or fall back to CPU on Coral Edge TPU, Rockchip RKNN (RK3588), Hailo-8 and Kneron KL720, and how to fix them before you compile.",
    crumb: "Unsupported ops on edge NPUs",
    body: `
<h1>Unsupported operators on Coral Edge TPU, RKNN, Hailo and Kneron</h1>
<p class="lede">Edge NPU compilers reject some ops outright and quietly push others to the CPU, often after a long compile. These are the rules ModelVisio's compiler pre-flight checks in your browser, taken from each vendor's documented limits.</p>
<p><a class="cta" href="/">Check your model now</a> <span class="muted">&nbsp;ONNX, TFLite, PyTorch, Keras, Caffe, OpenVINO and more. Problems are marked on the graph.</span></p>

<h2>Rules by target</h2>
<div class="scroll"><table>
<thead><tr><th>Rule</th><th>Coral Edge TPU</th><th>Rockchip RKNN (RK3588)</th><th>Hailo-8 (DFC)</th><th>Kneron KL720</th></tr></thead>
<tbody>
${matrix.map((r) => `<tr><td>${esc(r[0])}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td></tr>`).join("\n")}
</tbody></table></div>
<p class="muted">An error means the compiler rejects the op or the device can't run it. A warning means it may run on the CPU or needs an extra build option. "Not flagged" means ModelVisio has no rule for it, not a guarantee that it compiles.</p>

<h2>Static input shapes</h2>
<p>Every one of these compilers builds a fixed schedule, so every tensor size has to be known at compile time. A model exported with a dynamic height, width or sequence length fails on Coral, Hailo and Kneron. RKNN builds for fixed shapes or a preset list (<code>dynamic_input</code>). Re-export with a static input size, for example <code>1×3×640×640</code>. The batch dimension is fixed at export and is not flagged.</p>

<h2>Data-dependent ops</h2>
<p>These ops decide their output size or control flow from the input values, so no static NPU schedule exists for them. Cut the graph before them and do the step in post-processing on the host. Hailo handles NMS in its own post-process.</p>
${code(COMPAT_RULE_OPS.dataDependent)}

<h2>Coral Edge TPU</h2>
<p>The Edge TPU compiler maps the graph only up to the first unsupported op. Everything after it runs on the CPU, even ops the TPU supports, so one early unsupported op can cost most of the speed-up. After ONNX or Keras is lowered to TFLite, these ops become DIV, POW, EXP, LOG, SQRT, GATHER, CAST and similar ops outside the supported list:</p>
${code(COMPAT_RULE_OPS.coralUnsupported)}
<h3>Shape and pattern limits</h3>
<ul>
<li><strong>Softmax</strong> runs on the TPU only on a 1-D tensor of at most 16,000 elements.</li>
<li><strong>MatMul between two activations</strong> (attention-style batch matmul) is not supported. Only fully connected layers with constant weights map.</li>
<li><strong>Tensors</strong> may have a size above 1 only in their 3 innermost dimensions.</li>
<li><strong>Resize and Upsample</strong> may stay on the CPU depending on input and output size, to avoid precision loss. Check the compiler log.</li>
</ul>

<h2>Auto-fixes in ModelVisio</h2>
<p>For the common cases the app can rewrite the graph in place, show the new op-support result, and undo it:</p>
<ul>
${fixes}
</ul>
<p>The fixes change the in-browser graph so you can see the effect on compatibility and the estimated latency. Apply the same change in your training or export code before you compile.</p>

<h2>How the checks work</h2>
<p>Op names are normalized across ONNX, TFLite, Keras, Caffe, OpenVINO and other spellings, so <code>stablehlo.dot_general</code> and <code>MatMul</code> are matched the same way. The rules are deliberately conservative: an op ModelVisio doesn't recognize produces no finding. The check can miss a problem but won't invent one. Integer shape arithmetic that converters constant-fold away is skipped. The vendor compiler still has the final word.</p>
<p class="muted">Sources: <a href="https://coral.ai/docs/edgetpu/models-intro/">Coral: TensorFlow models on the Edge TPU</a> (supported operations and model requirements); the Rockchip RKNN, Hailo Dataflow Compiler and Kneron toolchain documentation. See also the <a href="/edge-ai-accelerators/">edge AI accelerator comparison</a>.</p>
`,
  });
}

// ── Model formats ───────────────────────────────────────────────────────────

function formatsPage(): SeoPage {
  const full = FORMAT_SUPPORT.filter((f) => f.status === "full");
  const meta = FORMAT_SUPPORT.filter((f) => f.status === "metadata");
  const exts = (xs: string[]) => xs.map((x) => `<code>.${esc(x)}</code>`).join(" ");
  const rows = (fs: typeof FORMAT_SUPPORT) => fs.map((f) => `<tr><td><strong>${esc(f.label)}</strong></td><td>${exts(f.exts)}</td></tr>`).join("\n");
  return layout({
    path: "/model-formats/",
    title: `${FORMAT_SUPPORT.length} Model Formats You Can Open Online | ModelVisio`,
    description: `Open ONNX, TFLite, PyTorch, GGUF, Safetensors, Core ML and ${FORMAT_SUPPORT.length - 6} more model formats in your browser. ${full.length} render a full graph; nothing is uploaded.`,
    crumb: "Model formats",
    body: `
<h1>Model formats you can open in the browser</h1>
<p class="lede">ModelVisio opens ${FORMAT_SUPPORT.length} neural-network model formats. ${full.length} are fully parsed into an interactive graph with per-layer shapes, weights and FLOPs. The other ${meta.length} are detected and show the file's real metadata, never a made-up graph. Parsing runs in a Web Worker in your browser, so the model file is not uploaded.</p>
<p><a class="cta" href="/">Open a model</a> <span class="muted">&nbsp;or load the YOLO26n demo without a file.</span></p>

<h2>Fully parsed: graph, inspector and pre-flight</h2>
<div class="scroll"><table><thead><tr><th>Format</th><th>File extensions</th></tr></thead><tbody>
${rows(full)}
</tbody></table></div>
<p>OpenVINO, ncnn, PaddlePaddle and TensorFlow.js keep their weights in a separate file. Select or drop the weights file together with the graph file (for example <code>model.xml</code> with <code>model.bin</code>).</p>

<h2>Detected: metadata only</h2>
<div class="scroll"><table><thead><tr><th>Format</th><th>File extensions</th></tr></thead><tbody>
${rows(meta)}
</tbody></table></div>

<h2>What you get for a fully parsed model</h2>
<ul>
<li>A Netron-style graph with a per-layer inspector, exportable to SVG or PNG.</li>
<li>A compiler pre-flight for Coral Edge TPU, RKNN, Hailo and Kneron (<a href="/edge-npu-unsupported-ops/">the rules</a>).</li>
<li>Hardware-fit scores for <a href="/edge-ai-accelerators/">${HW.length} edge accelerators</a>: estimated FPS, FPS/W and memory fit.</li>
<li>Exports to Graph JSON, Layers CSV, Safetensors and NumPy, plus TensorRT, HailoRT and RKNN deploy scripts.</li>
</ul>
<p class="muted">Missing a format? <a href="${REPO}/issues/new?template=feature_request.yml">Request it</a> with a sample file.</p>
`,
    jsonLd: [{
      "@type": "ItemList", name: "Model formats ModelVisio opens", numberOfItems: FORMAT_SUPPORT.length,
      itemListElement: FORMAT_SUPPORT.map((f, i) => ({ "@type": "ListItem", position: i + 1, name: f.label })),
    }],
  });
}

/** Every generated page. Paths end in "/" and are written as <path>index.html. */
export function buildSeoPages(): SeoPage[] {
  return [acceleratorsPage(), opsPage(), formatsPage()];
}

/** sitemap.xml for the home page plus the generated pages. */
export function buildSitemap(pages: Pick<SeoPage, "path">[], lastmod: string): string {
  const urls = ["/", ...pages.map((p) => p.path)];
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((u) => `  <url>\n    <loc>${SITE}${u}</loc>\n    <lastmod>${lastmod}</lastmod>\n  </url>`).join("\n")}
</urlset>
`;
}
