// Deployment recipe generator: for (model, device, runtime, precision) emit a
// set of runnable files (conversion script, calibration helper, inference +
// latency harness, accuracy check, benchmark command, requirements). Pure: no
// React, no I/O. Reuses the Converter's CONV_SCRIPTS templates where they fit
// (ONNX→RKNN, PyTorch→ONNX/TFLite, ONNX FP16) and the DeployRecipe panel's
// quick-start recipes as a reference file.

import type { Model } from "@modelvisio/parsers";
import type { DeviceSpec } from "../data/hardware";
import { CONV_SCRIPTS, type ConvOpts } from "../data/convScripts";
import { deployRecipeText } from "./recipeTemplates";
import { RUNTIMES, runtimesFor, type DeployPrecision, type RuntimeId } from "./runtimes";

export type PackageFile = { path: string; content: string; description: string };

export type RecipeBundle = {
  runtime: RuntimeId;
  runtimeLabel: string;
  precision: DeployPrecision;
  /** Artifact the conversion produces (model.engine, model.hef, …). */
  artifact: string;
  files: PackageFile[];
  /** One-line commands, in order, for the README. */
  steps: string[];
  benchmarkCmd: string;
  notes: string[];
};

const stem = (n: string) => n.replace(/\.[^.]+$/, "") || "model";

/** Is the source a PyTorch checkpoint (needs an ONNX export first)? */
export function isPytorchSource(model: Model): boolean {
  return /pytorch|torchscript|\.?pt$|\.?pth$|torch/i.test(model.format);
}

type Io = { h: number; w: number; c: number; rank: number; inputName: string; image: boolean; dims: number[] };

function ioOf(model: Model): Io {
  const s = model.inputShape ?? [];
  const rank = s.length;
  const image = rank === 4 && (s[1] === 1 || s[1] === 3);
  const h = image && s[2] > 0 ? s[2] : 640;
  const w = image && s[3] > 0 ? s[3] : h;
  const inLayer = model.layers.find((l) => l.group === "input");
  return {
    h, w, c: image ? s[1] : 3, rank, image,
    inputName: inLayer?.outs?.[0]?.n || inLayer?.name || "input",
    dims: s.map((d) => (d > 0 ? d : 1)),
  };
}

const convOpts = (o: Partial<ConvOpts>): ConvOpts => ({
  opset: 17, sz: 640, dyn: false, i8: false, fp16: true, inp: "model.onnx", out: "", prec: "fp16", plat: "rk3588", ...o,
});

// ─── Shared helpers emitted into the package ─────────────────────────────────

function calibScript(io: Io, layout: "nchw" | "nhwc"): string {
  return `"""Build an INT8 calibration set from representative images.

Usage: python calib/make_calib.py path/to/images/ [--n 300]
Writes calib/calib.npy (${layout.toUpperCase()}, float32 0..1) and calib/calib_list.txt.
Use 200-1000 images from the REAL deployment distribution (same camera,
lighting, resolution). Calibration quality dominates INT8 accuracy.
"""
import argparse, glob, os
import numpy as np
import cv2

H, W = ${io.h}, ${io.w}

def load(path):
    img = cv2.imread(path)[:, :, ::-1]                      # BGR -> RGB
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0                       # HWC
    return ${layout === "nchw" ? "x.transpose(2, 0, 1)" : "x"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    paths = sorted(p for e in ("jpg", "jpeg", "png", "bmp") for p in glob.glob(os.path.join(a.images, f"**/*.{e}"), recursive=True))[: a.n]
    if not paths:
        raise SystemExit(f"no images found under {a.images}")
    os.makedirs("calib", exist_ok=True)
    np.save("calib/calib.npy", np.stack([load(p) for p in paths]))
    with open("calib/calib_list.txt", "w") as f:
        f.write("\\n".join(os.path.abspath(p) for p in paths) + "\\n")
    print(f"calibration set: {len(paths)} images -> calib/calib.npy, calib/calib_list.txt")

if __name__ == "__main__":
    main()
`;
}

function referenceScript(onnxName: string, io: Io): string {
  return `"""FP32 reference outputs with ONNX Runtime (CPU), for the accuracy check.

Usage: python validate/reference.py path/to/val_images/ --out ref_outputs/
"""
import argparse, glob, os
import numpy as np
import onnxruntime as ort
${io.image ? "import cv2\n" : ""}
H, W = ${io.h}, ${io.w}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("--out", default="ref_outputs")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    sess = ort.InferenceSession("${onnxName}", providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.images, "*")))[: a.n]
    for i, p in enumerate(paths):
${io.image
    ? `        img = cv2.resize(cv2.imread(p)[:, :, ::-1], (W, H)).astype(np.float32) / 255.0
        x = img.transpose(2, 0, 1)[None]`
    : `        x = np.load(p).astype(np.float32)`}
        outs = sess.run(None, {name: x})
        np.savez(os.path.join(a.out, f"{i:05d}.npz"), *outs)
    print(f"wrote {len(paths)} reference outputs to {a.out}/")

if __name__ == "__main__":
    main()
`;
}

const ACCURACY_CHECK = `"""Compare deployed outputs against the FP32 reference.

Usage: python validate/accuracy_check.py ref_outputs/ dev_outputs/ [--min-cos 0.99]
Per output tensor: cosine similarity and max abs error, averaged over images.
Cosine >= 0.99 is a good first gate for INT8; ALSO run your task metric
(mAP / top-1 / WER) on a labelled validation set before shipping.
Note: outputs are compared flattened; if the deployed runtime emits NHWC
(TFLite/RKNN/Hailo) transpose to the reference layout first.
"""
import argparse, glob, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("dev")
    ap.add_argument("--min-cos", type=float, default=0.99)
    a = ap.parse_args()
    cos, err = {}, {}
    for rp in sorted(glob.glob(os.path.join(a.ref, "*.npz"))):
        dp = os.path.join(a.dev, os.path.basename(rp))
        if not os.path.exists(dp):
            continue
        r, d = np.load(rp), np.load(dp)
        for k in r.files:
            if k not in d.files:
                continue
            x = r[k].astype(np.float64).ravel()
            y = d[k].astype(np.float64).ravel()
            if x.size != y.size:
                print(f"{k}: size mismatch {x.size} vs {y.size} (layout/shape differs)")
                continue
            c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
            cos.setdefault(k, []).append(c)
            err.setdefault(k, []).append(float(np.max(np.abs(x - y))))
    if not cos:
        raise SystemExit("no matching outputs; run reference.py and infer.py --dump first")
    ok = True
    for k in cos:
        m = float(np.mean(cos[k]))
        ok &= m >= a.min_cos
        print(f"{k}: cosine {m:.5f} (min {min(cos[k]):.5f})  max|err| {np.mean(err[k]):.4g}  n={len(cos[k])}")
    print("PASS" if ok else f"FAIL: mean cosine below {a.min_cos}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
`;

/** Python inference + latency harness. `body` defines load() → run(x) → list[np.ndarray]. */
function inferHarness(opts: {
  title: string; io: Io; layout: "nchw" | "nhwc"; dtype: "float32" | "uint8"; targetFps?: number; body: string;
}): string {
  const { io } = opts;
  const shape = opts.layout === "nchw" ? `(1, ${io.c}, ${io.h}, ${io.w})` : `(1, ${io.h}, ${io.w}, ${io.c})`;
  return `"""${opts.title}

Usage:
  python runtime/infer.py --bench [--iters 300]            # latency p50/p95 + FPS gate
  python runtime/infer.py --images val/ --dump dev_outputs/ # outputs for validate/accuracy_check.py
"""
import argparse, glob, os, time
import numpy as np
${io.image ? "import cv2\n" : ""}
H, W = ${io.h}, ${io.w}
TARGET_FPS = float(os.environ.get("TARGET_FPS", "${opts.targetFps ?? 0}"))

def preprocess(path):
${io.image
    ? `    img = cv2.resize(cv2.imread(path)[:, :, ::-1], (W, H))   # RGB, HWC uint8
    x = ${opts.dtype === "uint8" ? "img" : "img.astype(np.float32) / 255.0"}
    return ${opts.layout === "nchw" ? "x.transpose(2, 0, 1)[None]" : "x[None]"}`
    : `    return np.load(path).astype(np.${opts.dtype})`}

def dummy():
    ${opts.dtype === "uint8" ? `return np.random.randint(0, 255, ${io.image ? shape : `(${io.dims.join(", ")})`}, dtype=np.uint8)` : `return np.random.rand(*${io.image ? shape : `(${io.dims.join(", ")},)`}).astype(np.float32)`}

${opts.body.trim()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--images")
    ap.add_argument("--dump")
    a = ap.parse_args()
    run = load()
    if a.images:
        os.makedirs(a.dump or "dev_outputs", exist_ok=True)
        for i, p in enumerate(sorted(glob.glob(os.path.join(a.images, "*")))):
            outs = run(preprocess(p))
            np.savez(os.path.join(a.dump or "dev_outputs", f"{i:05d}.npz"), *outs)
        print("outputs written")
    if a.bench or not a.images:
        x = dummy()
        for _ in range(30):
            run(x)                                           # warm-up (clocks, caches, lazy init)
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter(); run(x); ts.append((time.perf_counter() - t0) * 1000)
        p50, p95 = np.percentile(ts, 50), np.percentile(ts, 95)
        fps = 1000.0 / np.mean(ts)
        print(f"latency p50 {p50:.2f} ms  p95 {p95:.2f} ms  ~{fps:.1f} FPS (batch 1, {a.iters} iters)")
        if TARGET_FPS > 0:
            ok = 1000.0 / p95 >= TARGET_FPS
            print(("PASS" if ok else "FAIL") + f": p95-based FPS {1000.0 / p95:.1f} vs target {TARGET_FPS:g}")
            raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
`;
}

// ─── Per-runtime recipes ─────────────────────────────────────────────────────

type Ctx = {
  model: Model; dev: DeviceSpec; prec: DeployPrecision; io: Io;
  src: string; onnx: string; pt: boolean; targetFps?: number;
};

/** Step 0 for PyTorch sources: export to ONNX with the Converter's template. */
function exportStep(c: Ctx): PackageFile[] {
  if (!c.pt) return [];
  return [{
    path: "convert/export_onnx.py",
    content: CONV_SCRIPTS["pt→onnx"](convOpts({ inp: c.src, sz: c.io.h, opset: 17 })) + "\n",
    description: "PyTorch → ONNX export (run first)",
  }];
}

const reqs = (deps: string[]): PackageFile => ({ path: "runtime/requirements.txt", content: deps.join("\n") + "\n", description: "Python dependencies for the runtime harness" });

function tensorrt(c: Ctx): Omit<RecipeBundle, "runtime" | "runtimeLabel" | "precision"> {
  const flag = c.prec === "int8" ? "--int8 --fp16" : c.prec === "fp16" ? "--fp16" : "";
  const convert = `#!/usr/bin/env bash
# ${c.dev.n}: ONNX -> TensorRT engine (${c.prec.toUpperCase()}). Build ON the target
# (engines are tied to the GPU + TensorRT version). JetPack ships trtexec at
# /usr/src/tensorrt/bin/trtexec.
set -euo pipefail
TRTEXEC=\${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
${c.prec === "int8"
    ? `# INT8 needs calibration. Preferred: export a QDQ ONNX (explicit quantization) with
# NVIDIA TensorRT Model Optimizer / pytorch-quantization and build that; otherwise
# supply a calibration cache. Without either, trtexec --int8 uses PLACEHOLDER
# scales: fine for timing, NOT for accuracy.
CALIB=\${CALIB:-calib/calib.cache}
EXTRA=""
[ -f "$CALIB" ] && EXTRA="--calib=$CALIB" || echo "WARNING: no $CALIB; engine is timing-only until calibrated" >&2
`
    : `EXTRA=""
`}$TRTEXEC --onnx=${c.onnx} --saveEngine=model.engine ${flag} --memPoolSize=workspace:2048 $EXTRA
echo "built model.engine"
`;
  const body = `import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (creates the CUDA context)

def load():
    logger = trt.Logger(trt.Logger.WARNING)
    with open("model.engine", "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    stream = cuda.Stream()
    host, dev, inputs, outputs = {}, {}, [], []
    for i in range(engine.num_io_tensors):          # TensorRT >= 8.5 name-based I/O
        name = engine.get_tensor_name(i)
        shape = tuple(ctx.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host[name] = cuda.pagelocked_empty(trt.volume(shape), dtype)
        dev[name] = cuda.mem_alloc(host[name].nbytes)
        ctx.set_tensor_address(name, int(dev[name]))
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append((name, shape))
    def run(x):
        name, _ = inputs[0]
        np.copyto(host[name], x.ravel().astype(host[name].dtype))
        cuda.memcpy_htod_async(dev[name], host[name], stream)
        ctx.execute_async_v3(stream.handle)
        for n, _ in outputs:
            cuda.memcpy_dtoh_async(host[n], dev[n], stream)
        stream.synchronize()
        return [host[n].reshape(s).copy() for n, s in outputs]
    return run`;
  const files: PackageFile[] = [
    { path: "convert/convert.sh", content: convert, description: "ONNX → TensorRT engine build (trtexec)" },
    { path: "runtime/infer.py", content: inferHarness({ title: `TensorRT inference + latency harness on ${c.dev.n}.`, io: c.io, layout: "nchw", dtype: "float32", targetFps: c.targetFps, body }), description: "TensorRT inference + latency benchmark" },
    reqs(["numpy", "opencv-python-headless", "pycuda", "# tensorrt: provided by JetPack / the L4T container"]),
    {
      path: "runtime/Dockerfile",
      content: `# JetPack 6 / L4T r36 runtime; match the tag to the device's JetPack release.
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0
WORKDIR /app
COPY runtime/ /app/runtime/
COPY model.engine /app/model.engine
RUN pip3 install --no-cache-dir -r runtime/requirements.txt
CMD ["python3", "runtime/infer.py", "--bench"]
`,
      description: "Container for the Jetson runtime",
    },
  ];
  const ref = deployRecipeText(c.dev.id, c.onnx);
  if (ref) files.push({ path: "reference/quickstart.sh", content: ref + "\n", description: "Original quick-start recipe (DeployRecipe panel)" });
  return {
    artifact: "model.engine", files,
    steps: [...(c.pt ? ["python convert/export_onnx.py"] : []), ...(c.prec === "int8" ? ["python calib/make_calib.py images/   # then produce calib/calib.cache or a QDQ ONNX"] : []), "bash convert/convert.sh", "python3 runtime/infer.py --bench"],
    benchmarkCmd: "/usr/src/tensorrt/bin/trtexec --loadEngine=model.engine --warmUp=500 --iterations=1000 --percentile=95",
    notes: ["Run `sudo nvpmodel -m 0 && sudo jetson_clocks` before benchmarking for stable clocks.", "Rebuild the engine whenever JetPack/TensorRT changes."],
  };
}

function openvino(c: Ctx): Omit<RecipeBundle, "runtime" | "runtimeLabel" | "precision"> {
  const ovDevice = c.dev.id === "intel_npu" ? "NPU" : c.dev.id === "movidius" ? "MYRIAD" : "CPU";
  const convert = `"""ONNX -> OpenVINO IR (${c.prec.toUpperCase()}) for ${c.dev.n}."""
import numpy as np
import openvino as ov

model = ov.convert_model("${c.onnx}")
${c.prec === "int8"
    ? `import nncf  # pip install nncf
calib = np.load("calib/calib.npy")                       # from calib/make_calib.py (NCHW)
dataset = nncf.Dataset(list(calib), lambda x: x[None].astype(np.float32))
model = nncf.quantize(model, dataset, subset_size=min(300, len(calib)))
ov.save_model(model, "model.xml", compress_to_fp16=False)`
    : `ov.save_model(model, "model.xml", compress_to_fp16=${c.prec === "fp16" ? "True" : "False"})`}
print("saved model.xml + model.bin")
`;
  const body = `import openvino as ov

def load():
    core = ov.Core()
    compiled = core.compile_model("model.xml", "${ovDevice}")   # "CPU" / "GPU" / "NPU"
    req = compiled.create_infer_request()
    outs = compiled.outputs
    def run(x):
        req.infer({0: x})
        return [req.get_tensor(o).data.copy() for o in outs]
    return run`;
  return {
    artifact: "model.xml + model.bin",
    files: [
      { path: "convert/convert.py", content: convert, description: `ONNX → OpenVINO IR${c.prec === "int8" ? " + NNCF INT8 quantization" : ""}` },
      { path: "runtime/infer.py", content: inferHarness({ title: `OpenVINO inference + latency harness on ${c.dev.n} (${ovDevice}).`, io: c.io, layout: "nchw", dtype: "float32", targetFps: c.targetFps, body }), description: "OpenVINO inference + latency benchmark" },
      reqs(["openvino>=2024.0", ...(c.prec === "int8" ? ["nncf"] : []), "numpy", "opencv-python-headless"]),
    ],
    steps: [...(c.pt ? ["python convert/export_onnx.py"] : []), ...(c.prec === "int8" ? ["python calib/make_calib.py images/"] : []), "python convert/convert.py", "python runtime/infer.py --bench"],
    benchmarkCmd: `benchmark_app -m model.xml -d ${ovDevice} -hint latency -niter 1000`,
    notes: ovDevice === "MYRIAD" ? ["MYRIAD needs OpenVINO 2022.3 LTS (dropped in 2023+); use the legacy API there."] : ["The NPU plugin needs the Intel NPU driver installed on the host."],
  };
}

function rknn(c: Ctx): Omit<RecipeBundle, "runtime" | "runtimeLabel" | "precision"> {
  const base = CONV_SCRIPTS["onnx→rknn"](convOpts({ inp: c.onnx, out: "model.rknn", plat: "rk3588", i8: c.prec === "int8" }))
    .replace('dataset="calib_list.txt"', 'dataset="calib/calib_list.txt"');
  const convert = `"""ONNX -> RKNN (${c.prec === "int8" ? "INT8" : "FP16"}) for ${c.dev.n}. Run on an x86 Linux host with rknn-toolkit2."""
${base}
`;
  const body = `from rknnlite.api import RKNNLite   # on the board: pip install rknn-toolkit-lite2

def load():
    r = RKNNLite()
    assert r.load_rknn("model.rknn") == 0
    assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
    def run(x):
        return r.inference(inputs=[x])
    return run`;
  return {
    artifact: "model.rknn",
    files: [
      { path: "convert/convert.py", content: convert, description: "ONNX → RKNN (rknn-toolkit2)" },
      { path: "runtime/infer.py", content: inferHarness({ title: `RKNN-Lite inference + latency harness on ${c.dev.n}. Input: uint8 NHWC (normalization is baked into the model by rknn.config).`, io: c.io, layout: "nhwc", dtype: "uint8", targetFps: c.targetFps, body }), description: "RKNN-Lite inference + latency benchmark (on-board)" },
      reqs(["# host: rknn-toolkit2   board: rknn-toolkit-lite2", "numpy", "opencv-python-headless"]),
      ...(deployRecipeText(c.dev.id, c.onnx) ? [{ path: "reference/quickstart.sh", content: deployRecipeText(c.dev.id, c.onnx)! + "\n", description: "Original quick-start recipe (DeployRecipe panel)" }] : []),
    ],
    steps: [...(c.pt ? ["python convert/export_onnx.py"] : []), ...(c.prec === "int8" ? ["python calib/make_calib.py images/"] : []), "python convert/convert.py   # x86 host", "scp model.rknn runtime/ board:~/app/ && ssh board python3 app/runtime/infer.py --bench"],
    benchmarkCmd: "python3 runtime/infer.py --bench --iters 500",
    notes: ["RKNN Toolkit2 >= 2.0 runs SiLU natively; v1.x falls back to CPU.", "rknn.config mean/std assume 0..255 input; adjust to the model's training normalization."],
  };
}

function hailo(c: Ctx): Omit<RecipeBundle, "runtime" | "runtimeLabel" | "precision"> {
  const arch = c.dev.id === "hailo8l" ? "hailo8l" : "hailo8";
  const convert = `#!/usr/bin/env bash
# ONNX -> HEF for ${c.dev.n} with the Hailo Dataflow Compiler (x86 Linux host,
# Hailo AI Software Suite). Hailo is INT8-only; calibration is mandatory.
set -euo pipefail
python calib/make_calib.py "\${CALIB_IMAGES:?set CALIB_IMAGES=path/to/images}" --n 512
hailo parser onnx ${c.onnx} --hw-arch ${arch} --har-path model.har
# If the parser stops at unsupported post-processing, re-run with
#   --end-node-names <last backbone/head node>   and do the post-process on the host.
hailo optimize model.har --hw-arch ${arch} --calib-set-path calib/calib.npy --output-har-path model_optimized.har
hailo compiler model_optimized.har --hw-arch ${arch}
mv -f model_optimized.hef model.hef 2>/dev/null || true
echo "built model.hef"
`;
  const body = `from hailo_platform import VDevice   # HailoRT >= 4.18 InferModel API

def load():
    vdevice = VDevice()
    infer_model = vdevice.create_infer_model("model.hef")
    configured = infer_model.configure()
    names = infer_model.output_names
    def run(x):
        bindings = configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(x[0]))
        outs = {n: np.empty(infer_model.output(n).shape, dtype=np.float32) for n in names}
        for n in names:
            bindings.output(n).set_buffer(outs[n])
        configured.run([bindings], 10000)
        return [outs[n] for n in names]
    return run`;
  const files: PackageFile[] = [
    { path: "convert/convert.sh", content: convert, description: "ONNX → HEF (Hailo Dataflow Compiler)" },
    { path: "runtime/infer.py", content: inferHarness({ title: `HailoRT inference + latency harness on ${c.dev.n}. Input: uint8 NHWC.`, io: c.io, layout: "nhwc", dtype: "uint8", targetFps: c.targetFps, body }), description: "HailoRT inference + latency benchmark" },
    reqs(["# hailort python wheel from the Hailo developer zone (match the driver version)", "numpy", "opencv-python-headless"]),
  ];
  const ref = deployRecipeText(c.dev.id, c.onnx);
  if (ref) files.push({ path: "reference/quickstart.sh", content: ref + "\n", description: "Original quick-start recipe (DeployRecipe panel)" });
  return {
    artifact: "model.hef", files,
    steps: [...(c.pt ? ["python convert/export_onnx.py"] : []), "CALIB_IMAGES=images/ bash convert/convert.sh   # x86 host", "python3 runtime/infer.py --bench   # on the device"],
    benchmarkCmd: "hailortcli run model.hef --measure-power",
    notes: ["Hailo has no external DRAM: the compiler fails if the model doesn't fit on-chip.", "On Raspberry Pi 5 set PCIe Gen3 (dtparam=pciex1_gen=3) for full throughput."],
  };
}

function tfliteConvert(c: Ctx, edgetpu: boolean): string {
  const variant = c.prec === "int8" ? "full_integer_quant" : c.prec === "fp16" ? "float16" : "float32";
  if (c.pt) {
    const s = CONV_SCRIPTS["pt→tflite"](convOpts({ inp: c.src, sz: c.io.h, i8: c.prec === "int8" }));
    return `#!/usr/bin/env bash
# PyTorch -> TFLite (${c.prec.toUpperCase()}) via the Ultralytics exporter${edgetpu ? ", then Edge TPU compile" : ""}.
set -euo pipefail
python - << 'EOF'
${s}${c.prec === "int8" ? '  # pass data="your_dataset.yaml" for representative calibration images' : ""}
EOF
${edgetpu ? 'edgetpu_compiler -s -o . $(ls -1 *_saved_model/*full_integer_quant.tflite | head -1)\nmv -f *_edgetpu.tflite model_edgetpu.tflite\necho "built model_edgetpu.tflite"' : `cp $(ls -1 *_saved_model/*${variant}.tflite | head -1) model.tflite\necho "built model.tflite"`}
`;
  }
  return `#!/usr/bin/env bash
# ONNX -> TF SavedModel -> TFLite (${c.prec.toUpperCase()}) with onnx2tf${edgetpu ? ", then Edge TPU compile" : ""}.
# onnx2tf converts NCHW to NHWC, so the TFLite model takes NHWC input.
set -euo pipefail
${c.prec === "int8"
    ? `# INT8 calibration data: calib/calib.npy (NHWC float32 0..1) from calib/make_calib.py.
onnx2tf -i ${c.onnx} -o saved_model -oiqt -qt per-channel \\
  -cind "${c.io.inputName}" calib/calib.npy "[[[[0.0,0.0,0.0]]]]" "[[[[1.0,1.0,1.0]]]]"`
    : `onnx2tf -i ${c.onnx} -o saved_model`}
${edgetpu
    ? `edgetpu_compiler -s -o . saved_model/*_full_integer_quant.tflite
mv -f *_edgetpu.tflite model_edgetpu.tflite
echo "built model_edgetpu.tflite (check the log: ops after the first unmapped op run on CPU)"`
    : `cp saved_model/*_${variant}.tflite model.tflite
echo "built model.tflite"`}
`;
}

function tfliteBody(edgetpu: boolean, delegateNote: string): string {
  return `try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate   # LiteRT (pip install ai-edge-litert)
except ImportError:
    from tflite_runtime.interpreter import Interpreter, load_delegate  # legacy tflite-runtime

def load():
    ${edgetpu
    ? `interp = Interpreter("model_edgetpu.tflite", experimental_delegates=[load_delegate("libedgetpu.so.1")])`
    : `# ${delegateNote}
    interp = Interpreter("model.tflite", num_threads=4)`}
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    scale, zero = inp["quantization"]
    def run(x):
        if inp["dtype"] in (np.int8, np.uint8) and scale:
            x = np.clip(np.round(x / scale + zero), np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max).astype(inp["dtype"])
        interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
        interp.invoke()
        res = []
        for o in outs:
            y = interp.get_tensor(o["index"]).astype(np.float32)
            s, z = o["quantization"]
            res.append((y - z) * s if s else y)
        return res
    return run`;
}

const ANDROID_KT = (prec: DeployPrecision, vendor: string) => `// Android (Kotlin) LiteRT inference skeleton. Gradle:
//   implementation("com.google.ai.edge.litert:litert:1.0.1")
//   implementation("com.google.ai.edge.litert:litert-gpu:1.0.1")
// ${vendor}
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Detector(modelBuffer: ByteBuffer) {
    private val gpu = if (CompatibilityList().isDelegateSupportedOnThisDevice) GpuDelegate() else null
    private val interpreter = Interpreter(modelBuffer, Interpreter.Options().apply {
        ${prec === "int8" ? "// INT8 models run best on the NPU via the vendor delegate (see note above); GPU/XNNPACK otherwise.\n        " : ""}gpu?.let { addDelegate(it) } ?: setNumThreads(4)
    })
    private val input = ByteBuffer.allocateDirect(interpreter.getInputTensor(0).numBytes()).order(ByteOrder.nativeOrder())

    fun run(pixels: FloatArray): Array<Any> {
        input.rewind()
        ${prec === "int8" ? "val q = interpreter.getInputTensor(0).quantizationParams()\n        pixels.forEach { input.put(((it / q.scale) + q.zeroPoint).toInt().coerceIn(-128, 127).toByte()) }" : "pixels.forEach { input.putFloat(it) }"}
        val out = Array(interpreter.outputTensorCount) { i -> ByteBuffer.allocateDirect(interpreter.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder()) as Any }
        interpreter.runForMultipleInputsOutputs(arrayOf(input), out.withIndex().associate { it.index to it.value })
        return out
    }

    fun close() { interpreter.close(); gpu?.close() }
}
`;

function tflite(c: Ctx, edgetpu: boolean): Omit<RecipeBundle, "runtime" | "runtimeLabel" | "precision"> {
  const vendor = /qualcomm/i.test(c.dev.v) ? "Qualcomm: add the QNN TFLite delegate (Qualcomm AI Engine Direct) to reach the Hexagon NPU."
    : /mediatek/i.test(c.dev.v) ? "MediaTek: use the NeuroPilot TFLite delegate to reach the APU."
    : /nxp/i.test(c.dev.v) ? "NXP: load the NPU external delegate shipped with the eIQ BSP."
    : "Use the platform's NPU delegate where available; GPU/XNNPACK otherwise.";
  const files: PackageFile[] = [
    { path: "convert/convert.sh", content: tfliteConvert(c, edgetpu), description: edgetpu ? "→ full-INT8 TFLite → Edge TPU compile" : `→ TFLite (${c.prec.toUpperCase()})` },
    { path: "runtime/infer.py", content: inferHarness({ title: `${edgetpu ? "Edge TPU" : "LiteRT/TFLite"} inference + latency harness on ${c.dev.n}. Input: NHWC.`, io: c.io, layout: "nhwc", dtype: "float32", targetFps: c.targetFps, body: tfliteBody(edgetpu, vendor) }), description: "TFLite inference + latency benchmark" },
    reqs([edgetpu ? "tflite-runtime  # + libedgetpu1-std from the Coral apt repo" : "ai-edge-litert", "numpy", "opencv-python-headless"]),
  ];
  if (!edgetpu) files.push({ path: "runtime/android/Detector.kt", content: ANDROID_KT(c.prec, vendor), description: "Android LiteRT inference skeleton (Kotlin)" });
  const tfl = edgetpu ? "model_edgetpu.tflite" : "model.tflite";
  return {
    artifact: tfl, files,
    steps: [...(c.prec === "int8" && !c.pt ? ["python calib/make_calib.py images/   # NHWC calib.npy"] : []), "bash convert/convert.sh", "python runtime/infer.py --bench"],
    benchmarkCmd: edgetpu
      ? "python runtime/infer.py --bench --iters 500"
      : `adb push ${tfl} /data/local/tmp/ && adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/${tfl} --use_gpu=true --num_runs=200`,
    notes: edgetpu
      ? ["Edge TPU compiles one partition: everything after the first unsupported op runs on the host CPU.", "Keep weights under ~6.9 MiB to stay in the on-chip cache."]
      : [vendor, "Android 15 deprecates NNAPI; prefer vendor delegates or the GPU delegate."],
  };
}

function qnn(c: Ctx): Omit<RecipeBundle, "runtime" | "runtimeLabel" | "precision"> {
  const convert = `#!/usr/bin/env bash
# ONNX -> DLC for ${c.dev.n} with the Qualcomm AI Engine Direct / SNPE tools
# (x86 Linux host, SDK env sourced: source $SNPE_ROOT/bin/envsetup.sh).
# Tool names/flags follow SNPE 2.x; check them against your SDK version.
set -euo pipefail
snpe-onnx-to-dlc --input_network ${c.onnx} --output_path model.dlc
${c.prec === "int8"
    ? `# INT8 for the Hexagon HTP: calib/raw_list.txt lists raw float32 NHWC tensors (one per line).
snpe-dlc-quant --input_dlc model.dlc --input_list calib/raw_list.txt --output_dlc model_q.dlc
snpe-dlc-graph-prepare --input_dlc model_q.dlc --output_dlc model_htp.dlc --htp_socs ${c.dev.id === "rb5" ? "sm8250" : "qcs6490"}`
    : `# FP16 runs on the GPU runtime; the HTP needs a quantized DLC.
cp model.dlc model_htp.dlc`}
echo "built model_htp.dlc"
`;
  const run = `#!/usr/bin/env bash
# Run + time on the device (adb). Copy the SNPE runtime libs for the target arch first.
set -euo pipefail
adb push model_htp.dlc /data/local/tmp/app/
adb shell "cd /data/local/tmp/app && snpe-throughput-net-run --container model_htp.dlc --duration 30 --use_${c.prec === "int8" ? "dsp" : "gpu"} --perf_profile burst"
`;
  return {
    artifact: "model_htp.dlc",
    files: [
      { path: "convert/convert.sh", content: convert, description: "ONNX → DLC (Qualcomm SNPE/QNN tools)" },
      { path: "runtime/run_on_device.sh", content: run, description: "On-device throughput run (adb)" },
    ],
    steps: [...(c.pt ? ["python convert/export_onnx.py"] : []), "bash convert/convert.sh", "bash runtime/run_on_device.sh"],
    benchmarkCmd: "snpe-throughput-net-run --container model_htp.dlc --duration 30 --use_dsp",
    notes: ["Qualcomm tool names and flags change between SDK releases; verify against your SDK docs."],
  };
}

function ort(c: Ctx): Omit<RecipeBundle, "runtime" | "runtimeLabel" | "precision"> {
  const ep = /nvidia/i.test(c.dev.v) ? '["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]'
    : /intel/i.test(c.dev.v) ? '["OpenVINOExecutionProvider", "CPUExecutionProvider"]'
    : '["CPUExecutionProvider"]';
  const convert = c.prec === "fp16"
    ? CONV_SCRIPTS["onnx→onnx-fp16"](convOpts({ inp: c.onnx, out: "model_deploy.onnx" })) + "\n"
    : c.prec === "int8"
      ? `"""Static INT8 (QDQ) quantization with ONNX Runtime."""
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

class Reader(CalibrationDataReader):
    def __init__(self):
        import onnxruntime as ort
        self.name = ort.InferenceSession("${c.onnx}").get_inputs()[0].name
        self.it = iter(np.load("calib/calib.npy"))
    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {self.name: x[None].astype(np.float32)}

quantize_static("${c.onnx}", "model_deploy.onnx", Reader(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
print("saved model_deploy.onnx")
`
      : `import shutil\nshutil.copy("${c.onnx}", "model_deploy.onnx")\nprint("FP32: model used as-is")\n`;
  const body = `import onnxruntime as ort

def load():
    sess = ort.InferenceSession("model_deploy.onnx", providers=${ep})
    name = sess.get_inputs()[0].name
    def run(x):
        return sess.run(None, {name: x.astype(np.float32)})
    return run`;
  return {
    artifact: "model_deploy.onnx",
    files: [
      { path: "convert/convert.py", content: convert, description: `ONNX ${c.prec.toUpperCase()} preparation` },
      { path: "runtime/infer.py", content: inferHarness({ title: `ONNX Runtime inference + latency harness on ${c.dev.n}.`, io: c.io, layout: "nchw", dtype: "float32", targetFps: c.targetFps, body }), description: "ONNX Runtime inference + latency benchmark" },
      reqs([/nvidia/i.test(c.dev.v) ? "onnxruntime-gpu  # Jetson: use the JetPack wheel" : /intel/i.test(c.dev.v) ? "onnxruntime-openvino" : "onnxruntime", ...(c.prec === "fp16" ? ["onnx", "onnxconverter-common"] : []), "numpy", "opencv-python-headless"]),
    ],
    steps: [...(c.pt ? ["python convert/export_onnx.py"] : []), ...(c.prec === "int8" ? ["python calib/make_calib.py images/"] : []), "python convert/convert.py", "python runtime/infer.py --bench"],
    benchmarkCmd: "python runtime/infer.py --bench --iters 500",
    notes: ["ONNX Runtime is the portable path; the native runtime is usually faster on this device."],
  };
}

function vendor(c: Ctx): Omit<RecipeBundle, "runtime" | "runtimeLabel" | "precision"> {
  const tc = c.dev.toolchain.join(" / ");
  const convert = `#!/usr/bin/env bash
# ${c.dev.n}: compile ${c.onnx} with ${tc}.
# ModelVisio does not ship this vendor toolchain's exact CLI; follow the vendor
# SDK's ONNX import -> ${c.prec.toUpperCase()} quantize -> compile flow:
#   1. Import:     ${c.onnx} (opset 13-17 is the safest range for most vendor SDKs)
#   2. Quantize:   ${c.prec === "int8" ? "INT8 with calib/calib.npy (200-1000 representative images)" : `${c.prec.toUpperCase()} (no calibration needed)`}
#   3. Compile:    target = ${c.dev.n}
#   4. Deploy:     copy the compiled artifact + the vendor runtime to the device
set -euo pipefail
echo "Edit convert/convert.sh with the ${tc} commands for your SDK version." >&2
exit 1
`;
  return {
    artifact: `${c.dev.toolchain[0]} artifact`,
    files: [{ path: "convert/convert.sh", content: convert, description: `Vendor-SDK (${tc}) compile outline` }],
    steps: [...(c.pt ? ["python convert/export_onnx.py"] : []), ...(c.prec === "int8" ? ["python calib/make_calib.py images/"] : []), `bash convert/convert.sh   # fill in the ${tc} commands`],
    benchmarkCmd: `(vendor profiler for ${tc})`,
    notes: [`${c.dev.n} uses ${tc}; commands are an outline, not a verified script.`],
  };
}

/** Generate the full recipe for one deployment choice. */
export function generateRecipe(opts: {
  model: Model; device: DeviceSpec; precision: DeployPrecision; runtime?: RuntimeId; targetFps?: number;
}): RecipeBundle {
  const { model, device: dev, precision: prec } = opts;
  const runtime = opts.runtime && runtimesFor(dev).includes(opts.runtime) ? opts.runtime : runtimesFor(dev)[0];
  const pt = isPytorchSource(model);
  const src = model.name || "model.onnx";
  const onnx = pt ? `${stem(src)}.onnx` : /onnx/i.test(model.format) ? src : `${stem(src)}.onnx`;
  const c: Ctx = { model, dev, prec, io: ioOf(model), src, onnx, pt, targetFps: opts.targetFps };

  const r = runtime === "tensorrt" ? tensorrt(c)
    : runtime === "openvino" ? openvino(c)
    : runtime === "rknn" ? rknn(c)
    : runtime === "hailort" ? hailo(c)
    : runtime === "tflite-edgetpu" ? tflite(c, true)
    : runtime === "tflite" ? tflite(c, false)
    : runtime === "qnn" ? qnn(c)
    : runtime === "onnxruntime" ? ort(c)
    : vendor(c);

  const nhwc = runtime === "tflite" || runtime === "tflite-edgetpu";
  const before: PackageFile[] = [
    ...exportStep(c),
    ...(prec === "int8" ? [{ path: "calib/make_calib.py", content: calibScript(c.io, nhwc ? "nhwc" : "nchw"), description: "Build the INT8 calibration set" }] : []),
  ];
  const after: PackageFile[] = [
    { path: "validate/reference.py", content: referenceScript(onnx, c.io), description: "FP32 reference outputs (ONNX Runtime)" },
    { path: "validate/accuracy_check.py", content: ACCURACY_CHECK, description: "Deployed-vs-reference accuracy gate" },
  ];
  if (!/onnx/i.test(model.format) && !pt) {
    r.notes.unshift(`Source format is ${model.format}; these recipes start from ONNX. Export ${onnx} from the training framework first.`);
  }
  return {
    runtime, runtimeLabel: RUNTIMES[runtime].label, precision: prec,
    ...r,
    files: [...before, ...r.files, ...after],
  };
}
