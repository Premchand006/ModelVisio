import { writeZip, type ZipInput } from "@modelvisio/parsers";

// A "conversion kit" is a ZIP the user runs locally: their model + the exact
// conversion script + requirements.txt + Dockerfile + run.sh + README. Native
// toolchains (TensorRT/TFLite/CoreML/OpenVINO/RKNN/ncnn) can't run in a browser,
// so this gives a one-command path to the converted model.
type Kit = { deps: string[]; base: string; out: string; gpu?: boolean };

export const KIT_META: Record<string, Kit> = {
  "pt→onnx": { deps: ["ultralytics", "onnx", "onnxslim"], base: "python:3.11-slim", out: "*.onnx" },
  "pt→tflite": { deps: ["ultralytics", "tensorflow", "onnx2tf", "onnx", "onnxslim", "onnxruntime"], base: "python:3.11", out: "*_saved_model / *.tflite" },
  "pt→coreml": { deps: ["ultralytics", "coremltools"], base: "python:3.11", out: "*.mlpackage" },
  "onnx→tensorrt": { deps: [], base: "nvcr.io/nvidia/tensorrt:24.01-py3", out: "model.engine", gpu: true },
  "onnx→openvino": { deps: ["openvino"], base: "python:3.11", out: "model.xml + model.bin" },
  "onnx→rknn": { deps: ["rknn-toolkit2"], base: "python:3.10", out: "model.rknn" },
  "onnx→ncnn": { deps: ["ncnn"], base: "python:3.11", out: "model.param + model.bin" },
  "onnx→onnx-fp16": { deps: ["onnx", "onnxconverter-common"], base: "python:3.11", out: "model_fp16.onnx" },
  "onnx→tfjs": { deps: ["onnx2tf", "tensorflow", "tensorflowjs", "onnx", "onnxslim"], base: "python:3.11", out: "tfjs_model/" },
  "pt→torchscript": { deps: ["ultralytics", "torch"], base: "python:3.11", out: "*.torchscript" },
  "pt→executorch": { deps: ["ultralytics", "executorch"], base: "python:3.11", out: "*_executorch_model/" },
};

const enc = (s: string) => new TextEncoder().encode(s);
const base = (n: string) => n.replace(/\.[^.]+$/, "");

function dockerfile(k: Kit): string {
  return [
    `FROM ${k.base}`,
    "WORKDIR /work",
    "COPY . /work",
    k.deps.length ? `RUN pip install --no-cache-dir ${k.deps.join(" ")}` : "# (toolchain provided by the base image)",
    'CMD ["python", "convert.py"]',
    "",
  ].join("\n");
}

function runsh(k: Kit): string {
  return [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    k.deps.length ? `pip install ${k.deps.join(" ")}` : "# ensure the target toolchain is on PATH",
    "python convert.py",
    "",
  ].join("\n");
}

function readme(target: string, k: Kit, model: string): string {
  const dockerRun = k.gpu
    ? 'docker run --rm --gpus all -v "$PWD:/work" emstudio-convert'
    : 'docker run --rm -v "$PWD:/work" emstudio-convert';
  return [
    `# ModelVisio — ${target} conversion kit`,
    "",
    `Converts **${model}** (${target}). This uses native toolchains, so it runs`,
    "locally — not in the browser.",
    "",
    `**Output:** \`${k.out}\``,
    "",
    "## Option A — Docker (recommended)",
    "```bash",
    "docker build -t emstudio-convert .",
    dockerRun,
    "```",
    "",
    "## Option B — local Python",
    "```bash",
    k.deps.length ? `pip install ${k.deps.join(" ")}` : "# install the target toolchain",
    "python convert.py",
    "```",
    "",
    k.gpu ? "> Requires an NVIDIA GPU, drivers, and the NVIDIA Container Toolkit." : "",
    "",
  ].join("\n");
}

export function buildKit(
  target: string,
  scriptText: string,
  modelBytes: ArrayBuffer | null,
  modelName: string,
): { filename: string; data: Uint8Array } {
  const k = KIT_META[target] ?? { deps: [], base: "python:3.11", out: "output" };
  const model = modelName || "model";
  const entries: ZipInput[] = [];
  if (modelBytes) entries.push({ name: model, data: new Uint8Array(modelBytes) });
  entries.push({ name: "convert.py", data: enc(scriptText) });
  if (k.deps.length) entries.push({ name: "requirements.txt", data: enc(k.deps.join("\n") + "\n") });
  entries.push({ name: "Dockerfile", data: enc(dockerfile(k)) });
  entries.push({ name: "run.sh", data: enc(runsh(k)) });
  entries.push({ name: "README.md", data: enc(readme(target, k, model)) });
  return { filename: `${base(model)}-${target.replace("→", "-to-")}-kit.zip`, data: writeZip(entries) };
}
