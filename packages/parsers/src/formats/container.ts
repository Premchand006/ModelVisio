import type { Model, ModelLayer } from "../types";
import { isZip, readZipEntries } from "../common/zip";
import { weightsToModel, type TensorInfo } from "../common/weights";

// Honest handling for formats whose deep graph parsing isn't implemented yet
// (PyTorch pickle/zip, Core ML / TF / Caffe / Paddle protobuf, OpenVINO XML,
// RKNN/ncnn/MNN binaries, …). We DETECT the format and surface real metadata —
// never a fabricated graph.

const LABELS: Record<string, { format: string; framework: string }> = {
  pt: { format: "PyTorch", framework: "PyTorch" },
  pth: { format: "PyTorch", framework: "PyTorch" },
  ckpt: { format: "PyTorch checkpoint", framework: "PyTorch" },
  bin: { format: "PyTorch bin", framework: "PyTorch/HF" },
  mlmodel: { format: "Core ML", framework: "Core ML" },
  mlpackage: { format: "Core ML package", framework: "Core ML" },
  pb: { format: "TensorFlow", framework: "TensorFlow" },
  caffemodel: { format: "Caffe", framework: "Caffe" },
  prototxt: { format: "Caffe prototxt", framework: "Caffe" },
  xml: { format: "OpenVINO IR", framework: "OpenVINO" },
  pdmodel: { format: "PaddlePaddle", framework: "Paddle" },
  rknn: { format: "RKNN", framework: "Rockchip" },
  param: { format: "ncnn", framework: "ncnn" },
  mnn: { format: "MNN", framework: "Alibaba MNN" },
  mlir: { format: "MLIR", framework: "MLIR" },
  pkl: { format: "scikit-learn (pickle)", framework: "scikit-learn" },
  joblib: { format: "joblib", framework: "scikit-learn" },
};

function ext(name: string): string {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

export function parseContainer(input: ArrayBuffer | Uint8Array, name: string): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  const e = ext(name);
  const label = LABELS[e] ?? { format: e.toUpperCase() || "Unknown", framework: "unknown" };

  // PyTorch (and other) ZIP archives: list the real member entries.
  if (isZip(buf)) {
    const entries = readZipEntries(buf);
    const isTorch = entries.some((x) => /(^|\/)(data|constants)\.pkl$/.test(x.name) || /\/data\//.test(x.name));
    const tensors: TensorInfo[] = entries
      .filter((x) => !x.name.endsWith("/"))
      .map((x) => ({ name: x.name, shape: [], dtype: x.method === 0 ? "stored" : "deflate", bytes: x.size }));
    return weightsToModel(
      {
        name,
        format: isTorch ? "PyTorch (zip)" : `${label.format} (zip)`,
        framework: label.framework,
        sizeBytes: buf.byteLength,
        note: isTorch
          ? "PyTorch zip archive — tensor storages listed. Full graph needs unpickling (not yet implemented)."
          : "Archive members listed. Full graph parsing not yet implemented.",
      },
      tensors,
    );
  }

  // Everything else: one honest info node with detected format + magic + size.
  const magic = [...buf.subarray(0, 4)].map((b) => b.toString(16).padStart(2, "0")).join(" ");
  const node: ModelLayer = {
    id: 0,
    name,
    type: label.format,
    op: "Detected",
    shape: "?",
    dt: "—",
    params: 0,
    flops: 0,
    macs: 0,
    mem: buf.byteLength,
    group: "input",
    ins: [],
    outs: [],
    attr: { format: label.format, sizeBytes: buf.byteLength, magic },
    w: null,
    math: `Magic bytes: ${magic}`,
    insight: `Detected ${label.format}. Full graph parsing for this format is not yet implemented — ONNX, TFLite, Safetensors, GGUF, NumPy and Darknet are fully parsed today.`,
    qSens: 0,
    compIssues: [],
  };
  return {
    name, format: label.format, framework: label.framework, opset: 0,
    sizeBytes: buf.byteLength, inputShape: [], outputShape: [],
    layers: [node], edges: [],
  };
}
