import type { Model } from "./types";
import { parseOnnx } from "./onnx/parse";
import { parseSafetensors } from "./formats/safetensors";
import { parseNumpy } from "./formats/numpy";
import { parseGguf } from "./formats/gguf";
import { parseDarknet } from "./formats/darknet";
import { parseTflite } from "./formats/tflite";
import { parsePytorch } from "./formats/pytorch";
import { parseContainer } from "./formats/container";

export type ParserStatus = "full" | "metadata";

export type FormatInfo = { id: string; label: string; status: ParserStatus; exts: string[] };

// Which formats are fully parsed vs detection/metadata-only (honest UI signal).
// Coverage mirrors Netron's breadth: any of these file types can be opened — the
// "full" ones render a real graph, the "metadata" ones are detected and surface
// real file metadata (never a fabricated graph) until a deep parser lands.
export const FORMAT_SUPPORT: FormatInfo[] = [
  // ── Fully parsed (real graph) ──────────────────────────────────────────────
  { id: "onnx", label: "ONNX", status: "full", exts: ["onnx"] },
  { id: "tflite", label: "TFLite", status: "full", exts: ["tflite", "lite", "tfl"] },
  { id: "safetensors", label: "Safetensors", status: "full", exts: ["safetensors"] },
  { id: "gguf", label: "GGUF", status: "full", exts: ["gguf", "ggml"] },
  { id: "numpy", label: "NumPy", status: "full", exts: ["npy", "npz"] },
  { id: "darknet", label: "Darknet", status: "full", exts: ["cfg"] },
  { id: "pytorch", label: "PyTorch", status: "full", exts: ["pt", "pth", "ckpt", "bin"] },
  // ── Detected + metadata (graph parser incremental) ─────────────────────────
  { id: "onnxruntime", label: "ONNX Runtime", status: "metadata", exts: ["ort"] },
  { id: "torchscript", label: "TorchScript", status: "metadata", exts: ["ptl", "torchscript"] },
  { id: "torchexport", label: "torch.export", status: "metadata", exts: ["pt2"] },
  { id: "executorch", label: "ExecuTorch", status: "metadata", exts: ["pte"] },
  { id: "coreml", label: "Core ML", status: "metadata", exts: ["mlmodel", "mlpackage", "mlmodelc"] },
  { id: "openvino", label: "OpenVINO", status: "metadata", exts: ["xml"] },
  { id: "tensorflow", label: "TensorFlow", status: "metadata", exts: ["pb", "meta", "pbtxt"] },
  { id: "keras", label: "Keras", status: "metadata", exts: ["keras", "h5", "hdf5"] },
  { id: "tfjs", label: "TensorFlow.js", status: "metadata", exts: ["json"] },
  { id: "mxnet", label: "MXNet", status: "metadata", exts: ["params"] },
  { id: "caffe", label: "Caffe", status: "metadata", exts: ["caffemodel", "prototxt"] },
  { id: "paddle", label: "PaddlePaddle", status: "metadata", exts: ["pdmodel", "pdparams", "nb"] },
  { id: "ncnn", label: "ncnn", status: "metadata", exts: ["param"] },
  { id: "mnn", label: "MNN", status: "metadata", exts: ["mnn"] },
  { id: "tnn", label: "TNN", status: "metadata", exts: ["tnnproto", "tnnmodel"] },
  { id: "rknn", label: "RKNN", status: "metadata", exts: ["rknn"] },
  { id: "tensorrt", label: "TensorRT", status: "metadata", exts: ["engine", "plan", "trt"] },
  { id: "uff", label: "UFF", status: "metadata", exts: ["uff"] },
  { id: "mlir", label: "MLIR", status: "metadata", exts: ["mlir"] },
  { id: "cntk", label: "CNTK", status: "metadata", exts: ["cntk", "dnn"] },
  { id: "barracuda", label: "Barracuda", status: "metadata", exts: ["nn"] },
  { id: "megengine", label: "MegEngine", status: "metadata", exts: ["mge", "tm"] },
  { id: "nnabla", label: "NNabla", status: "metadata", exts: ["nntxt"] },
  { id: "hailo", label: "Hailo", status: "metadata", exts: ["har", "hn"] },
  { id: "om", label: "Huawei CANN (OM)", status: "metadata", exts: ["om"] },
  { id: "mlnet", label: "ML.NET", status: "metadata", exts: ["mlnet"] },
  { id: "bigdl", label: "BigDL", status: "metadata", exts: ["bigdl"] },
  { id: "catboost", label: "CatBoost", status: "metadata", exts: ["cbm"] },
  { id: "darknetw", label: "Darknet weights", status: "metadata", exts: ["weights"] },
  { id: "sklearn", label: "scikit-learn", status: "metadata", exts: ["pkl", "joblib", "pickle"] },
];

function ext(name: string): string {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

function magicEquals(buf: Uint8Array, bytes: number[]): boolean {
  return bytes.every((b, i) => buf[i] === b);
}

/** Best-effort format id from extension, refined by magic bytes. */
export function detectFormat(name: string, buf: Uint8Array): string {
  const e = ext(name);
  // Magic-byte overrides (extension can lie).
  if (buf.length >= 4) {
    if (magicEquals(buf, [0x47, 0x47, 0x55, 0x46])) return "gguf"; // GGUF
    if (magicEquals(buf, [0x93, 0x4e, 0x55, 0x4d])) return "numpy"; // \x93NUM(PY)
  }
  const hit = FORMAT_SUPPORT.find((f) => f.exts.includes(e));
  return hit?.id ?? "unknown";
}

/**
 * Parse any supported model file into the normalized Model. Fully-parsed
 * formats produce real graphs; others return detected metadata (see
 * FORMAT_SUPPORT). Pure function; safe to call from a Worker.
 */
export function parseModel(input: ArrayBuffer | Uint8Array, name: string): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  switch (detectFormat(name, buf)) {
    case "onnx": return parseOnnx(buf, { name });
    case "tflite": return parseTflite(buf, name);
    case "safetensors": return parseSafetensors(buf, name);
    case "gguf": return parseGguf(buf, name);
    case "numpy": return parseNumpy(buf, name);
    case "darknet": return parseDarknet(new TextDecoder().decode(buf), name);
    case "pytorch": return parsePytorch(buf, name);
    default: return parseContainer(buf, name);
  }
}
