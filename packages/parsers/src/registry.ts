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
export const FORMAT_SUPPORT: FormatInfo[] = [
  { id: "onnx", label: "ONNX", status: "full", exts: ["onnx"] },
  { id: "tflite", label: "TFLite", status: "full", exts: ["tflite"] },
  { id: "safetensors", label: "Safetensors", status: "full", exts: ["safetensors"] },
  { id: "gguf", label: "GGUF", status: "full", exts: ["gguf"] },
  { id: "numpy", label: "NumPy", status: "full", exts: ["npy", "npz"] },
  { id: "darknet", label: "Darknet", status: "full", exts: ["cfg"] },
  { id: "pytorch", label: "PyTorch", status: "full", exts: ["pt", "pth", "ckpt", "bin"] },
  { id: "coreml", label: "Core ML", status: "metadata", exts: ["mlmodel", "mlpackage"] },
  { id: "openvino", label: "OpenVINO", status: "metadata", exts: ["xml"] },
  { id: "tensorflow", label: "TensorFlow", status: "metadata", exts: ["pb"] },
  { id: "caffe", label: "Caffe", status: "metadata", exts: ["caffemodel", "prototxt"] },
  { id: "paddle", label: "PaddlePaddle", status: "metadata", exts: ["pdmodel"] },
  { id: "ncnn", label: "ncnn", status: "metadata", exts: ["param"] },
  { id: "rknn", label: "RKNN", status: "metadata", exts: ["rknn"] },
  { id: "mnn", label: "MNN", status: "metadata", exts: ["mnn"] },
  { id: "mlir", label: "MLIR", status: "metadata", exts: ["mlir"] },
  { id: "sklearn", label: "scikit-learn", status: "metadata", exts: ["pkl", "joblib"] },
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
