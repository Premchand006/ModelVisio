import type { Model } from "./types";
import { parseOnnx } from "./onnx/parse";
import { parseSafetensors } from "./formats/safetensors";
import { parseNumpy } from "./formats/numpy";
import { parseGguf } from "./formats/gguf";
import { parseDarknet } from "./formats/darknet";
import { parseTflite } from "./formats/tflite";
import { parsePytorch } from "./formats/pytorch";
import { parseContainer } from "./formats/container";
import { parseOpenvino, isOpenvinoXml } from "./formats/openvino";
import { parseCaffe } from "./formats/caffe";
import { parseCoreml, isCoremlPackageZip } from "./formats/coreml";
import { parseTensorflow } from "./formats/tensorflow";
import { parseKeras, parseKerasJson, isKerasJson, isHdf5 } from "./formats/keras";
import { parsePaddle } from "./formats/paddle";
import { parseNcnn, isNcnnParam } from "./formats/ncnn";
import { parseMnn } from "./formats/mnn";
import { parseRknn, isRknn } from "./formats/rknn";
import { parseMlir } from "./formats/mlir";
import { parseSklearn, isSklearnPickle } from "./formats/sklearn";
import { isZip } from "./common/zip";
import { annotateCompat } from "./compat";
import { companionNames, findCompanion, tfjsShardPaths, type Companions } from "./companions";
import { shardIndex } from "./common/tensorBundle";

export type ParserStatus = "full" | "metadata";

export type FormatInfo = { id: string; label: string; status: ParserStatus; exts: string[] };

// Which formats are fully parsed vs detection/metadata-only (honest UI signal).
// Coverage mirrors Netron's breadth: any of these file types can be opened. The
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
  { id: "coreml", label: "Core ML", status: "full", exts: ["mlmodel", "mlpackage"] },
  { id: "openvino", label: "OpenVINO", status: "full", exts: ["xml"] },
  { id: "tensorflow", label: "TensorFlow", status: "full", exts: ["pb", "meta", "pbtxt"] },
  { id: "keras", label: "Keras", status: "full", exts: ["keras", "h5", "hdf5"] },
  { id: "caffe", label: "Caffe", status: "full", exts: ["caffemodel", "prototxt"] },
  { id: "paddle", label: "PaddlePaddle (.pdmodel)", status: "full", exts: ["pdmodel"] },
  { id: "ncnn", label: "ncnn", status: "full", exts: ["param"] },
  { id: "mnn", label: "MNN", status: "full", exts: ["mnn"] },
  { id: "rknn", label: "RKNN", status: "full", exts: ["rknn"] },
  { id: "mlir", label: "MLIR", status: "full", exts: ["mlir"] },
  { id: "sklearn", label: "scikit-learn", status: "full", exts: ["pkl", "joblib", "pickle"] },
  // ── Detected + metadata (graph parser incremental) ─────────────────────────
  { id: "onnxruntime", label: "ONNX Runtime", status: "metadata", exts: ["ort"] },
  { id: "torchscript", label: "TorchScript", status: "metadata", exts: ["ptl", "torchscript"] },
  { id: "torchexport", label: "torch.export", status: "metadata", exts: ["pt2"] },
  { id: "executorch", label: "ExecuTorch", status: "metadata", exts: ["pte"] },
  { id: "coremlc", label: "Core ML (compiled)", status: "metadata", exts: ["mlmodelc"] },
  { id: "tfjs", label: "TensorFlow.js", status: "metadata", exts: ["json"] },
  { id: "mxnet", label: "MXNet", status: "metadata", exts: ["params"] },
  { id: "paddleparams", label: "PaddlePaddle params / Lite", status: "metadata", exts: ["pdparams", "nb"] },
  { id: "tnn", label: "TNN", status: "metadata", exts: ["tnnproto", "tnnmodel"] },
  { id: "tensorrt", label: "TensorRT", status: "metadata", exts: ["engine", "plan", "trt"] },
  { id: "uff", label: "UFF", status: "metadata", exts: ["uff"] },
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
  // .param and .param.bin (ext "bin" would otherwise go to PyTorch).
  if ((e === "param" || e === "bin") && isNcnnParam(buf)) return "ncnn";
  if ((e === "rknn" || buf[0] === 0x52) && isRknn(buf)) return "rknn";
  if (e === "pkl" || e === "pickle" || e === "joblib") {
    if (isZip(buf)) return "pytorch"; // torch.save zip archive
    if (isSklearnPickle(buf)) return "sklearn";
    return e === "joblib" ? "sklearn" : "unknown"; // generic pickle → metadata
  }
  if (e === "xml") return isOpenvinoXml(buf) ? "openvino" : "unknown";
  if ((e === "zip" || e === "mlpackage") && isCoremlPackageZip(buf)) return "coreml";
  if (isHdf5(buf) && (e === "h5" || e === "hdf5" || e === "keras")) return "keras";
  const hit = FORMAT_SUPPORT.find((f) => f.exts.includes(e));
  return hit?.id ?? "unknown";
}

/**
 * Parse any supported model file into the normalized Model. Fully-parsed
 * formats produce real graphs; others return detected metadata (see
 * FORMAT_SUPPORT). Pure function; safe to call from a Worker.
 */
export function parseModel(input: ArrayBuffer | Uint8Array, name: string, companions?: Companions): Model {
  const buf = input instanceof Uint8Array ? input : new Uint8Array(input);
  return annotateCompat(withWeightsNote(parseByFormat(buf, name, companions), buf, name, companions));
}

/**
 * When a format keeps its weights in a companion file the caller didn't supply,
 * say so: the graph and parameter counts are complete, the weight statistics
 * (min/max/mean/sparsity) are not.
 */
function withWeightsNote(m: Model, buf: Uint8Array, name: string, companions?: Companions): Model {
  if (!/OpenVINO|ncnn|Paddle|TensorFlow\.js|SavedModel/i.test(m.format)) return m;
  const wanted = companionNames(name, buf);
  if (!wanted.length || wanted.every((w) => findCompanion(companions, [w]))) return m;
  const missing = wanted.filter((w) => !findCompanion(companions, [w]));
  const note = `Weight statistics need ${missing.join(" + ")} from the same folder as ${name}. Open them together (multi-select or drop all files) to see weight min/max/mean/sparsity; the graph, shapes and parameter counts are already complete.`;
  return { ...m, notes: [...(m.notes ?? []), note] };
}

function parseByFormat(buf: Uint8Array, name: string, companions?: Companions): Model {
  const stem = name.replace(/\.[^.]+$/, "");
  switch (detectFormat(name, buf)) {
    case "onnx": return parseOnnx(buf, { name });
    case "tflite": return parseTflite(buf, name);
    case "safetensors": return parseSafetensors(buf, name);
    case "gguf": return parseGguf(buf, name);
    case "numpy": return parseNumpy(buf, name);
    case "darknet": return parseDarknet(new TextDecoder().decode(buf), name);
    case "pytorch": return parsePytorch(buf, name);
    case "openvino": return parseOpenvino(buf, name, findCompanion(companions, [`${stem}.bin`], [".bin"]));
    case "caffe": return parseCaffe(buf, name);
    case "coreml": return parseCoreml(buf, name);
    case "tensorflow": {
      // SavedModel checkpoint: variables/variables.index + variables.data-NNNNN-of-MMMMM.
      const index = findCompanion(companions, ["variables/variables.index"], [".index"]);
      const shards: Uint8Array[] = [];
      for (const k of Object.keys(companions ?? {})) {
        const i = shardIndex(k);
        if (i != null) shards[i] = findCompanion(companions, [k])!;
      }
      return parseTensorflow(buf, name, index ? { index, shards } : undefined);
    }
    // A .h5 may hold weights only (no model_config): fall back to metadata.
    case "keras": try { return parseKeras(buf, name); } catch { return parseContainer(buf, name); }
    case "paddle": return parsePaddle(buf, name, findCompanion(companions, [`${stem}.pdiparams`], [".pdiparams"]));
    case "ncnn": try {
      const bin = findCompanion(companions, companionNames(name), [".bin"]);
      return parseNcnn(buf, name, bin && bin !== buf ? { bin } : {});
    } catch { return parseContainer(buf, name); }
    case "mnn": try { return parseMnn(buf, name); } catch { return parseContainer(buf, name); }
    case "rknn": return parseRknn(buf, name);
    case "mlir": try { return parseMlir(buf, name); } catch { return parseContainer(buf, name); }
    case "sklearn": return parseSklearn(buf, name);
    case "tfjs": {
      const text = new TextDecoder().decode(buf);
      if (!isKerasJson(text)) return parseContainer(buf, name);
      const shards: Record<string, Uint8Array> = {};
      for (const p of tfjsShardPaths(text)) {
        const b = findCompanion(companions, [p]);
        if (b) shards[p] = b;
      }
      return parseKerasJson(text, name, shards);
    }
    default: return parseContainer(buf, name);
  }
}
