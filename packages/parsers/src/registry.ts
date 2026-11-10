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

export { FORMAT_SUPPORT, type FormatInfo, type ParserStatus } from "./formatSupport";
import { FORMAT_SUPPORT } from "./formatSupport";

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
