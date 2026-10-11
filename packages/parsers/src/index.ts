// @modelvisio/parsers: real model-format parsing → normalized graph objects.
// Pure TS, no React.

export type { Model, ModelLayer } from "./types";

// Unified entry point: detects the format and parses to the normalized Model.
export { parseModel, detectFormat, FORMAT_SUPPORT, type FormatInfo, type ParserStatus } from "./registry";

// Companion weight files (OpenVINO .bin, ncnn .bin, Paddle .pdiparams, ...).
export { companionNames, findCompanion, pickPrimary, type Companions } from "./companions";

// Rule-based per-op compatibility notes (parseModel applies these already).
export { annotateCompat, layerCompat, normOp, COMPAT_TARGETS, type CompatTarget } from "./compat";

// Individual parsers (also usable directly / in tests).
export { parseOnnx, type ParseOptions } from "./onnx/parse";
export { parseTflite } from "./formats/tflite";
export { parseSafetensors } from "./formats/safetensors";
export { parseNumpy } from "./formats/numpy";
export { parseGguf } from "./formats/gguf";
export { parseDarknet } from "./formats/darknet";
export { parsePytorch } from "./formats/pytorch";
export { parseContainer } from "./formats/container";
export { parseOpenvino } from "./formats/openvino";
export { parseCaffe, parseCaffePrototxt, parseCaffemodel } from "./formats/caffe";
export { parseCoreml } from "./formats/coreml";
export { parseTensorflow } from "./formats/tensorflow";
export { parseKeras, parseKerasJson } from "./formats/keras";
export { parsePaddle } from "./formats/paddle";
export { parseNcnn } from "./formats/ncnn";
export { parseMnn } from "./formats/mnn";
export { parseRknn } from "./formats/rknn";
export { parseMlir } from "./formats/mlir";
export { parseSklearn } from "./formats/sklearn";

// Progressive (streaming) parsing for large models.
export {
  parseModelStream, parseModelProgressive, STREAM_THRESHOLD_BYTES, DEFAULT_STREAM_BATCH, STREAMING_FORMATS,
  type ParseChunk, type ModelHeader, type ModelStream, type StreamOptions,
} from "./stream";
export { onnxStream } from "./onnx/parse";
export { tfliteStream } from "./formats/tflite";
export { ggufStream } from "./formats/gguf";
export { safetensorsStream } from "./formats/safetensors";

// Model conversion (browser-side data transforms) + ZIP writer for kits.
export { convertWeights, toGraphJson, toLayerCsv, type ConvertResult, type ConvTarget, type Precision } from "./convert";
export { writeZip, type ZipInput } from "./common/zipwriter";
export { ggufMetadataJson } from "./formats/gguf";
