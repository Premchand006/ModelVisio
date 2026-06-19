// @modelvisio/parsers — real model-format parsing → normalized graph objects.
// Pure TS, no React.

export type { Model, ModelLayer } from "./types";

// Unified entry point: detects the format and parses to the normalized Model.
export { parseModel, detectFormat, FORMAT_SUPPORT, type FormatInfo, type ParserStatus } from "./registry";

// Individual parsers (also usable directly / in tests).
export { parseOnnx, type ParseOptions } from "./onnx/parse";
export { parseTflite } from "./formats/tflite";
export { parseSafetensors } from "./formats/safetensors";
export { parseNumpy } from "./formats/numpy";
export { parseGguf } from "./formats/gguf";
export { parseDarknet } from "./formats/darknet";
export { parsePytorch } from "./formats/pytorch";
export { parseContainer } from "./formats/container";

// Model conversion (browser-side data transforms) + ZIP writer for kits.
export { convertWeights, toGraphJson, toLayerCsv, type ConvertResult, type ConvTarget, type Precision } from "./convert";
export { writeZip, type ZipInput } from "./common/zipwriter";
export { ggufMetadataJson } from "./formats/gguf";
