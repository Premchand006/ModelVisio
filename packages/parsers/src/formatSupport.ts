// Which model formats ModelVisio opens, and how deeply. Kept apart from the
// parsers (registry.ts re-exports it) so UI code can list formats without
// pulling every parser and protobufjs into its bundle.

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
