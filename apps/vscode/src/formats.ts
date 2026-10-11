// File associations for the two custom editors. Pure (no `vscode` import) so
// test/host-formats.test.ts can check these lists against both package.json's
// `customEditors` selectors and @modelvisio/parsers' FORMAT_SUPPORT registry.

/** Primary editor: claims these by default — no other tool owns the extension. */
export const VIEW_TYPE = "modelvisio.modelViewer";
/** Secondary editor: offered in "Reopen With…" only — the extension is generic
 *  (.json, .xml, .bin, …), so stealing it from the text editor would be rude. */
export const VIEW_TYPE_OPTION = "modelvisio.modelViewerOption";

export const DEFAULT_EXTS: readonly string[] = [
  "onnx", "ort",
  "tflite", "lite", "tfl",
  "safetensors", "gguf", "ggml", "npy", "npz",
  "pt", "pth", "ckpt", "ptl", "torchscript", "pt2", "pte",
  "mlmodel", "keras", "caffemodel", "pdmodel", "pdparams",
  "mnn", "tnnmodel", "tnnproto", "rknn", "uff", "cntk", "mge", "nntxt",
  "mlnet", "bigdl", "cbm",
];

export const OPTION_EXTS: readonly string[] = [
  "json", "xml", "bin", "cfg", "pb", "pbtxt", "meta", "h5", "hdf5",
  "params", "param", "nb", "weights", "nn", "plan", "engine", "trt",
  "prototxt", "pkl", "pickle", "joblib", "mlir", "dnn", "om", "tm", "hn",
  // Hailo archive — but also the far more common HTTP Archive (browser
  // devtools export), which must keep opening as JSON text.
  "har",
];

/** Registry extensions that name directories (bundles). Custom editors open
 *  files only, so these are deliberately not contributed. */
export const DIRECTORY_EXTS: readonly string[] = ["mlpackage", "mlmodelc"];

export const ALL_EXTS: readonly string[] = [...DEFAULT_EXTS, ...OPTION_EXTS];

/** Lower-cased extension of a file name or path ("" when there is none). */
export function extOf(nameOrPath: string): string {
  const base = basename(nameOrPath);
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot + 1).toLowerCase() : "";
}

/** Last path segment; accepts "/" and "\" separators. */
export function basename(p: string): string {
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] ?? "";
}

/** Custom-editor `selector` entries, in the shape package.json declares. */
export function selectorFor(exts: readonly string[]): { filenamePattern: string }[] {
  return exts.map((e) => ({ filenamePattern: `*.${e}` }));
}

/** Which of our editors to open a file with. Unknown extensions use the
 *  option editor: `vscode.openWith` accepts any registered viewType, and the
 *  option editor is the one that never claims files by default. */
export function viewTypeFor(nameOrPath: string): string {
  return DEFAULT_EXTS.includes(extOf(nameOrPath)) ? VIEW_TYPE : VIEW_TYPE_OPTION;
}

/** `showOpenDialog` filters: every supported model extension, then anything. */
export function openDialogFilters(): Record<string, string[]> {
  return { "Model files": [...ALL_EXTS], "All files": ["*"] };
}
