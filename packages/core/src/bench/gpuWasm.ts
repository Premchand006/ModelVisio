// Where the WebGPU/WebNN benchmark gets onnxruntime-web's GPU .wasm.
//
// The binary (ort-wasm-simd-threaded.asyncify.wasm, ~27 MB) is fetched from
// jsDelivr's copy of the exact onnxruntime-web release the app is built
// against, only when a GPU benchmark starts, and checked against the SHA-256
// of that release's file, so a tampered or mismatched download is refused.
// test/bench-gpuWasm.test.ts keeps both values in step with package.json and
// the installed package.

export const ORT_WEB_VERSION = "1.30.0";
export const GPU_WASM_FILE = "ort-wasm-simd-threaded.asyncify.wasm";
export const GPU_WASM_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_WEB_VERSION}/dist/${GPU_WASM_FILE}`;
export const GPU_WASM_SHA256 = "39f9f0894d478800487ed9f7dbe92618498db320cf55c8e3d89adff8dce658da";
export const GPU_WASM_MB = 27;

const hex = (b: ArrayBuffer) => [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, "0")).join("");

/** SHA-256 check of a downloaded runtime; false when it doesn't match (or can't be checked). */
export async function verifyGpuWasm(bytes: ArrayBuffer, subtle: SubtleCrypto | undefined = globalThis.crypto?.subtle): Promise<boolean> {
  if (!subtle) return false;
  return hex(await subtle.digest("SHA-256", bytes)) === GPU_WASM_SHA256;
}
