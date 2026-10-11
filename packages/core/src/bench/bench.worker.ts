/// <reference lib="webworker" />
// CPU benchmark worker: onnxruntime-web's WebAssembly-only build. Its "bundle"
// entry embeds the Emscripten glue, so the only file it fetches is its 14 MB
// .wasm (no dynamic import). That binary ships with the app, so CPU
// benchmarks work offline, and the worker can run from a blob: URL under the
// VS Code WebView's CSP.
import * as ort from "onnxruntime-web/wasm";
import { startBenchWorker } from "./workerMain";

startBenchWorker(ort);
