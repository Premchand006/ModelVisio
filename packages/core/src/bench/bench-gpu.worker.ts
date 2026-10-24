/// <reference lib="webworker" />
// WebGPU / WebNN benchmark worker: onnxruntime-web's GPU build. Its 27 MB
// .wasm is not shipped: it is over Cloudflare Pages' 25 MiB per-file limit and
// would nearly double the VS Code package. The main thread downloads it on
// first use from a version-pinned CDN URL, checks its SHA-256 and passes the
// bytes in (gpuWasm.ts); the build plugins keep the bundler from emitting it.
import * as ort from "onnxruntime-web/webgpu";
import { startBenchWorker } from "./workerMain";

startBenchWorker(ort);
