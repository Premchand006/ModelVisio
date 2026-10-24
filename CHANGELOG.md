# Changelog

Release notes for the ModelVisio website and desktop app. The VS Code extension has its own
[CHANGELOG](apps/vscode/CHANGELOG.md).

## [1.1.0] - unreleased

### Added

- **Grok as a second AI provider.** The copilot can use xAI Grok as well as Google Gemini. On the
  website the server picks it (`MODELVISIO_PROVIDER=grok`, `XAI_API_KEY`); in the desktop app you
  choose Gemini or Grok in the copilot's key panel and paste your own key.
- **Free web search.** When a question needs current facts (docs, versions, releases, benchmarks,
  op support), the copilot searches the web without an API key, reads the top results and cites
  them. Uses DuckDuckGo, or your own SearXNG server via `MODELVISIO_SEARXNG_URL`. Available on the
  website and in VS Code.
- **More model formats.** Core ML, OpenVINO IR, TensorFlow (GraphDef, SavedModel, `.pbtxt`),
  Keras, Caffe, PaddlePaddle, MNN, ncnn, RKNN, MLIR and scikit-learn, plus ONNX shape inference and
  companion weight files.
- **Quantization heatmap** with a mixed-precision plan export for ONNX Runtime or TFLite.
- **Model advisor** with ranked, device-specific recommendations.
- **Benchmark** that runs ONNX models on your device with `onnxruntime-web` and compares the result
  with the roofline estimate.
- **Performance investigator**, **optimization search**, **deployment agent** (produces a ZIP of
  conversion, inference and validation scripts), **fleet simulation** and a local **model
  registry** with version diffs.
- **Command-line tool and GitHub Action**: `modelvisio inspect / check / diff / registry`, with
  budget and regression gates for CI.
- **Desktop:** File → Open Recent, the copilot can read pasted URLs, and a native macOS menu bar
  (Edit menu shortcuts now work in text fields).

### Changed

- **macOS:** one universal installer for Apple Silicon and Intel
  (`ModelVisio-macOS-universal.dmg`) replaces the two separate `.dmg` files.
- The website's download button picks the installer for your operating system, with a panel for
  other platforms and macOS install help.
- Copy across the app and docs is plainer and easier to read.

## [1.0.0] - 2026-07-02

First release: graph and inspector, compiler pre-flight, auto-fixes, hardware-fit scoring for edge
accelerators, converters, deploy scripts and the Gemini copilot, on the web and as a desktop app
for Windows, macOS and Linux.

[1.1.0]: https://github.com/Premchand006/ModelVisio/compare/desktop-v1.0.0...main
[1.0.0]: https://github.com/Premchand006/ModelVisio/releases/tag/desktop-v1.0.0
