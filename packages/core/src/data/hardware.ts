// Edge-AI accelerator catalog used by the Hardware report.
//
// This is a SOURCED, roofline-ready device database — not a list of hand-tuned
// "suitability" numbers. The 0-100 score is COMPUTED per loaded model by
// packages/core/src/scoring (roofline + memory-fit + op-support + efficiency),
// so every field below feeds a defensible calculation rather than a guess.
//
// Confidence legend (`conf`):
//   V = vendor datasheet/official   B = independent benchmark
//   M = MLPerf result               ? = not found / estimated (treat as low-confidence)
// `bwConf` flags memory-bandwidth confidence separately because bandwidth is the
// single most load-bearing field for the roofline; an estimated bandwidth means
// the FPS estimate for memory-bound workloads is low-confidence.
//
// Sources are summarized inline (`src`) for auditability. See real_score report
// for the full citation set (roofline = Williams/Waterman/Patterson CACM 2009;
// MLPerf Inference v3.1; Hailo & vendor benchmark tables; nn-Meter MobiSys '21).

export type Precision = "int4" | "int8" | "fp16" | "bf16" | "fp32";

/** Bytes per element at a given deployment precision. */
export const BYTES: Record<Precision, number> = { int4: 0.5, int8: 1, fp16: 2, bf16: 2, fp32: 4 };

export type MemType = "LPDDR4" | "LPDDR4X" | "LPDDR5" | "LPDDR5X" | "GDDR6" | "HBM" | "onchip";

/** Per-workload-class utilization (achieved / peak). Calibrated from benchmarks:
 *  Orin ~37% dense INT8 on MLPerf ResNet-50; Hailo-8 ~42% ResNet-50 / ~24% YOLOv5m;
 *  Meteor Lake NPU ~24-29%; V10/ISCA'23: most single DNN workloads use <50% of peak. */
export type Util = { cnnCls: number; cnnDet: number; transformer: number };

export type DeviceSpec = {
  id: string;
  n: string;
  v: string;
  /** Maps to the `target` string used in ModelLayer.compIssues, when the parser
   *  emits per-op compatibility notes for this device (drives op-support score). */
  compatId?: string;

  // Compute ceilings (TOPS for int8/int4, TFLOPS for float). Store dense AND
  // sparse where applicable; the roofline ALWAYS uses dense to avoid 2x optimism.
  peak: { int4?: number; int8?: number; fp16?: number; bf16?: number; fp32?: number; sparseInt8?: number };

  // Memory
  ramGB: number;                 // external DRAM (0 for DRAM-free dataflow parts)
  memType: MemType;
  bandwidthGBs: number;          // CRITICAL for the roofline ridge point
  bwConf: "V" | "B" | "?";
  sramMB?: number;               // on-chip budget where disclosed (Coral 6.91, Hailo, etc.)
  externalMem?: boolean;         // onchip part can still spill to external DRAM?

  // Power & cost
  power: { typ: number; max: number };
  priceUSD?: number;

  // Software / op handling
  toolchain: string[];
  fmts: string[];                // formats the toolchain ingests directly
  quant: { int8Only: boolean; perChannel: boolean; mixedPrecision: boolean };
  cpuFallback: "graceful" | "single-partition" | "none";
  /** Fixed per-inference dispatch/host-sync overhead (ms). Caps FPS for tiny
   *  models on over-provisioned or host-attached (USB/PCIe) accelerators. */
  dispatchMs: number;

  // Calibration & provenance
  util: Util;
  /** Real measured benchmark anchor. When present for the model's workload class,
   *  the roofline uses the utilization IMPLIED by this datapoint instead of the
   *  generic estimate, and the UI marks the result "calibrated". refOps = 2×MACs
   *  of the reference network (ResNet-50 INT8 ≈ 8.2 GFLOP-ops). */
  measured?: { workload: "cnnCls" | "cnnDet" | "transformer"; fps: number; refOps: number; precision: Precision; note: string };
  formFactor: string;
  hostDependency: boolean;       // USB/M.2/PCIe coprocessor needs a host
  eol?: boolean;
  conf: "V" | "B" | "M" | "?";
  notes?: string;
  src: string[];
};

// Standard utilization profile for mature, well-supported toolchains.
const U_STD: Util = { cnnCls: 0.40, cnnDet: 0.25, transformer: 0.18 };
// Dataflow parts with very mature compilers tuned per-graph (Hailo).
const U_DATAFLOW: Util = { cnnCls: 0.42, cnnDet: 0.24, transformer: 0.15 };
// Small NPUs / VPUs / USB coprocessors: lower achieved util + high overhead.
const U_SMALL: Util = { cnnCls: 0.22, cnnDet: 0.14, transformer: 0.08 };

export const HW: DeviceSpec[] = [
  // ── NVIDIA Jetson (TensorRT/cuDNN/CUDA; INT8/FP16/FP32; 2:4 sparsity on Orin;
  //    unified CPU/GPU memory; graceful op fallback). ─────────────────────────
  {
    id: "agx_orin", n: "NVIDIA Jetson AGX Orin", v: "NVIDIA",
    peak: { int8: 137, sparseInt8: 275, fp16: 5.3 },
    ramGB: 64, memType: "LPDDR5", bandwidthGBs: 204.8, bwConf: "V",
    power: { typ: 40, max: 60 }, priceUSD: 1999,
    toolchain: ["TensorRT", "cuDNN", "CUDA"], fmts: ["ONNX", "TensorRT", "PyTorch"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.15, util: U_STD,
    measured: { workload: "cnnCls", fps: 6424, refOps: 8.2e9, precision: "int8", note: "MLPerf Inference v3.1 closed-edge ResNet-50 offline" },
    formFactor: "SoM", hostDependency: false, conf: "M",
    src: ["MLPerf Inference v3.1 closed-edge ResNet-50 ~6,424 fps → ~51 achieved INT8 TOPS (~37% of 137 dense)", "developer.nvidia.com/embedded/jetson-benchmarks"],
  },
  {
    id: "orin_nx", n: "NVIDIA Jetson Orin NX 16GB", v: "NVIDIA",
    peak: { int8: 100, sparseInt8: 100, fp16: 3.7 },
    ramGB: 16, memType: "LPDDR5", bandwidthGBs: 102, bwConf: "V",
    power: { typ: 20, max: 25 }, priceUSD: 699,
    toolchain: ["TensorRT", "cuDNN", "CUDA"], fmts: ["ONNX", "TensorRT", "PyTorch"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.2, util: U_STD,
    formFactor: "SoM", hostDependency: false, conf: "M",
    notes: "157 TOPS in JetPack 6.2 'super' mode.",
    src: ["NVIDIA Jetson datasheet; 100 TOPS (157 super); MLPerf-class ResNet-50 ~2,640 fps"],
  },
  {
    id: "orin_nano", n: "NVIDIA Jetson Orin Nano", v: "NVIDIA",
    peak: { int8: 40, sparseInt8: 67, fp16: 1.3 },
    ramGB: 8, memType: "LPDDR5", bandwidthGBs: 68, bwConf: "V",
    power: { typ: 12, max: 15 }, priceUSD: 249,
    toolchain: ["TensorRT", "cuDNN", "CUDA"], fmts: ["ONNX", "TensorRT", "PyTorch"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.25, util: U_STD,
    formFactor: "SoM", hostDependency: false, conf: "V",
    notes: "Super mode (JetPack 6.2): 67 sparse TOPS, 102 GB/s, $249 devkit.",
    src: ["NVIDIA Jetson Orin Nano datasheet; 40 TOPS / 68 GB/s (67 TOPS / 102 GB/s super)"],
  },

  // ── Google Coral Edge TPU (INT8-only; ~6.91 MiB param cache; severe
  //    single-partition CPU fallback). ────────────────────────────────────────
  {
    id: "coral", n: "Google Coral Edge TPU", v: "Google", compatId: "coral_tpu",
    peak: { int8: 4 },
    ramGB: 0, memType: "onchip", bandwidthGBs: 64, bwConf: "?", sramMB: 6.91, externalMem: true,
    power: { typ: 2, max: 2 }, priceUSD: 60,
    toolchain: ["EdgeTPU Compiler"], fmts: ["TFLite"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "single-partition", dispatchMs: 1.0, util: U_SMALL,
    formFactor: "USB/M.2/PCIe", hostDependency: true, conf: "B",
    notes: "INT8-only; 6.91 MiB usable param cache; once an unsupported op appears, it + the rest of the graph run on host CPU.",
    src: ["coral.ai/docs/edgetpu/models-intro (single-partition fallback)", "coral.ai/docs/edgetpu/compiler (6.91 MiB RAM)", "MobileNet v2 ~400 fps"],
  },

  // ── Qualcomm (SNPE / QNN; INT8/INT16/FP16). ──────────────────────────────────
  {
    id: "rb5", n: "Qualcomm RB5", v: "Qualcomm",
    peak: { int8: 15 },
    ramGB: 8, memType: "LPDDR5", bandwidthGBs: 44, bwConf: "?",
    power: { typ: 10, max: 12 }, priceUSD: 250,
    toolchain: ["SNPE", "QNN"], fmts: ["ONNX", "TFLite", "SNPE", "QNN"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.4, util: U_STD,
    formFactor: "SoM/board", hostDependency: false, conf: "V",
    notes: "QRB5165 (Snapdragon 865-class). Achieved benchmarks not published — peak TOPS [V], bandwidth estimated.",
    src: ["Qualcomm RB5 datasheet (15 TOPS AI Engine)"],
  },
  {
    id: "rb6", n: "Qualcomm RB6 (QCS6490)", v: "Qualcomm",
    peak: { int8: 12 },
    ramGB: 12, memType: "LPDDR5", bandwidthGBs: 51, bwConf: "?",
    power: { typ: 12, max: 15 }, priceUSD: 250,
    toolchain: ["SNPE", "QNN"], fmts: ["ONNX", "TFLite", "QNN"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.4, util: U_STD,
    formFactor: "SoM/board", hostDependency: false, conf: "V",
    notes: "QCS6490 6th-gen AI Engine = 12 DENSE TOPS (corrects the common 20-TOPS marketing figure). Bandwidth estimated.",
    src: ["Qualcomm QCS6490/Dragonwing datasheet (12 dense TOPS)"],
  },

  // ── Intel (OpenVINO / DirectML / ONNX RT; retains FP16). ─────────────────────
  {
    id: "intel_npu", n: "Intel Core Ultra NPU", v: "Intel",
    peak: { int8: 11, fp16: 5.7 },
    ramGB: 32, memType: "LPDDR5X", bandwidthGBs: 120, bwConf: "B",
    power: { typ: 20, max: 30 }, priceUSD: 400,
    toolchain: ["OpenVINO", "DirectML", "ONNX RT"], fmts: ["ONNX", "OpenVINO"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.3, util: { cnnCls: 0.27, cnnDet: 0.18, transformer: 0.12 },
    formFactor: "Integrated (laptop SoC)", hostDependency: false, conf: "B",
    notes: "Meteor Lake NPU 3720 'AI Boost'. PRoof: achieved ~24-29% of peak; many models failed to run under OpenVINO 2024 → low-confidence (nn-Meter: analytical prediction least accurate on NPU/VPU).",
    src: ["Intel Core Ultra spec; PRoof ICPP'24 (perf 'significantly deviated from theoretical')"],
  },
  {
    id: "movidius", n: "Intel Movidius Myriad X", v: "Intel",
    peak: { int8: 1 },
    ramGB: 0.5, memType: "LPDDR4", bandwidthGBs: 6.4, bwConf: "?",
    power: { typ: 1.5, max: 2 }, priceUSD: 70,
    toolchain: ["OpenVINO"], fmts: ["OpenVINO", "ONNX"],
    quant: { int8Only: false, perChannel: false, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 1.2, util: U_SMALL,
    formFactor: "USB/M.2", hostDependency: true, eol: true, conf: "V",
    notes: "EOL (~2020). ~1 TOPS DNN (4 TOPS aggregate). Legacy analysis only.",
    src: ["Intel Movidius Myriad X (MA2485) datasheet"],
  },

  // ── AMD/Xilinx (Vitis-AI; DPU soft IP; INT8). ────────────────────────────────
  {
    id: "kv260", n: "AMD Xilinx KV260", v: "AMD/Xilinx",
    peak: { int8: 1.4 },
    ramGB: 4, memType: "LPDDR4", bandwidthGBs: 19.2, bwConf: "B",
    power: { typ: 10, max: 15 }, priceUSD: 249,
    toolchain: ["Vitis-AI"], fmts: ["ONNX", "Vitis-AI"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "graceful", dispatchMs: 0.5, util: { cnnCls: 0.30, cnnDet: 0.20, transformer: 0.10 },
    formFactor: "SoM", hostDependency: false, conf: "B",
    notes: "Zynq UltraScale+ DPU (~1.4 TOPS-class for typical B3136 config; corrects the 3.2 figure). Custom (non-Model-Zoo) models documented as hard to deploy.",
    src: ["Kria KV260 / K26 datasheet; Xilinx DPU B4096 benchmarks; SOM power measured >10W [B]"],
  },

  // ── Hailo (HailoRT + Dataflow Compiler; all-on-chip SRAM, DRAM-free; INT8).
  //    Weights are resident on-chip → per-frame traffic is activations only, so
  //    these parts are effectively compute-bound when the model fits. ──────────
  {
    id: "hailo8", n: "Hailo-8", v: "Hailo",
    peak: { int8: 26 },
    ramGB: 0, memType: "onchip", bandwidthGBs: 1000, bwConf: "?", sramMB: 32, externalMem: false,
    power: { typ: 2.5, max: 3.5 }, priceUSD: 90,
    toolchain: ["HailoRT", "Dataflow Compiler"], fmts: ["ONNX", "TFLite", "HailoRT"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "none", dispatchMs: 0.2, util: U_DATAFLOW,
    measured: { workload: "cnnCls", fps: 1371, refOps: 8.2e9, precision: "int8", note: "Hailo AI SW Suite v2024-10 ResNet-50 (batch 8)" },
    formFactor: "M.2/mPCIe/PCIe", hostDependency: true, conf: "B",
    notes: "DRAM-free dataflow: weights resident on-chip. Official: ResNet-50 1,371 fps @3.7ms 375 FPS/W (~42% util); YOLOv5m-640 242 fps 45 FPS/W (~24% util). Models too large for on-chip SRAM cannot deploy (no external memory).",
    src: ["Hailo AI SW Suite v2024-10 benchmark table; HailoRT log (1371.26 fps, 3.39 ms, 3.96 W)"],
  },
  {
    id: "hailo8l", n: "Hailo-8L (RPi AI Kit)", v: "Hailo",
    peak: { int8: 13 },
    ramGB: 0, memType: "onchip", bandwidthGBs: 1000, bwConf: "?", sramMB: 16, externalMem: false,
    power: { typ: 1.5, max: 2.5 }, priceUSD: 70,
    toolchain: ["HailoRT", "Dataflow Compiler"], fmts: ["ONNX", "TFLite", "HailoRT"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "none", dispatchMs: 0.25, util: U_DATAFLOW,
    formFactor: "M.2", hostDependency: true, conf: "B",
    notes: "Entry-grade Hailo-8. YOLOv8s-640 ~80-120 fps on RPi5 PCIe Gen3 [B].",
    src: ["Hailo-8L datasheet; community RPi5 benchmarks"],
  },

  // ── Rockchip (RKNN; INT4/8/16/FP16; PyTorch→ONNX→RKNN AOT). ──────────────────
  {
    id: "rk3588", n: "Rockchip RK3588 NPU", v: "Rockchip", compatId: "rk3588",
    peak: { int8: 6 },
    ramGB: 8, memType: "LPDDR5", bandwidthGBs: 19, bwConf: "B",
    power: { typ: 8, max: 10 }, priceUSD: 160,
    toolchain: ["RKNN"], fmts: ["ONNX", "RKNN", "TFLite"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.4, util: { cnnCls: 0.30, cnnDet: 0.20, transformer: 0.10 },
    formFactor: "SoM/board", hostDependency: false, conf: "B",
    notes: "6 TOPS (3-core). ResNet18 ~244 fps @4.09ms; TinyLlama-1.1B ~10-15 tok/s. RKNN v1.x uses CPU fallback for SiLU (v2.0+ native).",
    src: ["Rockchip RK3588 TRM; community RKNN benchmarks (~19 GB/s LPDDR)"],
  },

  // ── SiMa.ai (Palette/ONE; INT8; MLPerf winner). ──────────────────────────────
  {
    id: "sima", n: "SiMa.ai MLSoC", v: "SiMa.ai",
    peak: { int8: 50 },
    ramGB: 16, memType: "LPDDR4", bandwidthGBs: 30, bwConf: "?",
    power: { typ: 10, max: 15 }, priceUSD: 500,
    toolchain: ["Palette", "ONE Platform"], fmts: ["ONNX", "TFLite", "SiMa"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "graceful", dispatchMs: 0.3, util: U_STD,
    formFactor: "SoM/PCIe", hostDependency: false, conf: "M",
    notes: "Gen1 MLSoC won MLPerf v3.0 closed-edge ResNet-50 single-stream vs Orin. Bandwidth estimated.",
    src: ["MLPerf Inference v3.0 closed-edge; SiMa.ai press"],
  },

  // ── DeepX (DXNN SDK; INT8). ──────────────────────────────────────────────────
  {
    id: "deepx", n: "DeepX DX-M1", v: "DeepX",
    peak: { int8: 25 },
    ramGB: 4, memType: "LPDDR5", bandwidthGBs: 40, bwConf: "?",
    power: { typ: 3, max: 5 }, priceUSD: 95,
    toolchain: ["DXNN"], fmts: ["ONNX", "TFLite", "DeepX"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "graceful", dispatchMs: 0.4, util: U_STD,
    formFactor: "M.2 2280", hostDependency: true, conf: "V",
    notes: "25 INT8 TOPS; M1 = 4-8GB external LPDDR5, PCIe Gen3 x4. Bandwidth estimated.",
    src: ["DeepX DX-M1 datasheet; DXNN SDK docs"],
  },

  // ── Other accelerators ───────────────────────────────────────────────────────
  {
    id: "mobilint", n: "Mobilint MA35D", v: "Mobilint",
    peak: { int8: 30 },
    ramGB: 0, memType: "onchip", bandwidthGBs: 600, bwConf: "?", sramMB: 20, externalMem: true,
    power: { typ: 5, max: 7 }, priceUSD: 200,
    toolchain: ["Mobilint SDK"], fmts: ["ONNX", "Mobilint"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "graceful", dispatchMs: 0.4, util: U_SMALL,
    formFactor: "M.2/PCIe", hostDependency: true, conf: "?",
    notes: "Peak TOPS [V]; bandwidth & achieved throughput not sourced — low-confidence.",
    src: ["Mobilint MA35D product brief"],
  },
  {
    id: "kneron", n: "Kneron KL720", v: "Kneron", compatId: "kneron_kl720",
    peak: { int8: 1 },
    ramGB: 0.5, memType: "LPDDR4", bandwidthGBs: 4, bwConf: "?",
    power: { typ: 1, max: 1.5 }, priceUSD: 80,
    toolchain: ["Kneron SDK"], fmts: ["ONNX", "TFLite"],
    quant: { int8Only: true, perChannel: false, mixedPrecision: false },
    cpuFallback: "graceful", dispatchMs: 1.0, util: U_SMALL,
    formFactor: "USB/M.2", hostDependency: true, conf: "?",
    notes: "~0.9-1.4 TOPS class; bandwidth not sourced — low-confidence.",
    src: ["Kneron KL720 product page"],
  },
  {
    id: "blaize", n: "Blaize Pathfinder P1600", v: "Blaize",
    peak: { int8: 16 },
    ramGB: 8, memType: "LPDDR4", bandwidthGBs: 25, bwConf: "?",
    power: { typ: 7, max: 9 }, priceUSD: 300,
    toolchain: ["Blaize Picasso"], fmts: ["ONNX", "TFLite"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.4, util: U_SMALL,
    formFactor: "SoM", hostDependency: false, conf: "?",
    notes: "~16 TOPS class; key specs incompletely sourced — low-confidence.",
    src: ["Blaize Pathfinder P1600 brief"],
  },
  {
    id: "nxp", n: "NXP i.MX 93 Neutron NPU", v: "NXP",
    peak: { int8: 0.5 },
    ramGB: 2, memType: "LPDDR4X", bandwidthGBs: 6.4, bwConf: "B",
    power: { typ: 2, max: 3 }, priceUSD: 50,
    toolchain: ["eIQ"], fmts: ["TFLite", "ONNX"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "graceful", dispatchMs: 0.6, util: U_SMALL,
    formFactor: "SoM", hostDependency: false, conf: "V",
    notes: "eIQ Neutron NPU 0.5 TOPS (256 MAC/cycle @1GHz). Documented CPU fallback for ARG_MAX, RESIZE_BILINEAR, etc.",
    src: ["NXP i.MX 93 datasheet; eIQ docs (op CPU-fallback list)"],
  },
  {
    id: "amlogic", n: "Amlogic A311D2 NPU", v: "Amlogic",
    peak: { int8: 3.2 },
    ramGB: 4, memType: "LPDDR4", bandwidthGBs: 12, bwConf: "?",
    power: { typ: 6, max: 8 }, priceUSD: 120,
    toolchain: ["Tengine", "Vivante"], fmts: ["ONNX", "TFLite", "Tengine"],
    quant: { int8Only: true, perChannel: true, mixedPrecision: false },
    cpuFallback: "graceful", dispatchMs: 0.5, util: U_SMALL,
    formFactor: "SoM", hostDependency: false, conf: "V",
    notes: "3.2 TOPS Vivante NPU. Bandwidth estimated.",
    src: ["Amlogic A311D2 datasheet"],
  },
  {
    id: "ti_tda4", n: "TI TDA4VM (Jacinto)", v: "TI",
    peak: { int8: 8 },
    ramGB: 4, memType: "LPDDR4", bandwidthGBs: 17, bwConf: "B",
    power: { typ: 15, max: 20 }, priceUSD: 250,
    toolchain: ["TIDL"], fmts: ["ONNX", "TFLite", "TIDL"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.5, util: { cnnCls: 0.28, cnnDet: 0.18, transformer: 0.10 },
    formFactor: "SoM", hostDependency: false, conf: "V",
    notes: "~8 TOPS DLA (C7x DSP + MMA). Automotive-grade.",
    src: ["TI TDA4VM datasheet; TIDL docs"],
  },
  {
    id: "mtk_genio", n: "MediaTek Genio 1200 APU", v: "MediaTek",
    peak: { int8: 4.8 },
    ramGB: 8, memType: "LPDDR4X", bandwidthGBs: 17, bwConf: "?",
    power: { typ: 6, max: 8 }, priceUSD: 180,
    toolchain: ["NeuroPilot"], fmts: ["ONNX", "TFLite", "NeuroPilot"],
    quant: { int8Only: false, perChannel: true, mixedPrecision: true },
    cpuFallback: "graceful", dispatchMs: 0.5, util: U_SMALL,
    formFactor: "SMARC SoM", hostDependency: false, conf: "?",
    notes: "~4.8 TOPS APU. MediaTek does not disclose its TOPS methodology — low-confidence.",
    src: ["MediaTek Genio 1200 (MT8395) brief"],
  },
];

export const FORMATS: string[] = [
  "ONNX", "ONNX Runtime", "TFLite", "PyTorch (.pt)", "TorchScript", "torch.export",
  "ExecuTorch", "TensorFlow", "TF.js", "Keras", "Core ML", "OpenVINO", "Caffe",
  "Darknet", "MXNet", "PaddlePaddle", "MNN", "ncnn", "TNN", "RKNN", "Hailo",
  "TensorRT (.engine)", "UFF", "CNTK", "MegEngine", "NNabla", "Barracuda",
  "Huawei OM", "ML.NET", "BigDL", "CatBoost", "scikit-learn", "Safetensors",
  "NumPy (.npz)", "GGUF", "MLIR",
];
