# ModelVisio: A Defensible, Roofline-Grounded `scoreDevice(model, device)` Function

## TL;DR
- Replace the fudged 0–100 score with a **roofline-based estimate**: compute per-layer arithmetic intensity, take the lower of the compute-bound and memory-bound throughput ceilings, apply a workload-class utilization de-rating (~0.30–0.40 for CNNs, ~0.15–0.25 for transformers — both empirically grounded), then fold in a hard memory-fit guard, a format/op-support coverage sub-score, and an efficiency (FPS/W, FPS/$) sub-score.
- The single most important architectural fix is the **memory-fit hard-fail guard**: a model whose weights + peak activation working set exceeds device RAM must never show green, regardless of compute headroom.
- Peak TOPS are marketing numbers; real achieved throughput is ~19–42% of peak even on the best-tuned CNNs (Jetson AGX Orin reaches ~51 of 275 sparse TOPS on MLPerf ResNet-50; Hailo-8 reaches ~11 of 26 TOPS), so the utilization factor must be a per-device-class constant derived from benchmarks, not a guess.

## Key Findings

### Roofline is the right backbone
The roofline model — Samuel Williams, Andrew Waterman, and David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, Vol. 52, No. 4 (April 2009), pp. 65–76 (ACM DOI 10.1145/1498765.1498785) — bounds attainable performance as `min(peak_compute, bandwidth × arithmetic_intensity)`. For DNNs this is applied per-layer/per-kernel and aggregated. Academic and industry tooling all build on exactly this decomposition: PRoof (ICPP '24, ACM 10.1145/3673038.3673116), the "Time-Based Roofline for Deep Learning" paper (arXiv 2009.04598), the LLM-Viewer/roofline survey (arXiv 2402.16363), and the Pagoda Jetson roofline study (dream-lab/pagoda on GitHub).

Microsoft's nn-Meter — Zhang, Han, Wei, Zheng, Cao, Yang, Liu, "nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices," MobiSys '21 Best Paper (ACM DOI 10.1145/3458864.3467882) — shows pure FLOP-counting is insufficient: its kernel-level predictor "achieves 99.0% (mobile CPU), 99.1% (mobile Adreno 640 GPU), 99.0% (mobile Adreno 630 GPU), and 83.4% (Intel VPU) prediction accuracy," versus only 22.1% for a FLOPs-based predictor and 17.1% for FLOPs+MAC. The lower VPU/NPU accuracy confirms that an analytical roofline with a calibrated per-device utilization factor is the most defensible *implementable* compromise for ModelVisio (which has model shapes but cannot run every model on every chip).

### Achieved throughput is a fraction of peak TOPS
- **Jetson AGX Orin**, MLPerf Inference v3.1 closed-edge ResNet-50 offline: ~6,424 samples/sec → ~51 achieved INT8 TOPS, i.e., ~19% of the 275 sparse-TOPS marketing figure but ~37% of the ~137 dense INT8 peak. The result was achieved on the Jetson AGX Orin Developer Kit running JetPack 5.1.1, TensorRT 9.0.1, CUDA 11.4 (NVIDIA's MLCommons Inference v3.1 closed-edge submission, mlcommons/inference_results_v3.1 README_Jetson.md; developer.nvidia.com/embedded/jetson-benchmarks).
- **Hailo-8** (26 TOPS): Hailo's official benchmark table (hailo.ai, Hailo AI SW Suite v2024-10, Dataflow Compiler v3.29.0, batch 8, PCIe, host Intel Core i5-9400) lists "ResNet-50 v1 · 224×224 · 1371 FPS · 3.7 ms · 375 FPS/W," corroborated by a HailoRT log ("FPS (hw_only) = 1371.26 ... Latency (hw) = 3.39 ms ... Power ≈ 3.96 W"). That is ~11 TOPS achieved → ~42% of peak. The same table reports the heavier detector "YOLOv5m · 640×640 · 242 FPS · 5.3 ms · 45 FPS/W" and segmentation "stdc1 · 1024×1920 · 58 FPS · 3.1 ms · 19 FPS/W," i.e., heavy CNNs land near ~20–25% utilization.
- **Intel Meteor Lake NPU** (NPU 3720, 11.5 INT8 TOPS / 5.7 FP16 TFLOPS theoretical): PRoof reports performance "significantly deviated from its theoretical value"; an independent MatMul measurement reached ~1.35 FP16 TFLOPS (~24–29% of peak).
- General rule from V10 (ISCA '23): most single DNN workloads use <50% of a TPU core's FLOPS.

This justifies device-class utilization constants: **dense well-supported CNN classifiers ~0.35–0.45, CNN detectors/segmenters ~0.20–0.30, transformers/LLM decode ~0.10–0.25 (memory-bound).**

### Memory bandwidth is the hidden bottleneck
Edge accelerators range from ~17 GB/s (single LPDDR4X interfaces) to 204.8 GB/s (Jetson AGX Orin). Bandwidth, not TOPS, dominates transformer decode and high-resolution CNN workloads. Hailo's own analysis notes a 1B-parameter LLM tops out at ~40 tokens/s on a single 17 GB/s LPDDR4X interface, and Token-by-Token performance does not scale linearly with model size unless bandwidth scales too. The roofline ridge point (peak_compute / bandwidth) tells you the arithmetic intensity a model must exceed to be compute-bound on a given device.

### Op-support and quantization gating is binary, not gradual, on some devices
The Google Coral Edge TPU is the canonical example: it is INT8-only, requires constant tensor sizes at compile time, and — critically — once the compiler hits an unsupported op, per coral.ai/docs/edgetpu/models-intro: "the Edge TPU compiler cannot partition the model more than once, so as soon as an unsupported operation occurs, that operation and everything after it executes on the CPU, even if supported operations occur later... you should expect a significantly degraded inference speed compared to a model that executes entirely on the Edge TPU." This can collapse achieved FPS by an order of magnitude. The scoring function must model op coverage and CPU-fallback penalties, not just compute.

## Details

### Part 1 — Device Parameter Catalog

Confidence legend: **[V]** vendor datasheet/official, **[B]** independent benchmark, **[M]** MLPerf, **[?]** unknown/not found. Where a spec could not be located, it is explicitly marked `[?]` — do not guess.

#### NVIDIA Jetson family
Toolchain TensorRT/cuDNN/CUDA; INT8/FP16/FP32; per-tensor & per-channel quantization; structured 2:4 sparsity on Orin (Ampere). Unified CPU/GPU memory; graceful op handling (DLA falls back to GPU; GPU runs almost all ops).
- **Jetson AGX Orin 64GB** [V]: 275 sparse INT8 TOPS (≈137 dense); ~5.3 FP32 TFLOPS; 64GB LPDDR5 256-bit; **204.8 GB/s**; 15–60W configurable; ~$1,999 module class; 100×87mm SoM. MLPerf v3.1 ResNet-50 offline ~6,424 fps [M].
- **Jetson Orin NX 16GB** [V]: 100 TOPS (157 in JetPack 6.2 "super" mode); 16GB LPDDR5 128-bit; **102 GB/s**; 10–40W; ~$699 class. MLPerf-class ResNet-50 ~2,640 fps [M].
- **Jetson Orin Nano 8GB / Super** [V]: 40 TOPS / 67 sparse TOPS (super); 8GB LPDDR5; **68 GB/s → 102 GB/s** (super, JetPack 6.2; GPU clock 635→1,020 MHz, CPU 1.5→1.7 GHz); 7–25W; $249 devkit (super), down from $499. Pure software unlock — existing kits upgrade via JetPack 6.2.
- **Jetson AGX Xavier** [V]: 32 INT8 TOPS (22 GPU + 2×5 DLA); 11 FP16 TFLOPS; 16/32GB LPDDR4x 256-bit; **137 GB/s**; 10/15/30W; 512-core Volta + 64 Tensor cores + dual NVDLA v1; SoM.

#### Google Coral Edge TPU [V/B]
4 INT8 TOPS, 2W (2 TOPS/W); **INT8-only**; ~8MB SRAM with **6.91 MiB usable for parameter caching** (per coral.ai/docs/edgetpu/compiler, which shows a model "uses 4.21 MiB of the available 6.91 MiB of RAM"); form factors USB (3.0/2.0, host-dependent), M.2, PCIe Gen2 x1, Dev Board (i.MX 8M, 1–4GB LPDDR4). Toolchain: Edge TPU Compiler on fully-quantized TFLite. **Op constraints**: constant tensor sizes at compile time; tensors effectively ≤3 dims; single-partition CPU fallback (severe). MobileNet v2 ≈400 fps [V]. Price ~$60 (USB), ~$130 (Dev Board). CPU-fallback class: **single-partition** (worst case for scoring).

#### Qualcomm
- **QCS6490 / RB3 Gen 2 / Dragonwing** [V]: 12 dense TOPS (6th-gen AI Engine: Hexagon DSP + HVX + Hexagon Tensor Accelerator); LPDDR5/LPDDR4x, up to 16GB on SBCs; toolchain SNPE / Qualcomm AI Engine Direct (QNN), ONNX; INT8/INT16/FP16. Octa-core Kryo 670 (A78/A55). SBC/board ~$60–250.
- **RB5 (QRB5165)** [V]: 15 TOPS AI Engine (Snapdragon 865-class); 8GB LPDDR5; SNPE/QNN; 5G. (Peak TOPS [V]; achieved benchmarks not compiled [?].)
- **RB6** [V]: enhanced AI Engine, Kryo 585, sub-6/mmWave 5G; clean published NPU TOPS figure not found [?].

#### Intel
- **Core Ultra (Meteor Lake) NPU** [V/B]: NPU 3720 "AI Boost" ~10–11.5 INT8 TOPS / ~5.7 FP16 TFLOPS theoretical; platform ~34 TOPS total; OpenVINO/DirectML/ONNX RT; retains FP16. Achieved ~24–29% of peak [B]; only a small fraction of models ran successfully under OpenVINO 2024 in the PRoof study.
- **Core Ultra 200V (Lunar Lake) NPU** [V]: NPU 4 = 47–48 INT8 TOPS (Intel spec sheet lists "NPU Peak TOPS (Int8) 47"); GPU 64–67 INT8 TOPS; platform 120 TOPS; on-package LPDDR5X (≤32GB); sparsity support; meets Microsoft Copilot+ 40-TOPS bar; OpenVINO/WindowsML/DirectML/ONNX RT/WebNN; 17–30W TDP.
- **Movidius Myriad X (MA2485)** [V]: ~1 TOPS DNN (4 TOPS aggregate); 16 SHAVE cores; 1.5W; 4Gbit in-package LPDDR4; OpenVINO; USB/M.2/PCIe. **Discontinued (EOL ~2020)** — flag as legacy/low-confidence.

#### AMD/Xilinx (Vitis-AI; DPU soft IP; INT8; per-channel)
- **Kria KV260 (K26 SoM)** [V/B]: Zynq UltraScale+ MPSoC; DPU configurable (apps use B3136; Xilinx benchmarks used B4096) at 300/600 MHz; ~1.4 TOPS-class per typical config; 4GB LPDDR4; ~$199–349. Toolchain Vitis-AI (Quantizer/Compiler/Model Zoo). SOM power measured >10W under heavy load [B]. Note real-world ML deployment of custom (non–Model-Zoo) models is documented as difficult [B].
- **VCK190 (Versal AI Core)** [V]: AI Engine vector array, hundreds-of-INT8-TOPS class; high-end eval board (multi-thousand USD). Achieved benchmarks not compiled here [?].

#### Hailo (HailoRT + Dataflow Compiler; on-chip SRAM, DRAM-free; INT8; TF/TFLite/Keras/PyTorch/ONNX)
- **Hailo-8** [V/B]: 26 INT8 TOPS; 2.5W typical; all-on-chip SRAM (no external DRAM — a distinctive architecture); M.2/mPCIe/PCIe (Century card). Official: ResNet-50 1,371 fps @3.7 ms, 375 FPS/W; YOLOv5m-640 242 fps, 45 FPS/W. Industrial −40–85°C. ~$70–110 module.
- **Hailo-8L** [V]: 13 TOPS; same architecture, entry grade; YOLOv8s-640 ~80–120 fps on RPi5 PCIe Gen3 [B]; lower cost.
- **Hailo-10H** [V]: 40 INT4 TOPS; second-gen neural core targeting edge GenAI/LLM/VLM; M.2; external memory (LPDDR4X) option for large models.

#### Rockchip RK3588 / RK3576 (RKNN toolkit; INT4/INT8/INT16/FP16/BF16/TF32; PyTorch→ONNX→RKNN AOT flow)
- **RK3588 NPU** [V/B]: 6 TOPS INT8 (3-core, in-house IP); LPDDR4/4X/5 quad-channel (54-bit), up to ~19 GB/s [B]; LLMs limited to W8A8. ResNet18 ~244 fps @4.09 ms; TinyLlama-1.1B ~10–15 tok/s [B]. Boards $150–180.
- **RK3576 NPU** [V]: 6 TOPS INT8; LPDDR4/4X/5 (32-bit) up to ~12 GB/s; supports W4A16 for LLMs; ~30% slower than RK3588 on YOLO due to lower bandwidth/scheduling [B]. Boards ~$103.

#### SiMa.ai MLSoC [V/M]
- **Gen1 MLSoC** [V/M]: 16nm; won MLPerf v3.0 closed-edge ResNet-50 single-stream vs NVIDIA Orin using off-the-shelf software; ~20% MLPerf power improvement v3.0→v3.1; Palette/ONE Platform toolchain; INT8.
- **Modalix (Gen2)** [V]: M25/M50/M100/M200 = 25/50/100/200 TOPS configs (single-chip 50 TOPS; cluster 2× for 100, 4× for 200); MLA with mixed precision (BFLOAT16 + INT8 + INT16); Arm A65 application cluster; supports CNNs/transformers/LLMs/LMMs (Llama-7B, GPT-J, Llava-7B demonstrated); SoM (Enclustra, Jetson-pin-compatible), PCIe Gen5; "10× perf/W of alternatives" (vendor claim — flag).

#### Other credible accelerators
- **Axelera Metis AIPU** [V/B]: 214 INT8 TOPS (quad-core RISC-V-controlled digital in-memory compute); 15 TOPS/W; ~5–9W typical; ResNet-50 ~3,200 fps and MobileNetV2 ~38,884 fps (vendor); M.2 ($149, 1–4GB on-card DRAM) / PCIe ($499, 4-chip = 856 TOPS, 16/64GB). Voyager SDK; TF/PyTorch/ONNX. Note: 214 TOPS not reachable within M.2 thermal envelope (~8W) [B].
- **EdgeCortix SAKURA-II** [V]: 60 INT8 TOPS / 30 BF16 TFLOPS (dual-chip module 120 TOPS); 8W; 8/16GB; second-gen DNA architecture; MERA compiler; M.2/PCIe; targets LLM/LVM/multimodal.
- **DeepX DX-M1 / DX-M1M** [V]: 25 INT8 TOPS; 1–5W (M1) / 3W typical (M1M); M1 = 4–8GB external LPDDR5, PCIe Gen3 x4, M.2 2280; M1M = 1GB integrated LPDDR4X (4266 MT/s), PCIe Gen3 x2, M.2 2242; DXNN SDK (DX-COM compiler, DX-RT runtime, DX-STREAM GStreamer); ONNX/PyTorch/TF/Keras. ~$85–97 (M1M module). DX-M1ML light variant = 13 TOPS. Quattro PCIe = 4× M1 = 100 TOPS @20W.
- **MediaTek Genio 1200 (MT8395)** [V]: ~4.8–5 TOPS APU; octa-core A78/A55 + Mali-G57; NeuroPilot SDK; SMARC SoM. (TOPS calculation method not publicly disclosed by MediaTek [?].)
- **NXP i.MX 93 (eIQ Neutron / Ethos-U65 NPU)** [V]: 0.5 TOPS (256 MAC/cycle @1 GHz); TFLite + eIQ; documented CPU-fallback for unsupported ops (e.g., ARG_MAX, RESIZE_BILINEAR fall back to Cortex-A). LPDDR4X. (Note: i.MX 95 adds a larger Neutron NPU; i.MX 8M Plus = 2.3 TOPS.)
- **Amlogic A311D2** [V]: 3.2 TOPS NPU (A311D = 5 TOPS; newer A311Y3 = 8 TOPS, LPDDR5/5X); LPDDR4/5; Vivante NPU (open-source Mesa driver upstreamed).
- **TI TDA4VM (Jacinto)** [V]: ~8 TOPS deep-learning accelerator (C7x DSP + MMA); TIDL toolchain; automotive-grade.
- **Memryx MX3** [B]: ~40 TFLOPS-class quad M.2 module. **Kneron KL720** ~0.9–1.4 TOPS / **KL730** ~4 TOPS class [?]. **Blaize Pathfinder P1600 / Xplorer** ~16 TOPS class [?]. **Furiosa RNGD/Warboss**, **Brainchip Akida**, **Sophgo/Sophon**, **Untether**, **Esperanto**, **Google Coral NPU** (new IP), **RK3568** (~0.8 TOPS) — key specs not fully sourced in this pass; mark `[?]` and exclude from high-confidence scoring until verified.

### Part 2 — Modeling Logic

**Roofline core.** For each layer L with MACs `m_L` (2 FLOPs each) and bytes moved `b_L` (input activations + output activations + weights, each at deployment precision):
- Arithmetic intensity `AI_L = 2·m_L / b_L` (FLOP/byte).
- Per-device ridge point `AI* = peak_compute / bandwidth`.
- Layer is **compute-bound** if `AI_L ≥ AI*`, else **memory-bound**.
- Attainable layer throughput `P_L = min(peak_compute, bandwidth × AI_L)` FLOP/s.
- Ideal layer time `t_L = 2·m_L / P_L`. Aggregate `t_ideal = Σ t_L`.
- Real latency `t_real = t_ideal / utilization`; `FPS = 1000 / t_real_ms`.

This is the same per-layer-then-aggregate scheme used by PRoof and the LLM-Viewer roofline survey; PRoof's key insight is that DNN operator implementations reuse on-chip SRAM well, so memory bytes can be estimated from operator input/output sizes (rules per op type) without hardware counters — directly implementable from ONNX shapes.

**Utilization factor (calibrated, per device-class × workload-class):** CNN-classification 0.35–0.45; CNN-detection/segmentation 0.20–0.30; transformer/attention 0.10–0.25. Justified by: MLPerf-derived Orin (~37% of dense INT8), Hailo-8 (~42% on ResNet-50, ~20–25% on YOLOv5m/segmentation), Meteor Lake NPU (~24–29%), and V10/ISCA '23 (<50% generally). Use **dense** peak as the denominator to avoid 2× optimism on NVIDIA sparse figures.

**Memory fit.** `footprint = weights_bytes + peak_activation_bytes + runtime_overhead`. Weights = params × bytes-per-element at deployment precision (INT8=1, FP16=2, FP32=4). Peak activation = the **largest single live working set** — the maximum over the layer graph of the sum of concurrently-live tensors — NOT the sum of all activations. CNN rule of thumb: the largest (input_activation + output_activation) pair plus that layer's weights, scaled by batch size. Then:
- `footprint > device_RAM` → **hard fail** (overall score capped ≤15, never green).
- For SRAM-resident devices (Hailo, Coral), `footprint > on-chip SRAM` → penalize (spill/tiling overhead), don't fail if external DRAM exists; **hard fail** if no external memory path (e.g., Hailo-8 with a model exceeding its on-chip budget).
- Weight cache > SRAM budget (Coral 6.91 MiB) → throughput penalty proportional to the spilled fraction.

**Format & op support.** Sub-score = fraction of model ops natively supported by the device toolchain. Unsupported ops → CPU-fallback latency penalty. Model the fallback class: **graceful** (Jetson, most NPUs — penalize per-op), **single-partition** (Coral — once an unsupported op appears, it and all subsequent ops run on CPU, so the penalty is the cumulative cost of the entire tail of the graph), or **none** (compile fails → hard fail). Quantization gate: if a device is INT8-only and the model can't be quantized within accuracy tolerance, apply a sensitivity penalty.

**Efficiency.** `FPS/W = est_FPS / typical_power`; `FPS/$ = est_FPS / device_price`. Weight these per deployment profile — heavily for power/battery-constrained or cost-sensitive targets, lightly for max-throughput targets.

### Part 3 — Synthesis Deliverable

**TypeScript device schema:**
```ts
interface DeviceSpec {
  id: string; name: string; vendor: string;
  // Compute (TOPS / TFLOPS) — store dense AND sparse where applicable
  peak: {
    int4?: number; int8?: number; fp16?: number; bf16?: number; fp32?: number;
    sparseInt8?: number;          // NVIDIA quotes sparse; default roofline to dense
  };
  // Memory
  ramGB: number;
  memType: 'LPDDR4'|'LPDDR4X'|'LPDDR5'|'LPDDR5X'|'GDDR6'|'HBM'|'onchip';
  bandwidthGBs: number;           // CRITICAL for roofline; if unknown -> degrade + low confidence
  sramMB?: number;                // on-chip budget where disclosed (Coral 6.91, Hailo, SiMa)
  // Power & cost
  powerW: { idle?: number; typ?: number; max: number };
  priceUSD?: number;
  // Software
  toolchain: string[];            // TensorRT, EdgeTPU, RKNN, HailoRT, OpenVINO, SNPE/QNN, Vitis-AI, Palette, MERA, DXNN, NeuroPilot, TIDL
  formats: string[];              // ONNX, TFLite, PyTorch, TF, Keras
  quant: { int8Only: boolean; perChannel: boolean; mixedPrecision: boolean };
  unsupportedOps?: string[];
  cpuFallback: 'graceful'|'single-partition'|'none';
  // Calibration
  utilization: { cnnCls: number; cnnDet: number; transformer: number };
  formFactor: string;             // M.2, USB, SoM, PCIe, mPCIe
  hostDependency: boolean;        // true for USB/M.2/PCIe coprocessors
  confidence: Record<string, 'V'|'B'|'M'|'?'>;
  sources: Record<string, string>; // per-field URL/citation for auditability
}
```

**Roofline latency/FPS function (codeable directly):**
```ts
function estimateFps(model: ModelGraph, dev: DeviceSpec, prec: Precision): number {
  const peakFlops = pickPeak(dev, prec) * 1e12;      // dense TOPS -> FLOP/s
  const bw = dev.bandwidthGBs * 1e9;                  // bytes/s
  let tIdeal = 0;                                     // seconds
  for (const L of model.layers) {
    const flops = 2 * L.macs;
    const bytes = L.inputBytes(prec) + L.outputBytes(prec) + L.weightBytes(prec);
    const ai = flops / bytes;
    const P = Math.min(peakFlops, bw * ai);            // roofline
    tIdeal += flops / P;
  }
  const u = utilFor(model.workloadClass, dev.utilization); // 0.10–0.45
  return 1 / (tIdeal / u);                            // FPS
}
```

**Sub-score weighting (default throughput profile):** computeFit/latency 40%, memoryFit 25% (with hard-fail override), formatOpSupport 20%, efficiency 15%. Power-constrained profile: efficiency 30%, latency 30%, memoryFit 25%, opSupport 15%. The memory-fit hard-fail forces overall ≤15 ("won't deploy") regardless of other sub-scores; a compile-fail (cpuFallback === 'none' with unsupported core ops) does the same.

**UI surfacing for explainability:**
- Show the 0–100 number with a **stacked breakdown bar** of the four sub-scores.
- Show the **binding roofline regime** per the bottleneck layer(s): e.g., "memory-bound: model AI = 8 FLOP/B < device ridge AI* = 27 FLOP/B."
- Show estimated **FPS, latency (ms), FPS/W, FPS/$**, and the utilization constant used.
- A **red banner** when the memory-fit guard trips ("Model footprint 12.4 GB > device RAM 8 GB — cannot deploy"), and an **amber banner** for partial CPU fallback ("17 ops fall back to CPU on Coral after first unsupported op → ~X× slowdown").
- Per-spec **confidence badge and source link** on hover, so the score is auditable and `[?]` fields are visibly distinguished from `[V]`/`[M]` ones.

## Recommendations
1. **Ship the memory-fit hard-fail guard first.** Highest-value, lowest-effort fix; it eliminates the "impossible deployment shows green" failure mode that most embarrasses a compatibility score. Use peak-activation working set (not sum) plus weights at deployment precision.
2. **Implement the per-layer roofline** from ONNX graph shapes you already parse; fall back to whole-model MAC/byte aggregates only when per-layer shapes are unavailable, flagging lower confidence.
3. **Seed utilization constants from the catalog** (defaults: CNN-cls 0.40, CNN-det 0.25, transformer 0.18) and refine per device as you collect your own measurements. Recalibration threshold: if predicted FPS deviates >2× from any measured datapoint, adjust that device's utilization constant.
4. **Store peak TOPS as dense-and-sparse explicitly** and default the roofline to dense INT8; never feed NVIDIA's 275 sparse TOPS into the compute ceiling.
5. **Mark every unknown spec `[?]`** in the data file and exclude unknown-bandwidth devices from full roofline scoring (use a degraded heuristic with an explicit low-confidence flag) rather than inventing bandwidth.
6. **Encode the CPU-fallback class** per device; the Coral single-partition rule should produce a much steeper penalty than a graceful per-op fallback, because the entire graph tail moves to the host CPU.

## Caveats
- Vendor TOPS are best-case synthetic peaks; several (NVIDIA sparse 275, Axelera 214 in M.2) are not reachable in the advertised form factor's thermal envelope. Always cross-check against MLPerf or independent FPS where available.
- nn-Meter shows analytical prediction is least accurate on VPU/NPU-class hardware (83.4% vs ~99% on CPU/GPU); treat scores for Myriad X, Coral, and obscure NPUs as lower-confidence and surface that.
- RB6, VCK190 achieved throughput, Kneron KL720/KL730, Blaize P1600, Furiosa, Brainchip Akida, Sophgo, Untether, Esperanto, and MediaTek's TOPS methodology are incompletely sourced and marked `[?]`; do not present their scores as high-confidence.
- Some community/forum YOLO FPS numbers conflate Hailo-8 / Hailo-8L / Hailo-10H; only Hailo's official Hailo-8 benchmark table and HailoRT logs were used for Hailo-8 figures here.
- The MLPerf-derived Orin utilization (~37% dense) is for a tuned ResNet-50; your default constants are deliberately conservative because most user models are less hardware-friendly than MLPerf submissions.
- Movidius Myriad X is EOL; include it only for legacy analysis and flag accordingly.