import type { ModelLayer } from "../types";

// Per-op display metadata + quantization-sensitivity priors. Kept small and
// declarative; the AI copilot can later enrich `insight` per actual model.
type Meta = { math: string; insight: string; qSens: number; group: ModelLayer["group"] };

const TABLE: Record<string, Meta> = {
  Conv: { math: "y = W ∗ x + b", insight: "Convolution — usually the FLOP hotspot; quantization-sensitive.", qSens: 0.7, group: "backbone" },
  ConvTranspose: { math: "y = Wᵀ ∗ x", insight: "Transposed conv (upsampling). Watch output-shape rounding across compilers.", qSens: 0.7, group: "backbone" },
  Gemm: { math: "y = α·(A·B) + β·C", insight: "Dense/linear layer. Largest weight tensors; the classifier head.", qSens: 0.6, group: "head" },
  MatMul: { math: "y = A · B", insight: "Matrix multiply — attention/projection cost center in transformers.", qSens: 0.6, group: "head" },
  BatchNormalization: { math: "y = γ·(x−μ)/√(σ²+ε) + β", insight: "Fold into the preceding Conv at inference for best INT8 accuracy.", qSens: 0.35, group: "backbone" },
  Relu: { math: "y = max(0, x)", insight: "Activation; fuses into the preceding op on every edge runtime.", qSens: 0.15, group: "backbone" },
  LeakyRelu: { math: "y = max(αx, x)", insight: "Leaky activation; fuses on most runtimes.", qSens: 0.15, group: "backbone" },
  Sigmoid: { math: "y = 1/(1+e^{-x})", insight: "Saturating activation; quantize with care near the tails.", qSens: 0.3, group: "backbone" },
  Tanh: { math: "y = tanh(x)", insight: "Saturating activation.", qSens: 0.3, group: "backbone" },
  MaxPool: { math: "y = max over window", insight: "Spatial downsample, no params.", qSens: 0.1, group: "backbone" },
  AveragePool: { math: "y = mean over window", insight: "Spatial downsample, no params.", qSens: 0.1, group: "backbone" },
  GlobalAveragePool: { math: "y_c = mean_{h,w} x", insight: "Collapses spatial dims — bridge from backbone to head.", qSens: 0.2, group: "neck" },
  GlobalMaxPool: { math: "y_c = max_{h,w} x", insight: "Global pooling bridge.", qSens: 0.2, group: "neck" },
  Flatten: { math: "reshape to 2-D", insight: "No-op on most runtimes; pure reshape.", qSens: 0.0, group: "neck" },
  Reshape: { math: "reshape", insight: "Metadata-only on most runtimes.", qSens: 0.0, group: "neck" },
  Transpose: { math: "permute axes", insight: "Layout change; can be costly if it blocks fusion.", qSens: 0.0, group: "neck" },
  Concat: { math: "concat along axis", insight: "Joins feature maps; common in skip/neck connections.", qSens: 0.1, group: "neck" },
  Add: { math: "y = a + b", insight: "Elementwise add — residual connections.", qSens: 0.2, group: "backbone" },
  Mul: { math: "y = a · b", insight: "Elementwise multiply.", qSens: 0.2, group: "backbone" },
  Softmax: { math: "y_i = e^{x_i}/Σe^{x_j}", insight: "Normalizes logits to probabilities. Verify axis after conversion.", qSens: 0.3, group: "output" },
  LogSoftmax: { math: "log softmax", insight: "Log-probabilities.", qSens: 0.3, group: "output" },
};

const DEFAULT: Meta = { math: "—", insight: "", qSens: 0.2, group: "backbone" };

export function opMeta(op: string): Meta {
  return TABLE[op] ?? { ...DEFAULT, insight: `${op} operator.` };
}
