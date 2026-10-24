import { attrInt, attrInts } from "./attrs";
import { DYN, prod, type Shape } from "./shapes";

export type Cost = { flops: number; macs: number };

const ZERO: Cost = { flops: 0, macs: 0 };

// One FLOP per output element: elementwise arithmetic and activations.
const ELEMENTWISE = new Set([
  "Add", "Sub", "Mul", "Div", "Relu", "LeakyRelu", "PRelu", "Sigmoid", "HardSigmoid", "HardSwish", "Tanh",
  "Clip", "Elu", "Selu", "Celu", "Erf", "Gelu", "Mish", "Softplus", "Pow", "Sqrt", "Exp", "Log", "Neg", "Abs",
  "Reciprocal", "Max", "Min", "Mean", "Sum", "Where",
]);

/**
 * Estimate compute cost for a node. Uses output shape + input/weight shapes.
 * Counts MACs for the linear ops (the cost that dominates), FLOPs = 2·MACs
 * plus per-element work for elementwise ops. Per-sample (dynamic batch = 1).
 */
export function computeCost(
  op: string,
  attrs: Record<string, unknown>,
  inShapes: (Shape | undefined)[],
  outShape: Shape | undefined,
): Cost {
  const x = inShapes[0];
  const w = inShapes[1];
  const outN = outShape ? prod(outShape) : 0;
  const linear = (macs: number): Cost => ({ macs, flops: 2 * macs });

  if (ELEMENTWISE.has(op)) return { macs: 0, flops: outN };

  switch (op) {
    case "Conv":
    case "ConvInteger": {
      if (!outShape) return ZERO;
      // Weight is [Cout, Cin/group, k...]: every output element costs Cin/group·k MACs.
      // Reading it from the weight keeps the count right when the input shape is unknown.
      if (w && w.length >= 3 && w.every((d) => d !== DYN)) return linear(outN * prod(w.slice(1)));
      if (!x) return ZERO;
      const cin = x[1] === DYN ? 1 : x[1];
      const group = attrInt(attrs, "group", 1);
      const kernel = attrInts(attrs, "kernel_shape") ?? [w?.[2] ?? 1, w?.[3] ?? 1];
      return linear(outN * (cin / group) * prod(kernel));
    }

    case "ConvTranspose": {
      // Weight is [Cin, Cout/group, k...]: every input element scatters into Cout/group·k outputs.
      if (!x || !w || w.length < 3 || w.some((d) => d === DYN)) return ZERO;
      return linear(prod(x) * prod(w.slice(1)));
    }

    case "Gemm": {
      if (!outShape) return ZERO;
      const transA = attrInt(attrs, "transA", 0);
      const transB = attrInt(attrs, "transB", 0);
      const K = x ? (transA ? x[0] : x[1]) : w ? (transB ? w[1] : w[0]) : DYN;
      return linear(outN * (K === DYN || K == null ? 1 : K));
    }

    case "MatMul":
    case "MatMulInteger": {
      if (!outShape) return ZERO;
      const K = x ? x[x.length - 1] : w ? w[Math.max(0, w.length - 2)] : DYN;
      return linear(outN * (K === DYN || K == null ? 1 : K));
    }

    case "MaxPool":
    case "AveragePool": {
      if (!outShape) return ZERO;
      const kernel = attrInts(attrs, "kernel_shape") ?? [1, 1];
      return { macs: 0, flops: outN * prod(kernel) };
    }

    case "GlobalAveragePool":
    case "GlobalMaxPool":
      return { macs: 0, flops: x ? prod(x) : 0 };

    case "BatchNormalization":
    case "InstanceNormalization":
    case "LayerNormalization":
    case "GroupNormalization":
      return { macs: 0, flops: 2 * outN };

    case "Resize":
    case "Upsample":
      // Nearest is a gather; linear modes blend 2^spatial neighbours. Count the blend.
      return { macs: 0, flops: String(attrs.mode ?? "nearest") === "nearest" ? 0 : 4 * outN };

    case "Softmax":
    case "LogSoftmax":
      return { macs: 0, flops: 3 * outN };

    default:
      return ZERO;
  }
}
