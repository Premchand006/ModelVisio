import { attrInt, attrInts } from "./attrs";
import { DYN, prod, type Shape } from "./shapes";

export type Cost = { flops: number; macs: number };

const ZERO: Cost = { flops: 0, macs: 0 };

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

  switch (op) {
    case "Conv":
    case "ConvInteger": {
      if (!x || !w || !outShape) return ZERO;
      const cin = x[1] === DYN ? 1 : x[1];
      const group = attrInt(attrs, "group", 1);
      const kernel = attrInts(attrs, "kernel_shape") ?? [w[2] ?? 1, w[3] ?? 1];
      const kArea = kernel.reduce((a, b) => a * b, 1);
      const macs = outN * (cin / group) * kArea;
      return { macs, flops: 2 * macs };
    }

    case "Gemm": {
      if (!x || !outShape) return ZERO;
      const transA = attrInt(attrs, "transA", 0);
      const K = (transA ? x[0] : x[1]) ?? DYN;
      const k = K === DYN ? 1 : K;
      const macs = outN * k;
      return { macs, flops: 2 * macs };
    }

    case "MatMul": {
      if (!x || !outShape) return ZERO;
      const K = x[x.length - 1] === DYN ? 1 : x[x.length - 1];
      const macs = outN * K;
      return { macs, flops: 2 * macs };
    }

    case "MaxPool":
    case "AveragePool": {
      if (!outShape) return ZERO;
      const kernel = attrInts(attrs, "kernel_shape") ?? [1, 1];
      const kArea = kernel.reduce((a, b) => a * b, 1);
      return { macs: 0, flops: outN * kArea };
    }

    case "GlobalAveragePool":
    case "GlobalMaxPool":
      return { macs: 0, flops: x ? prod(x) : 0 };

    case "BatchNormalization":
    case "InstanceNormalization":
    case "LayerNormalization":
      return { macs: 0, flops: 2 * outN };

    case "Add":
    case "Sub":
    case "Mul":
    case "Div":
    case "Relu":
    case "LeakyRelu":
    case "Sigmoid":
    case "Tanh":
    case "Clip":
    case "Elu":
    case "Erf":
    case "Gelu":
    case "Pow":
      return { macs: 0, flops: outN };

    case "Softmax":
    case "LogSoftmax":
      return { macs: 0, flops: 3 * outN };

    default:
      return ZERO;
  }
}
