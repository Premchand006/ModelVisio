// Latency statistics for benchmark samples. Pure: no React, no DOM.

export type LatencyStats = {
  count: number;
  mean: number;
  median: number;
  p90: number;
  p99: number;
  min: number;
  max: number;
  /** Sample standard deviation (n-1); 0 for a single sample. */
  std: number;
  /** Sequential inferences per second at the mean latency (1000 / mean). */
  throughput: number;
};

/**
 * Percentile of an ASCENDING-sorted array with linear interpolation between
 * closest ranks (NumPy's default "linear" method). `p` is in [0, 100].
 */
export function percentileSorted(sorted: readonly number[], p: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const q = Math.min(100, Math.max(0, p)) / 100;
  const pos = (n - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Percentile of unsorted samples (copies before sorting). */
export function percentile(samples: readonly number[], p: number): number {
  return percentileSorted([...samples].sort((a, b) => a - b), p);
}

/** Summarize latency samples in milliseconds. Non-finite samples are ignored. */
export function summarize(samplesMs: readonly number[]): LatencyStats {
  const xs = samplesMs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  const n = xs.length;
  if (n === 0) {
    return { count: 0, mean: NaN, median: NaN, p90: NaN, p99: NaN, min: NaN, max: NaN, std: NaN, throughput: 0 };
  }
  let sum = 0;
  for (const x of xs) sum += x;
  const mean = sum / n;
  let ss = 0;
  for (const x of xs) ss += (x - mean) * (x - mean);
  const std = n > 1 ? Math.sqrt(ss / (n - 1)) : 0;
  return {
    count: n,
    mean,
    median: percentileSorted(xs, 50),
    p90: percentileSorted(xs, 90),
    p99: percentileSorted(xs, 99),
    min: xs[0],
    max: xs[n - 1],
    std,
    throughput: mean > 0 ? 1000 / mean : 0,
  };
}
