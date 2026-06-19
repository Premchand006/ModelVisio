// Compact display formatting shared across components.

/** 1234567 → "1.23M", 1200 → "1.2K", small → the number. */
export function fmt(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "G";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return "" + n;
}

/** Byte count → "4.7MB" / "18.4KB" / "512B". */
export function fB(b: number): string {
  if (b >= 1e6) return (b / 1e6).toFixed(1) + "MB";
  if (b >= 1e3) return (b / 1e3).toFixed(1) + "KB";
  return b + "B";
}
