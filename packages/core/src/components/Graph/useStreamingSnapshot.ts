import { useEffect, useReducer, useRef } from "react";

/**
 * While `streaming`, returns a snapshot of `value` that refreshes at most once
 * per `gapMs()` (trailing, so the latest value always lands); otherwise returns
 * `value` as-is. Used to rate-limit graph layout while a large model streams
 * in: each refresh costs a full dagre pass, so the gap adapts to how long the
 * last layout took. When streaming ends the final value passes straight through.
 */
export function useStreamingSnapshot<T>(value: T, streaming: boolean, gapMs: () => number): T {
  const snap = useRef(value);
  const lastAt = useRef(-Infinity);
  const [, bump] = useReducer((n: number) => n + 1, 0);

  if (!streaming) snap.current = value;

  useEffect(() => {
    if (!streaming || snap.current === value) return;
    const wait = lastAt.current + gapMs() - Date.now();
    const land = () => {
      snap.current = value;
      lastAt.current = Date.now();
      bump();
    };
    if (wait <= 0) {
      land();
      return;
    }
    const timer = setTimeout(land, wait);
    return () => clearTimeout(timer);
    // gapMs is read at schedule time on purpose; it is not a trigger.
  }, [value, streaming]);

  return streaming ? snap.current : value;
}
