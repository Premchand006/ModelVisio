// Session store for the Benchmark tab. App unmounts inactive sidebar tabs, so
// run history, form settings and an in-flight run live here (module scope, per
// page session) instead of in component state. Framework-free: the component
// reads it through useSyncExternalStore.

import { startBenchmark, type BenchHandle, type StartArgs } from "./benchClient";
import { DEFAULT_CONFIG, type BenchConfig, type BenchProgress, type BenchResult } from "./types";

export type BenchState = {
  runs: Record<string, BenchResult[]>;
  configs: Record<string, BenchConfig>;
  errors: Record<string, string | null>;
  active: { modelKey: string; progress: BenchProgress | null; cancelling: boolean } | null;
  /** Hardware-DB device id used for the predicted-latency comparison. */
  refDevice: string;
};

/** Identity of a loaded model for history purposes. */
export const modelKey = (m: { name: string; sizeBytes: number }) => `${m.name}:${m.sizeBytes}`;

export type BenchStore = ReturnType<typeof createBenchStore>;

export function createBenchStore(start: (a: StartArgs) => BenchHandle = startBenchmark, refDevice = "orin_nano") {
  let state: BenchState = { runs: {}, configs: {}, errors: {}, active: null, refDevice };
  let handle: BenchHandle | null = null;
  const subs = new Set<() => void>();
  const set = (patch: Partial<BenchState>) => {
    state = { ...state, ...patch };
    subs.forEach((f) => f());
  };

  return {
    subscribe(f: () => void) {
      subs.add(f);
      return () => { subs.delete(f); };
    },
    getSnapshot: () => state,
    configFor: (key: string): BenchConfig => state.configs[key] ?? DEFAULT_CONFIG,
    setConfig(key: string, cfg: BenchConfig) {
      set({ configs: { ...state.configs, [key]: cfg } });
    },
    setRefDevice(id: string) {
      set({ refDevice: id });
    },
    /** Start a run. Only one runs at a time (concurrent runs would skew each other). */
    run(key: string, args: Omit<StartArgs, "onProgress">): boolean {
      if (state.active) return false;
      set({ active: { modelKey: key, progress: null, cancelling: false }, errors: { ...state.errors, [key]: null } });
      const h = start({
        ...args,
        onProgress: (p) => {
          if (state.active?.modelKey === key) set({ active: { ...state.active, progress: p } });
        },
      });
      handle = h;
      h.promise.then(
        (r) => {
          handle = null;
          set({ active: null, runs: { ...state.runs, [key]: [...(state.runs[key] ?? []), r] } });
        },
        (e: unknown) => {
          handle = null;
          const msg = e instanceof Error ? e.message : String(e);
          set({ active: null, errors: { ...state.errors, [key]: msg === "Cancelled" ? null : msg } });
        },
      );
      return true;
    },
    cancel() {
      if (!handle || !state.active) return;
      set({ active: { ...state.active, cancelling: true } });
      handle.cancel();
    },
    remove(key: string, id: string) {
      set({ runs: { ...state.runs, [key]: (state.runs[key] ?? []).filter((r) => r.id !== id) } });
    },
    clear(key: string) {
      set({ runs: { ...state.runs, [key]: [] } });
    },
  };
}

/** The app-wide store used by the Benchmark component. */
export const benchStore = /* @__PURE__ */ createBenchStore();
