import { parseModelStream, type Companions, type Model, type ModelLayer, type ParseChunk } from "@modelvisio/parsers";

// Helpers for progressive parsing, shared by the worker, the main-thread
// fallback and the UI. Pure (no React), so they are unit-testable.

/**
 * Merges small parser chunks into fewer, larger messages: flushes once
 * `maxLayers` layers are buffered or `maxMs` has passed since the last flush.
 * The first push always flushes, so the header reaches the UI immediately.
 */
export class ChunkCoalescer {
  private pending: ParseChunk | null = null;
  private last = -Infinity;

  constructor(
    private readonly emit: (chunk: ParseChunk) => void,
    private readonly maxLayers = 400,
    private readonly maxMs = 50,
    private readonly now: () => number = () => Date.now(),
  ) {}

  push(c: ParseChunk): void {
    const p = this.pending;
    if (!p) {
      this.pending = { layers: [...c.layers], edges: [...c.edges], progress: c.progress, ...(c.header ? { header: { ...c.header } } : {}) };
    } else {
      for (const l of c.layers) p.layers.push(l);
      for (const e of c.edges) p.edges.push(e);
      if (c.header) p.header = { ...p.header, ...c.header };
      p.progress = Math.max(p.progress, c.progress);
    }
    if (this.pending!.layers.length >= this.maxLayers || this.now() - this.last >= this.maxMs) this.flush();
  }

  flush(): void {
    const p = this.pending;
    this.pending = null;
    this.last = this.now();
    if (p) this.emit(p);
  }
}

const EMPTY_HEADER = {
  name: "", format: "", framework: "", opset: 0, sizeBytes: 0, inputShape: [] as number[], outputShape: [] as number[],
};

/**
 * Accumulates streamed chunks into a renderable partial Model. Edges whose
 * endpoints haven't arrived yet are held back until they do, so a snapshot
 * never references a missing node; duplicate layer ids are ignored.
 */
export class PartialModelBuilder {
  header: Partial<Model> = {};
  readonly layers: ModelLayer[] = [];
  readonly edges: [number, number][] = [];
  progress = 0;
  private ids = new Set<number>();
  private held: [number, number][] = [];

  push(c: ParseChunk): void {
    if (c.header) this.header = { ...this.header, ...c.header };
    for (const l of c.layers) {
      if (this.ids.has(l.id)) continue;
      this.ids.add(l.id);
      this.layers.push(l);
    }
    const pending = this.held.length ? [...this.held, ...c.edges] : c.edges;
    this.held = [];
    for (const e of pending) {
      if (this.ids.has(e[0]) && this.ids.has(e[1])) this.edges.push(e);
      else this.held.push(e);
    }
    this.progress = Math.max(this.progress, c.progress);
  }

  /** A fresh Model object (new arrays) for React state. */
  snapshot(): Model {
    return { ...EMPTY_HEADER, ...this.header, layers: this.layers.slice(), edges: this.edges.slice() };
  }
}

/** Yield to the event loop so the browser can handle input and paint. */
export function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Main-thread progressive parse (used when no Worker can be created): drives
 * the parser generator a batch at a time, emitting coalesced chunks and
 * yielding to the event loop whenever it has worked for `sliceMs`, so the
 * UI keeps painting while a large model loads.
 */
export async function parseStreamingOnMainThread(
  buffer: ArrayBuffer,
  name: string,
  onChunk: (chunk: ParseChunk) => void,
  opts: { batchSize?: number; sliceMs?: number; yieldFn?: () => Promise<void>; now?: () => number; companions?: Companions } = {},
): Promise<Model> {
  const now = opts.now ?? (() => Date.now());
  const sliceMs = opts.sliceMs ?? 12;
  const yieldFn = opts.yieldFn ?? yieldToEventLoop;
  const co = new ChunkCoalescer(onChunk, 400, 50, now);
  const it = parseModelStream(buffer, name, { batchSize: opts.batchSize ?? 64, companions: opts.companions });
  let sliceStart = now();
  for (;;) {
    const r = it.next();
    if (r.done) {
      co.flush();
      return r.value;
    }
    co.push(r.value);
    if (now() - sliceStart >= sliceMs) {
      co.flush();
      await yieldFn();
      sliceStart = now();
    }
  }
}

/**
 * Calls `fn` at most once per `ms`, trailing (the latest state always lands),
 * aligned to an animation frame when available. `cancel()` drops a pending call.
 */
export function throttleFrames(fn: () => void, ms = 250): { (): void; cancel: () => void } {
  let last = -Infinity;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let frame: number | null = null;
  const raf = typeof requestAnimationFrame === "function" ? requestAnimationFrame : null;
  const run = () => {
    timer = null;
    const go = () => { frame = null; last = Date.now(); fn(); };
    if (raf) frame = raf(go);
    else go();
  };
  const call = () => {
    if (timer != null || frame != null) return;
    timer = setTimeout(run, Math.max(0, last + ms - Date.now()));
  };
  call.cancel = () => {
    if (timer != null) clearTimeout(timer);
    if (frame != null && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
    timer = null;
    frame = null;
  };
  return call;
}
