import type { Model, ModelLayer } from "../types";

// Shared plumbing for progressive (streaming) parsing. A streaming parser is a
// generator that yields ParseChunks as it works and returns the final Model.
// Chunks are provisional: the returned Model is always authoritative (it is
// exactly what the one-shot parser returns), so a consumer swaps it in at the end.

/** Model-level fields (everything except the graph itself). */
export type ModelHeader = Omit<Model, "layers" | "edges">;

export type ParseChunk = {
  /** New layers, in emission order; ids never repeat across chunks. */
  layers: ModelLayer[];
  /** New edges. Both endpoints are in this or an earlier chunk. */
  edges: [number, number][];
  /** Header fields known so far (sent early, then any fix-ups at the end). */
  header?: Partial<ModelHeader>;
  /** Fraction of the work done, 0..1. */
  progress: number;
};

export type ModelStream = Generator<ParseChunk, Model, void>;

/** Run a streaming parser to completion, ignoring its chunks. */
export function drain(stream: ModelStream): Model {
  for (;;) {
    const r = stream.next();
    if (r.done) return r.value;
  }
}

/** The header part of a finished Model. */
export function headerOf(m: Model): ModelHeader {
  const header: Partial<Model> = { ...m };
  delete header.layers;
  delete header.edges;
  return header as ModelHeader;
}

/**
 * Collects layers/edges as a parser produces them and cuts them into chunks.
 * With batchSize 0 it is inert (the one-shot path pays nothing for it).
 */
export class ChunkBuffer {
  private layers: ModelLayer[] = [];
  private edges: [number, number][] = [];
  private header: Partial<ModelHeader> | undefined;
  private readonly seen = new Set<string>();

  constructor(readonly batchSize: number) {}

  get enabled(): boolean {
    return this.batchSize > 0;
  }

  /** True once enough layers are buffered to emit a chunk. */
  get full(): boolean {
    return this.enabled && this.layers.length >= this.batchSize;
  }

  setHeader(h: Partial<ModelHeader>): void {
    if (this.enabled) this.header = { ...this.header, ...h };
  }

  layer(l: ModelLayer): void {
    if (this.enabled) this.layers.push(l);
  }

  edge(from: number, to: number): void {
    if (!this.enabled) return;
    const key = `${from}->${to}`;
    if (this.seen.has(key)) return;
    this.seen.add(key);
    this.edges.push([from, to]);
  }

  /** Cut the buffered layers/edges into a chunk and reset. */
  take(progress: number): ParseChunk {
    const c: ParseChunk = { layers: this.layers, edges: this.edges, progress };
    if (this.header) c.header = this.header;
    this.layers = [];
    this.edges = [];
    this.header = undefined;
    return c;
  }

  /**
   * Final chunk: whatever is still buffered, plus any edge of the final model
   * that was not emitted incrementally, plus the complete header (fix-ups such
   * as an inferred output shape).
   */
  finish(model: Model): ParseChunk {
    for (const [a, b] of model.edges) this.edge(a, b);
    this.setHeader(headerOf(model));
    return this.take(1);
  }
}

/**
 * Stream an already-built Model in slices of `batchSize` layers. Each edge is
 * emitted with the slice in which its later endpoint appears, so no chunk ever
 * references a layer that hasn't been sent yet. The first chunk carries the
 * header.
 */
export function* sliceModel(model: Model, batchSize: number): Generator<ParseChunk, void, void> {
  const n = model.layers.length;
  const size = Math.max(1, batchSize);
  const pos = new Map<number, number>();
  model.layers.forEach((l, i) => pos.set(l.id, i));
  const slices = Math.max(1, Math.ceil(n / size));
  const edgeBuckets: [number, number][][] = Array.from({ length: slices }, () => []);
  for (const e of model.edges) {
    const a = pos.get(e[0]), b = pos.get(e[1]);
    if (a == null || b == null) continue;
    edgeBuckets[Math.floor(Math.max(a, b) / size)].push(e);
  }
  for (let s = 0; s < slices; s++) {
    const c: ParseChunk = {
      layers: model.layers.slice(s * size, (s + 1) * size),
      edges: edgeBuckets[s],
      progress: (s + 1) / slices,
    };
    if (s === 0) c.header = headerOf(model);
    yield c;
  }
}
