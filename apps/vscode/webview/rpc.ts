import type { HostToWebview, WebviewToHost } from "../src/protocol";
import type { Post } from "./vscodeApi";

// Id-correlated request/response over postMessage. Each request gets a fresh id;
// the host echoes it on the reply, which resolves the matching promise.

type Req = Extract<WebviewToHost, { id: number }>;
type Rep = Extract<HostToWebview, { id: number }>;

/** Which reply message types answer each request type. */
type ReplyMap = { chat: "chatResult" | "chatError"; scrape: "scrapeResult" | "scrapeError"; save: "saveResult" };

export type RequestType = Req["type"];
/** A request minus its id (distributes over the union so each variant keeps its own fields). */
type OmitId<T> = T extends unknown ? Omit<T, "id"> : never;
export type RequestMsg = OmitId<Req>;
export type ReplyOf<T extends RequestType> = Extract<Rep, { type: ReplyMap[T] }>;

const REPLY_TYPES = new Set<string>(["chatResult", "chatError", "scrapeResult", "scrapeError", "saveResult"]);

/** Same shape fetch() rejects with on abort, so callers' AbortError checks work unchanged. */
export function abortError(): DOMException {
  return new DOMException("The operation was aborted.", "AbortError");
}

export type Rpc = {
  /** Post `msg` with a new id and resolve with the host's reply. Rejects with AbortError if `signal` fires first. */
  request<M extends RequestMsg>(msg: M, signal?: AbortSignal): Promise<ReplyOf<M["type"]>>;
  /** Feed a host message in; returns true if it was a reply this rpc consumed. */
  handle(msg: unknown): boolean;
  /** Outstanding requests (for tests / diagnostics). */
  pending(): number;
};

export function createRpc(post: Post): Rpc {
  // Random start: if the WebView reloads, the host may still answer requests
  // from the previous page; ids counting from 1 again would let those stale
  // replies resolve the new page's requests.
  let seq = Math.floor(Math.random() * 1e9);
  const waiting = new Map<number, (r: Rep) => void>();

  return {
    request<M extends RequestMsg>(msg: M, signal?: AbortSignal) {
      if (signal?.aborted) return Promise.reject(abortError());
      const id = ++seq;
      return new Promise<ReplyOf<M["type"]>>((resolve, reject) => {
        // On abort, drop the entry so a late reply is ignored (the protocol has
        // no cancel message; the host simply finishes and its answer is discarded).
        const onAbort = () => {
          waiting.delete(id);
          reject(abortError());
        };
        waiting.set(id, (r) => {
          signal?.removeEventListener("abort", onAbort);
          resolve(r as ReplyOf<M["type"]>);
        });
        signal?.addEventListener("abort", onAbort, { once: true });
        try {
          post({ ...msg, id } as unknown as Req);
        } catch (e) {
          waiting.delete(id);
          signal?.removeEventListener("abort", onAbort);
          reject(e);
        }
      });
    },
    handle(msg) {
      if (!msg || typeof msg !== "object") return false;
      const m = msg as { type?: unknown; id?: unknown };
      if (typeof m.type !== "string" || !REPLY_TYPES.has(m.type) || typeof m.id !== "number") return false;
      const done = waiting.get(m.id);
      if (!done) return true; // a reply to an aborted/unknown request: consumed, ignored
      waiting.delete(m.id);
      done(msg as Rep);
      return true;
    },
    pending: () => waiting.size,
  };
}
