import type { AppApi } from "@modelvisio/core";

// Bridges host `model` messages to the App's openFile handle. The first model
// can arrive before the App calls onReady, so it is buffered until then. Every
// `model` message (including `reload: true` after an on-disk change) carries
// the latest bytes and re-opens them; the App re-parses.
export type ModelSink = {
  setApi(api: AppApi): void;
  push(name: string, bytes: Uint8Array): void;
};

export function createModelSink(toFile: (bytes: Uint8Array, name: string) => File = defaultFile): ModelSink {
  let api: AppApi | null = null;
  let pending: { name: string; bytes: Uint8Array } | null = null;

  return {
    setApi(a) {
      // onReady can fire more than once (core re-runs it when its handler
      // identity changes); only a buffered, not-yet-opened model is opened.
      api = a;
      if (pending) {
        const { name, bytes } = pending;
        pending = null;
        a.openFile(toFile(bytes, name));
      }
    },
    push(name, bytes) {
      if (api) api.openFile(toFile(bytes, name));
      else pending = { name, bytes };
    },
  };
}

function defaultFile(bytes: Uint8Array, name: string): File {
  return new File([bytes as Uint8Array<ArrayBuffer>], name);
}
