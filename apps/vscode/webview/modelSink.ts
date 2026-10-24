import type { AppApi } from "@modelvisio/core";

// Bridges host `model` messages to the App's openFile handle. The first model
// can arrive before the App calls onReady, so it is buffered until then. Every
// `model` message (including `reload: true` after an on-disk change) carries
// the latest bytes and re-opens them; the App re-parses.
type Companion = { name: string; bytes: Uint8Array };

export type ModelSink = {
  setApi(api: AppApi): void;
  push(name: string, bytes: Uint8Array, companions?: Companion[]): void;
};

export function createModelSink(toFile: (bytes: Uint8Array, name: string) => File = defaultFile): ModelSink {
  let api: AppApi | null = null;
  let pending: { name: string; bytes: Uint8Array; companions?: Companion[] } | null = null;
  const open = (a: AppApi, name: string, bytes: Uint8Array, companions?: Companion[]) => {
    const extra = companions?.map((c) => toFile(c.bytes, c.name));
    if (extra?.length) a.openFile(toFile(bytes, name), extra);
    else a.openFile(toFile(bytes, name));
  };

  return {
    setApi(a) {
      // onReady can fire more than once (core re-runs it when its handler
      // identity changes); only a buffered, not-yet-opened model is opened.
      api = a;
      if (pending) {
        const { name, bytes, companions } = pending;
        pending = null;
        open(a, name, bytes, companions);
      }
    },
    push(name, bytes, companions) {
      if (api) open(api, name, bytes, companions);
      else pending = { name, bytes, companions };
    },
  };
}

function defaultFile(bytes: Uint8Array, name: string): File {
  return new File([bytes as Uint8Array<ArrayBuffer>], name);
}
