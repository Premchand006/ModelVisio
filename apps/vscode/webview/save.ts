import type { SaveHandler } from "@modelvisio/core";
import type { Rpc } from "./rpc";
import type { Post } from "./vscodeApi";

// `<a download>` does nothing in a webview, so every core download is routed to
// the host, which shows a native Save dialog. The host notifies on success and
// a cancelled dialog is silent; only failures are surfaced from here.
export function createSaveHandler(rpc: Rpc, post: Post): SaveHandler {
  return (data, filename, mime) => {
    // A view into a larger buffer would ship the whole buffer; send just the bytes.
    const payload = typeof data !== "string" && (data.byteOffset !== 0 || data.byteLength !== data.buffer.byteLength) ? data.slice() : data;
    rpc.request({ type: "save", filename, mime, data: payload }).then(
      (r) => {
        if (r.error) post({ type: "notify", level: "error", message: `Couldn't save ${filename}: ${r.error}` });
      },
      (e: unknown) => post({ type: "notify", level: "error", message: `Couldn't save ${filename}: ${e instanceof Error ? e.message : String(e)}` }),
    );
  };
}
