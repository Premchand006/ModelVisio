export type SaveHandler = (data: Uint8Array | string, filename: string, mime: string) => void;

// Shells without working `<a download>` (VS Code WebView, Tauri) install a
// handler that forwards saves to a native Save dialog.
let saveHandler: SaveHandler | null = null;

/** Route every app download through `h` (e.g. a VS Code/Tauri native save). Pass null to restore the browser download. */
export function setSaveHandler(h: SaveHandler | null): void {
  saveHandler = h;
}

/** Trigger a download of binary or text data (browser download unless a save handler is set). */
export function downloadFile(data: Uint8Array | string, filename: string, mime = "application/octet-stream"): void {
  if (saveHandler) {
    saveHandler(data, filename, mime);
    return;
  }
  const part: BlobPart = typeof data === "string" ? data : (data.slice().buffer as ArrayBuffer);
  const blob = new Blob([part], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
