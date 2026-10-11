import { afterEach, describe, expect, it, vi } from "vitest";
import { downloadFile, setSaveHandler } from "../src/utils/download";

// Node env has no DOM — stub just enough of `document` / `URL` for the
// browser-download path so we can see which route downloadFile took.
function stubBrowser() {
  const anchor = { href: "", download: "", click: vi.fn(), remove: vi.fn() };
  vi.stubGlobal("document", {
    createElement: vi.fn(() => anchor),
    body: { appendChild: vi.fn() },
  });
  const createObjectURL = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:stub");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  return { anchor, createObjectURL };
}

afterEach(() => {
  setSaveHandler(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("downloadFile save hook", () => {
  it("delegates to the save handler with data, filename and mime", () => {
    const { createObjectURL } = stubBrowser();
    const handler = vi.fn();
    setSaveHandler(handler);
    const bytes = new Uint8Array([1, 2, 3]);
    downloadFile(bytes, "model.onnx", "application/x-onnx");
    downloadFile("print('hi')", "deploy.py", "text/x-python");
    expect(handler).toHaveBeenNthCalledWith(1, bytes, "model.onnx", "application/x-onnx");
    expect(handler).toHaveBeenNthCalledWith(2, "print('hi')", "deploy.py", "text/x-python");
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it("passes the default octet-stream mime when none is given", () => {
    const handler = vi.fn();
    setSaveHandler(handler);
    downloadFile("x", "a.bin");
    expect(handler).toHaveBeenCalledWith("x", "a.bin", "application/octet-stream");
  });

  it("setSaveHandler(null) restores the browser <a download> path", () => {
    const { anchor, createObjectURL } = stubBrowser();
    const handler = vi.fn();
    setSaveHandler(handler);
    setSaveHandler(null);
    downloadFile("hello", "notes.txt", "text/plain");
    expect(handler).not.toHaveBeenCalled();
    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(anchor.download).toBe("notes.txt");
    expect(anchor.href).toBe("blob:stub");
    expect(anchor.click).toHaveBeenCalledOnce();
  });
});
