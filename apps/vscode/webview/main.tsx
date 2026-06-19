import { useCallback, useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { App, type AppApi, type ThemeName } from "@modelvisio/core";

// Minimal VS Code WebView API surface we use.
type VsCodeApi = { postMessage: (msg: unknown) => void };
declare function acquireVsCodeApi(): VsCodeApi;
const vscode = acquireVsCodeApi();

// --- AI copilot bridge ----------------------------------------------------
// The copilot POSTs to /api/chat, which doesn't exist inside a webview. We
// intercept just that request and forward it to the extension host (which holds
// the Gemini key from VS Code settings and makes the real API call). Everything
// else uses the genuine fetch. This keeps core's Chat/sendChat unchanged.
type ChatReply = { text?: string; sources?: { title: string; url: string }[]; error?: string };
let chatSeq = 0;
const chatPending = new Map<number, (r: ChatReply) => void>();
const realFetch = window.fetch.bind(window);
window.fetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const method = (init?.method || (typeof input === "object" && "method" in input ? input.method : "GET") || "GET").toUpperCase();
  if (url && /\/api\/chat$/.test(url) && method === "POST") {
    let body: { system?: string; messages?: unknown[] } = {};
    try { body = JSON.parse((init?.body as string) || "{}"); } catch { /* ignore */ }
    const id = ++chatSeq;
    return new Promise<ChatReply>((resolve) => {
      chatPending.set(id, resolve);
      vscode.postMessage({ type: "chat", id, system: body.system, messages: body.messages });
    }).then((reply) => new Response(JSON.stringify(reply), {
      status: reply.error ? 500 : 200,
      headers: { "content-type": "application/json" },
    }));
  }
  return realFetch(input, init);
};

function b64ToBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return buf;
}

function Root() {
  const [theme, setTheme] = useState<ThemeName>("dark");
  const apiRef = useRef<AppApi | null>(null);
  const pending = useRef<{ name: string; b64: string } | null>(null);

  const open = (name: string, b64: string) =>
    apiRef.current?.openFile(new File([b64ToBuffer(b64)], name));

  const onReady = useCallback((a: AppApi) => {
    apiRef.current = a;
    if (pending.current) {
      const { name, b64 } = pending.current;
      pending.current = null;
      a.openFile(new File([b64ToBuffer(b64)], name));
    }
  }, []);

  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      const msg = e.data as { type?: string; name?: string; b64?: string; theme?: ThemeName; id?: number; text?: string; sources?: { title: string; url: string }[]; error?: string };
      if (msg.type === "theme" && msg.theme) {
        setTheme(msg.theme);
      } else if (msg.type === "model" && msg.name && msg.b64) {
        // The model may arrive before the app calls onReady — buffer it.
        if (apiRef.current) open(msg.name, msg.b64);
        else pending.current = { name: msg.name, b64: msg.b64 };
      } else if ((msg.type === "chatResult" || msg.type === "chatError") && msg.id != null) {
        const resolve = chatPending.get(msg.id);
        if (resolve) {
          chatPending.delete(msg.id);
          resolve(msg.type === "chatError" ? { error: msg.error } : { text: msg.text, sources: msg.sources });
        }
      }
    };
    window.addEventListener("message", onMessage);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", onMessage);
  }, []);

  return <App onReady={onReady} themeOverride={theme} initialTheme={theme} />;
}

ReactDOM.createRoot(document.getElementById("root")!).render(<Root />);
