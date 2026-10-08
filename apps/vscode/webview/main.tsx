import { useCallback, useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { App, setSaveHandler, type AppApi, type ThemeName } from "@modelvisio/core";
import type { HostToWebview } from "../src/protocol";
import { ErrorBanner } from "./ErrorBanner";
import { installFetchBridge } from "./fetchBridge";
import { createModelSink } from "./modelSink";
import { createRpc } from "./rpc";
import { createSaveHandler } from "./save";
import { post } from "./vscodeApi";
import { installWorkerShim, prefetchWorkerScripts, workerAssets } from "./workerShim";

// Platform glue only — all UI is core's App. Order matters: the fetch bridge,
// save handler and Worker shim are installed before the App mounts so core
// never sees the unpatched globals.
const rpc = createRpc(post);
const realFetch = installFetchBridge(window, rpc);
setSaveHandler(createSaveHandler(rpc, post));
window.addEventListener("message", (e: MessageEvent) => rpc.handle(e.data));

const sink = createModelSink();

// VS Code tags <body> with its theme kind; use it so the first paint matches.
const initialTheme: ThemeName = /\bvscode-(high-contrast-)?light\b/.test(document.body.className) ? "light" : "dark";

function Root() {
  const [theme, setTheme] = useState<ThemeName>(initialTheme);
  const [modelError, setModelError] = useState<{ name: string; error: string } | null>(null);

  const onReady = useCallback((a: AppApi) => sink.setApi(a), []);

  // Native form controls / scrollbars follow the app theme.
  useEffect(() => {
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      const msg = e.data as HostToWebview;
      if (!msg || typeof msg !== "object") return;
      if (msg.type === "theme") {
        setTheme(msg.theme);
      } else if (msg.type === "model") {
        setModelError(null);
        sink.push(msg.name, msg.bytes);
      } else if (msg.type === "modelError") {
        setModelError({ name: msg.name, error: msg.error });
      }
    };
    window.addEventListener("message", onMessage);
    post({ type: "ready" });
    return () => window.removeEventListener("message", onMessage);
  }, []);

  return (
    <>
      <App onReady={onReady} themeOverride={theme} initialTheme={initialTheme} />
      {modelError && (
        <ErrorBanner
          theme={theme}
          title={`Couldn't open ${modelError.name}`}
          message={modelError.error}
          onDismiss={() => setModelError(null)}
        />
      )}
    </>
  );
}

function mount() {
  ReactDOM.createRoot(document.getElementById("root")!).render(<Root />);
}

// Prefetch the parse worker before mounting so the shim can hand core a real,
// synchronously-constructed Worker (see workerShim.ts). import.meta.url is
// webview.js's URL — the same base core resolves its worker URL against. The
// prefetch is a local resource read (~ms); it is time-boxed, and on any
// failure we mount anyway and core parses on the main thread as before.
const hrefs = workerAssets().map((p) => new URL(p, import.meta.url).href);
prefetchWorkerScripts(hrefs, realFetch)
  .then((scripts) => installWorkerShim(window, scripts))
  .catch(() => {})
  .finally(mount);
