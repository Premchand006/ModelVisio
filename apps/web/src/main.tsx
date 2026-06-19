import React, { useCallback } from "react";
import ReactDOM from "react-dom/client";
import { App, type AppApi } from "@modelvisio/core";
import { installTauriChatProxy, isTauri, wireTauriMenu } from "./tauri";
import "./index.css";

// Under the desktop shell, route the copilot's /api/chat through the Rust side.
installTauriChatProxy();

let wired = false;

function Root() {
  // When running under Tauri, connect the native menu once the app is ready.
  const onReady = useCallback((api: AppApi) => {
    if (!wired && isTauri()) {
      wired = true;
      void wireTauriMenu(api.openFile);
    }
  }, []);
  return <App onReady={onReady} />;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
