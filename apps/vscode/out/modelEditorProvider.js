"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.ModelEditorProvider = void 0;
const vscode = __importStar(require("vscode"));
/**
 * Opens model files in a WebView running the @modelvisio/core app. Read-only:
 * we never write the model back. File bytes are read on the extension host and
 * pushed to the WebView, which parses them in its worker.
 */
class ModelEditorProvider {
    static register(context) {
        return vscode.window.registerCustomEditorProvider(ModelEditorProvider.viewType, new ModelEditorProvider(context), {
            webviewOptions: { retainContextWhenHidden: true },
            supportsMultipleEditorsPerDocument: false,
        });
    }
    constructor(context) {
        this.context = context;
    }
    openCustomDocument(uri) {
        return { uri, dispose: () => undefined };
    }
    async resolveCustomEditor(document, panel) {
        const webview = panel.webview;
        webview.options = {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")],
        };
        webview.html = this.getHtml(webview);
        const sendModel = async () => {
            const bytes = await vscode.workspace.fs.readFile(document.uri);
            const name = document.uri.path.split("/").pop() || "model";
            // base64 survives the JSON message bridge intact (Uint8Array would not).
            webview.postMessage({ type: "model", name, b64: Buffer.from(bytes).toString("base64") });
        };
        const subs = [];
        subs.push(webview.onDidReceiveMessage(async (msg) => {
            if (msg?.type === "ready") {
                webview.postMessage({ type: "theme", theme: currentTheme() });
                await sendModel();
            }
            else if (msg?.type === "chat") {
                await handleChat(webview, msg);
            }
        }), vscode.window.onDidChangeActiveColorTheme(() => {
            webview.postMessage({ type: "theme", theme: currentTheme() });
        }));
        panel.onDidDispose(() => subs.forEach((s) => s.dispose()));
    }
    getHtml(webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "webview.js"));
        const nonce = makeNonce();
        const csp = [
            `default-src 'none'`,
            `img-src ${webview.cspSource} data: blob:`,
            `script-src 'nonce-${nonce}'`,
            `style-src ${webview.cspSource} 'unsafe-inline' https://fonts.googleapis.com`,
            `font-src https://fonts.gstatic.com`,
            `connect-src https:`,
            `worker-src ${webview.cspSource} blob:`,
        ].join("; ");
        return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="${csp}" />
    <style>html,body,#root{height:100%;margin:0;padding:0}</style>
  </head>
  <body>
    <div id="root"></div>
    <script nonce="${nonce}" src="${scriptUri}"></script>
  </body>
</html>`;
    }
}
exports.ModelEditorProvider = ModelEditorProvider;
ModelEditorProvider.viewType = "modelvisio.modelViewer";
/**
 * AI copilot bridge: the webview forwards its /api/chat POST here. We hold the
 * Gemini key (from VS Code settings, never the webview) and call the API on the
 * extension host, then post the result back. Mirrors the web serverless proxy.
 */
async function handleChat(webview, msg) {
    const cfg = vscode.workspace.getConfiguration("modelvisio");
    const key = cfg.get("geminiApiKey")?.trim() || process.env.GEMINI_API_KEY;
    if (!key) {
        webview.postMessage({
            type: "chatError",
            id: msg.id,
            error: "No Gemini API key. Set 'ModelVisio: Gemini API Key' in VS Code Settings — get a free key at https://aistudio.google.com/apikey",
        });
        return;
    }
    const model = cfg.get("geminiModel")?.trim() || "gemini-2.5-flash";
    const webSearch = cfg.get("webSearch") ?? false;
    try {
        const out = await callGemini(key, model, webSearch, msg.system ?? "", msg.messages ?? []);
        webview.postMessage({ type: "chatResult", id: msg.id, text: out.text, sources: out.sources });
    }
    catch (e) {
        webview.postMessage({ type: "chatError", id: msg.id, error: e instanceof Error ? e.message : String(e) });
    }
}
async function callGemini(key, model, webSearch, system, messages) {
    const contents = messages
        .filter((m) => m && typeof m.content === "string" && m.content.length > 0)
        .map((m) => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] }));
    const base = { contents, generationConfig: { maxOutputTokens: 4096, temperature: 0.4 } };
    if (system)
        base.systemInstruction = { parts: [{ text: system }] };
    const payloads = webSearch ? [{ ...base, tools: [{ google_search: {} }] }, base] : [base];
    let lastErr = "Gemini request failed";
    for (const payload of payloads) {
        const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
            method: "POST",
            headers: { "content-type": "application/json", "x-goog-api-key": key },
            body: JSON.stringify(payload),
        });
        const data = (await r.json().catch(() => ({})));
        if (!r.ok) {
            lastErr = `Gemini API error ${r.status}: ${data?.error?.message || "request failed"}`;
            continue;
        }
        const cand = data.candidates?.[0];
        const text = (cand?.content?.parts ?? []).map((p) => p.text).filter(Boolean).join("");
        const chunks = cand?.groundingMetadata?.groundingChunks ?? [];
        const seen = new Set();
        const sources = [];
        for (const c of chunks) {
            const u = c?.web?.uri;
            if (u && !seen.has(u)) {
                seen.add(u);
                sources.push({ title: c.web?.title || u, url: u });
            }
        }
        return { text: text || "(empty response)", sources };
    }
    throw new Error(lastErr);
}
function currentTheme() {
    const k = vscode.window.activeColorTheme.kind;
    return k === vscode.ColorThemeKind.Light || k === vscode.ColorThemeKind.HighContrastLight
        ? "light"
        : "dark";
}
function makeNonce() {
    let text = "";
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (let i = 0; i < 32; i++)
        text += chars.charAt(Math.floor(Math.random() * chars.length));
    return text;
}
//# sourceMappingURL=modelEditorProvider.js.map