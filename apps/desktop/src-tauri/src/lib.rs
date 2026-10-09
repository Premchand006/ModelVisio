use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tauri::menu::{MenuBuilder, MenuItemBuilder, SubmenuBuilder};
use tauri::Emitter;
use tauri_plugin_dialog::DialogExt;

/// Extensions offered by File → Open Model. Mirrors the `exts` lists in
/// packages/parsers/src/registry.ts — keep the two in sync when adding a format.
const MODEL_EXTENSIONS: &[&str] = &[
    "onnx", "ort", "tflite", "lite", "tfl", "safetensors", "gguf", "ggml", "npy", "npz", "cfg",
    "weights", "pt", "pth", "ckpt", "bin", "ptl", "torchscript", "pt2", "pte", "mlmodel",
    "mlmodelc", "xml", "pb", "meta", "pbtxt", "keras", "h5", "hdf5", "json", "params",
    "caffemodel", "prototxt", "pdmodel", "pdparams", "nb", "param", "mnn", "tnnproto",
    "tnnmodel", "rknn", "engine", "plan", "trt", "uff", "mlir", "cntk", "dnn", "nn", "mge", "tm",
    "nntxt", "har", "hn", "om", "mlnet", "bigdl", "cbm", "pkl", "joblib", "pickle",
];

/// Payload emitted to the WebView when a model is opened via the native menu.
/// The web layer (apps/web/src/tauri.ts) turns this into a File and parses it.
#[derive(Clone, Serialize)]
struct ModelOpened {
    name: String,
    bytes: Vec<u8>,
}

#[derive(Deserialize)]
struct ChatMsg {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Source {
    title: String,
    url: String,
}

#[derive(Serialize)]
struct ChatResponse {
    text: String,
    sources: Vec<Source>,
}

/// AI copilot proxy for the desktop app — the desktop mirror of the web
/// /api/chat serverless function. The frontend (apps/web/src/tauri.ts) routes
/// its /api/chat POST here via `invoke("chat", ...)`, naming the provider the
/// user picked in the AI Copilot ("gemini" or "grok"; absent means Gemini, so
/// older frontends keep working). Keys stay on the Rust side of each call and
/// are never bundled with the app.
#[tauri::command]
async fn chat(
    provider: Option<String>,
    api_key: String,
    system: String,
    messages: Vec<ChatMsg>,
) -> Result<ChatResponse, String> {
    match provider.as_deref() {
        Some("grok") => grok_chat(&api_key, &system, &messages).await,
        _ => gemini_chat(&api_key, &system, &messages).await,
    }
}

/// Bring-your-own-key: the user's own key is entered in the app's AI settings,
/// stored locally in the WebView, and passed in per call. The environment is
/// only a convenience fallback for local dev (a maintainer running with the
/// provider's key variable set).
fn resolve_key(api_key: &str, env_vars: &[&str], missing: &str) -> Result<String, String> {
    let k = api_key.trim();
    if !k.is_empty() {
        return Ok(k.to_string());
    }
    env_vars
        .iter()
        .find_map(|v| std::env::var(v).ok().filter(|s| !s.trim().is_empty()))
        .ok_or_else(|| missing.to_string())
}

/// Google Gemini (generateContent), with Google Search grounding when enabled.
async fn gemini_chat(api_key: &str, system: &str, messages: &[ChatMsg]) -> Result<ChatResponse, String> {
    let key = resolve_key(
        api_key,
        &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "No Gemini API key set. Click the key icon in the AI Copilot and paste \
         your free key from https://aistudio.google.com/apikey.",
    )?;
    let model = std::env::var("MODELVISIO_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());
    // Grounding ON unless MODELVISIO_WEB_SEARCH=off, matching the web/VS Code
    // shells; the call falls back to ungrounded if grounding is unavailable.
    let web_search = std::env::var("MODELVISIO_WEB_SEARCH")
        .map(|v| v != "off")
        .unwrap_or(true);

    let contents: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| !m.content.is_empty())
        .map(|m| {
            serde_json::json!({
                "role": if m.role == "assistant" { "model" } else { "user" },
                "parts": [{ "text": m.content }],
            })
        })
        .collect();

    let mut base = serde_json::json!({
        "contents": contents,
        "generationConfig": { "maxOutputTokens": 4096, "temperature": 0.4 },
    });
    if !system.is_empty() {
        base["systemInstruction"] = serde_json::json!({ "parts": [{ "text": system }] });
    }

    // Try grounded first (if enabled), then fall back to ungrounded.
    let mut payloads: Vec<serde_json::Value> = Vec::new();
    if web_search {
        let mut grounded = base.clone();
        grounded["tools"] = serde_json::json!([{ "google_search": {} }]);
        payloads.push(grounded);
    }
    payloads.push(base);

    let client = reqwest::Client::new();
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        model
    );
    let mut last_err = String::from("Gemini request failed");
    for payload in payloads {
        let resp = client
            .post(&url)
            .header("x-goog-api-key", &key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = resp.status();
        let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            last_err = format!(
                "Gemini API error {}: {}",
                status.as_u16(),
                data["error"]["message"].as_str().unwrap_or("request failed")
            );
            continue;
        }
        let cand = &data["candidates"][0];
        let text: String = cand["content"]["parts"]
            .as_array()
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        let mut sources: Vec<Source> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        if let Some(chunks) = cand["groundingMetadata"]["groundingChunks"].as_array() {
            for c in chunks {
                if let Some(u) = c["web"]["uri"].as_str() {
                    if seen.insert(u.to_string()) {
                        let title = c["web"]["title"].as_str().unwrap_or(u).to_string();
                        sources.push(Source {
                            title,
                            url: u.to_string(),
                        });
                    }
                }
            }
        }
        return Ok(ChatResponse {
            text: if text.is_empty() {
                "(empty response)".to_string()
            } else {
                text
            },
            sources,
        });
    }
    Err(last_err)
}

/// xAI's OpenAI-compatible chat-completions endpoint. Keys: https://console.x.ai .
const XAI_CHAT_URL: &str = "https://api.x.ai/v1/chat/completions";

/// Models that accept `reasoning_effort`: the grok-4.x line and *-reasoning ids
/// (mirrors `supportsReasoningEffort` in packages/ai/src/grok.ts).
fn grok_supports_reasoning_effort(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    let grok_4x = m
        .strip_prefix("grok-4.")
        .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
    grok_4x || (m.contains("reasoning") && !m.contains("non-reasoning"))
}

/// xAI answers errors as `{ code, error: "msg" }`; OpenAI-style
/// `{ error: { message } }` is also seen.
fn grok_error_text(data: &serde_json::Value) -> String {
    data["error"]
        .as_str()
        .or_else(|| data["error"]["message"].as_str())
        .map(str::to_string)
        .unwrap_or_else(|| data.to_string())
}

/// xAI Grok — the desktop mirror of `runGrokChat` (packages/ai/src/grok.ts).
/// Grok's chat-completions endpoint has no free built-in web search and returns
/// no citations for plain calls, so `sources` is always empty here (the web
/// shell's free crawler is not ported to desktop).
async fn grok_chat(api_key: &str, system: &str, messages: &[ChatMsg]) -> Result<ChatResponse, String> {
    let key = resolve_key(
        api_key,
        &["XAI_API_KEY"],
        "No Grok (xAI) API key set. Click the key icon in the AI Copilot and paste \
         your key from https://console.x.ai.",
    )?;
    let model = std::env::var("MODELVISIO_GROK_MODEL").unwrap_or_else(|_| "grok-4.7".to_string());

    // OpenAI-style messages with the system prompt first.
    let mut wire: Vec<serde_json::Value> = Vec::new();
    if !system.is_empty() {
        wire.push(serde_json::json!({ "role": "system", "content": system }));
    }
    wire.extend(messages.iter().filter(|m| !m.content.is_empty()).map(|m| {
        serde_json::json!({
            "role": if m.role == "assistant" { "assistant" } else { "user" },
            "content": m.content,
        })
    }));

    let mut payload = serde_json::json!({
        "model": model,
        "messages": wire,
        "temperature": 0.4,
        "stream": false,
        "max_tokens": 4096,
    });
    // Low reasoning effort keeps answers fast on models that think by default.
    if grok_supports_reasoning_effort(&model) {
        payload["reasoning_effort"] = serde_json::json!("low");
    }

    let client = reqwest::Client::new();
    loop {
        let resp = client
            .post(XAI_CHAT_URL)
            .bearer_auth(&key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = resp.status().as_u16();
        let data: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
        if (200..300).contains(&status) {
            let text = data["choices"][0]["message"]["content"].as_str().unwrap_or("");
            return Ok(ChatResponse {
                text: if text.is_empty() {
                    "(empty response)".to_string()
                } else {
                    text.to_string()
                },
                sources: vec![],
            });
        }
        let msg = grok_error_text(&data);
        // A model that rejects reasoning_effort: drop it once and resend at once.
        if status == 400 && msg.to_ascii_lowercase().contains("reasoning") {
            if let Some(obj) = payload.as_object_mut() {
                if obj.remove("reasoning_effort").is_some() {
                    continue;
                }
            }
        }
        return Err(match status {
            401 | 403 => format!(
                "Grok API key rejected ({status}). Check your key at https://console.x.ai. Original: {msg}"
            ),
            404 => format!("Grok model \"{model}\" is not available to this key (404): {msg}"),
            429 => format!("Grok rate limit (429) on \"{model}\": {msg}"),
            _ => format!("Grok API error {status}: {msg}"),
        });
    }
}

/// Check the configured updater endpoint (the GitHub Release `latest.json`) and,
/// if a newer signed build is available, ask the user before downloading and
/// installing it (installing restarts the app, and on Windows closes it to run
/// the installer — never do that under someone mid-analysis). Verified against
/// the public key in tauri.conf.json. Desktop-only.
#[cfg(desktop)]
async fn check_for_updates(app: tauri::AppHandle) -> tauri_plugin_updater::Result<()> {
    use tauri_plugin_dialog::{MessageDialogButtons, MessageDialogKind};
    use tauri_plugin_updater::UpdaterExt;

    let Some(update) = app.updater()?.check().await? else {
        return Ok(());
    };
    let prompt = format!(
        "ModelVisio {} is available (you have {}).

Install it now? The app will restart.",
        update.version, update.current_version
    );
    let dialog_app = app.clone();
    let accepted = tauri::async_runtime::spawn_blocking(move || {
        dialog_app
            .dialog()
            .message(prompt)
            .title("Update available")
            .kind(MessageDialogKind::Info)
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Update now".to_string(),
                "Later".to_string(),
            ))
            .blocking_show()
    })
    .await
    .unwrap_or(false);
    if !accepted {
        return Ok(());
    }
    update
        .download_and_install(|_chunk, _total| {}, || {})
        .await?;
    app.restart();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![chat])
        .setup(|app| {
            // Desktop auto-update: register the updater and check the GitHub
            // Release `latest.json` in the background, mirroring Netron's
            // self-update. Mobile has no updater, so this is desktop-only.
            #[cfg(desktop)]
            {
                app.handle()
                    .plugin(tauri_plugin_updater::Builder::new().build())?;
                let handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = check_for_updates(handle).await {
                        eprintln!("ModelVisio: update check failed: {e}");
                    }
                });
            }

            let open_item = MenuItemBuilder::with_id("open_model", "Open Model…")
                .accelerator("CmdOrCtrl+O")
                .build(app)?;
            let file_menu = SubmenuBuilder::new(app, "File")
                .item(&open_item)
                .separator()
                .quit()
                .build()?;
            let menu = MenuBuilder::new(app).item(&file_menu).build()?;
            app.set_menu(menu)?;
            Ok(())
        })
        .on_menu_event(|app, event| {
            if event.id().as_ref() == "open_model" {
                let app_handle = app.clone();
                app.dialog()
                    .file()
                    .add_filter("Models", MODEL_EXTENSIONS)
                    .add_filter("All files", &["*"])
                    .pick_file(move |path| {
                        let Some(fp) = path else { return };
                        let Some(p) = fp.as_path() else { return };
                        let Ok(bytes) = std::fs::read(p) else { return };
                        let name = p
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("model.onnx")
                            .to_string();
                        let _ = app_handle.emit("model-opened", ModelOpened { name, bytes });
                    });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
