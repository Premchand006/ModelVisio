use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tauri::menu::{MenuBuilder, MenuItemBuilder, SubmenuBuilder};
use tauri::Emitter;
use tauri_plugin_dialog::DialogExt;

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

/// AI copilot proxy for the desktop app. Holds GEMINI_API_KEY on the Rust side
/// (read from the environment that launched the app) so it never reaches the
/// WebView, and calls the Gemini API — the desktop mirror of the web /api/chat
/// serverless function. The frontend (apps/web/src/tauri.ts) routes its
/// /api/chat POST here via `invoke("chat", ...)`.
#[tauri::command]
async fn chat(api_key: String, system: String, messages: Vec<ChatMsg>) -> Result<ChatResponse, String> {
    // Bring-your-own-key: the user's own Gemini key is entered in the app's AI
    // settings, stored locally in the WebView, and passed in here per call. No
    // key is bundled in the app. The environment is only a convenience fallback
    // for local dev (a maintainer running with GEMINI_API_KEY set).
    let key = {
        let k = api_key.trim();
        if !k.is_empty() {
            k.to_string()
        } else {
            std::env::var("GEMINI_API_KEY")
                .or_else(|_| std::env::var("GOOGLE_API_KEY"))
                .map_err(|_| {
                    "No Gemini API key set. Click the key icon in the AI Copilot and paste \
                     your free key from https://aistudio.google.com/apikey."
                        .to_string()
                })?
        }
    };
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

/// Check the configured updater endpoint (the GitHub Release `latest.json`) and,
/// if a newer signed build is available, download and install it, then restart.
/// Verified against the public key in tauri.conf.json. Desktop-only.
#[cfg(desktop)]
async fn check_for_updates(app: tauri::AppHandle) -> tauri_plugin_updater::Result<()> {
    use tauri_plugin_updater::UpdaterExt;
    if let Some(update) = app.updater()?.check().await? {
        update
            .download_and_install(|_chunk, _total| {}, || {})
            .await?;
        app.restart();
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
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
                    .add_filter("Models", &["onnx"])
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
