use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::menu::{Menu, MenuBuilder, MenuItemBuilder, PredefinedMenuItem, Submenu, SubmenuBuilder};
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tauri_plugin_dialog::DialogExt;

mod companions;
mod recent;
mod scrape;

/// Extensions offered by File → Open Model. Mirrors the `exts` lists in
/// packages/parsers/src/registry.ts. Keep the two in sync when adding a format.
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
    /// Sibling weight files (OpenVINO .bin, Paddle .pdiparams, ...) read from the same folder.
    companions: Vec<Companion>,
}

#[derive(Clone, Serialize)]
struct Companion {
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

/// AI copilot proxy for the desktop app, the desktop mirror of the web /api/chat
/// serverless function. The frontend (apps/web/src/tauri.ts) routes its /api/chat
/// POST here via `invoke("chat", ...)`, naming the provider the user picked in
/// the AI Copilot ("gemini" or "grok"; absent means Gemini, so older frontends
/// keep working). Keys stay on the Rust side of each call and are never bundled
/// with the app.
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

/// URL reader for the copilot: the desktop mirror of the web /api/scrape
/// function (packages/ai/src/scrape.ts). The frontend (apps/web/src/tauri.ts)
/// routes its `POST /api/scrape {url}` here via `invoke("scrape", { url })`.
/// Returns the same JSON the web proxy does: a page
/// `{url, finalUrl, title, text, truncated, bytes, contentType}`, or rejects
/// with `{error, status}` (400 = blocked/bad URL, 502 = origin failure).
/// SSRF guard, limits and MODELVISIO_SCRAPE_* knobs: see src/scrape.rs.
#[tauri::command]
async fn scrape(url: String) -> Result<scrape::ScrapedPage, scrape::ScrapeError> {
    scrape::scrape_url(&url, &scrape::ScrapeOptions::from_env()).await
}

// ─── Recent models (File → Open Recent) ──────────────────────────────────────

const FILE_MENU_ID: &str = "file";
const OPEN_RECENT_ID: &str = "open_recent";
const RECENT_ITEM_PREFIX: &str = "recent_model:";
const RECENT_CLEAR_ID: &str = "recent_clear";

/// The recent list plus where it is saved (None if the config dir is unknown;
/// the list then lives for this session only).
struct RecentState {
    list: Mutex<recent::RecentList>,
    file: Option<PathBuf>,
}

impl RecentState {
    fn snapshot(&self) -> recent::RecentList {
        self.list.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Replace the Open Recent submenu's items with `paths` (most recent first),
/// then a separator and "Clear Recently Opened". Each entry's menu id carries
/// its full path, so a click opens exactly what was shown.
fn fill_recent_menu<R: Runtime>(sub: &Submenu<R>, paths: &[String]) -> tauri::Result<()> {
    let app = sub.app_handle();
    while sub.remove_at(0)?.is_some() {}
    if paths.is_empty() {
        let none = MenuItemBuilder::with_id("recent_none", "No Recent Models")
            .enabled(false)
            .build(app)?;
        sub.append(&none)?;
    }
    for p in paths {
        let item = MenuItemBuilder::with_id(format!("{RECENT_ITEM_PREFIX}{p}"), recent::menu_label(p))
            .build(app)?;
        sub.append(&item)?;
    }
    sub.append(&PredefinedMenuItem::separator(app)?)?;
    let clear = MenuItemBuilder::with_id(RECENT_CLEAR_ID, "Clear Recently Opened")
        .enabled(!paths.is_empty())
        .build(app)?;
    sub.append(&clear)?;
    Ok(())
}

/// Rebuild File → Open Recent from the current list. Runs as one main-thread
/// task that reads the list itself, so concurrent updates can't interleave
/// their menu edits or leave an older snapshot on screen.
fn refresh_recent_menu<R: Runtime>(app: &AppHandle<R>) {
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        let Some(state) = handle.try_state::<RecentState>() else { return };
        let list = state.snapshot();
        let sub = handle
            .menu()
            .and_then(|m| m.get(FILE_MENU_ID))
            .and_then(|f| f.as_submenu().and_then(|f| f.get(OPEN_RECENT_ID)));
        if let Some(sub) = sub.as_ref().and_then(|s| s.as_submenu()) {
            if let Err(e) = fill_recent_menu(sub, list.paths()) {
                eprintln!("ModelVisio: couldn't rebuild Open Recent: {e}");
            }
        }
    });
}

/// Apply `change` to the recent list; if it changed, save it, rebuild the
/// menu and tell the WebView (`recent-models-changed`, payload: the entries).
fn update_recent<R: Runtime>(app: &AppHandle<R>, change: impl FnOnce(&mut recent::RecentList) -> bool) {
    let Some(state) = app.try_state::<RecentState>() else { return };
    let entries = {
        let mut list = state.list.lock().unwrap_or_else(|e| e.into_inner());
        if !change(&mut list) {
            return;
        }
        if let Some(file) = &state.file {
            if let Err(e) = recent::save(file, &list) {
                eprintln!("ModelVisio: couldn't save the recent models list: {e}");
            }
        }
        list.entries()
    };
    refresh_recent_menu(app);
    let _ = app.emit("recent-models-changed", entries);
}

/// Read a model from disk and hand it to the WebView (`model-opened`, the same
/// event File → Open Model uses), recording it in the recent list. A file that
/// no longer exists is dropped from the list. Errors are user-facing messages.
fn open_model_path<R: Runtime>(app: &AppHandle<R>, path: &Path) -> Result<(), String> {
    let shown = path.display().to_string();
    match std::fs::read(path) {
        Ok(bytes) => {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model.onnx")
                .to_string();
            let companions = companions::read_companions(path, &bytes)
                .into_iter()
                .map(|(name, bytes)| Companion { name, bytes })
                .collect();
            app.emit("model-opened", ModelOpened { name, bytes, companions })
                .map_err(|e| format!("Couldn't open {shown}: {e}"))?;
            if let Some(p) = path.to_str() {
                update_recent(app, |l| l.push(p));
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            update_recent(app, |l| l.remove(&shown));
            Err(format!("{shown} no longer exists, so it was removed from Open Recent."))
        }
        Err(e) => Err(format!("Couldn't open {shown}: {e}")),
    }
}

/// Non-blocking native warning dialog.
fn show_warning<R: Runtime>(app: &AppHandle<R>, title: &str, message: String) {
    use tauri_plugin_dialog::MessageDialogKind;
    app.dialog()
        .message(message)
        .title(title)
        .kind(MessageDialogKind::Warning)
        .show(|_| {});
}

/// The recently opened models, most recent first: `[{ path, name }]`.
#[tauri::command]
fn recent_models(state: tauri::State<'_, RecentState>) -> Vec<recent::RecentModel> {
    state.snapshot().entries()
}

/// Open a model from the recent list (emits `model-opened`, like the menu).
/// Only listed paths are accepted, so the WebView can't use this to read
/// arbitrary files. Rejects with a message if the file is gone (it is then
/// removed from the list) or unreadable.
#[tauri::command]
async fn open_recent(app: AppHandle, path: String) -> Result<(), String> {
    let listed = app.state::<RecentState>().snapshot().contains(&path);
    if !listed {
        return Err(format!("{path} isn't in the recently opened list."));
    }
    tauri::async_runtime::spawn_blocking(move || open_model_path(&app, Path::new(&path)))
        .await
        .map_err(|e| e.to_string())?
}

/// Empty the recent list (same as File → Open Recent → Clear Recently Opened).
#[tauri::command]
fn clear_recent_models(app: AppHandle) {
    update_recent(&app, |l| l.clear());
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

/// xAI Grok: the desktop mirror of `runGrokChat` (packages/ai/src/grok.ts).
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
/// the installer; never do that under someone mid-analysis). Verified against
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

/// Native menu bar. Windows/Linux get a single File menu: WebView2 and
/// WebKitGTK handle the clipboard/undo shortcuts in text inputs natively.
/// macOS is different: keyboard shortcuts are dispatched through the menu bar,
/// so without an Edit menu Cmd+C/V/X/A/Z are dead in the WebView (users could
/// not paste their Gemini key), and the first submenu is always taken as the
/// application menu. So macOS gets the standard app/File/Edit/View/Window set.
/// File → Open Recent is filled from `recent` and rebuilt by `refresh_recent_menu`.
fn build_menu<R: Runtime>(app: &tauri::App<R>, recent: &[String]) -> tauri::Result<Menu<R>> {
    let open_item = MenuItemBuilder::with_id("open_model", "Open Model…")
        .accelerator("CmdOrCtrl+O")
        .build(app)?;
    let recent_menu = SubmenuBuilder::with_id(app, OPEN_RECENT_ID, "Open Recent").build()?;
    fill_recent_menu(&recent_menu, recent)?;

    #[cfg(target_os = "macos")]
    {
        use tauri::menu::{AboutMetadata, WINDOW_SUBMENU_ID};

        let pkg = app.package_info();
        let about = AboutMetadata {
            name: Some(pkg.name.clone()),
            version: Some(pkg.version.to_string()),
            copyright: app.config().bundle.copyright.clone(),
            ..Default::default()
        };
        let app_menu = SubmenuBuilder::new(app, "ModelVisio")
            .about(Some(about))
            .separator()
            .services()
            .separator()
            .hide()
            .hide_others()
            .show_all()
            .separator()
            .quit()
            .build()?;
        let file_menu = SubmenuBuilder::with_id(app, FILE_MENU_ID, "File")
            .item(&open_item)
            .item(&recent_menu)
            .separator()
            .close_window()
            .build()?;
        let edit_menu = SubmenuBuilder::new(app, "Edit")
            .undo()
            .redo()
            .separator()
            .cut()
            .copy()
            .paste()
            .select_all()
            .build()?;
        let view_menu = SubmenuBuilder::new(app, "View").fullscreen().build()?;
        // WINDOW_SUBMENU_ID makes `set_menu` register this as NSApp's windows
        // menu, so macOS appends its window list / Bring All to Front.
        let window_menu = SubmenuBuilder::with_id(app, WINDOW_SUBMENU_ID, "Window")
            .minimize()
            .maximize()
            .separator()
            .close_window()
            .build()?;
        MenuBuilder::new(app)
            .item(&app_menu)
            .item(&file_menu)
            .item(&edit_menu)
            .item(&view_menu)
            .item(&window_menu)
            .build()
    }

    #[cfg(not(target_os = "macos"))]
    {
        let file_menu = SubmenuBuilder::with_id(app, FILE_MENU_ID, "File")
            .item(&open_item)
            .item(&recent_menu)
            .separator()
            .quit()
            .build()?;
        MenuBuilder::new(app).item(&file_menu).build()
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            chat,
            scrape,
            recent_models,
            open_recent,
            clear_recent_models
        ])
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

            // Recent models: <app config dir>/recent-models.json.
            let file = app
                .path()
                .app_config_dir()
                .ok()
                .map(|d| d.join(recent::FILE_NAME));
            let list = file.as_deref().map(recent::load).unwrap_or_default();
            let paths = list.paths().to_vec();
            app.manage(RecentState { list: Mutex::new(list), file });

            app.set_menu(build_menu(app, &paths)?)?;
            Ok(())
        })
        .on_menu_event(|app, event| {
            let id = event.id().as_ref();
            if id == "open_model" {
                let app_handle = app.clone();
                app.dialog()
                    .file()
                    .add_filter("Models", MODEL_EXTENSIONS)
                    .add_filter("All files", &["*"])
                    .pick_file(move |path| {
                        let Some(fp) = path else { return };
                        let Some(p) = fp.as_path() else { return };
                        if let Err(msg) = open_model_path(&app_handle, p) {
                            show_warning(&app_handle, "Couldn't open model", msg);
                        }
                    });
            } else if let Some(path) = id.strip_prefix(RECENT_ITEM_PREFIX) {
                // Read off the main thread: models can be hundreds of MB.
                let app_handle = app.clone();
                let path = PathBuf::from(path);
                tauri::async_runtime::spawn_blocking(move || {
                    if let Err(msg) = open_model_path(&app_handle, &path) {
                        show_warning(&app_handle, "Couldn't open model", msg);
                    }
                });
            } else if id == RECENT_CLEAR_ID {
                update_recent(app, |l| l.clear());
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
