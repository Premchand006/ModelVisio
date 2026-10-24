//! Recently opened models (File → Open Recent).
//!
//! The list holds the last `MAX_RECENT` model paths, most recent first, and is
//! saved as JSON (`{"paths": [...]}`) in the app config dir
//! (`<config>/com.modelvisio.app/recent-models.json`). Only files opened through
//! Rust are recorded: File → Open Model… and Open Recent itself. Files dropped
//! on, or picked in, the WebView's own drop zone never reach Rust (the browser
//! gives no path), so they are simply not listed.
//!
//! This file is the pure list logic plus persistence; the menu and the Tauri
//! commands that use it live in lib.rs.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MAX_RECENT: usize = 10;
pub const FILE_NAME: &str = "recent-models.json";

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecentList {
    #[serde(default)]
    paths: Vec<String>,
}

/// One entry as the WebView sees it (`recent_models` command and the
/// `recent-models-changed` event).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecentModel {
    pub path: String,
    pub name: String,
}

/// Paths compare case-insensitively on Windows and macOS (default
/// filesystems), exactly elsewhere.
fn same_path(a: &str, b: &str) -> bool {
    if cfg!(any(windows, target_os = "macos")) {
        a.to_lowercase() == b.to_lowercase()
    } else {
        a == b
    }
}

pub fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string()
}

impl RecentList {
    /// Parse a saved list, tolerating junk (a corrupt file starts empty) and
    /// re-applying dedupe + cap in case it was edited by hand.
    pub fn from_json(s: &str) -> Self {
        let saved: RecentList = serde_json::from_str(s).unwrap_or_default();
        let mut list = RecentList::default();
        for p in saved.paths.iter().rev() {
            list.push(p);
        }
        list
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{\"paths\":[]}".to_string())
    }

    pub fn paths(&self) -> &[String] {
        &self.paths
    }

    pub fn contains(&self, path: &str) -> bool {
        self.paths.iter().any(|p| same_path(p, path))
    }

    /// Move `path` to the front (adding it if new), keeping at most MAX_RECENT.
    /// Returns whether the list changed.
    pub fn push(&mut self, path: &str) -> bool {
        if path.trim().is_empty() {
            return false;
        }
        if self.paths.first().is_some_and(|p| p == path) {
            return false;
        }
        self.paths.retain(|p| !same_path(p, path));
        self.paths.insert(0, path.to_string());
        self.paths.truncate(MAX_RECENT);
        true
    }

    pub fn remove(&mut self, path: &str) -> bool {
        let before = self.paths.len();
        self.paths.retain(|p| !same_path(p, path));
        self.paths.len() != before
    }

    pub fn clear(&mut self) -> bool {
        let changed = !self.paths.is_empty();
        self.paths.clear();
        changed
    }

    pub fn entries(&self) -> Vec<RecentModel> {
        self.paths
            .iter()
            .map(|p| RecentModel { path: p.clone(), name: file_name(p) })
            .collect()
    }
}

/// Load the saved list; a missing or unreadable file is an empty list.
pub fn load(file: &Path) -> RecentList {
    std::fs::read_to_string(file)
        .map(|s| RecentList::from_json(&s))
        .unwrap_or_default()
}

/// Save via a temp file + rename so a crash mid-write can't leave half a file.
pub fn save(file: &Path, list: &RecentList) -> std::io::Result<()> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp: PathBuf = file.with_extension("json.tmp");
    std::fs::write(&tmp, list.to_json())?;
    std::fs::rename(&tmp, file)
}

/// Menu label for a path: "yolov8n.onnx (C:\models\yolo)". Long folders are
/// shortened in the middle. Menus have no tooltips, so the folder is the
/// disambiguator. `&` is doubled because the native menus (muda) treat a single
/// `&` as a mnemonic marker on every platform.
pub fn menu_label(path: &str) -> String {
    let name = file_name(path);
    let dir = Path::new(path)
        .parent()
        .and_then(|d| d.to_str())
        .filter(|d| !d.is_empty())
        .map(|d| shorten_middle(d, 60));
    let label = match dir {
        Some(d) => format!("{name} ({d})"),
        None => name,
    };
    label.replace('&', "&&")
}

fn shorten_middle(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let head = max / 3;
    let tail = max - head - 1;
    let mut out: String = chars[..head].iter().collect();
    out.push('…');
    out.extend(&chars[chars.len() - tail..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_moves_to_front_dedupes_and_caps() {
        let mut l = RecentList::default();
        for i in 0..12 {
            assert!(l.push(&format!("/m/{i}.onnx")));
        }
        assert_eq!(l.paths().len(), MAX_RECENT);
        assert_eq!(l.paths()[0], "/m/11.onnx");
        assert_eq!(l.paths()[9], "/m/2.onnx");
        assert!(l.push("/m/5.onnx"));
        assert_eq!(l.paths()[0], "/m/5.onnx");
        assert_eq!(l.paths().iter().filter(|p| *p == "/m/5.onnx").count(), 1);
        assert!(!l.push("/m/5.onnx"), "already first: no change");
        assert!(!l.push("  "));
    }

    #[test]
    fn remove_and_clear_report_changes() {
        let mut l = RecentList::default();
        l.push("/a.onnx");
        l.push("/b.onnx");
        assert!(l.remove("/a.onnx"));
        assert!(!l.remove("/a.onnx"));
        assert!(l.contains("/b.onnx"));
        assert!(l.clear());
        assert!(!l.clear());
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_compare_case_insensitively() {
        let mut l = RecentList::default();
        l.push(r"C:\Models\A.onnx");
        l.push(r"c:\models\a.onnx");
        assert_eq!(l.paths(), [r"c:\models\a.onnx"]);
        assert!(l.contains(r"C:\MODELS\A.ONNX"));
    }

    #[test]
    fn json_round_trip_tolerates_junk() {
        let mut l = RecentList::default();
        l.push("/a.onnx");
        l.push("/b.tflite");
        assert_eq!(RecentList::from_json(&l.to_json()), l);
        assert_eq!(RecentList::from_json("not json"), RecentList::default());
        assert_eq!(RecentList::from_json("{}"), RecentList::default());
        let dup = RecentList::from_json(r#"{"paths":["/x","/y","/x"]}"#);
        assert_eq!(dup.paths(), ["/x", "/y"]);
    }

    #[test]
    fn save_and_load() {
        let dir = std::env::temp_dir().join(format!("modelvisio-recent-test-{}", std::process::id()));
        let file = dir.join(FILE_NAME);
        let mut l = RecentList::default();
        l.push("/a.onnx");
        save(&file, &l).unwrap();
        assert_eq!(load(&file), l);
        assert_eq!(load(&dir.join("missing.json")), RecentList::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn labels_and_entries() {
        let p = if cfg!(windows) { r"C:\models\R&D\yolo.onnx" } else { "/models/R&D/yolo.onnx" };
        let label = menu_label(p);
        assert!(label.starts_with("yolo.onnx ("), "{label}");
        assert!(label.contains("R&&D"), "{label}");
        let long = format!("/{}/net.onnx", "deep/".repeat(30));
        let label = menu_label(&long);
        assert!(label.contains('…') && label.chars().count() < 80, "{label}");
        let mut l = RecentList::default();
        l.push(p);
        assert_eq!(l.entries(), vec![RecentModel { path: p.to_string(), name: "yolo.onnx".into() }]);
    }
}
