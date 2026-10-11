//! Sibling weight files a model format keeps next to its graph file. Mirrors
//! `companionNames` in packages/parsers/src/companions.ts: keep the two in step.

use std::path::Path;

/// Relative paths (from the graph file's folder) to load alongside `name`.
pub fn companion_names(name: &str, head: &[u8]) -> Vec<String> {
    let lower = name.to_ascii_lowercase();
    let stem = match name.rfind('.') {
        Some(i) => &name[..i],
        None => name,
    };
    if lower.ends_with(".xml") || (lower.ends_with(".param") && !lower.ends_with(".param.bin")) {
        return vec![format!("{stem}.bin")];
    }
    if lower.ends_with(".param.bin") {
        return vec![format!("{}.bin", &name[..name.len() - ".param.bin".len()])];
    }
    if lower.ends_with(".pdmodel") {
        return vec![format!("{stem}.pdiparams")];
    }
    if lower == "saved_model.pb" {
        return vec![
            "variables/variables.index".into(),
            "variables/variables.data-00000-of-00001".into(),
        ];
    }
    if lower.ends_with(".json") {
        // TF.js layers/graph model: weightsManifest[].paths, in order.
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(head) else { return vec![] };
        return v
            .get("weightsManifest")
            .and_then(|m| m.as_array())
            .map(|groups| {
                groups
                    .iter()
                    .filter_map(|g| g.get("paths").and_then(|p| p.as_array()))
                    .flatten()
                    .filter_map(|p| p.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
    }
    vec![]
}

/// Read the companions that exist next to `path`, skipping missing ones and
/// any relative path that would escape the model's folder.
pub fn read_companions(path: &Path, bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
    let Some(dir) = path.parent() else { return vec![] };
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    companion_names(name, bytes)
        .into_iter()
        .filter(|rel| !rel.split(['/', '\\']).any(|c| c == ".." || c.is_empty()) && !Path::new(rel).is_absolute())
        .filter_map(|rel| std::fs::read(dir.join(&rel)).ok().map(|b| (rel, b)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_per_format() {
        assert_eq!(companion_names("tinynet.xml", b""), vec!["tinynet.bin"]);
        assert_eq!(companion_names("net.param", b""), vec!["net.bin"]);
        assert_eq!(companion_names("net.param.bin", b""), vec!["net.bin"]);
        assert_eq!(companion_names("m.pdmodel", b""), vec!["m.pdiparams"]);
        assert_eq!(companion_names("model.onnx", b""), Vec::<String>::new());
        let json = br#"{"weightsManifest":[{"paths":["group1-shard1of2.bin","group1-shard2of2.bin"]}]}"#;
        assert_eq!(companion_names("model.json", json), vec!["group1-shard1of2.bin", "group1-shard2of2.bin"]);
    }

    #[test]
    fn refuses_paths_outside_the_folder() {
        let dir = std::env::temp_dir().join("mv-companions-test");
        std::fs::create_dir_all(&dir).unwrap();
        let json = br#"{"weightsManifest":[{"paths":["../secret.bin","/etc/passwd"]}]}"#;
        std::fs::write(dir.join("model.json"), json).unwrap();
        assert!(read_companions(&dir.join("model.json"), json).is_empty());
    }
}
