//! Universal Import & Export Bridge: seamlessly converts across all blueprint formats
//! (.json, .toml, .yaml, .bson, .ron, .aid), weights containers (.pkg, .safetensors),
//! unified packages (.aidpkg), compressed archives (.tar, .tar.gz, .tgz), and project directories.

use super::{
    aid_bson,
    aid_envelope::{AidFormatTag, pack_aid, unpack_aid},
    aid_json, aid_ron, aid_toml, aid_yaml,
    aidpkg::{create_aidpkg, read_aidpkg},
    archive::{TarEntry, create_tar_gz, read_tar, read_tar_gz},
    pkg_envelope::{PkgWeightsFormat, pack_pkg, unpack_pkg},
};
use crate::spec::AiDesignSpec;
use std::fs;
use std::path::Path;

/// Supported blueprint file formats for direct input and output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueprintFileFormat {
    Yaml,
    Json,
    Toml,
    Bson,
    Ron,
    NativeAid,
}

impl BlueprintFileFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "yaml" | "yml" => Some(Self::Yaml),
            "json" => Some(Self::Json),
            "toml" => Some(Self::Toml),
            "bson" => Some(Self::Bson),
            "ron" => Some(Self::Ron),
            "aid" => Some(Self::NativeAid),
            _ => None,
        }
    }

    pub fn default_extension(&self) -> &'static str {
        match self {
            Self::Yaml => "yaml",
            Self::Json => "json",
            Self::Toml => "toml",
            Self::Bson => "bson",
            Self::Ron => "ron",
            Self::NativeAid => "aid",
        }
    }
}

/// Automatically detects format and deserializes AiDesignSpec from any file path.
pub fn load_blueprint_auto(path: &Path) -> Result<AiDesignSpec, String> {
    let bytes =
        fs::read(path).map_err(|e| format!("Failed to read file '{}': {}", path.display(), e))?;
    load_blueprint_from_bytes(&bytes, path.to_str())
}

/// Automatically detects format and deserializes AiDesignSpec from raw bytes.
pub fn load_blueprint_from_bytes(
    bytes: &[u8],
    filename_hint: Option<&str>,
) -> Result<AiDesignSpec, String> {
    // 1. If it's a tar.gz / tar archive, unpack entries and find blueprint
    if let Ok(entries) = read_tar_gz(bytes)
        && let Some(spec_entry) = entries.iter().find(|e| {
            e.path.contains(".aid")
                || e.path.contains("blueprint")
                || e.path.ends_with(".yaml")
                || e.path.ends_with(".json")
                || e.path.ends_with(".toml")
                || e.path.ends_with(".bson")
        })
    {
        return load_blueprint_from_bytes(&spec_entry.data, Some(&spec_entry.path));
    }
    if let Ok(entries) = read_tar(bytes)
        && let Some(spec_entry) = entries.iter().find(|e| {
            e.path.contains(".aid")
                || e.path.contains("blueprint")
                || e.path.ends_with(".yaml")
                || e.path.ends_with(".json")
                || e.path.ends_with(".toml")
                || e.path.ends_with(".bson")
        })
    {
        return load_blueprint_from_bytes(&spec_entry.data, Some(&spec_entry.path));
    }

    // 2. If it's an .aidpkg binary package, extract embedded spec
    if let Ok(contents) = read_aidpkg(bytes) {
        return Ok(contents.spec);
    }

    // 3. Try native .aid container envelope or direct unpack
    if let Ok((spec, _)) = unpack_aid(bytes) {
        return Ok(spec);
    }

    // 4. Try format specific to filename hint if provided
    if let Some(name) = filename_hint {
        let lower = name.to_lowercase();
        if (lower.ends_with(".yaml") || lower.ends_with(".yml"))
            && let Ok(spec) = aid_yaml::from_yaml_slice(bytes)
        {
            return Ok(spec);
        }
        if lower.ends_with(".json")
            && let Ok(spec) = aid_json::from_json_slice(bytes)
        {
            return Ok(spec);
        }
        if lower.ends_with(".toml")
            && let Ok(spec) = aid_toml::from_toml_slice(bytes)
        {
            return Ok(spec);
        }
        if lower.ends_with(".bson")
            && let Ok(spec) = aid_bson::from_bson_slice(bytes)
        {
            return Ok(spec);
        }
        if lower.ends_with(".ron")
            && let Ok(spec) = aid_ron::from_ron_slice(bytes)
        {
            return Ok(spec);
        }
        if lower.ends_with(".py")
            && let Ok(src) = std::str::from_utf8(bytes)
            && let Ok(spec) = crate::importer::import_from_python_source(src, None)
        {
            return Ok(spec);
        }
    }

    // 5. Fallback cascade: try all individual deserializers
    if let Ok(spec) = aid_yaml::from_yaml_slice(bytes) {
        return Ok(spec);
    }
    if let Ok(spec) = aid_json::from_json_slice(bytes) {
        return Ok(spec);
    }
    if let Ok(spec) = aid_toml::from_toml_slice(bytes) {
        return Ok(spec);
    }
    if let Ok(spec) = aid_bson::from_bson_slice(bytes) {
        return Ok(spec);
    }
    if let Ok(spec) = aid_ron::from_ron_slice(bytes) {
        return Ok(spec);
    }
    if let Ok(src) = std::str::from_utf8(bytes)
        && let Ok(spec) = crate::importer::import_from_python_source(src, None)
    {
        return Ok(spec);
    }

    Err("Could not deserialize blueprint from bytes as any supported format (AID, AIDPKG, YAML, JSON, TOML, BSON, RON, TAR.GZ, PY)".to_string())
}

/// Saves an AiDesignSpec into any destination path, automatically determining the format
/// from the file extension (.yaml, .json, .toml, .bson, .ron, .aid, .aidpkg, .tar.gz).
pub fn save_blueprint_auto(
    spec: &AiDesignSpec,
    path: &Path,
    weights: Option<&[u8]>,
) -> Result<(), String> {
    let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let lower_name = filename.to_lowercase();

    // 1. Compressed TAR.GZ archive
    if lower_name.ends_with(".tar.gz") || lower_name.ends_with(".tgz") {
        let yaml_bytes = aid_yaml::to_yaml_bytes(spec).map_err(|e| e.to_string())?;
        let mut entries = vec![TarEntry {
            path: "blueprint.yaml".to_string(),
            data: yaml_bytes,
        }];
        if let Some(w) = weights {
            entries.push(TarEntry {
                path: "weights.safetensors".to_string(),
                data: w.to_vec(),
            });
        }
        let tar_gz = create_tar_gz(&entries);
        fs::write(path, tar_gz).map_err(|e| format!("Failed to write TAR.GZ archive: {}", e))?;
        return Ok(());
    }

    // 2. Native single-file unified container (.aidpkg)
    if lower_name.ends_with(".aidpkg") {
        let pkg_bytes = create_aidpkg(spec, weights, None)
            .map_err(|e| format!("Failed to create .aidpkg: {}", e))?;
        fs::write(path, pkg_bytes).map_err(|e| format!("Failed to write .aidpkg: {}", e))?;
        return Ok(());
    }

    // 3. Native weights container (.pkg)
    if lower_name.ends_with(".pkg") {
        if let Some(w) = weights {
            let pkg_bytes = pack_pkg(w, PkgWeightsFormat::Safetensors, spec.graph.nodes.len());
            fs::write(path, pkg_bytes).map_err(|e| format!("Failed to write .pkg: {}", e))?;
            return Ok(());
        } else {
            return Err("Cannot export .pkg without weights payload".to_string());
        }
    }

    // 4. Native blueprint container (.aid)
    if lower_name.ends_with(".aid") {
        let aid_bytes = pack_aid(spec, AidFormatTag::Yaml)?;
        fs::write(path, aid_bytes).map_err(|e| format!("Failed to write .aid: {}", e))?;
        return Ok(());
    }

    // 5. Direct file extensions
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("yaml")
        .to_lowercase();
    let bytes = match ext.as_str() {
        "yaml" | "yml" => aid_yaml::to_yaml_bytes(spec).map_err(|e| e.to_string())?,
        "json" => aid_json::to_json_bytes(spec)?,
        "toml" => aid_toml::to_toml_bytes(spec)?,
        "bson" => aid_bson::to_bson_bytes(spec)?,
        "ron" => aid_ron::to_ron_bytes(spec).map_err(|e| e.to_string())?,
        _ => aid_yaml::to_yaml_bytes(spec).map_err(|e| e.to_string())?,
    };

    fs::write(path, bytes)
        .map_err(|e| format!("Failed to write output to '{}': {}", path.display(), e))?;
    Ok(())
}

/// Exports a complete turnkey project directory containing blueprint in selected format, weights, manifest, and multi-backend code.
pub fn export_project_directory(
    spec: &AiDesignSpec,
    out_dir: &Path,
    format: BlueprintFileFormat,
    weights: Option<&[u8]>,
) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| {
        format!(
            "Failed to create project directory '{}': {}",
            out_dir.display(),
            e
        )
    })?;

    // 1. Write blueprint in requested format (.aid for native byte format, .yaml for YAML exports)
    let (spec_filename, spec_bytes) = match format {
        BlueprintFileFormat::Yaml => (
            "blueprint.yaml",
            aid_yaml::to_yaml_bytes(spec).map_err(|e| e.to_string())?,
        ),
        BlueprintFileFormat::Json => ("blueprint.json", aid_json::to_json_bytes(spec)?),
        BlueprintFileFormat::Toml => ("blueprint.toml", aid_toml::to_toml_bytes(spec)?),
        BlueprintFileFormat::Bson => ("blueprint.bson", aid_bson::to_bson_bytes(spec)?),
        BlueprintFileFormat::Ron => (
            "blueprint.ron",
            aid_ron::to_ron_bytes(spec).map_err(|e| e.to_string())?,
        ),
        BlueprintFileFormat::NativeAid => ("blueprint.aid", pack_aid(spec, AidFormatTag::Bson)?),
    };
    fs::write(out_dir.join(spec_filename), &spec_bytes)
        .map_err(|e| format!("Failed to write blueprint file: {}", e))?;

    // 2. Write manifest.json
    let manifest = serde_json::json!({
        "model_id": spec.metadata.model_id,
        "display_name": spec.metadata.display_name,
        "blueprint_file": spec_filename,
        "has_weights": weights.is_some(),
        "created_at_utc": "2026-10-26T12:00:00Z",
    });
    fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .map_err(|e| format!("Failed to write manifest.json: {}", e))?;

    // 3. Write weights if present (both raw safetensors and .pkg container)
    if let Some(w) = weights {
        fs::write(out_dir.join("weights.safetensors"), w)
            .map_err(|e| format!("Failed to write weights.safetensors: {}", e))?;
        let pkg_bytes = pack_pkg(w, PkgWeightsFormat::Safetensors, spec.graph.nodes.len());
        fs::write(out_dir.join("weights.pkg"), pkg_bytes)
            .map_err(|e| format!("Failed to write weights.pkg: {}", e))?;
    }

    // 4. Generate & write complete multi-backend code bundle
    let bundle = crate::codegen::generate_all_backends(spec);
    bundle
        .save_to_disk(out_dir)
        .map_err(|e| format!("Failed to emit generated code: {}", e))?;

    Ok(())
}

/// Imports blueprint and optional weights from an existing project directory.
pub fn import_project_directory(dir: &Path) -> Result<(AiDesignSpec, Option<Vec<u8>>), String> {
    if !dir.is_dir() {
        return Err(format!("'{}' is not a valid directory", dir.display()));
    }

    let candidates = [
        "blueprint.aid",
        "blueprint.yaml",
        "blueprint.json",
        "blueprint.toml",
        "blueprint.bson",
        "blueprint.ron",
        "model.aid",
        "model.yaml",
        "model.json",
        "blueprint.aid.yaml",
        "blueprint.aid.json",
        "blueprint.aid.toml",
        "blueprint.aid.bson",
        "blueprint.aid.ron",
        "model.aid.yaml",
        "model.aid.json",
        "model.aid.toml",
        "model.aid.bson",
        "model.aid.ron",
        "config.json",
    ];

    let mut found_spec = None;
    for cand in candidates {
        let cand_path = dir.join(cand);
        if cand_path.is_file()
            && let Ok(spec) = load_blueprint_auto(&cand_path)
        {
            found_spec = Some(spec);
            break;
        }
    }

    let spec = found_spec.ok_or_else(|| {
        format!(
            "No recognizable blueprint found in directory '{}'",
            dir.display()
        )
    })?;

    // Look for weights
    let weights_candidates = [
        "weights.safetensors",
        "model.safetensors",
        "weights.pkg",
        "model.pkg",
        "weights.bson",
        "model.bson",
    ];
    let mut found_weights = None;
    for w_cand in weights_candidates {
        let w_path = dir.join(w_cand);
        if w_path.is_file()
            && let Ok(bytes) = fs::read(&w_path)
        {
            if w_cand.ends_with(".pkg")
                && let Ok((_, raw_w)) = unpack_pkg(&bytes)
            {
                found_weights = Some(raw_w);
                break;
            }
            found_weights = Some(bytes);
            break;
        }
    }

    Ok((spec, found_weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_formats_roundtrip() {
        let spec = AiDesignSpec::default();

        // 1. JSON
        let json_bytes = aid_json::to_json_bytes(&spec).unwrap();
        let loaded_json = load_blueprint_from_bytes(&json_bytes, Some("model.json")).unwrap();
        assert_eq!(loaded_json.metadata.model_id, spec.metadata.model_id);

        // 2. TOML
        let toml_bytes = aid_toml::to_toml_bytes(&spec).unwrap();
        let loaded_toml = load_blueprint_from_bytes(&toml_bytes, Some("model.toml")).unwrap();
        assert_eq!(loaded_toml.metadata.model_id, spec.metadata.model_id);

        // 3. BSON
        let bson_bytes = aid_bson::to_bson_bytes(&spec).unwrap();
        let loaded_bson = load_blueprint_from_bytes(&bson_bytes, Some("model.bson")).unwrap();
        assert_eq!(loaded_bson.metadata.model_id, spec.metadata.model_id);

        // 4. Native .aid envelope
        let aid_bytes = pack_aid(&spec, AidFormatTag::Bson).unwrap();
        let loaded_aid = load_blueprint_from_bytes(&aid_bytes, Some("model.aid")).unwrap();
        assert_eq!(loaded_aid.metadata.model_id, spec.metadata.model_id);

        // 5. TAR.GZ bundle
        let entries = vec![TarEntry {
            path: "blueprint.aid.bson".to_string(),
            data: bson_bytes,
        }];
        let tar_gz = create_tar_gz(&entries);
        let loaded_targz = load_blueprint_from_bytes(&tar_gz, Some("archive.tar.gz")).unwrap();
        assert_eq!(loaded_targz.metadata.model_id, spec.metadata.model_id);
    }
}
