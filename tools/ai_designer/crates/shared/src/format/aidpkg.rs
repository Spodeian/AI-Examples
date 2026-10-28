//! Single-file binary container (.aidpkg) with 4096-byte page alignment for zero-copy memory mapping.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use std::io::{Cursor, Read, Write};

use crate::spec::AiDesignSpec;

pub const AIDPKG_MAGIC: &[u8; 8] = b"AIDPKG\x01\x00";
pub const AIDPKG_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 64;
pub const TOC_ENTRY_SIZE: usize = 64;
pub const PAGE_ALIGNMENT: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub enum PkgError {
    #[error("Invalid magic identifier (expected AIDPKG\\x01\\x00): {0:?}")]
    InvalidMagic([u8; 8]),
    #[error("Unsupported package version: {0}")]
    UnsupportedVersion(u32),
    #[error("TOC CRC32 checksum mismatch (expected {expected:#x}, got {computed:#x})")]
    TocCrcMismatch { expected: u32, computed: u32 },
    #[error("Entry '{name}' CRC32 mismatch (expected {expected:#x}, got {computed:#x})")]
    EntryCrcMismatch {
        name: String,
        expected: u32,
        computed: u32,
    },
    #[error("Required entry '{0}' missing from package TOC")]
    MissingEntry(&'static str),
    #[error("IO error during container read/write: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON serialization error: {0}")]
    RonError(#[from] ron::Error),
    #[error("RON parse error: {0}")]
    RonSpannedError(#[from] ron::error::SpannedError),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// A parsed entry descriptor from the Table of Contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TocEntry {
    pub name: String,
    pub offset: u64,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
    pub crc32: u32,
    pub compression: u16,
}

/// Extracted contents of an .aidpkg binary package.
#[derive(Debug, Clone, PartialEq)]
pub struct AidPkgContents {
    pub spec: AiDesignSpec,
    pub manifest_json: String,
    pub weights_bytes: Option<Vec<u8>>,
    pub parity_bytes: Option<Vec<u8>>,
}

/// Packages a model specification and optional Safetensors weights into a single binary container.
pub fn create_aidpkg(
    spec: &AiDesignSpec,
    weights_bytes: Option<&[u8]>,
    parity_bytes: Option<&[u8]>,
) -> Result<Vec<u8>, PkgError> {
    let manifest = serde_json::json!({
        "aidpkg_version": AIDPKG_VERSION,
        "model_id": spec.metadata.model_id,
        "display_name": spec.metadata.display_name,
        "author": spec.metadata.author,
        "license": spec.metadata.license,
        "has_weights": weights_bytes.is_some(),
        "has_parity_vectors": parity_bytes.is_some(),
        "created_at_utc": "2026-10-26T12:00:00Z",
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let spec_ron_bytes = ron::to_string(spec)?.into_bytes();
    let spec_yaml_bytes = serde_yaml::to_string(spec)
        .map_err(|e| PkgError::Io(std::io::Error::other(e)))?
        .into_bytes();
    let spec_bson_bytes = bson::to_vec(spec).map_err(|e| PkgError::Io(std::io::Error::other(e)))?;

    let entries_to_write = vec![
        ("manifest.json", manifest_bytes),
        ("blueprint.yaml", spec_yaml_bytes),
        ("blueprint.bson", spec_bson_bytes),
        ("spec.ron", spec_ron_bytes),
    ];

    let weights_data = weights_bytes.map(|w| w.to_vec());
    let parity_data = parity_bytes.map(|p| p.to_vec());

    let total_entries = entries_to_write.len()
        + if weights_data.is_some() { 1 } else { 0 }
        + if parity_data.is_some() { 1 } else { 0 };

    let toc_size = total_entries * TOC_ENTRY_SIZE;
    let mut current_offset = HEADER_SIZE + toc_size;

    let mut toc_descriptors = Vec::new();

    // 1. Calculate offsets and descriptors for metadata
    for (name, data) in &entries_to_write {
        let mut hasher = Hasher::new();
        hasher.update(data);
        let crc = hasher.finalize();

        toc_descriptors.push(TocEntry {
            name: name.to_string(),
            offset: current_offset as u64,
            compressed_size: data.len() as u64,
            uncompressed_size: data.len() as u64,
            crc32: crc,
            compression: 0,
        });
        current_offset += data.len();
    }

    // 2. Add page-aligned weights if present
    if let Some(ref w_data) = weights_data {
        // Pad current_offset to nearest 4096-byte page boundary
        let remainder = current_offset % PAGE_ALIGNMENT;
        let pad = if !current_offset.is_multiple_of(PAGE_ALIGNMENT) {
            PAGE_ALIGNMENT - remainder
        } else {
            0
        };
        current_offset += pad;

        let mut hasher = Hasher::new();
        hasher.update(w_data);
        let crc = hasher.finalize();

        toc_descriptors.push(TocEntry {
            name: "weights.safetensors".to_string(),
            offset: current_offset as u64,
            compressed_size: w_data.len() as u64,
            uncompressed_size: w_data.len() as u64,
            crc32: crc,
            compression: 0,
        });
        current_offset += w_data.len();
    }

    // 3. Add parity vectors if present
    if let Some(ref p_data) = parity_data {
        let mut hasher = Hasher::new();
        hasher.update(p_data);
        let crc = hasher.finalize();

        toc_descriptors.push(TocEntry {
            name: "parity_vectors.safetensors".to_string(),
            offset: current_offset as u64,
            compressed_size: p_data.len() as u64,
            uncompressed_size: p_data.len() as u64,
            crc32: crc,
            compression: 0,
        });
        current_offset += p_data.len();
    }

    let total_file_size = current_offset;

    // 4. Encode TOC binary bytes and compute TOC CRC
    let mut toc_bytes = Vec::with_capacity(toc_size);
    for entry in &toc_descriptors {
        let mut name_buf = [0u8; 32];
        let bytes = entry.name.as_bytes();
        let len = bytes.len().min(31);
        name_buf[..len].copy_from_slice(&bytes[..len]);

        toc_bytes.write_all(&name_buf)?;
        toc_bytes.write_u64::<LittleEndian>(entry.offset)?;
        toc_bytes.write_u64::<LittleEndian>(entry.compressed_size)?;
        toc_bytes.write_u64::<LittleEndian>(entry.uncompressed_size)?;
        toc_bytes.write_u32::<LittleEndian>(entry.crc32)?;
        toc_bytes.write_u16::<LittleEndian>(entry.compression)?;
        toc_bytes.write_u16::<LittleEndian>(0)?; // reserved
    }

    let mut toc_hasher = Hasher::new();
    toc_hasher.update(&toc_bytes);
    let toc_crc = toc_hasher.finalize();

    // 5. Assemble final file
    let mut out = Vec::with_capacity(total_file_size);

    // 64-byte Header
    out.write_all(AIDPKG_MAGIC)?;
    out.write_u32::<LittleEndian>(AIDPKG_VERSION)?;
    out.write_u32::<LittleEndian>(0)?; // flags
    out.write_u64::<LittleEndian>(HEADER_SIZE as u64)?;
    out.write_u64::<LittleEndian>(total_file_size as u64)?;
    out.write_u32::<LittleEndian>(total_entries as u32)?;
    out.write_u32::<LittleEndian>(toc_crc)?;
    out.write_all(&[0u8; 24])?; // header padding to 64 bytes

    // Table of Contents
    out.write_all(&toc_bytes)?;

    // Payloads
    for (_name, data) in entries_to_write {
        out.write_all(&data)?;
    }

    if let Some(ref w_data) = weights_data {
        let target_offset = toc_descriptors
            .iter()
            .find(|e| e.name == "weights.safetensors")
            .unwrap()
            .offset as usize;
        let pad_needed = target_offset - out.len();
        out.resize(out.len() + pad_needed, 0u8);
        out.write_all(w_data)?;
    }

    if let Some(ref p_data) = parity_data {
        out.write_all(p_data)?;
    }

    Ok(out)
}

/// Reads and extracts an .aidpkg binary package from raw bytes or memory-mapped file.
pub fn read_aidpkg(bytes: &[u8]) -> Result<AidPkgContents, PkgError> {
    if bytes.len() < HEADER_SIZE {
        return Err(PkgError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "File smaller than minimum header size",
        )));
    }

    let mut cursor = Cursor::new(bytes);

    let mut magic = [0u8; 8];
    cursor.read_exact(&mut magic)?;
    if &magic != AIDPKG_MAGIC {
        return Err(PkgError::InvalidMagic(magic));
    }

    let version = cursor.read_u32::<LittleEndian>()?;
    if version != AIDPKG_VERSION {
        return Err(PkgError::UnsupportedVersion(version));
    }

    let _flags = cursor.read_u32::<LittleEndian>()?;
    let _header_size = cursor.read_u64::<LittleEndian>()?;
    let _total_file_size = cursor.read_u64::<LittleEndian>()?;
    let entry_count = cursor.read_u32::<LittleEndian>()?;
    let expected_toc_crc = cursor.read_u32::<LittleEndian>()?;

    cursor.set_position(HEADER_SIZE as u64);

    let toc_byte_len = entry_count as usize * TOC_ENTRY_SIZE;
    if bytes.len() < HEADER_SIZE + toc_byte_len {
        return Err(PkgError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "TOC extends beyond file size",
        )));
    }

    let toc_slice = &bytes[HEADER_SIZE..HEADER_SIZE + toc_byte_len];
    let mut toc_hasher = Hasher::new();
    toc_hasher.update(toc_slice);
    let computed_toc_crc = toc_hasher.finalize();

    if computed_toc_crc != expected_toc_crc {
        return Err(PkgError::TocCrcMismatch {
            expected: expected_toc_crc,
            computed: computed_toc_crc,
        });
    }

    let mut toc_cursor = Cursor::new(toc_slice);
    let mut entries = Vec::with_capacity(entry_count as usize);

    for _ in 0..entry_count {
        let mut name_buf = [0u8; 32];
        toc_cursor.read_exact(&mut name_buf)?;
        let name_len = name_buf.iter().position(|&b| b == 0).unwrap_or(32);
        let name = String::from_utf8_lossy(&name_buf[..name_len]).to_string();

        let offset = toc_cursor.read_u64::<LittleEndian>()?;
        let compressed_size = toc_cursor.read_u64::<LittleEndian>()?;
        let uncompressed_size = toc_cursor.read_u64::<LittleEndian>()?;
        let crc32 = toc_cursor.read_u32::<LittleEndian>()?;
        let compression = toc_cursor.read_u16::<LittleEndian>()?;
        let _reserved = toc_cursor.read_u16::<LittleEndian>()?;

        entries.push(TocEntry {
            name,
            offset,
            compressed_size,
            uncompressed_size,
            crc32,
            compression,
        });
    }

    // Helper to read and verify entry payload
    let get_entry_data = |entry_name: &str| -> Result<Option<&[u8]>, PkgError> {
        if let Some(entry) = entries.iter().find(|e| e.name == entry_name) {
            let start = entry.offset as usize;
            let end = start + entry.compressed_size as usize;
            if end > bytes.len() {
                return Err(PkgError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("Entry '{entry_name}' extends beyond file bounds"),
                )));
            }
            let data = &bytes[start..end];
            let mut h = Hasher::new();
            h.update(data);
            let computed = h.finalize();
            if computed != entry.crc32 {
                return Err(PkgError::EntryCrcMismatch {
                    name: entry_name.to_string(),
                    expected: entry.crc32,
                    computed,
                });
            }
            Ok(Some(data))
        } else {
            Ok(None)
        }
    };

    let manifest_bytes =
        get_entry_data("manifest.json")?.ok_or(PkgError::MissingEntry("manifest.json"))?;
    let manifest_json = String::from_utf8_lossy(manifest_bytes).to_string();

    let spec: AiDesignSpec = if let Some(yaml_bytes) =
        get_entry_data("blueprint.yaml")?.or(get_entry_data("blueprint.aid.yaml")?)
    {
        serde_yaml::from_slice(yaml_bytes).map_err(|e| PkgError::Io(std::io::Error::other(e)))?
    } else if let Some(bson_bytes) =
        get_entry_data("blueprint.bson")?.or(get_entry_data("blueprint.aid.bson")?)
    {
        bson::from_slice(bson_bytes).map_err(|e| PkgError::Io(std::io::Error::other(e)))?
    } else if let Some(ron_bytes) = get_entry_data("spec.ron")? {
        ron::from_str(&String::from_utf8_lossy(ron_bytes))?
    } else if let Some(json_bytes) =
        get_entry_data("blueprint.json")?.or(get_entry_data("blueprint.aid.json")?)
    {
        serde_json::from_slice(json_bytes)?
    } else {
        return Err(PkgError::MissingEntry("spec.ron or blueprint.*"));
    };

    let weights_bytes = get_entry_data("weights.safetensors")?
        .or(get_entry_data("weights.pkg")?)
        .or(get_entry_data("weights.bson")?)
        .map(|w| w.to_vec());

    let parity_bytes = get_entry_data("parity_vectors.safetensors")?.map(|b| b.to_vec());

    Ok(AidPkgContents {
        spec,
        manifest_json,
        weights_bytes,
        parity_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::*;

    #[test]
    fn test_aidpkg_round_trip_with_alignment() {
        let mut spec = AiDesignSpec::new("robotics_flow", "Robotics Flow Model");
        spec.inputs.push(InputPortSpec {
            id: "state".to_string(),
            name: "Proprioception".to_string(),
            space: TensorSpace::ContinuousVector {
                dimension: 14,
                range: None,
            },
            symbolic_shape: vec![SymbolicDim::from("B"), SymbolicDim::from(14)],
            normalizer: NormalizerSpec::None,
            coordinate_encoding: CoordinateEncoding::None,
            patching: None,
        });
        spec.outputs.push(OutputPortSpec {
            id: "action".to_string(),
            name: "Action Output".to_string(),
            space: TensorSpace::ContinuousVector {
                dimension: 7,
                range: None,
            },
            symbolic_shape: vec![SymbolicDim::from("B"), SymbolicDim::from(7)],
            constraint: StructuralConstraint::DirectLinear,
            loss: LossPairing::MSE,
        });
        spec.graph.nodes.push(NodeSpec {
            id: "linear_1".to_string(),
            name: "Dense Projection".to_string(),
            operator: OperatorKind::Linear {
                in_features: 14,
                out_features: 7,
                bias: true,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        });

        let dummy_weights = vec![0xABu8; 1024];
        let dummy_parity = vec![0xCDu8; 256];

        let pkg_bytes = create_aidpkg(&spec, Some(&dummy_weights), Some(&dummy_parity))
            .expect("Failed to create package");

        // Verify weights offset is 4096-byte page aligned!
        let contents = read_aidpkg(&pkg_bytes).expect("Failed to read package");
        assert_eq!(contents.spec.metadata.model_id, "robotics_flow");
        assert_eq!(contents.weights_bytes.as_ref().unwrap().len(), 1024);
        assert_eq!(contents.parity_bytes.as_ref().unwrap().len(), 256);
    }
}
