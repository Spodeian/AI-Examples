//! .aid Format Container: safely wraps any blueprint format (YAML, JSON, TOML, BSON, RON)
//! with magic bytes, format descriptor, and CRC32 integrity check.

use super::{aid_bson, aid_json, aid_ron, aid_toml, aid_yaml};
use crate::spec::AiDesignSpec;
use crc32fast::Hasher;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AidFormatTag {
    Yaml = 0x01,
    Json = 0x02,
    Toml = 0x03,
    Bson = 0x04,
    Ron = 0x05,
}

impl AidFormatTag {
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Yaml),
            0x02 => Some(Self::Json),
            0x03 => Some(Self::Toml),
            0x04 => Some(Self::Bson),
            0x05 => Some(Self::Ron),
            _ => None,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Yaml => "yaml",
            Self::Json => "json",
            Self::Toml => "toml",
            Self::Bson => "bson",
            Self::Ron => "ron",
        }
    }
}

pub const AID_MAGIC: &[u8; 4] = b"AID\x01";

/// Packs an AiDesignSpec into a native .aid envelope using the specified format.
pub fn pack_aid(spec: &AiDesignSpec, format: AidFormatTag) -> Result<Vec<u8>, String> {
    let payload = match format {
        AidFormatTag::Yaml => aid_yaml::to_yaml_bytes(spec).map_err(|e| e.to_string())?,
        AidFormatTag::Json => aid_json::to_json_bytes(spec)?,
        AidFormatTag::Toml => aid_toml::to_toml_bytes(spec)?,
        AidFormatTag::Bson => aid_bson::to_bson_bytes(spec)?,
        AidFormatTag::Ron => aid_ron::to_ron_bytes(spec).map_err(|e| e.to_string())?,
    };

    let mut hasher = Hasher::new();
    hasher.update(&payload);
    let crc = hasher.finalize();

    let mut out = Vec::with_capacity(4 + 1 + 4 + payload.len());
    out.extend_from_slice(AID_MAGIC);
    out.push(format as u8);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Unpacks an AiDesignSpec from raw bytes, supporting native .aid container
/// as well as transparent fallback to raw YAML, JSON, TOML, BSON, or RON.
pub fn unpack_aid(bytes: &[u8]) -> Result<(AiDesignSpec, AidFormatTag), String> {
    if bytes.len() >= 9 && &bytes[0..4] == AID_MAGIC {
        let tag = AidFormatTag::from_u8(bytes[4])
            .ok_or_else(|| format!("Unknown .aid inner format tag: 0x{:02x}", bytes[4]))?;
        let expected_crc = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
        let payload = &bytes[9..];

        let mut hasher = Hasher::new();
        hasher.update(payload);
        let actual_crc = hasher.finalize();

        if actual_crc != expected_crc {
            return Err(format!(
                "CRC32 checksum mismatch in .aid package: expected {:08x}, got {:08x}",
                expected_crc, actual_crc
            ));
        }

        let spec = match tag {
            AidFormatTag::Yaml => aid_yaml::from_yaml_slice(payload).map_err(|e| e.to_string())?,
            AidFormatTag::Json => aid_json::from_json_slice(payload)?,
            AidFormatTag::Toml => aid_toml::from_toml_slice(payload)?,
            AidFormatTag::Bson => aid_bson::from_bson_slice(payload)?,
            AidFormatTag::Ron => aid_ron::from_ron_slice(payload).map_err(|e| e.to_string())?,
        };
        return Ok((spec, tag));
    }

    // Transparent fallback: try each format directly
    if let Ok(spec) = aid_yaml::from_yaml_slice(bytes) {
        return Ok((spec, AidFormatTag::Yaml));
    }
    if let Ok(spec) = aid_json::from_json_slice(bytes) {
        return Ok((spec, AidFormatTag::Json));
    }
    if let Ok(spec) = aid_toml::from_toml_slice(bytes) {
        return Ok((spec, AidFormatTag::Toml));
    }
    if let Ok(spec) = aid_bson::from_bson_slice(bytes) {
        return Ok((spec, AidFormatTag::Bson));
    }
    if let Ok(spec) = aid_ron::from_ron_slice(bytes) {
        return Ok((spec, AidFormatTag::Ron));
    }

    Err("Failed to parse blueprint bytes as any known format (AID envelope, YAML, JSON, TOML, BSON, RON)".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aid_envelope_all_formats() {
        let mut spec = AiDesignSpec::new("test_model", "Test Architecture");
        spec.inputs.push(crate::spec::InputPortSpec {
            id: "x".to_string(),
            name: "Input".to_string(),
            space: crate::spec::TensorSpace::ContinuousVector {
                dimension: 64,
                range: None,
            },
            symbolic_shape: vec![
                crate::spec::SymbolicDim::from("B"),
                crate::spec::SymbolicDim::from(64),
            ],
            normalizer: crate::spec::NormalizerSpec::None,
            coordinate_encoding: crate::spec::CoordinateEncoding::None,
            patching: None,
        });
        spec.outputs.push(crate::spec::OutputPortSpec {
            id: "y".to_string(),
            name: "Output".to_string(),
            space: crate::spec::TensorSpace::ContinuousVector {
                dimension: 10,
                range: None,
            },
            symbolic_shape: vec![
                crate::spec::SymbolicDim::from("B"),
                crate::spec::SymbolicDim::from(10),
            ],
            constraint: crate::spec::StructuralConstraint::DirectLinear,
            loss: crate::spec::LossPairing::CrossEntropy,
        });
        spec.graph.nodes.push(crate::spec::NodeSpec {
            id: "linear_1".to_string(),
            name: "Dense".to_string(),
            operator: crate::spec::OperatorKind::Linear {
                in_features: 64,
                out_features: 10,
                bias: true,
            },
            precision: Some(crate::spec::PrecisionFormat::BF16),
            metadata: Default::default(),
        });

        for tag in [
            AidFormatTag::Yaml,
            AidFormatTag::Json,
            AidFormatTag::Toml,
            AidFormatTag::Bson,
            AidFormatTag::Ron,
        ] {
            let packed = pack_aid(&spec, tag).unwrap();
            let (unpacked, unpacked_tag) = unpack_aid(&packed).unwrap();
            assert_eq!(unpacked_tag, tag);
            assert_eq!(unpacked.metadata.model_id, spec.metadata.model_id);
        }
    }
}
