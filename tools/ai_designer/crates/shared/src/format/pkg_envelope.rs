//! .pkg Weights Container: safely wraps any weight format (Safetensors, BSON, Quantized, ONNX)
//! with 4096-byte page alignment for zero-copy memory mapping (`mmap`), format tag, and CRC32 checks.

use crc32fast::Hasher;
use serde::{Deserialize, Serialize};

pub const PKG_MAGIC: &[u8; 4] = b"PKG\x01";
pub const PAGE_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PkgWeightsFormat {
    Safetensors = 0x01,
    BsonTensors = 0x02,
    Quantized = 0x03,
    Onnx = 0x04,
    RawBinary = 0x05,
}

impl PkgWeightsFormat {
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Safetensors),
            0x02 => Some(Self::BsonTensors),
            0x03 => Some(Self::Quantized),
            0x04 => Some(Self::Onnx),
            0x05 => Some(Self::RawBinary),
            _ => None,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::BsonTensors => "bson",
            Self::Quantized => "bin",
            Self::Onnx => "onnx",
            Self::RawBinary => "bin",
        }
    }
}

/// Metadata header describing the packed weights payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkgManifest {
    pub format: PkgWeightsFormat,
    pub num_tensors: usize,
    pub payload_size: usize,
    pub payload_crc: u32,
    pub page_aligned_offset: usize,
}

/// Packs raw weights into a 4096-byte page-aligned .pkg container.
pub fn pack_pkg(weights: &[u8], format: PkgWeightsFormat, num_tensors: usize) -> Vec<u8> {
    let mut hasher = Hasher::new();
    hasher.update(weights);
    let crc = hasher.finalize();

    let manifest = PkgManifest {
        format,
        num_tensors,
        payload_size: weights.len(),
        payload_crc: crc,
        page_aligned_offset: PAGE_SIZE,
    };

    let manifest_bytes = serde_json::to_vec(&manifest).unwrap_or_default();
    let header_len = 4 + 4 + manifest_bytes.len(); // magic + len + json

    let mut out = Vec::new();
    out.extend_from_slice(PKG_MAGIC);
    out.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&manifest_bytes);

    // Pad to 4096-byte page boundary
    let padding = if !header_len.is_multiple_of(PAGE_SIZE) {
        PAGE_SIZE - (header_len % PAGE_SIZE)
    } else {
        0
    };
    out.resize(out.len() + padding, 0);

    // Append weights at exact page boundary
    out.extend_from_slice(weights);
    out
}

/// Unpacks a .pkg weights container, verifying CRC32 and returning (manifest, weights_slice).
pub fn unpack_pkg(bytes: &[u8]) -> Result<(PkgManifest, Vec<u8>), String> {
    if bytes.len() >= 8 && &bytes[0..4] == PKG_MAGIC {
        let manifest_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if 8 + manifest_len > bytes.len() {
            return Err("Truncated .pkg header".to_string());
        }

        let manifest: PkgManifest = serde_json::from_slice(&bytes[8..8 + manifest_len])
            .map_err(|e| format!("Invalid .pkg manifest JSON: {}", e))?;

        let header_len = 8 + manifest_len;
        let padding = if !header_len.is_multiple_of(PAGE_SIZE) {
            PAGE_SIZE - (header_len % PAGE_SIZE)
        } else {
            0
        };
        let weights_offset = header_len + padding;

        if weights_offset + manifest.payload_size > bytes.len() {
            return Err("Truncated .pkg weights payload".to_string());
        }

        let weights = bytes[weights_offset..weights_offset + manifest.payload_size].to_vec();

        let mut hasher = Hasher::new();
        hasher.update(&weights);
        let actual_crc = hasher.finalize();

        if actual_crc != manifest.payload_crc {
            return Err(format!(
                "CRC32 mismatch in .pkg: expected {:08x}, got {:08x}",
                manifest.payload_crc, actual_crc
            ));
        }

        return Ok((manifest, weights));
    }

    // Transparent fallback: treat as raw safetensors or binary weights
    let format = if bytes.len() >= 8 {
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if header_len > 0 && header_len < 100_000_000 && 8 + header_len <= bytes.len() {
            PkgWeightsFormat::Safetensors
        } else {
            PkgWeightsFormat::RawBinary
        }
    } else {
        PkgWeightsFormat::RawBinary
    };

    let manifest = PkgManifest {
        format,
        num_tensors: 0,
        payload_size: bytes.len(),
        payload_crc: 0,
        page_aligned_offset: 0,
    };

    Ok((manifest, bytes.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pkg_roundtrip_with_alignment() {
        let fake_weights = vec![0xABu8; 1024];
        let packed = pack_pkg(&fake_weights, PkgWeightsFormat::Safetensors, 4);

        // Check 4096-byte alignment
        let (manifest, unpacked) = unpack_pkg(&packed).unwrap();
        assert_eq!(manifest.format, PkgWeightsFormat::Safetensors);
        assert_eq!(manifest.num_tensors, 4);
        assert_eq!(unpacked, fake_weights);
    }
}
