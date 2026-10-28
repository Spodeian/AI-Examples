//! BSON (Binary JSON) codec for .bson and .aid.bson architecture blueprints.

use crate::spec::AiDesignSpec;

/// Serializes an AiDesignSpec into raw BSON binary bytes.
pub fn to_bson_bytes(spec: &AiDesignSpec) -> Result<Vec<u8>, String> {
    bson::to_vec(spec).map_err(|e| format!("Failed to serialize AiDesignSpec to BSON: {}", e))
}

/// Deserializes an AiDesignSpec from BSON binary bytes.
pub fn from_bson_slice(bytes: &[u8]) -> Result<AiDesignSpec, String> {
    bson::from_slice(bytes)
        .map_err(|e| format!("Failed to parse AiDesignSpec from BSON bytes: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bson_roundtrip() {
        let spec = AiDesignSpec::default();
        let bytes = to_bson_bytes(&spec).unwrap();
        assert!(!bytes.is_empty());
        let parsed = from_bson_slice(&bytes).unwrap();
        assert_eq!(spec.metadata.model_id, parsed.metadata.model_id);
    }
}
