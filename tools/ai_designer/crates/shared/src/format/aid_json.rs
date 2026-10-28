//! JSON codec for .json and .aid.json architecture blueprints.

use crate::spec::AiDesignSpec;

/// Serializes an AiDesignSpec into pretty-printed JSON string.
pub fn to_json_string(spec: &AiDesignSpec) -> Result<String, String> {
    serde_json::to_string_pretty(spec)
        .map_err(|e| format!("Failed to serialize AiDesignSpec to JSON: {}", e))
}

/// Serializes an AiDesignSpec into compact JSON bytes.
pub fn to_json_bytes(spec: &AiDesignSpec) -> Result<Vec<u8>, String> {
    serde_json::to_vec(spec)
        .map_err(|e| format!("Failed to serialize AiDesignSpec to JSON bytes: {}", e))
}

/// Deserializes an AiDesignSpec from a JSON string.
pub fn from_json_str(json_str: &str) -> Result<AiDesignSpec, String> {
    serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse AiDesignSpec from JSON: {}", e))
}

/// Deserializes an AiDesignSpec from JSON bytes.
pub fn from_json_slice(bytes: &[u8]) -> Result<AiDesignSpec, String> {
    serde_json::from_slice(bytes)
        .map_err(|e| format!("Failed to parse AiDesignSpec from JSON bytes: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_roundtrip() {
        let spec = AiDesignSpec::default();
        let json_str = to_json_string(&spec).unwrap();
        let parsed = from_json_str(&json_str).unwrap();
        assert_eq!(spec.metadata.model_id, parsed.metadata.model_id);
    }
}
