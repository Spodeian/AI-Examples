//! TOML codec for .toml and .aid.toml architecture blueprints.

use crate::spec::AiDesignSpec;

/// Serializes an AiDesignSpec into pretty-printed TOML string.
pub fn to_toml_string(spec: &AiDesignSpec) -> Result<String, String> {
    toml::to_string_pretty(spec)
        .map_err(|e| format!("Failed to serialize AiDesignSpec to TOML: {}", e))
}

/// Serializes an AiDesignSpec into TOML bytes.
pub fn to_toml_bytes(spec: &AiDesignSpec) -> Result<Vec<u8>, String> {
    to_toml_string(spec).map(|s| s.into_bytes())
}

/// Deserializes an AiDesignSpec from a TOML string.
pub fn from_toml_str(toml_str: &str) -> Result<AiDesignSpec, String> {
    toml::from_str(toml_str).map_err(|e| format!("Failed to parse AiDesignSpec from TOML: {}", e))
}

/// Deserializes an AiDesignSpec from TOML bytes.
pub fn from_toml_slice(bytes: &[u8]) -> Result<AiDesignSpec, String> {
    let toml_str =
        std::str::from_utf8(bytes).map_err(|e| format!("Invalid UTF-8 in TOML bytes: {}", e))?;
    from_toml_str(toml_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_roundtrip() {
        let spec = AiDesignSpec::default();
        let toml_str = to_toml_string(&spec).unwrap();
        let parsed = from_toml_str(&toml_str).unwrap();
        assert_eq!(spec.metadata.model_id, parsed.metadata.model_id);
    }
}
