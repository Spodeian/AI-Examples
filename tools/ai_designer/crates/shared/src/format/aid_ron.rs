//! RON (Rusty Object Notation) serializer and deserializer for fast, lossless Rust AST caching.

use crate::spec::AiDesignSpec;

#[derive(Debug, thiserror::Error)]
pub enum RonFormatError {
    #[error("Failed to serialize model design to RON: {0}")]
    SerializeError(#[from] ron::Error),
    #[error("Failed to parse .aid RON specification: {0}")]
    ParseError(#[from] ron::error::SpannedError),
    #[error("Validation failed: {0}")]
    ValidationError(String),
}

/// Serializes an AiDesignSpec into pretty-printed RON format.
pub fn to_ron_string(spec: &AiDesignSpec) -> Result<String, RonFormatError> {
    spec.validate().map_err(RonFormatError::ValidationError)?;
    let pretty = ron::ser::PrettyConfig::default();
    let s = ron::ser::to_string_pretty(spec, pretty)?;
    Ok(s)
}

/// Deserializes an AiDesignSpec from RON format.
pub fn from_ron_str(ron_str: &str) -> Result<AiDesignSpec, RonFormatError> {
    let spec: AiDesignSpec = ron::from_str(ron_str)?;
    spec.validate().map_err(RonFormatError::ValidationError)?;
    Ok(spec)
}

/// Serializes an AiDesignSpec into RON bytes.
pub fn to_ron_bytes(spec: &AiDesignSpec) -> Result<Vec<u8>, RonFormatError> {
    to_ron_string(spec).map(|s| s.into_bytes())
}

/// Deserializes an AiDesignSpec from RON bytes.
pub fn from_ron_slice(bytes: &[u8]) -> Result<AiDesignSpec, RonFormatError> {
    let s =
        std::str::from_utf8(bytes).map_err(|e| RonFormatError::ValidationError(e.to_string()))?;
    from_ron_str(s)
}
