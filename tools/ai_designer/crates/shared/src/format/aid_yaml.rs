//! YAML serializer and deserializer for the human-readable .aid format.

use crate::spec::AiDesignSpec;

#[derive(Debug, thiserror::Error)]
pub enum YamlFormatError {
    #[error("Failed to serialize model design to YAML: {0}")]
    SerializeError(#[from] serde_yaml::Error),
    #[error("Failed to parse .aid YAML specification: {0}")]
    ParseError(serde_yaml::Error),
    #[error("Validation failed: {0}")]
    ValidationError(String),
}

/// Serializes an AiDesignSpec into clean, human-readable .aid YAML.
pub fn to_yaml_string(spec: &AiDesignSpec) -> Result<String, YamlFormatError> {
    spec.validate().map_err(YamlFormatError::ValidationError)?;
    let yaml = serde_yaml::to_string(spec)?;
    Ok(yaml)
}

/// Deserializes and validates an AiDesignSpec from a .aid YAML string.
pub fn from_yaml_str(yaml_str: &str) -> Result<AiDesignSpec, YamlFormatError> {
    let spec: AiDesignSpec = serde_yaml::from_str(yaml_str).map_err(YamlFormatError::ParseError)?;
    spec.validate().map_err(YamlFormatError::ValidationError)?;
    Ok(spec)
}

/// Serializes an AiDesignSpec into YAML bytes.
pub fn to_yaml_bytes(spec: &AiDesignSpec) -> Result<Vec<u8>, YamlFormatError> {
    to_yaml_string(spec).map(|s| s.into_bytes())
}

/// Deserializes an AiDesignSpec from YAML bytes.
pub fn from_yaml_slice(bytes: &[u8]) -> Result<AiDesignSpec, YamlFormatError> {
    let spec: AiDesignSpec = serde_yaml::from_slice(bytes).map_err(YamlFormatError::ParseError)?;
    spec.validate().map_err(YamlFormatError::ValidationError)?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::*;

    #[test]
    fn test_yaml_round_trip() {
        let mut spec = AiDesignSpec::new("test_model", "Test Architecture");
        spec.inputs.push(InputPortSpec {
            id: "x".to_string(),
            name: "Input Vector".to_string(),
            space: TensorSpace::ContinuousVector {
                dimension: 64,
                range: None,
            },
            symbolic_shape: vec![SymbolicDim::from("B"), SymbolicDim::from(64)],
            normalizer: NormalizerSpec::None,
            coordinate_encoding: CoordinateEncoding::None,
            patching: None,
        });
        spec.outputs.push(OutputPortSpec {
            id: "y".to_string(),
            name: "Output Vector".to_string(),
            space: TensorSpace::ContinuousVector {
                dimension: 10,
                range: None,
            },
            symbolic_shape: vec![SymbolicDim::from("B"), SymbolicDim::from(10)],
            constraint: StructuralConstraint::DirectLinear,
            loss: LossPairing::CrossEntropy,
        });
        spec.graph.nodes.push(NodeSpec {
            id: "linear_1".to_string(),
            name: "Dense Projection".to_string(),
            operator: OperatorKind::Linear {
                in_features: 64,
                out_features: 10,
                bias: true,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        });

        let yaml = to_yaml_string(&spec).expect("YAML serialization failed");
        let deserialized = from_yaml_str(&yaml).expect("YAML deserialization failed");
        assert_eq!(spec, deserialized);
    }
}
