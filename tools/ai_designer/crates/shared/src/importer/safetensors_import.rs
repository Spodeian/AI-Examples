//! Safetensors Importer: extracts tensor metadata, detects layer topology, and synthesizes an AiDesignSpec.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use crate::spec::{
    AiDesignSpec, BlueprintMetadata, ComputeGraph, ConnectionSpec, HardwareTargetProfile,
    InputPortSpec, LossPairing, NodeSpec, NormalizerSpec, OperatorKind, OutputPortSpec,
    PrecisionFormat, StructuralConstraint, SymbolicDim, TensorSpace,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsTensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Parse Safetensors header JSON directly from bytes without loading tensors into RAM.
pub fn parse_safetensors_header(
    bytes: &[u8],
) -> Result<BTreeMap<String, SafetensorsTensorInfo>, String> {
    if bytes.len() < 8 {
        return Err("File too short for Safetensors header".to_string());
    }
    let header_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + header_size {
        return Err(format!(
            "Truncated Safetensors header: expected {} bytes, got {}",
            header_size,
            bytes.len() - 8
        ));
    }
    let header_slice = &bytes[8..8 + header_size];
    let mut raw_map: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header_slice)
        .map_err(|e| format!("Failed to parse Safetensors JSON header: {}", e))?;

    // Drop optional __metadata__ key
    raw_map.remove("__metadata__");

    let mut result = BTreeMap::new();
    for (k, v) in raw_map {
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .unwrap_or("F32")
            .to_string();
        let shape: Vec<usize> = v
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|val| val.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let offsets: [usize; 2] = v
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .and_then(|arr| {
                if arr.len() == 2 {
                    Some([
                        arr[0].as_u64().unwrap_or(0) as usize,
                        arr[1].as_u64().unwrap_or(0) as usize,
                    ])
                } else {
                    None
                }
            })
            .unwrap_or([0, 0]);

        result.insert(
            k,
            SafetensorsTensorInfo {
                dtype,
                shape,
                data_offsets: offsets,
            },
        );
    }

    Ok(result)
}

/// Reverse-engineers an AiDesignSpec from Safetensors tensor definitions.
pub fn import_spec_from_safetensors(
    model_id: &str,
    tensors: &BTreeMap<String, SafetensorsTensorInfo>,
) -> Result<AiDesignSpec, String> {
    let mut nodes = Vec::new();
    let mut input_dim = 768usize;
    let mut output_dim = 768usize;
    let mut detected_precision = PrecisionFormat::BF16;

    // Detect precision from majority dtype
    for info in tensors.values() {
        match info.dtype.to_uppercase().as_str() {
            "BF16" => {
                detected_precision = PrecisionFormat::BF16;
                break;
            }
            "F16" => {
                detected_precision = PrecisionFormat::FP16;
                break;
            }
            "F32" => {
                detected_precision = PrecisionFormat::FP32;
            }
            "F8_E4M3" | "F8E4M3" => {
                detected_precision = PrecisionFormat::FP8E4M3FN;
                break;
            }
            _ => {}
        }
    }

    // Group tensor keys by common prefix
    let mut layer_groups: HashMap<String, Vec<(String, SafetensorsTensorInfo)>> = HashMap::new();
    for (name, info) in tensors {
        let parts: Vec<&str> = name.split('.').collect();
        let group_key = if parts.len() >= 2 {
            parts[..parts.len() - 1].join(".")
        } else {
            name.clone()
        };
        layer_groups
            .entry(group_key)
            .or_default()
            .push((name.clone(), info.clone()));
    }

    let mut sorted_groups: Vec<_> = layer_groups.into_iter().collect();
    sorted_groups.sort_by(|a, b| a.0.cmp(&b.0));

    let mut node_idx = 0;
    for (group_name, tensor_list) in &sorted_groups {
        let mut node_spec = None;

        for (tname, tinfo) in tensor_list {
            let tlower = tname.to_lowercase();
            if tlower.contains("embed") && tinfo.shape.len() == 2 {
                input_dim = tinfo.shape[1];
                node_spec = Some(NodeSpec {
                    id: format!("embed_{}", node_idx),
                    name: format!("Embedding ({})", group_name),
                    operator: OperatorKind::Embedding {
                        num_embeddings: tinfo.shape[0],
                        embedding_dim: tinfo.shape[1],
                    },
                    precision: Some(detected_precision),
                    metadata: HashMap::new(),
                });
                break;
            } else if (tlower.contains("attn")
                || tlower.contains("attention")
                || tlower.contains("q_proj"))
                && tinfo.shape.len() >= 2
            {
                let d_model = tinfo.shape[1];
                let num_heads = (d_model / 64).max(1);
                input_dim = d_model;
                output_dim = d_model;
                node_spec = Some(NodeSpec {
                    id: format!("attn_{}", node_idx),
                    name: format!("Attention Layer ({})", group_name),
                    operator: OperatorKind::MHA {
                        num_heads,
                        d_k: d_model / num_heads,
                        d_v: d_model / num_heads,
                        dropout: 0.0,
                    },
                    precision: Some(detected_precision),
                    metadata: HashMap::new(),
                });
                break;
            } else if (tlower.contains("norm") || tlower.contains("ln") || tlower.contains("rms"))
                && !tinfo.shape.is_empty()
            {
                let d = tinfo.shape[0];
                node_spec = Some(NodeSpec {
                    id: format!("norm_{}", node_idx),
                    name: format!("RMSNorm ({})", group_name),
                    operator: OperatorKind::RMSNorm { dim: d, eps: 1e-5 },
                    precision: Some(detected_precision),
                    metadata: HashMap::new(),
                });
                break;
            } else if (tlower.contains("mlp")
                || tlower.contains("ffn")
                || tlower.contains("fc")
                || tlower.contains("dense"))
                && tinfo.shape.len() == 2
            {
                let d_in = tinfo.shape[1];
                let d_out = tinfo.shape[0];
                output_dim = d_out;
                node_spec = Some(NodeSpec {
                    id: format!("dense_{}", node_idx),
                    name: format!("Dense ({})", group_name),
                    operator: OperatorKind::Linear {
                        in_features: d_in,
                        out_features: d_out,
                        bias: true,
                    },
                    precision: Some(detected_precision),
                    metadata: HashMap::new(),
                });
                break;
            } else if tlower.contains("mamba") || tlower.contains("ssm") {
                node_spec = Some(NodeSpec {
                    id: format!("ssm_{}", node_idx),
                    name: format!("Mamba SSM ({})", group_name),
                    operator: OperatorKind::Mamba2SSD {
                        d_model: 768,
                        d_state: 128,
                        d_conv: 4,
                        expand: 2,
                    },
                    precision: Some(detected_precision),
                    metadata: HashMap::new(),
                });
                break;
            }
        }

        if let Some(spec) = node_spec {
            nodes.push(spec);
            node_idx += 1;
        }
    }

    if nodes.is_empty() {
        nodes.push(NodeSpec {
            id: "imported_backbone".to_string(),
            name: "Imported Backbone".to_string(),
            operator: OperatorKind::Linear {
                in_features: input_dim,
                out_features: output_dim,
                bias: true,
            },
            precision: Some(detected_precision),
            metadata: HashMap::new(),
        });
    }

    // Connect nodes in linear pipeline
    let mut connections = Vec::new();
    for i in 0..nodes.len().saturating_sub(1) {
        connections.push(ConnectionSpec {
            from_node: nodes[i].id.clone(),
            from_port: 0,
            to_node: nodes[i + 1].id.clone(),
            to_port: 0,
        });
    }

    let mut spec = AiDesignSpec::new(model_id, &format!("Imported: {}", model_id));
    spec.metadata = BlueprintMetadata {
        model_id: model_id.to_string(),
        display_name: format!("Imported: {}", model_id),
        domain: "Imported Models".to_string(),
        author: "Safetensors Ingest Engine".to_string(),
        license: "Apache-2.0".to_string(),
        target_hardware: HardwareTargetProfile::rtx_4090(),
        tags: vec!["safetensors".to_string(), "imported".to_string()],
    };
    spec.inputs = vec![InputPortSpec {
        id: "input_features".to_string(),
        name: "Input Features".to_string(),
        space: TensorSpace::ContinuousVector {
            dimension: input_dim,
            range: None,
        },
        symbolic_shape: vec![
            SymbolicDim::Symbol("B".into()),
            SymbolicDim::Symbol("T".into()),
            SymbolicDim::Known(input_dim),
        ],
        normalizer: NormalizerSpec::Standardize { mean_fill: 0.0 },
        coordinate_encoding: crate::spec::CoordinateEncoding::None,
        patching: None,
    }];
    spec.outputs = vec![OutputPortSpec {
        id: "output_predictions".to_string(),
        name: "Output Predictions".to_string(),
        space: TensorSpace::ContinuousVector {
            dimension: output_dim,
            range: None,
        },
        symbolic_shape: vec![
            SymbolicDim::Symbol("B".into()),
            SymbolicDim::Symbol("T".into()),
            SymbolicDim::Known(output_dim),
        ],
        constraint: StructuralConstraint::DirectLinear,
        loss: LossPairing::MSE,
    }];
    spec.graph = ComputeGraph {
        topology: crate::spec::WiringPattern::SerialStack,
        nodes,
        connections,
    };

    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_header_parsing_and_spec_synthesis() {
        let json_header = serde_json::json!({
            "__metadata__": { "format": "pt" },
            "transformer.layer.0.attn.q_proj.weight": {
                "dtype": "BF16",
                "shape": [512, 512],
                "data_offsets": [0, 524288]
            },
            "transformer.layer.0.norm.weight": {
                "dtype": "BF16",
                "shape": [512],
                "data_offsets": [524288, 525312]
            },
            "transformer.layer.0.mlp.fc1.weight": {
                "dtype": "BF16",
                "shape": [2048, 512],
                "data_offsets": [525312, 2622464]
            }
        });

        let json_bytes = serde_json::to_vec(&json_header).unwrap();
        let header_len = (json_bytes.len() as u64).to_le_bytes();

        let mut raw_file = Vec::new();
        raw_file.extend_from_slice(&header_len);
        raw_file.extend_from_slice(&json_bytes);
        raw_file.resize(raw_file.len() + 1024, 0);

        let parsed = parse_safetensors_header(&raw_file).expect("Parse header failed");
        assert_eq!(parsed.len(), 3);
        assert!(parsed.contains_key("transformer.layer.0.attn.q_proj.weight"));

        let spec =
            import_spec_from_safetensors("mock_transformer", &parsed).expect("Synthesis failed");
        assert_eq!(spec.metadata.model_id, "mock_transformer");
        assert!(!spec.graph.nodes.is_empty());
        assert_eq!(spec.graph.connections.len(), 2);
    }
}
