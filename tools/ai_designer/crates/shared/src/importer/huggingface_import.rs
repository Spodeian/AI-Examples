//! Hugging Face config.json Importer: converts standard HF architecture definitions into AiDesignSpec blueprints.

use serde_json::Value;
use std::collections::HashMap;

use crate::spec::{
    AiDesignSpec, BlueprintMetadata, ComputeGraph, ConnectionSpec, CoordinateEncoding,
    HardwareTargetProfile, InputPortSpec, LossPairing, NodeSpec, NormalizerSpec, OperatorKind,
    OutputPortSpec, PrecisionFormat, StructuralConstraint, SymbolicDim, TensorSpace, WiringPattern,
};

/// Imports an AiDesignSpec from a HuggingFace config.json string.
pub fn import_from_huggingface_config(config_json: &str) -> Result<AiDesignSpec, String> {
    let root: Value = serde_json::from_str(config_json)
        .map_err(|e| format!("Invalid HuggingFace config.json: {}", e))?;

    let model_type = root
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("transformer")
        .to_string();

    let hidden_size = root
        .get("hidden_size")
        .or_else(|| root.get("d_model"))
        .and_then(|v| v.as_u64())
        .unwrap_or(768) as usize;

    let num_heads = root
        .get("num_attention_heads")
        .or_else(|| root.get("n_head"))
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;

    let num_layers = root
        .get("num_hidden_layers")
        .or_else(|| root.get("n_layer"))
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;

    let intermediate_size = root
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size * 4);

    let vocab_size = root
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as usize;

    let rms_norm_eps = root
        .get("rms_norm_eps")
        .or_else(|| root.get("layer_norm_eps"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5);

    let mut nodes = Vec::new();

    // 1. Embedding / Tokenizer Projection
    nodes.push(NodeSpec {
        id: "embed_tokens".to_string(),
        name: format!("Embedding (Vocab: {})", vocab_size),
        operator: OperatorKind::Embedding {
            num_embeddings: vocab_size,
            embedding_dim: hidden_size,
        },
        precision: Some(PrecisionFormat::BF16),
        metadata: HashMap::new(),
    });

    // 2. Unrolled or representative layers
    let layers_to_emit = num_layers.min(4);
    for layer_idx in 0..layers_to_emit {
        let norm_id = format!("layer_{}_norm", layer_idx);
        nodes.push(NodeSpec {
            id: norm_id.clone(),
            name: format!("Layer {} Pre-Norm", layer_idx),
            operator: OperatorKind::RMSNorm {
                dim: hidden_size,
                eps: rms_norm_eps,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: HashMap::new(),
        });

        let core_id = format!("layer_{}_core", layer_idx);
        if model_type.contains("mamba") {
            nodes.push(NodeSpec {
                id: core_id.clone(),
                name: format!("Layer {} Mamba-2 SSM", layer_idx),
                operator: OperatorKind::Mamba2SSD {
                    d_model: hidden_size,
                    d_state: 128,
                    d_conv: 4,
                    expand: 2,
                },
                precision: Some(PrecisionFormat::BF16),
                metadata: HashMap::new(),
            });
        } else {
            let head_dim = hidden_size / num_heads;
            nodes.push(NodeSpec {
                id: core_id.clone(),
                name: format!("Layer {} Attention", layer_idx),
                operator: OperatorKind::MHA {
                    num_heads,
                    d_k: head_dim,
                    d_v: head_dim,
                    dropout: 0.0,
                },
                precision: Some(PrecisionFormat::BF16),
                metadata: HashMap::new(),
            });
        }

        let ffn_id = format!("layer_{}_ffn", layer_idx);
        nodes.push(NodeSpec {
            id: ffn_id.clone(),
            name: format!("Layer {} SwiGLU / MLP", layer_idx),
            operator: OperatorKind::SwiGLU {
                in_features: hidden_size,
                hidden_features: intermediate_size,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: HashMap::new(),
        });
    }

    // 3. Final Norm and Head
    nodes.push(NodeSpec {
        id: "final_norm".to_string(),
        name: "Final RMSNorm".to_string(),
        operator: OperatorKind::RMSNorm {
            dim: hidden_size,
            eps: rms_norm_eps,
        },
        precision: Some(PrecisionFormat::BF16),
        metadata: HashMap::new(),
    });

    nodes.push(NodeSpec {
        id: "lm_head".to_string(),
        name: "LM Head Projection".to_string(),
        operator: OperatorKind::Linear {
            in_features: hidden_size,
            out_features: vocab_size,
            bias: false,
        },
        precision: Some(PrecisionFormat::BF16),
        metadata: HashMap::new(),
    });

    // Wire sequential pipeline
    let mut connections = Vec::new();
    for i in 0..nodes.len().saturating_sub(1) {
        connections.push(ConnectionSpec {
            from_node: nodes[i].id.clone(),
            from_port: 0,
            to_node: nodes[i + 1].id.clone(),
            to_port: 0,
        });
    }

    let mut spec = AiDesignSpec::new(
        &format!("hf_{}", model_type),
        &format!(
            "HuggingFace {} ({}L, {}D)",
            model_type.to_uppercase(),
            num_layers,
            hidden_size
        ),
    );
    spec.metadata = BlueprintMetadata {
        model_id: format!("hf_{}", model_type),
        display_name: format!(
            "HuggingFace {} ({}L, {}D)",
            model_type.to_uppercase(),
            num_layers,
            hidden_size
        ),
        domain: "Natural Language Processing".to_string(),
        author: "HuggingFace Config Importer".to_string(),
        license: "Apache-2.0".to_string(),
        target_hardware: HardwareTargetProfile::rtx_4090(),
        tags: vec!["huggingface".to_string(), model_type.clone()],
    };
    spec.inputs = vec![InputPortSpec {
        id: "input_tokens".to_string(),
        name: "Input Tokens".to_string(),
        space: TensorSpace::ContinuousVector {
            dimension: hidden_size,
            range: None,
        },
        symbolic_shape: vec![
            SymbolicDim::Symbol("B".into()),
            SymbolicDim::Symbol("T".into()),
        ],
        normalizer: NormalizerSpec::Standardize { mean_fill: 0.0 },
        coordinate_encoding: CoordinateEncoding::None,
        patching: None,
    }];
    spec.outputs = vec![OutputPortSpec {
        id: "logits".to_string(),
        name: "Logits".to_string(),
        space: TensorSpace::ContinuousVector {
            dimension: vocab_size,
            range: None,
        },
        symbolic_shape: vec![
            SymbolicDim::Symbol("B".into()),
            SymbolicDim::Symbol("T".into()),
            SymbolicDim::Known(vocab_size),
        ],
        constraint: StructuralConstraint::DirectLinear,
        loss: LossPairing::CrossEntropy,
    }];
    spec.graph = ComputeGraph {
        topology: WiringPattern::SerialStack,
        nodes,
        connections,
    };

    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_llama_huggingface_config() {
        let llama_config = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000
        }"#;

        let spec = import_from_huggingface_config(llama_config).expect("Import failed");
        assert_eq!(spec.metadata.model_id, "hf_llama");
        assert!(spec.graph.nodes.iter().any(|n| n.id == "lm_head"));
        assert!(
            spec.graph
                .nodes
                .iter()
                .any(|n| matches!(n.operator, OperatorKind::SwiGLU { .. }))
        );
        assert_eq!(spec.outputs[0].loss, LossPairing::CrossEntropy);
    }
}
