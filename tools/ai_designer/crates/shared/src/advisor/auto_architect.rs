//! One-click Auto-Architect synthesis engine: derives optimal model architectures from I/O declarations.

use crate::spec::AiDesignSpec;
use crate::spec::graph::{NodeSpec, NormPlacement, WiringPattern};
use crate::spec::io::{
    CoordinateEncoding, InputPortSpec, LossPairing, NormalizerSpec, OutputPortSpec, PatchingSpec,
    StructuralConstraint, TensorSpace,
};
use crate::spec::operators::OperatorKind;
use crate::spec::precision::PrecisionFormat;

/// Synthesizes an optimal AiDesignSpec based on the domain inductive biases of declared inputs and outputs.
pub fn synthesize_architecture(
    model_id: &str,
    display_name: &str,
    inputs: Vec<InputPortSpec>,
    outputs: Vec<OutputPortSpec>,
) -> AiDesignSpec {
    let mut spec = AiDesignSpec::new(model_id, display_name);
    spec.inputs = inputs;
    spec.outputs = outputs;

    // Detect primary problem domain from inputs and outputs
    if has_video_or_image(&spec.inputs) {
        synthesize_vision_flow_matching(&mut spec);
    } else if has_time_series(&spec.inputs) {
        synthesize_timeseries_forecaster(&mut spec);
    } else if has_robot_trajectory(&spec.outputs) {
        synthesize_robotic_diffusion_policy(&mut spec);
    } else if has_tabular_features(&spec.inputs) {
        synthesize_tabular_prior_network(&mut spec);
    } else if has_discrete_tokens(&spec.inputs) {
        synthesize_reasoning_hybrid_backbone(&mut spec);
    } else if has_coordinate_field(&spec.inputs) {
        synthesize_fourier_neural_operator(&mut spec);
    } else {
        synthesize_general_dense_residual(&mut spec);
    }

    spec
}

fn has_video_or_image(inputs: &[InputPortSpec]) -> bool {
    inputs.iter().any(|i| {
        matches!(
            i.space,
            TensorSpace::Video3D { .. } | TensorSpace::ImageRGB { .. }
        )
    })
}

fn has_time_series(inputs: &[InputPortSpec]) -> bool {
    inputs
        .iter()
        .any(|i| matches!(i.space, TensorSpace::TimeSeries { .. }))
}

fn has_robot_trajectory(outputs: &[OutputPortSpec]) -> bool {
    outputs
        .iter()
        .any(|o| matches!(o.space, TensorSpace::RigidBodyTrajectory { .. }))
}

fn has_tabular_features(inputs: &[InputPortSpec]) -> bool {
    inputs
        .iter()
        .any(|i| matches!(i.space, TensorSpace::TabularFeatures { .. }))
}

fn has_discrete_tokens(inputs: &[InputPortSpec]) -> bool {
    inputs
        .iter()
        .any(|i| matches!(i.space, TensorSpace::DiscreteTokens { .. }))
}

fn has_coordinate_field(inputs: &[InputPortSpec]) -> bool {
    inputs
        .iter()
        .any(|i| matches!(i.space, TensorSpace::CoordinateField { .. }))
}

fn synthesize_vision_flow_matching(spec: &mut AiDesignSpec) {
    spec.metadata.domain = "ComputerVision".to_string();
    spec.metadata.tags = vec![
        "vision".to_string(),
        "flow-matching".to_string(),
        "dit".to_string(),
        "diffusion".to_string(),
    ];

    // Ensure 3D RoPE or Patching on inputs
    for input in &mut spec.inputs {
        if matches!(input.space, TensorSpace::Video3D { .. }) {
            input.coordinate_encoding = CoordinateEncoding::RoPE3D { base: 10000.0 };
            input.patching = Some(PatchingSpec {
                patch_size: [2, 2],
                flatten_channels: true,
            });
        }
    }

    for output in &mut spec.outputs {
        output.loss = LossPairing::OptimalTransportConditionalFlowMatching;
    }

    spec.graph.topology = WiringPattern::ResidualStream {
        norm_placement: NormPlacement::PreLN,
        layer_scale_init: Some(1e-2),
    };

    spec.graph.nodes = vec![
        NodeSpec {
            id: "patch_embed".to_string(),
            name: "Patch Linear Embedding".to_string(),
            operator: OperatorKind::Linear {
                in_features: 64,
                out_features: 512,
                bias: true,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "dit_block_1".to_string(),
            name: "DiT MHA Backbone".to_string(),
            operator: OperatorKind::MHA {
                num_heads: 8,
                d_k: 64,
                d_v: 64,
                dropout: 0.0,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "swiglu_ffn".to_string(),
            name: "SwiGLU Feedforward".to_string(),
            operator: OperatorKind::SwiGLU {
                in_features: 512,
                hidden_features: 1376,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "velocity_head".to_string(),
            name: "Flow Velocity Vector Field Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: 512,
                out_features: 64,
                bias: true,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
    ];
}

fn synthesize_timeseries_forecaster(spec: &mut AiDesignSpec) {
    spec.metadata.domain = "TimeSeriesAndFinance".to_string();
    spec.metadata.tags = vec![
        "time-series".to_string(),
        "xlstm".to_string(),
        "revin".to_string(),
        "crps".to_string(),
    ];

    for input in &mut spec.inputs {
        if matches!(input.space, TensorSpace::TimeSeries { .. }) {
            input.normalizer = NormalizerSpec::RevIN {
                eps: 1e-5,
                affine: true,
            };
        }
    }

    for output in &mut spec.outputs {
        output.constraint = StructuralConstraint::QuantileCRPS_GPD {
            quantiles: vec![0.1, 0.5, 0.9],
        };
        output.loss = LossPairing::QuantileCRPS;
    }

    spec.graph.nodes = vec![
        NodeSpec {
            id: "ts_proj".to_string(),
            name: "Temporal Input Projection".to_string(),
            operator: OperatorKind::Linear {
                in_features: 32,
                out_features: 128,
                bias: true,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "xlstm_core".to_string(),
            name: "Extended LSTM (mLSTM Matrix Memory)".to_string(),
            operator: OperatorKind::xLSTM {
                d_model: 128,
                num_heads: 4,
                variant: "mLSTM".to_string(),
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "quantile_head".to_string(),
            name: "Probabilistic Quantile Readout Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: 128,
                out_features: 3,
                bias: true,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
    ];
}

fn synthesize_robotic_diffusion_policy(spec: &mut AiDesignSpec) {
    spec.metadata.domain = "RoboticsAndEmbodiedAI".to_string();
    spec.metadata.tags = vec![
        "robotics".to_string(),
        "cbf-qp".to_string(),
        "continuous-6d".to_string(),
        "act-policy".to_string(),
    ];

    for output in &mut spec.outputs {
        if matches!(output.space, TensorSpace::RigidBodyTrajectory { .. }) {
            output.constraint = StructuralConstraint::ContinuousRotation6D;
        }
    }

    spec.graph.nodes = vec![
        NodeSpec {
            id: "policy_trunk".to_string(),
            name: "Action Chunking Trunk".to_string(),
            operator: OperatorKind::Linear {
                in_features: 128,
                out_features: 256,
                bias: true,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "cbf_qp_safety".to_string(),
            name: "Differentiable Control Barrier Filter".to_string(),
            operator: OperatorKind::Linear {
                in_features: 256,
                out_features: 7,
                bias: true,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
    ];
}

fn synthesize_tabular_prior_network(spec: &mut AiDesignSpec) {
    spec.metadata.domain = "MedicalAndHealthcare".to_string();
    spec.metadata.tags = vec![
        "tabular".to_string(),
        "ple".to_string(),
        "tabpfn".to_string(),
        "monotonic".to_string(),
    ];

    for input in &mut spec.inputs {
        if matches!(input.space, TensorSpace::TabularFeatures { .. }) {
            input.normalizer = NormalizerSpec::PLE { num_bins: 16 };
        }
    }

    spec.graph.nodes = vec![
        NodeSpec {
            id: "ple_embed".to_string(),
            name: "Piecewise Linear Feature Projection".to_string(),
            operator: OperatorKind::Linear {
                in_features: 256,
                out_features: 128,
                bias: true,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "sparse_entmax".to_string(),
            name: "Entmax 1.5 Sparse Selection Gate".to_string(),
            operator: OperatorKind::Entmax15Gate,
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "tabpfn_attn".to_string(),
            name: "TabPFN Prior In-Context Attention".to_string(),
            operator: OperatorKind::MHA {
                num_heads: 4,
                d_k: 32,
                d_v: 32,
                dropout: 0.05,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "monotonic_head".to_string(),
            name: "Monotonic Output Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: 128,
                out_features: 1,
                bias: true,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
    ];
}

fn synthesize_reasoning_hybrid_backbone(spec: &mut AiDesignSpec) {
    spec.metadata.domain = "NaturalLanguageProcessing".to_string();
    spec.metadata.tags = vec![
        "nlp".to_string(),
        "mamba-2".to_string(),
        "mla".to_string(),
        "deepseek-moe".to_string(),
    ];

    for input in &mut spec.inputs {
        if matches!(input.space, TensorSpace::DiscreteTokens { .. }) {
            input.coordinate_encoding = CoordinateEncoding::DecoupledRoPE;
        }
    }

    spec.graph.nodes = vec![
        NodeSpec {
            id: "embed".to_string(),
            name: "Token Embedding".to_string(),
            operator: OperatorKind::Embedding {
                num_embeddings: 32000,
                embedding_dim: 1024,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "mamba2_ssd".to_string(),
            name: "Mamba-2 State Space Duality Block".to_string(),
            operator: OperatorKind::Mamba2SSD {
                d_model: 1024,
                d_state: 64,
                d_conv: 4,
                expand: 2,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "mla_attn".to_string(),
            name: "Multi-Head Latent Attention (MLA)".to_string(),
            operator: OperatorKind::MLA {
                num_heads: 8,
                d_latent_kv: 256,
                d_rope: 32,
                d_v: 64,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "moe_ffn".to_string(),
            name: "DeepSeek Fine-Grained MoE".to_string(),
            operator: OperatorKind::DeepSeekMoE {
                total_experts: 16,
                routed_k: 4,
                shared_experts: 2,
                d_model: 1024,
                d_ffn: 768,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "lm_head".to_string(),
            name: "Language Modeling Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: 1024,
                out_features: 32000,
                bias: false,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
    ];
}

fn synthesize_fourier_neural_operator(spec: &mut AiDesignSpec) {
    spec.metadata.domain = "PhysicsAndScientificAI".to_string();
    spec.metadata.tags = vec![
        "physics".to_string(),
        "pde".to_string(),
        "fno".to_string(),
        "curl-free".to_string(),
    ];

    for output in &mut spec.outputs {
        output.constraint = StructuralConstraint::CurlFreeVectorField;
    }

    spec.graph.nodes = vec![
        NodeSpec {
            id: "fno_core".to_string(),
            name: "2D Fourier Neural Operator (FNO)".to_string(),
            operator: OperatorKind::FourierNeuralOperator {
                modes: vec![16, 16],
                width: 64,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "field_head".to_string(),
            name: "Potential Field Projection Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: 64,
                out_features: 2,
                bias: true,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
    ];
}

fn synthesize_general_dense_residual(spec: &mut AiDesignSpec) {
    spec.graph.nodes = vec![
        NodeSpec {
            id: "in_proj".to_string(),
            name: "Input Projection".to_string(),
            operator: OperatorKind::Linear {
                in_features: 64,
                out_features: 128,
                bias: true,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "swiglu".to_string(),
            name: "SwiGLU Non-Linear Block".to_string(),
            operator: OperatorKind::SwiGLU {
                in_features: 128,
                hidden_features: 256,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: Default::default(),
        },
        NodeSpec {
            id: "out_proj".to_string(),
            name: "Readout Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: 128,
                out_features: 10,
                bias: true,
            },
            precision: Some(PrecisionFormat::FP32),
            metadata: Default::default(),
        },
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::io::SymbolicDim;

    #[test]
    fn test_auto_architect_time_series_synthesis() {
        let input = InputPortSpec {
            id: "prices".to_string(),
            name: "Prices".to_string(),
            space: TensorSpace::TimeSeries {
                channels: 16,
                horizon: 256,
                stationary: false,
            },
            symbolic_shape: vec![
                SymbolicDim::Symbol("B".to_string()),
                SymbolicDim::Known(16),
                SymbolicDim::Known(256),
            ],
            normalizer: NormalizerSpec::None,
            coordinate_encoding: CoordinateEncoding::None,
            patching: None,
        };

        let output = OutputPortSpec {
            id: "forecast".to_string(),
            name: "Forecast".to_string(),
            space: TensorSpace::TimeSeries {
                channels: 16,
                horizon: 64,
                stationary: false,
            },
            symbolic_shape: vec![
                SymbolicDim::Symbol("B".to_string()),
                SymbolicDim::Known(16),
                SymbolicDim::Known(64),
            ],
            constraint: StructuralConstraint::DirectLinear,
            loss: LossPairing::MSE,
        };

        let synthesized =
            synthesize_architecture("auto_ts", "Auto Time Series", vec![input], vec![output]);

        // Must automatically inject RevIN on input
        assert!(matches!(
            synthesized.inputs[0].normalizer,
            NormalizerSpec::RevIN { .. }
        ));
        // Must automatically inject QuantileCRPS on output
        assert!(matches!(
            synthesized.outputs[0].loss,
            LossPairing::QuantileCRPS
        ));
        // Must contain xLSTM core
        assert!(
            synthesized
                .graph
                .nodes
                .iter()
                .any(|n| matches!(n.operator, OperatorKind::xLSTM { .. }))
        );
    }
}
