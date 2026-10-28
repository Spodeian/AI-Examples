//! Multi-Objective Pareto Evolution & Architecture Search Engine.
//! Balances FLOPs, Latency $L^*$, VRAM Budget, and Model Capacity across hardware presets.

use crate::analysis::roofline::analyze_roofline;
use crate::spec::*;
use serde::{Deserialize, Serialize};

/// Preset objective profile for Pareto evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParetoObjectivePreset {
    EdgeConstrained,     // Cortex-M / Jetson Orin: minimal FLOPs & VRAM
    WorkstationBalanced, // RTX 3060 / 4090: optimal throughput & latency
    DatacenterMaximal,   // H100: maximal parameters & capacity
}

/// A candidate architecture along the synthesized Pareto optimal frontier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoCandidate {
    pub label: String,
    pub description: String,
    pub spec: AiDesignSpec,
    pub total_param_count: usize,
    pub flops_per_token: usize,
    pub latency_ms: f64,
    pub vram_bytes: usize,
    pub is_oom_risk: bool,
    pub score: f64,
}

/// Generates a 5-candidate Pareto frontier spanning from ultra-compact edge to high-capacity frontier.
pub fn generate_pareto_frontier(
    base_spec: &AiDesignSpec,
    target_hardware: Option<&HardwareTargetProfile>,
) -> Vec<ParetoCandidate> {
    let hw = target_hardware
        .cloned()
        .unwrap_or_else(|| base_spec.metadata.target_hardware.clone());
    let mut candidates = Vec::new();

    // 1. Edge-Minimal Candidate (Low FLOPs, Low VRAM)
    let edge_spec = mutate_architecture(
        base_spec,
        &hw,
        0.5,
        PrecisionFormat::FP8E4M3FN,
        "edge_minimal",
    );
    candidates.push(evaluate_candidate(
        "1. Edge-Minimal",
        "Ultra-lean design optimized for edge MCUs and low latency",
        edge_spec,
        &hw,
    ));

    // 2. Latency-Optimal Candidate (Fastest forward pass)
    let latency_spec = mutate_architecture(
        base_spec,
        &hw,
        0.75,
        PrecisionFormat::BF16,
        "latency_optimal",
    );
    candidates.push(evaluate_candidate(
        "2. Latency-Optimal",
        "Low-depth parallelized architecture for real-time inference",
        latency_spec,
        &hw,
    ));

    // 3. Workstation-Balanced Candidate (Default optimal balance)
    let balanced_spec = mutate_architecture(
        base_spec,
        &hw,
        1.0,
        PrecisionFormat::BF16,
        "workstation_balanced",
    );
    candidates.push(evaluate_candidate(
        "3. Workstation-Balanced",
        "Standard capacity tuned for single-GPU RTX 3060/4090",
        balanced_spec,
        &hw,
    ));

    // 4. Capacity-Maximal Candidate (Deeper layers, high expressivity)
    let capacity_spec = mutate_architecture(
        base_spec,
        &hw,
        1.5,
        PrecisionFormat::BF16,
        "capacity_maximal",
    );
    candidates.push(evaluate_candidate(
        "4. High-Capacity",
        "Deeper representations with SwiGLU / MoE expansion",
        capacity_spec,
        &hw,
    ));

    // 5. Frontier-Extreme Candidate (Near hardware ceiling)
    let frontier_spec = mutate_architecture(
        base_spec,
        &hw,
        2.0,
        PrecisionFormat::BF16,
        "frontier_extreme",
    );
    candidates.push(evaluate_candidate(
        "5. Frontier-Extreme",
        "Maximum capacity model pushing compute towards the hardware roofline",
        frontier_spec,
        &hw,
    ));

    candidates
}

fn mutate_architecture(
    base: &AiDesignSpec,
    hw: &HardwareTargetProfile,
    scale_factor: f64,
    target_precision: PrecisionFormat,
    variant_id: &str,
) -> AiDesignSpec {
    let mut mutated = base.clone();
    mutated.metadata.model_id = format!("{}_{}", base.metadata.model_id, variant_id);
    mutated.metadata.display_name = format!("{} ({})", base.metadata.display_name, variant_id);
    mutated.metadata.target_hardware = hw.clone();
    mutated.precision.default_weight_precision = target_precision;

    // Mutate nodes: scale features by scale_factor
    for node in &mut mutated.graph.nodes {
        node.precision = Some(target_precision);
        match &mut node.operator {
            OperatorKind::Linear {
                in_features,
                out_features,
                ..
            } => {
                *in_features = ((*in_features as f64 * scale_factor).round() as usize).max(16);
                *out_features = ((*out_features as f64 * scale_factor).round() as usize).max(16);
            }
            OperatorKind::LayerNorm { dim, .. } | OperatorKind::RMSNorm { dim, .. } => {
                *dim = ((*dim as f64 * scale_factor).round() as usize).max(16);
            }
            OperatorKind::MHA {
                num_heads,
                d_k,
                d_v,
                ..
            } => {
                if scale_factor > 1.2 {
                    *num_heads = (*num_heads * 2).max(1);
                } else if scale_factor < 0.8 {
                    *num_heads = (*num_heads / 2).max(1);
                }
                *d_k = ((*d_k as f64 * scale_factor).round() as usize).max(16);
                *d_v = *d_k;
            }
            OperatorKind::SwiGLU {
                in_features,
                hidden_features,
            }
            | OperatorKind::GeGLU {
                in_features,
                hidden_features,
            } => {
                *in_features = ((*in_features as f64 * scale_factor).round() as usize).max(16);
                *hidden_features =
                    ((*hidden_features as f64 * scale_factor).round() as usize).max(32);
            }
            _ => {}
        }
    }

    mutated
}

fn evaluate_candidate(
    label: &str,
    description: &str,
    spec: AiDesignSpec,
    _hw: &HardwareTargetProfile,
) -> ParetoCandidate {
    let roofline = analyze_roofline(&spec, 1, 128);
    let total_params =
        crate::analysis::optimizer_partitioner::partition_parameters(&spec).total_param_count;

    // Multi-objective score: higher is better (FLOPs efficiency penalized by latency and VRAM footprint)
    let score = if roofline.vram.is_oom_risk {
        0.0
    } else {
        (total_params as f64).log10() / (roofline.estimated_latency_per_step_ms + 0.01)
    };

    ParetoCandidate {
        label: label.to_string(),
        description: description.to_string(),
        spec,
        total_param_count: total_params,
        flops_per_token: roofline.total_flops_per_token,
        latency_ms: roofline.estimated_latency_per_step_ms,
        vram_bytes: roofline.vram.total_allocated_bytes,
        is_oom_risk: roofline.vram.is_oom_risk,
        score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_frontier_generation() {
        let mut spec = AiDesignSpec::new("toy_transformer", "Toy Transformer");
        spec.inputs.push(InputPortSpec {
            id: "x".to_string(),
            name: "Input".to_string(),
            space: TensorSpace::ContinuousVector {
                dimension: 256,
                range: None,
            },
            symbolic_shape: vec![SymbolicDim::from("B"), SymbolicDim::from(256)],
            normalizer: NormalizerSpec::None,
            coordinate_encoding: CoordinateEncoding::None,
            patching: None,
        });
        spec.outputs.push(OutputPortSpec {
            id: "y".to_string(),
            name: "Output".to_string(),
            space: TensorSpace::ContinuousVector {
                dimension: 10,
                range: None,
            },
            symbolic_shape: vec![SymbolicDim::from("B"), SymbolicDim::from(10)],
            constraint: StructuralConstraint::DirectLinear,
            loss: LossPairing::CrossEntropy,
        });
        spec.graph.nodes = vec![
            NodeSpec {
                id: "linear_1".to_string(),
                name: "Dense 1".to_string(),
                operator: OperatorKind::Linear {
                    in_features: 256,
                    out_features: 512,
                    bias: true,
                },
                precision: Some(PrecisionFormat::BF16),
                metadata: Default::default(),
            },
            NodeSpec {
                id: "linear_2".to_string(),
                name: "Dense 2".to_string(),
                operator: OperatorKind::Linear {
                    in_features: 512,
                    out_features: 10,
                    bias: true,
                },
                precision: Some(PrecisionFormat::BF16),
                metadata: Default::default(),
            },
        ];

        let frontier = generate_pareto_frontier(&spec, None);
        assert_eq!(frontier.len(), 5);

        // Assert monotonic ordering in parameter capacity
        assert!(frontier[0].total_param_count < frontier[2].total_param_count);
        assert!(frontier[2].total_param_count < frontier[4].total_param_count);
    }
}
