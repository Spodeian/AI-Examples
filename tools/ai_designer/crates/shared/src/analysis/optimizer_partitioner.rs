//! Automated partitioning of model parameters into Muon 2D GEMMs, AdamW 1D/heads, and 0-WD groups.

use crate::spec::AiDesignSpec;
use crate::spec::operators::OperatorKind;
use serde::{Deserialize, Serialize};

/// Categorical optimizer group assignment based on tensor dimensionality and inductive bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizerGroup {
    /// Internal 2D weight matrices (d_out >= 64, d_in >= 64) updated via Newton-Schulz orthogonalization.
    Muon2DGEMM,
    /// Token embeddings, small 1D vectors, and terminal readout projection heads.
    AdamW1D,
    /// Scale vectors (RMSNorm/LayerNorm), biases, router load-balancing biases, and KAN spline knots.
    ZeroWeightDecay,
}

/// Metadata and parameter configuration for an individual learnable tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamDescriptor {
    pub name: String,
    pub node_id: String,
    pub shape: Vec<usize>,
    pub param_count: usize,
    pub group: OptimizerGroup,
    pub suggested_lr: f64,
    pub suggested_weight_decay: f64,
}

/// Consolidated parameter partitioning report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionReport {
    pub muon_params: Vec<ParamDescriptor>,
    pub adamw_params: Vec<ParamDescriptor>,
    pub zero_wd_params: Vec<ParamDescriptor>,
    pub total_param_count: usize,
    pub muon_param_count: usize,
    pub adamw_param_count: usize,
    pub zero_wd_param_count: usize,
    pub muon_ratio: f32,
}

impl PartitionReport {
    pub fn new() -> Self {
        Self {
            muon_params: Vec::new(),
            adamw_params: Vec::new(),
            zero_wd_params: Vec::new(),
            total_param_count: 0,
            muon_param_count: 0,
            adamw_param_count: 0,
            zero_wd_param_count: 0,
            muon_ratio: 0.0,
        }
    }
}

impl Default for PartitionReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Inspects an AiDesignSpec and computes the three-way optimizer parameter partitioning.
pub fn partition_parameters(spec: &AiDesignSpec) -> PartitionReport {
    let mut report = PartitionReport::new();

    let muon_base_lr = spec.training.muon.lr;
    let adamw_base_lr = spec.training.adamw.lr;
    let adamw_wd = spec.training.adamw.weight_decay;

    let node_count = spec.graph.nodes.len();

    for (node_idx, node) in spec.graph.nodes.iter().enumerate() {
        let is_last_node = node_idx == node_count - 1;

        match &node.operator {
            OperatorKind::Linear {
                in_features,
                out_features,
                bias,
            } => {
                let is_gemm_large = *in_features >= 64 && *out_features >= 64;
                let weight_count = in_features * out_features;

                // Terminal readout head goes to AdamW to preserve calibrated loss scale
                if is_last_node || !is_gemm_large {
                    report.adamw_params.push(ParamDescriptor {
                        name: format!("{}.weight", node.id),
                        node_id: node.id.clone(),
                        shape: vec![*out_features, *in_features],
                        param_count: weight_count,
                        group: OptimizerGroup::AdamW1D,
                        suggested_lr: adamw_base_lr,
                        suggested_weight_decay: adamw_wd,
                    });
                } else {
                    report.muon_params.push(ParamDescriptor {
                        name: format!("{}.weight", node.id),
                        node_id: node.id.clone(),
                        shape: vec![*out_features, *in_features],
                        param_count: weight_count,
                        group: OptimizerGroup::Muon2DGEMM,
                        suggested_lr: muon_base_lr,
                        suggested_weight_decay: 0.0,
                    });
                }

                if *bias {
                    report.zero_wd_params.push(ParamDescriptor {
                        name: format!("{}.bias", node.id),
                        node_id: node.id.clone(),
                        shape: vec![*out_features],
                        param_count: *out_features,
                        group: OptimizerGroup::ZeroWeightDecay,
                        suggested_lr: adamw_base_lr,
                        suggested_weight_decay: 0.0,
                    });
                }
            }

            OperatorKind::Embedding {
                num_embeddings,
                embedding_dim,
            } => {
                let count = num_embeddings * embedding_dim;
                report.adamw_params.push(ParamDescriptor {
                    name: format!("{}.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*num_embeddings, *embedding_dim],
                    param_count: count,
                    group: OptimizerGroup::AdamW1D,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: adamw_wd,
                });
            }

            OperatorKind::MHA {
                num_heads,
                d_k,
                d_v,
                ..
            } => {
                let d_model = num_heads * d_k;
                let qkv_weights = 3 * d_model * d_model;
                let out_weights = (num_heads * d_v) * d_model;

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.qkv_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * d_model, d_model],
                    param_count: qkv_weights,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_model, num_heads * d_v],
                    param_count: out_weights,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::MLA {
                num_heads,
                d_latent_kv,
                d_rope,
                d_v,
            } => {
                let d_model = num_heads * d_v;
                let q_down_count = d_model * d_latent_kv;
                let kv_down_count = d_model * (d_latent_kv + d_rope);
                let out_count = d_model * (num_heads * d_v);

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.q_down.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_latent_kv, d_model],
                    param_count: q_down_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.kv_down.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_latent_kv + d_rope, d_model],
                    param_count: kv_down_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_model, num_heads * d_v],
                    param_count: out_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::Mamba2SSD {
                d_model,
                d_state: _,
                d_conv,
                expand,
            } => {
                let d_inner = expand * d_model;
                let in_proj_count = 2 * d_inner * d_model;
                let out_proj_count = d_model * d_inner;
                let conv_count = d_inner * d_conv;
                let dt_count = d_inner;

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.in_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![2 * d_inner, *d_model],
                    param_count: in_proj_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_model, d_inner],
                    param_count: out_proj_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.conv1d.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_inner, 1, *d_conv],
                    param_count: conv_count,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.dt_bias", node.id),
                    node_id: node.id.clone(),
                    shape: vec![dt_count],
                    param_count: dt_count,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::RMSNorm { dim, .. } | OperatorKind::LayerNorm { dim, .. } => {
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*dim],
                    param_count: *dim,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::KANLayer {
                in_features,
                out_features,
                grid_size,
                spline_order,
            } => {
                let base_weights = in_features * out_features;
                let spline_weights = in_features * out_features * (grid_size + spline_order);

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.base_weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*out_features, *in_features],
                    param_count: base_weights,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                // Spline knot coefficients receive zero weight decay to prevent collapsing grid basis
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.spline_weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*out_features, *in_features, grid_size + spline_order],
                    param_count: spline_weights,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::DeepSeekMoE {
                total_experts,
                shared_experts,
                d_model,
                d_ffn,
                ..
            } => {
                let expert_gate_count = total_experts * d_model;
                let per_expert_ffn = 2 * d_model * d_ffn; // gate + up + down
                let total_routed_ffn = (total_experts + shared_experts) * per_expert_ffn;

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.expert_weights", node.id),
                    node_id: node.id.clone(),
                    shape: vec![total_experts + shared_experts, 2 * d_ffn, *d_model],
                    param_count: total_routed_ffn,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                // Router gating logits
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.router_gate.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*total_experts, *d_model],
                    param_count: expert_gate_count,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            _ => {}
        }
    }

    report.muon_param_count = report.muon_params.iter().map(|p| p.param_count).sum();
    report.adamw_param_count = report.adamw_params.iter().map(|p| p.param_count).sum();
    report.zero_wd_param_count = report.zero_wd_params.iter().map(|p| p.param_count).sum();

    report.total_param_count =
        report.muon_param_count + report.adamw_param_count + report.zero_wd_param_count;
    if report.total_param_count > 0 {
        report.muon_ratio = report.muon_param_count as f32 / report.total_param_count as f32;
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::graph::NodeSpec;

    #[test]
    fn test_optimizer_partitioning_rules() {
        let mut spec = AiDesignSpec::new("test_opt", "Test Optimizer Model");
        spec.graph.nodes.push(NodeSpec {
            id: "embed".to_string(),
            name: "Embedding".to_string(),
            operator: OperatorKind::Embedding {
                num_embeddings: 1000,
                embedding_dim: 128,
            },
            precision: None,
            metadata: Default::default(),
        });
        spec.graph.nodes.push(NodeSpec {
            id: "internal_gemm".to_string(),
            name: "Internal GEMM".to_string(),
            operator: OperatorKind::Linear {
                in_features: 128,
                out_features: 256,
                bias: true,
            },
            precision: None,
            metadata: Default::default(),
        });
        spec.graph.nodes.push(NodeSpec {
            id: "norm".to_string(),
            name: "RMS Norm".to_string(),
            operator: OperatorKind::RMSNorm {
                dim: 256,
                eps: 1e-5,
            },
            precision: None,
            metadata: Default::default(),
        });
        spec.graph.nodes.push(NodeSpec {
            id: "head".to_string(),
            name: "Final Readout Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: 256,
                out_features: 10,
                bias: true,
            },
            precision: None,
            metadata: Default::default(),
        });

        let report = partition_parameters(&spec);
        assert_eq!(report.muon_params.len(), 1);
        assert_eq!(report.muon_params[0].name, "internal_gemm.weight");

        assert_eq!(report.adamw_params.len(), 2);
        assert_eq!(report.adamw_params[0].name, "embed.weight");
        assert_eq!(report.adamw_params[1].name, "head.weight");

        assert_eq!(report.zero_wd_params.len(), 3); // internal_gemm.bias, norm.weight, head.bias
    }
}
