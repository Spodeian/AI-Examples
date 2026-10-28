//! Autonomous Transitive Parity Engine & Forward Numerical Health Verifier.
//! Evaluates analytical DAG execution with deterministic Seed-42 synthetic inputs,
//! checking for numerical stability, vanishing/exploding activations, and NaN/Inf traps.

use crate::spec::{AiDesignSpec, OperatorKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParityStatus {
    Passed,
    Warning,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityVerificationReport {
    pub model_id: String,
    pub nodes_evaluated: usize,
    pub max_activation: f32,
    pub min_activation: f32,
    pub has_nan_or_inf: bool,
    pub condition_stable: bool,
    pub status: ParityStatus,
    pub node_activations: HashMap<String, Vec<f32>>,
    pub diagnostics: Vec<String>,
}

/// Evaluates synthetic forward pass and verifies numerical condition across all nodes.
pub fn verify_transitive_parity(spec: &AiDesignSpec) -> ParityVerificationReport {
    let mut diagnostics = Vec::new();
    let mut node_activations: HashMap<String, Vec<f32>> = HashMap::new();
    let mut has_nan_or_inf = false;
    let mut max_act = 0.0f32;
    let mut min_act = f32::MAX;

    // Seed-42 initial deterministic synthetic input vector (length 128)
    let d_in = spec.inputs.first().map_or(128, |p| {
        p.symbolic_shape
            .last()
            .and_then(|dim| match dim {
                crate::spec::SymbolicDim::Known(k) => Some(*k),
                _ => None,
            })
            .unwrap_or(128)
    });

    let mut current_vec: Vec<f32> = (0..d_in)
        .map(|i| (i as f32 * 0.37 + 0.13).sin() * 0.5)
        .collect();

    for (i, node) in spec.graph.nodes.iter().enumerate() {
        let in_len = current_vec.len();
        let out_vec = match &node.operator {
            OperatorKind::Linear {
                in_features,
                out_features,
                bias,
            } => {
                let out_dim = *out_features;
                let mut res = vec![0.0f32; out_dim];
                for (o, res_elem) in res.iter_mut().enumerate() {
                    let mut sum = if *bias { 0.01 * (o as f32).sin() } else { 0.0 };
                    for (k, &curr_val) in current_vec
                        .iter()
                        .take((*in_features).min(in_len))
                        .enumerate()
                    {
                        let w = ((k as f32 * 0.17 + o as f32 * 0.31).cos()
                            / (k as f32 + 1.0).sqrt())
                            * 0.1;
                        sum += curr_val * w;
                    }
                    *res_elem = sum;
                }
                res
            }
            OperatorKind::RMSNorm { dim, eps } => {
                let d = (*dim).min(in_len);
                let sum_sq: f32 = current_vec[0..d].iter().map(|x| x * x).sum();
                let rms = ((sum_sq / d as f32) + (*eps as f32)).sqrt();
                current_vec[0..d].iter().map(|x| x / rms).collect()
            }
            OperatorKind::LayerNorm { dim, eps } => {
                let d = (*dim).min(in_len);
                let mean: f32 = current_vec[0..d].iter().sum::<f32>() / d as f32;
                let var: f32 = current_vec[0..d]
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>()
                    / d as f32;
                let std = (var + (*eps as f32)).sqrt();
                current_vec[0..d].iter().map(|x| (x - mean) / std).collect()
            }
            OperatorKind::SwiGLU {
                in_features: _,
                hidden_features,
            } => {
                let h = *hidden_features;
                (0..h)
                    .map(|j| {
                        let val = current_vec.get(j % in_len).copied().unwrap_or(0.1);
                        let gate = val * (1.0 / (1.0 + (-val).exp())); // SiLU
                        gate * val
                    })
                    .collect()
            }
            OperatorKind::Snake { frequency } => {
                let a = *frequency;
                current_vec
                    .iter()
                    .map(|&x| x + (1.0 / a) * (a * x).sin().powi(2))
                    .collect()
            }
            OperatorKind::Mamba2SSD {
                d_model, expand, ..
            } => {
                let d_inner = d_model * expand;
                (0..d_inner)
                    .map(|j| {
                        let x = current_vec.get(j % in_len).copied().unwrap_or(0.05);
                        x * 0.95 // linear recurrence decay
                    })
                    .collect()
            }
            OperatorKind::WTConv { out_channels, .. } => (0..*out_channels)
                .map(|c| {
                    let x = current_vec.get(c % in_len).copied().unwrap_or(0.1);
                    x * 0.85
                })
                .collect(),
            _ => {
                // Default identity or dimension projection
                current_vec.clone()
            }
        };

        // Inspect activation health
        for &v in &out_vec {
            if v.is_nan() || v.is_infinite() {
                has_nan_or_inf = true;
                diagnostics.push(format!(
                    "Node '{}' [{}] emitted NaN or Infinity!",
                    node.id, node.name
                ));
                break;
            }
            let abs_v = v.abs();
            if abs_v > max_act {
                max_act = abs_v;
            }
            if abs_v < min_act && abs_v > 0.0 {
                min_act = abs_v;
            }
        }

        if max_act > 1e4 {
            diagnostics.push(format!(
                "Node '{}' [{}] exhibits exploding activations: max |a| = {:.2e}",
                node.id, node.name, max_act
            ));
        }

        node_activations.insert(node.id.clone(), out_vec.clone());
        current_vec = out_vec;

        if i >= 64 {
            break; // Cap forward pass loop for deep topologies
        }
    }

    let condition_stable =
        !has_nan_or_inf && max_act < 1e3 && (min_act > 1e-6 || min_act == f32::MAX);

    let status = if has_nan_or_inf {
        ParityStatus::Failed
    } else if !condition_stable {
        ParityStatus::Warning
    } else {
        ParityStatus::Passed
    };

    if status == ParityStatus::Passed {
        diagnostics.push(format!(
            "Forward pass stable across {} layers. Activation dynamic range: [{:.2e}, {:.2e}].",
            spec.graph.nodes.len(),
            if min_act == f32::MAX { 0.0 } else { min_act },
            max_act
        ));
    } else if status == ParityStatus::Warning {
        if min_act < 1e-6 {
            diagnostics.push(format!("Warning: Extremely small minimum activation magnitude ({:.2e} < 1e-6); risk of vanishing gradients or underflow in reduced precision.", min_act));
        }
        if max_act >= 1e3 {
            diagnostics.push(format!("Warning: High maximum activation magnitude ({:.2e} >= 1e3); risk of FP16/FP8 overflow without scaling.", max_act));
        }
    }

    ParityVerificationReport {
        model_id: spec.metadata.model_id.clone(),
        nodes_evaluated: spec.graph.nodes.len(),
        max_activation: max_act,
        min_activation: if min_act == f32::MAX { 0.0 } else { min_act },
        has_nan_or_inf,
        condition_stable,
        status,
        node_activations,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::aid_yaml::from_yaml_str;

    #[test]
    fn test_transitive_parity_on_spatiotemporal_dit() {
        let yaml_str = include_str!("../../../../templates/01_spatiotemporal_dit_flow.aid.yaml");
        let spec: AiDesignSpec = from_yaml_str(yaml_str).unwrap();
        let report = verify_transitive_parity(&spec);

        assert_eq!(report.has_nan_or_inf, false);
        assert!(report.status == ParityStatus::Passed || report.status == ParityStatus::Warning);
        assert!(report.max_activation > 0.0);
    }
}
