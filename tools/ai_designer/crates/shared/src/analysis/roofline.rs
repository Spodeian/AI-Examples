//! Analytical Roofline Model, FLOP calculation, Operational Intensity, and VRAM Waterfall.

use super::optimizer_partitioner::partition_parameters;
use crate::spec::AiDesignSpec;
use crate::spec::precision::PrecisionFormat;
use serde::{Deserialize, Serialize};

/// Roofline regime classification indicating execution bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RooflineRegime {
    MemoryBandwidthBound,
    ComputeBound,
}

/// Detailed VRAM memory footprint waterfall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VramWaterfall {
    pub weights_bytes: usize,
    pub master_weights_bytes: usize,
    pub optimizer_states_bytes: usize,
    pub activation_bytes: usize,
    pub kv_cache_bytes: usize,
    pub total_allocated_bytes: usize,
    pub max_hardware_vram_bytes: usize,
    pub headroom_bytes: i64,
    pub is_oom_risk: bool,
}

/// Consolidated Roofline analysis report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RooflineReport {
    pub hardware_profile: String,
    pub total_flops_per_token: usize,
    pub total_bytes_transferred_per_token: usize,
    pub operational_intensity: f64,
    pub machine_balance: f64,
    pub regime: RooflineRegime,
    pub peak_tflops: f64,
    pub memory_bandwidth_gbs: f64,
    pub estimated_latency_per_step_ms: f64,
    pub critical_horizon_tokens: usize,
    pub vram: VramWaterfall,
}

/// Analyzes an AiDesignSpec against its configured target hardware profile.
pub fn analyze_roofline(spec: &AiDesignSpec, batch_size: usize, seq_len: usize) -> RooflineReport {
    let hw = &spec.metadata.target_hardware;
    let partition = partition_parameters(spec);

    let default_prec = spec.precision.default_weight_precision;
    let bytes_per_elem = (default_prec.bits() / 8.0).max(0.5);

    // 1. Model Weights VRAM
    let weights_bytes = (partition.total_param_count as f64 * bytes_per_elem as f64) as usize;

    // 2. Master Weights VRAM (FP32 copy for mixed-precision training: 4 bytes per param)
    let master_weights_bytes = partition.total_param_count * 4;

    // 3. Optimizer State VRAM
    // Muon state = 1 FP32 momentum buffer per param (4 bytes)
    // AdamW / Zero-WD state = 2 FP32 moment buffers per param (8 bytes)
    let muon_opt_bytes = partition.muon_param_count * 4;
    let adamw_opt_bytes = (partition.adamw_param_count + partition.zero_wd_param_count) * 8;
    let optimizer_states_bytes = muon_opt_bytes + adamw_opt_bytes;

    // 4. Activation Memory
    // Base activation estimate = sum of hidden dimensions across layers * batch * seq_len * 2 bytes
    let total_hidden_dim: usize = spec
        .graph
        .nodes
        .iter()
        .map(|n| match &n.operator {
            crate::spec::operators::OperatorKind::Linear { out_features, .. } => *out_features,
            crate::spec::operators::OperatorKind::MHA { num_heads, d_v, .. } => num_heads * d_v,
            crate::spec::operators::OperatorKind::MLA { num_heads, d_v, .. } => num_heads * d_v,
            crate::spec::operators::OperatorKind::Mamba2SSD { d_model, .. } => *d_model,
            crate::spec::operators::OperatorKind::xLSTM { d_model, .. } => *d_model,
            _ => 128,
        })
        .sum();

    let raw_activation_bytes = batch_size * seq_len * total_hidden_dim * 2;
    let activation_bytes = if spec.training.hygiene.activation_checkpointing {
        raw_activation_bytes / 3 // ~3x activation memory reduction via checkpointing
    } else {
        raw_activation_bytes
    };

    // 5. KV Cache Memory (if attention layers exist)
    let kv_cache_bytes: usize = spec
        .graph
        .nodes
        .iter()
        .map(|n| {
            match &n.operator {
                crate::spec::operators::OperatorKind::MHA { num_heads, d_k, .. } => {
                    2 * batch_size * seq_len * num_heads * d_k * 2 // 2 (K+V) * elements * 2 bytes
                }
                crate::spec::operators::OperatorKind::GQA {
                    num_kv_heads, d_k, ..
                } => 2 * batch_size * seq_len * num_kv_heads * d_k * 2,
                crate::spec::operators::OperatorKind::MLA {
                    d_latent_kv,
                    d_rope,
                    ..
                } => {
                    // MLA KV cache compresses KV to (d_latent_kv + d_rope) per token!
                    batch_size * seq_len * (d_latent_kv + d_rope) * 2
                }
                _ => 0,
            }
        })
        .sum();

    let max_vram_usize = hw.max_vram_bytes as usize;
    let total_allocated_bytes = weights_bytes
        + master_weights_bytes
        + optimizer_states_bytes
        + activation_bytes
        + kv_cache_bytes;
    let headroom_bytes = max_vram_usize as i64 - total_allocated_bytes as i64;
    let is_oom_risk = headroom_bytes < 0;

    let vram = VramWaterfall {
        weights_bytes,
        master_weights_bytes,
        optimizer_states_bytes,
        activation_bytes,
        kv_cache_bytes,
        total_allocated_bytes,
        max_hardware_vram_bytes: max_vram_usize,
        headroom_bytes,
        is_oom_risk,
    };

    // 6. Operational FLOPs and Memory Intensity
    let total_flops = spec.graph.total_flops(seq_len);
    let total_flops_per_token = total_flops.checked_div(seq_len).unwrap_or(total_flops);

    // Memory bytes transferred during forward pass (weights read + inputs/outputs streaming)
    let memory_bytes_per_token = weights_bytes + (total_hidden_dim * 4);
    let operational_intensity = if memory_bytes_per_token > 0 {
        total_flops_per_token as f64 / memory_bytes_per_token as f64
    } else {
        0.0
    };

    // Hardware machine balance: Peak FLOPs / Peak Bytes/s
    // Peak TFLOPS * 1e12 / (Memory Bandwidth GB/s * 1e9) = (TFLOPS * 1000) / Bandwidth
    let peak_tflops = match default_prec {
        PrecisionFormat::FP8E4M3FN | PrecisionFormat::FP8E5M2 => hw.peak_fp8_tflops,
        _ => hw.peak_fp16_tflops,
    };
    let machine_balance = (peak_tflops * 1000.0) / hw.memory_bandwidth_gbs.max(1.0);

    let regime = if operational_intensity >= machine_balance {
        RooflineRegime::ComputeBound
    } else {
        RooflineRegime::MemoryBandwidthBound
    };

    // Latency estimate (ms)
    let compute_time_ms = (total_flops as f64 / (peak_tflops * 1e12)) * 1000.0;
    let memory_time_ms =
        (memory_bytes_per_token as f64 * seq_len as f64 / (hw.memory_bandwidth_gbs * 1e9)) * 1000.0;
    let estimated_latency_per_step_ms = compute_time_ms.max(memory_time_ms);

    // Critical horizon L*: maximum sequence length fitting in remaining VRAM
    let static_vram = weights_bytes + master_weights_bytes + optimizer_states_bytes;
    let available_dynamic_vram = max_vram_usize.saturating_sub(static_vram);
    let per_token_dynamic_bytes =
        ((total_hidden_dim * 2) + (kv_cache_bytes / seq_len.max(1))).max(1);
    let critical_horizon_tokens =
        (available_dynamic_vram / per_token_dynamic_bytes) / batch_size.max(1);

    RooflineReport {
        hardware_profile: hw.profile_name.clone(),
        total_flops_per_token,
        total_bytes_transferred_per_token: memory_bytes_per_token,
        operational_intensity,
        machine_balance,
        regime,
        peak_tflops,
        memory_bandwidth_gbs: hw.memory_bandwidth_gbs,
        estimated_latency_per_step_ms,
        critical_horizon_tokens,
        vram,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::graph::NodeSpec;
    use crate::spec::operators::OperatorKind;

    #[test]
    fn test_roofline_and_vram_calculation() {
        let mut spec = AiDesignSpec::new("test_roofline", "Roofline Test");
        spec.graph.nodes.push(NodeSpec {
            id: "linear_layer".to_string(),
            name: "Dense".to_string(),
            operator: OperatorKind::Linear {
                in_features: 1024,
                out_features: 1024,
                bias: true,
            },
            precision: None,
            metadata: Default::default(),
        });

        let report = analyze_roofline(&spec, 8, 512);
        assert!(!report.vram.is_oom_risk);
        assert!(report.vram.weights_bytes > 0);
        assert!(report.vram.optimizer_states_bytes > 0);
        assert!(report.critical_horizon_tokens > 1000);
    }
}
