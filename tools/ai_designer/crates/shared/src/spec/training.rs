//! Training specification, optimizer parameter partitioning, schedules, and gradient hygiene.

use serde::{Deserialize, Serialize};

/// Optimizer configuration for internal 2D weight matrices (GEMMs) via Newton-Schulz matrix orthogonalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuonConfig {
    #[serde(default = "default_muon_lr")]
    pub lr: f64,
    #[serde(default = "default_muon_momentum")]
    pub momentum: f64,
    #[serde(default = "default_muon_steps")]
    pub newton_schulz_steps: usize,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: default_muon_lr(),
            momentum: default_muon_momentum(),
            newton_schulz_steps: default_muon_steps(),
            weight_decay: default_weight_decay(),
        }
    }
}

fn default_muon_lr() -> f64 {
    0.02
}
fn default_muon_momentum() -> f64 {
    0.95
}
fn default_muon_steps() -> usize {
    5
}
fn default_weight_decay() -> f64 {
    0.01
}

/// Optimizer configuration for token embeddings, 1D vectors, and final output heads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamWConfig {
    #[serde(default = "default_adamw_lr")]
    pub lr: f64,
    #[serde(default = "default_adamw_betas")]
    pub betas: (f64, f64),
    #[serde(default = "default_adamw_eps")]
    pub eps: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: default_adamw_lr(),
            betas: default_adamw_betas(),
            eps: default_adamw_eps(),
            weight_decay: default_weight_decay(),
        }
    }
}

fn default_adamw_lr() -> f64 {
    0.001
}
fn default_adamw_betas() -> (f64, f64) {
    (0.9, 0.95)
}
fn default_adamw_eps() -> f64 {
    1e-8
}

/// Learning rate schedule configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SchedulerSpec {
    WarmupStableDecay {
        #[serde(default = "default_wsd_warmup")]
        warmup_ratio: f64,
        #[serde(default = "default_wsd_stable")]
        stable_ratio: f64,
        #[serde(default = "default_wsd_decay")]
        decay_ratio: f64,
        #[serde(default = "default_min_lr_ratio")]
        min_lr_ratio: f64,
    },
    #[serde(rename = "cosine_with_warmup", alias = "cosine")]
    CosineWithWarmup {
        #[serde(default = "default_wsd_warmup")]
        warmup_ratio: f64,
        #[serde(default = "default_min_lr_ratio")]
        min_lr_ratio: f64,
        #[serde(default)]
        warmup_steps: Option<usize>,
        #[serde(default)]
        total_steps: Option<usize>,
        #[serde(default)]
        min_lr: Option<f64>,
    },
    Constant {
        lr: f64,
    },
}

impl Default for SchedulerSpec {
    fn default() -> Self {
        Self::WarmupStableDecay {
            warmup_ratio: default_wsd_warmup(),
            stable_ratio: default_wsd_stable(),
            decay_ratio: default_wsd_decay(),
            min_lr_ratio: default_min_lr_ratio(),
        }
    }
}

fn default_wsd_warmup() -> f64 {
    0.05
}
fn default_wsd_stable() -> f64 {
    0.75
}
fn default_wsd_decay() -> f64 {
    0.20
}
fn default_min_lr_ratio() -> f64 {
    0.01
}

/// Gradient clipping, weight averaging, and numerical stabilization rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientHygieneSpec {
    /// Adaptive percentile-based gradient clipping (AutoClip).
    #[serde(default = "default_true")]
    pub auto_clip: bool,
    #[serde(default = "default_clip_percentile")]
    pub clip_percentile: f64,
    #[serde(default = "default_clip_max_norm")]
    pub max_norm: f64,
    /// Exponential Moving Average (EMA) weight decay factor.
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f64,
    /// Grokfast gradient filtering accelerating algorithmic generalization.
    #[serde(default)]
    pub grokfast: bool,
    /// Activation checkpointing to trade recompute for VRAM.
    #[serde(default)]
    pub activation_checkpointing: bool,
}

impl Default for GradientHygieneSpec {
    fn default() -> Self {
        Self {
            auto_clip: true,
            clip_percentile: default_clip_percentile(),
            max_norm: default_clip_max_norm(),
            ema_decay: default_ema_decay(),
            grokfast: false,
            activation_checkpointing: false,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_clip_percentile() -> f64 {
    90.0
}
fn default_clip_max_norm() -> f64 {
    1.0
}
fn default_ema_decay() -> f64 {
    0.999
}

/// Multi-task and multi-objective loss balancing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LossBalancingStrategy {
    #[default]
    FixedWeights,
    /// Projected Conflicting Gradients (PCGrad) projecting gradients with negative inner product.
    #[serde(rename = "pcgrad", alias = "p_c_grad")]
    PCGrad,
    /// Neural Tangent Kernel (NTK) eigenvalue balancing for physics-informed neural networks.
    #[serde(rename = "ntk_balancing", alias = "n_t_k_balancing")]
    NTKBalancing,
    /// Dynamic gradient norm balancing.
    GradNorm,
}

/// Complete training regime specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingSpec {
    #[serde(default = "default_true")]
    pub auto_partition_parameters: bool,
    #[serde(default)]
    pub muon: MuonConfig,
    #[serde(default)]
    pub adamw: AdamWConfig,
    #[serde(default)]
    pub scheduler: SchedulerSpec,
    #[serde(default)]
    pub hygiene: GradientHygieneSpec,
    #[serde(default)]
    pub loss_balancing: LossBalancingStrategy,
}

impl Default for TrainingSpec {
    fn default() -> Self {
        Self {
            auto_partition_parameters: true,
            muon: MuonConfig::default(),
            adamw: AdamWConfig::default(),
            scheduler: SchedulerSpec::default(),
            hygiene: GradientHygieneSpec::default(),
            loss_balancing: LossBalancingStrategy::default(),
        }
    }
}
