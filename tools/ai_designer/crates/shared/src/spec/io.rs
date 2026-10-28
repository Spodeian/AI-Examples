//! Input and Output Port specifications, typed tensor spaces, coordinate encodings, normalizers, and structural constraints.

use serde::{Deserialize, Serialize};

/// Symbolic or concrete dimension for dynamic tensor shape reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SymbolicDim {
    Known(usize),
    Symbol(String),
}

impl SymbolicDim {
    pub fn is_batch(&self) -> bool {
        match self {
            Self::Symbol(s) => s == "B" || s == "batch" || s == "batch_size",
            _ => false,
        }
    }

    pub fn is_sequence(&self) -> bool {
        match self {
            Self::Symbol(s) => s == "T" || s == "seq" || s == "seq_len" || s == "L",
            _ => false,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Known(k) => k.to_string(),
            Self::Symbol(s) => s.clone(),
        }
    }
}

impl From<usize> for SymbolicDim {
    fn from(v: usize) -> Self {
        Self::Known(v)
    }
}

impl From<&str> for SymbolicDim {
    fn from(s: &str) -> Self {
        Self::Symbol(s.to_string())
    }
}

/// Typed representation space defining domain inductive biases and statistical distributions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensorSpace {
    ContinuousVector {
        dimension: usize,
        #[serde(default)]
        range: Option<(f64, f64)>,
    },
    #[serde(rename = "image_rgb", alias = "image_r_g_b")]
    ImageRGB {
        resolution: [usize; 2],
        #[serde(default = "default_rgb_channels")]
        channels: usize,
    },
    #[serde(rename = "video_3d", alias = "video3_d")]
    Video3D {
        frames: usize,
        resolution: [usize; 2],
        #[serde(default = "default_rgb_channels")]
        channels: usize,
    },
    TimeSeries {
        channels: usize,
        horizon: usize,
        #[serde(default)]
        stationary: bool,
    },
    GraphRelational {
        node_features: usize,
        edge_features: usize,
        #[serde(default)]
        heterophilic: bool,
    },
    DiscreteTokens {
        vocab_size: usize,
        #[serde(default)]
        is_causal: bool,
    },
    CoordinateField {
        domain_dim: usize, // e.g. 2 for (x, y), 3 for (x, y, z), 4 for (x, y, z, t)
        #[serde(default)]
        time_dependent: bool,
    },
    RigidBodyTrajectory {
        horizon_steps: usize,
        dof_per_step: usize, // e.g. 7 for translation(3) + rotation(3) + gripper(1)
    },
    TabularFeatures {
        num_continuous: usize,
        num_categorical: usize,
        #[serde(default)]
        skewness: Option<f64>,
    },
}

fn default_rgb_channels() -> usize {
    3
}

/// Preprocessing transformations applied to raw input streams.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormalizerSpec {
    #[default]
    None,
    Standardize {
        #[serde(default)]
        mean_fill: f64,
    },
    MinMax {
        min: f64,
        max: f64,
    },
    /// Reversible Instance Normalization (RevIN) strips instance mean and variance before the backbone,
    /// and restores the exact trend statistics post-backbone.
    #[serde(rename = "rev_in")]
    RevIN {
        #[serde(default = "default_revin_eps")]
        eps: f64,
        #[serde(default = "default_true")]
        affine: bool,
    },
    /// Symmetric logarithm scaling: sign(x) * ln(|x| + 1) for wide-dynamic-range and heavy-tailed signals.
    Symlog,
    /// Piecewise Linear Embeddings (PLE) for continuous tabular features, preventing MLP low-frequency bias.
    #[serde(rename = "ple")]
    PLE {
        #[serde(default = "default_ple_bins")]
        num_bins: usize,
    },
    #[serde(rename = "imagenet_normalize", alias = "image_net_normalize")]
    ImageNetNormalize,
}

fn default_revin_eps() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}
fn default_ple_bins() -> usize {
    16
}

/// Coordinate, positional, and geometric encodings injected into input representations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(non_camel_case_types)]
pub enum CoordinateEncoding {
    #[default]
    None,
    #[serde(rename = "rope_1d", alias = "ro_pe1_d")]
    RoPE1D {
        #[serde(default = "default_rope_base")]
        base: f64,
    },
    #[serde(rename = "rope_2d_dynamic", alias = "ro_pe2_d_dynamic")]
    RoPE2D_Dynamic {
        #[serde(default = "default_rope_base")]
        base: f64,
    },
    #[serde(rename = "rope_3d", alias = "ro_pe3_d")]
    RoPE3D {
        #[serde(default = "default_rope_base")]
        base: f64,
    },
    #[serde(rename = "decoupled_rope", alias = "decoupled_ro_pe")]
    DecoupledRoPE,
    #[serde(rename = "alibi")]
    ALiBi,
    /// 6D Plücker coordinates r = (d, o x d) for camera rays and rigid body screw axes.
    PluckerRays,
    /// Random Walk Positional Encodings (RWPE) for graphs.
    #[serde(rename = "rwpe_graph", alias = "r_w_p_e_graph")]
    RWPE_Graph { walk_steps: usize },
    /// Random Fourier Features mapping continuous coordinates (x, y, z, t) to multiscale sinusoids.
    RandomFourierFeatures {
        num_features: usize,
        #[serde(default = "default_fourier_scale")]
        scale: f64,
    },
}

fn default_rope_base() -> f64 {
    10000.0
}
fn default_fourier_scale() -> f64 {
    1.0
}

/// Configuration for spatial patching (e.g. Vision Transformer / Audio Spectrogram patches).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchingSpec {
    pub patch_size: [usize; 2],
    #[serde(default = "default_true")]
    pub flatten_channels: bool,
}

/// Declaration of a single named input stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputPortSpec {
    pub id: String,
    pub name: String,
    pub space: TensorSpace,
    pub symbolic_shape: Vec<SymbolicDim>,
    #[serde(default)]
    pub normalizer: NormalizerSpec,
    #[serde(default)]
    pub coordinate_encoding: CoordinateEncoding,
    #[serde(default)]
    pub patching: Option<PatchingSpec>,
}

/// Structural mathematical constraints enforced by output heads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(non_camel_case_types)]
pub enum StructuralConstraint {
    #[default]
    DirectLinear,
    /// Enforces continuous 6D Gram-Schmidt orthogonalization into SO(3), eliminating gimbal lock and quaternion double-cover.
    #[serde(rename = "continuous_rotation_6d", alias = "continuous_rotation6_d")]
    ContinuousRotation6D,
    /// Predicts a vector potential A and outputs analytic curl v = curl(A), guaranteeing divergence-free fluid velocity.
    CurlFreeVectorField,
    /// Scalar Hamiltonian H(q, p) predicting symplectic energy-conserving phase-space flows.
    SymplecticHamiltonianField,
    /// Fenchel-Young projection onto the simplex allowing exact 0.0 sparse probabilities.
    Sparsemax,
    #[serde(rename = "entmax_15", alias = "entmax15")]
    Entmax15,
    /// Enforces partial or full monotonicity via non-negative weight parameterization.
    Monotonic { positive_weights: bool },
    /// Input-Convex Neural Network (ICNN) guaranteeing convex optimization surfaces.
    #[serde(rename = "input_convex_icnn")]
    InputConvexICNN,
    /// Differentiable KKT quadratic program solving hard convex constraints at inference.
    #[serde(rename = "optnet_kkt")]
    OptNetKKT,
    /// Differentiable Control Barrier Function Quadratic Program ensuring forward-invariant safe trajectories.
    #[serde(rename = "cbf_qp")]
    CBF_QP {
        safe_distance: f64,
        max_velocity: f64,
    },
    /// Evidential Dirichlet distribution modeling epistemic and aleatoric classification uncertainty.
    EvidentialDirichlet { num_classes: usize },
    /// Continuous Ranked Probability Score (CRPS) with Generalized Pareto Distribution (GPD) tail risk quantiles.
    #[serde(rename = "quantile_crps_gpd")]
    QuantileCRPS_GPD { quantiles: Vec<f64> },
}

/// Canonical loss function paired with output heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LossPairing {
    #[default]
    #[serde(rename = "mse")]
    MSE,
    CrossEntropy,
    #[serde(rename = "smooth_l1")]
    SmoothL1,
    Focal,
    FenchelYoung,
    #[serde(
        rename = "ot_cfm",
        alias = "optimal_transport_conditional_flow_matching"
    )]
    OptimalTransportConditionalFlowMatching,
    ScoreMatchingSDE,
    SobolevH1PhysicsResidual,
    SymplecticEnergyConservation,
    #[serde(rename = "fape", alias = "frame_aligned_point_error_fape")]
    FrameAlignedPointErrorFAPE,
    #[serde(rename = "quantile_crps")]
    QuantileCRPS,
    #[serde(rename = "vicreg", alias = "vic_reg_anti_collapse")]
    VICRegAntiCollapse,
    #[serde(
        rename = "gflownet_trajectory_balance",
        alias = "g_flow_net_trajectory_balance"
    )]
    GFlowNetTrajectoryBalance,
    #[serde(rename = "grpo_outcome_reward", alias = "g_r_p_o_outcome_reward")]
    GRPOOutcomeReward,
}

/// Declaration of a single named output stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputPortSpec {
    pub id: String,
    pub name: String,
    pub space: TensorSpace,
    pub symbolic_shape: Vec<SymbolicDim>,
    #[serde(default)]
    pub constraint: StructuralConstraint,
    #[serde(default)]
    pub loss: LossPairing,
}
