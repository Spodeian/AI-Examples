//! Primitive neural, geometric, spectral, state-space, and unrolled optimization operators.

use serde::{Deserialize, Serialize};

/// Enumeration of all primitive operators available in the Designer catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(non_camel_case_types)]
pub enum OperatorKind {
    // 1. Attention & Associative Memories
    #[serde(rename = "mha")]
    MHA {
        num_heads: usize,
        d_k: usize,
        d_v: usize,
        #[serde(default)]
        dropout: f32,
    },
    #[serde(rename = "gqa")]
    GQA {
        num_q_heads: usize,
        num_kv_heads: usize,
        d_k: usize,
    },
    #[serde(rename = "mla")]
    MLA {
        num_heads: usize,
        d_latent_kv: usize,
        d_rope: usize,
        d_v: usize,
    },
    #[serde(rename = "diff_attn")]
    DiffAttn { num_heads: usize, d_k: usize },
    #[serde(rename = "linear_attention")]
    LinearAttention {
        num_heads: usize,
        d_k: usize,
        feature_map: String, // e.g. "elu+1" or "cos_sim"
    },
    #[serde(rename = "sliding_window_sink")]
    SlidingWindowSink {
        window_size: usize,
        num_sink_tokens: usize,
    },
    #[serde(rename = "evoformer_triangle")]
    EvoformerTriangle { d_pair: usize },
    #[serde(rename = "invariant_point_attention")]
    InvariantPointAttention { num_heads: usize, num_points: usize },

    // 2. State Space, Recurrent & Reservoir
    #[serde(rename = "mamba_s6")]
    MambaS6 {
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
    },
    #[serde(rename = "mamba2_ssd", alias = "mamba_2_s_s_d", alias = "mamba2")]
    Mamba2SSD {
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
    },
    #[serde(rename = "s4d", alias = "s4_d")]
    S4D { d_model: usize, d_state: usize },
    #[serde(rename = "lru")]
    LRU { d_model: usize, d_state: usize },
    #[serde(rename = "x_lstm", alias = "xlstm")]
    xLSTM {
        d_model: usize,
        num_heads: usize,
        variant: String, // "mLSTM" or "sLSTM"
    },
    #[serde(rename = "titans_memory")]
    TitansMemory { d_model: usize, memory_depth: usize },
    #[serde(rename = "echo_state_network")]
    EchoStateNetwork {
        reservoir_size: usize,
        spectral_radius: f64,
    },
    #[serde(rename = "next_gen_reservoir")]
    NextGenReservoir {
        delay_taps: usize,
        poly_degree: usize,
    },

    // 3. Optimization-as-Architecture (Deep Unfolding & Implicit)
    #[serde(rename = "lista")]
    LISTA {
        num_unrolled_steps: usize,
        lambda: f64,
    },
    #[serde(rename = "admm_net", alias = "a_d_m_m_net")]
    ADMMNet { num_stages: usize, rho: f64 },
    #[serde(rename = "crate")]
    CRATE {
        d_model: usize,
        num_heads: usize,
        lambda_step: f64,
    },
    #[serde(rename = "deq")]
    DEQ {
        d_model: usize,
        max_iters: usize,
        tolerance: f64,
    },
    #[serde(rename = "ttt")]
    TTT {
        d_model: usize,
        mini_batch_size: usize,
    },
    #[serde(rename = "modern_hopfield")]
    ModernHopfield { num_heads: usize, beta: f64 },

    // 4. Spatial, Spectral, Geometric & Physics
    #[serde(rename = "convnext_v2_block", alias = "conv_ne_xt_v2_block")]
    ConvNeXtV2Block {
        dim: usize,
        kernel_size: usize,
        mlp_ratio: f32,
    },
    #[serde(rename = "wt_conv", alias = "w_t_conv")]
    WTConv {
        in_channels: usize,
        out_channels: usize,
        wavelet_levels: usize,
    },
    #[serde(rename = "fourier_neural_operator", alias = "fno")]
    FourierNeuralOperator { modes: Vec<usize>, width: usize },
    #[serde(rename = "deeponet", alias = "deep_o_net")]
    DeepONet {
        branch_dim: usize,
        trunk_dim: usize,
        basis_dim: usize,
    },
    #[serde(rename = "egnn", alias = "e_g_n_n")]
    EGNN { node_dim: usize, edge_dim: usize },
    #[serde(rename = "cellular_sheaf_conv")]
    CellularSheafConv { node_dim: usize, stalk_dim: usize },
    #[serde(rename = "clifford_geometric_product")]
    CliffordGeometricProduct { p: usize, q: usize },
    #[serde(rename = "gaussian_splatting_3d", alias = "gaussian_splatting3_d")]
    GaussianSplatting3D { num_gaussians: usize },

    // 5. Splines, Feedforwards & Activations
    #[serde(rename = "linear")]
    Linear {
        in_features: usize,
        out_features: usize,
        #[serde(default = "default_true")]
        bias: bool,
    },
    #[serde(rename = "embedding")]
    Embedding {
        num_embeddings: usize,
        embedding_dim: usize,
    },
    #[serde(rename = "kan_layer", alias = "k_a_n_layer")]
    KANLayer {
        in_features: usize,
        out_features: usize,
        grid_size: usize,
        spline_order: usize,
    },
    #[serde(rename = "swiglu", alias = "swi_g_l_u")]
    SwiGLU {
        in_features: usize,
        hidden_features: usize,
    },
    #[serde(rename = "geglu", alias = "ge_g_l_u")]
    GeGLU {
        in_features: usize,
        hidden_features: usize,
    },
    #[serde(rename = "snake")]
    Snake { frequency: f32 },

    // 6. Normalization & Stability Guards
    #[serde(rename = "rms_norm", alias = "r_m_s_norm")]
    RMSNorm {
        dim: usize,
        #[serde(default = "default_norm_eps")]
        eps: f64,
    },
    #[serde(rename = "layer_norm")]
    LayerNorm {
        dim: usize,
        #[serde(default = "default_norm_eps")]
        eps: f64,
    },
    #[serde(rename = "qk_norm", alias = "q_k_norm")]
    QKNorm { d_head: usize },
    #[serde(rename = "logit_soft_cap")]
    LogitSoftCap { cap_val: f32 },
    #[serde(rename = "rev_in_layer", alias = "rev_i_n_layer")]
    RevINLayer { channels: usize },

    // 7. Latent Gating, Routing & Combinatorial
    #[serde(rename = "deepseek_moe", alias = "deep_seek_mo_e")]
    DeepSeekMoE {
        total_experts: usize,
        routed_k: usize,
        shared_experts: usize,
        d_model: usize,
        d_ffn: usize,
    },
    #[serde(rename = "slot_attention")]
    SlotAttention {
        num_slots: usize,
        slot_dim: usize,
        iters: usize,
    },
    #[serde(rename = "hard_concrete_l0", alias = "hard_concrete_l_0")]
    HardConcreteL0 { dim: usize },
    #[serde(rename = "sparsemax_gate")]
    SparsemaxGate,
    #[serde(
        rename = "entmax15_gate",
        alias = "entmax_15_gate",
        alias = "entmax1_5_gate"
    )]
    Entmax15Gate,
    #[serde(rename = "soft_sort_permutahedron")]
    SoftSortPermutahedron { num_items: usize },
    #[serde(rename = "mixture_of_depths")]
    MixtureOfDepths { capacity_factor: f32 },
}

fn default_true() -> bool {
    true
}
fn default_norm_eps() -> f64 {
    1e-5
}

/// Architectural overview and composition guide for an operator node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorOverview {
    pub category: &'static str,
    pub complexity_tag: &'static str,
    pub what_it_does: &'static str,
    pub typical_usage: &'static str,
    pub typically_followed_by: &'static str,
}

impl OperatorKind {
    /// Functional category label used for catalog filtering.
    pub fn category_tag(&self) -> &'static str {
        self.overview().category
    }

    /// Computational complexity / paradigm filter tag.
    pub fn complexity_tag(&self) -> &'static str {
        self.overview().complexity_tag
    }

    /// Clickable URL to the corresponding reference topic in the AI-Examples repository.
    pub fn curriculum_url(&self) -> String {
        let slug = self.curriculum_slug();
        if slug.contains('/') {
            format!("https://github.com/Spodeian/AI-Examples/blob/main/markdown/{slug}.md")
        } else {
            format!("https://github.com/Spodeian/AI-Examples/tree/main/notebooks/{slug}")
        }
    }

    /// Architectural overview describing what the operator computes, when to use it, and what typically follows it.
    pub fn overview(&self) -> OperatorOverview {
        match self {
            Self::MHA { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Quadratic / Global Mixing",
                what_it_does: "Computes scaled dot-product attention across multiple query, key, and value heads in parallel, allowing every token or spatial patch to attend globally.",
                typical_usage: "Standard Transformer blocks, multimodal cross-attention fusion, and short-to-medium sequence encoders/decoders.",
                typically_followed_by: "Residual addition -> Pre-FFN RMSNorm -> SwiGLU Gated MLP or DeepSeek Sparse MoE.",
            },
            Self::GQA { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "KV-Compressed Attention",
                what_it_does: "Shares key and value heads across groups of query heads, reducing KV-cache memory bandwidth by 4x-8x while preserving multi-head expressivity.",
                typical_usage: "High-throughput autoregressive LLMs (Llama-3, Mistral, Qwen) where decode latency is memory-bandwidth bound.",
                typically_followed_by: "Residual addition -> RMSNorm -> SwiGLU Gated MLP.",
            },
            Self::MLA { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "KV-Compressed Attention",
                what_it_does: "Projects keys and values into a low-rank latent vector c_KV with decoupled Rotary Positional Embeddings (RoPE), enabling matrix absorption at inference.",
                typical_usage: "Long-context frontier reasoning models (DeepSeek-V2/V3) requiring minimal KV-cache VRAM footprint.",
                typically_followed_by: "Residual addition -> RMSNorm -> DeepSeek Sparse MoE or SwiGLU MLP.",
            },
            Self::DiffAttn { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Quadratic / Global Mixing",
                what_it_does: "Subtracts two softmax attention maps to cancel common-mode attention noise and amplify sparse relevant context.",
                typical_usage: "Long-context retrieval, needle-in-a-haystack reasoning, and hallucination mitigation.",
                typically_followed_by: "GroupNorm / RMSNorm per head -> Output Linear projection -> SwiGLU MLP.",
            },
            Self::LinearAttention { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Replaces the exp(QK^T) kernel with associatively factorized feature maps phi(Q)(phi(K)^T V), reducing time and memory complexity from O(T^2) to O(T).",
                typical_usage: "Ultra-long sequences, streaming audio/video, and hybrid linear-full attention stacks.",
                typically_followed_by: "RMSNorm / Gate -> SwiGLU Gated MLP.",
            },
            Self::SlidingWindowSink { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Restricts attention to a local sliding window while retaining permanent initial 'sink' tokens to absorb surplus softmax probability mass.",
                typical_usage: "Infinite-length streaming language and audio models (StreamingLLM) with constant KV cache.",
                typically_followed_by: "RMSNorm -> SwiGLU MLP or periodic global MLA layer.",
            },
            Self::EvoformerTriangle { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Updates pairwise residue/atom representations z_ij via triangular multiplicative and axial attention updates enforcing 3D triangle inequality consistency.",
                typical_usage: "Biomolecular structure prediction (AlphaFold2/3) and 3D distance geometry networks.",
                typically_followed_by: "SE(3) Invariant Point Attention (IPA) -> E(3)-Equivariant GNN or FAPE output head.",
            },
            Self::InvariantPointAttention { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Computes attention in local 3D rigid-body residue frames T_i = (R_i, t_i), guaranteeing exact invariance under global SE(3) rotations and translations.",
                typical_usage: "Protein backbone refinement, molecular docking, and SE(3) robotic manipulation policies.",
                typically_followed_by: "Backbone frame update layer -> Dense Linear projection with ContinuousRotation6D constraint.",
            },
            Self::Mamba2SSD { .. } | Self::MambaS6 { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Evolves a selective latent state h_t = A_t h_{t-1} + B_t x_t using input-dependent step sizes and Structured State Space Duality (SSD) block GEMMs.",
                typical_usage: "Linear-time O(T) sequence modeling across language, raw audio waveforms, genomics, and long-horizon control.",
                typically_followed_by: "RMSNorm -> SwiGLU MLP, or interleaved 7:1 with Multi-Head Latent Attention (Jamba hybrid).",
            },
            Self::S4D { .. } | Self::LRU { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Applies a diagonalized continuous-time linear state-space recurrence initialized via HiPPO/exponential eigenvalues for long-range memory.",
                typical_usage: "Continuous-time signal processing, physiological time series, and long-range arena tasks.",
                typically_followed_by: "Pointwise GLU / SwiGLU channel mixing -> LayerNorm / RMSNorm.",
            },
            Self::xLSTM { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Combines exponential input/forget gating with a matrix-valued covariance memory cell (mLSTM) or scalar recurrent state mixing (sLSTM).",
                typical_usage: "Probabilistic time-series forecasting, financial econometrics, and fast O(1)-state sequence inference.",
                typically_followed_by: "RMSNorm -> SwiGLU or Linear projection head (often paired with RevIN).",
            },
            Self::TitansMemory { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Updates a deep neural long-term memory module at test time using surprise-driven (gradient norm) momentum updates.",
                typical_usage: "Million-token context retention and continual test-time adaptation.",
                typically_followed_by: "Sliding-window or Multi-Head Attention branch -> SwiGLU MLP.",
            },
            Self::EchoStateNetwork { .. } | Self::NextGenReservoir { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Projects time series into a fixed high-dimensional dynamical reservoir or nonlinear time-delay polynomial feature space.",
                typical_usage: "Ultra-low-latency chaotic dynamical system forecasting and edge DSP.",
                typically_followed_by: "Ridge-regressed Dense Linear readout layer.",
            },
            Self::LISTA { .. } | Self::ADMMNet { .. } | Self::CRATE { .. } | Self::DEQ { .. } | Self::TTT { .. } | Self::ModernHopfield { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Optimization-as-Architecture",
                what_it_does: "Embeds an unrolled proximal optimization solver (ISTA/ADMM/Rate-Reduction) or implicit fixed-point equilibrium directly as a differentiable neural layer.",
                typical_usage: "Compressed sensing, inverse problems, white-box interpretable transformers, and adaptive test-time compute.",
                typically_followed_by: "RMSNorm -> Linear readout or OptNet / CBF-QP constrained output head.",
            },
            Self::ConvNeXtV2Block { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Applies 7x7 depthwise spatial convolution followed by inverted bottleneck pointwise MLP with Global Response Normalization (GRN).",
                typical_usage: "High-efficiency 2D/3D vision backbones, dense segmentation, and hybrid vision-robotics encoders.",
                typically_followed_by: "Downsampling / Patchify -> Slot Attention or Multi-Head Attention.",
            },
            Self::WTConv { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Performs cascaded discrete wavelet decomposition, applying small kernels across multi-frequency wavelet subbands to achieve an O(2^L) receptive field.",
                typical_usage: "Multi-scale vision, medical imaging, seismic/audio DSP, and DiT patch stems.",
                typically_followed_by: "RMSNorm -> Multi-Head Attention or ConvNeXtV2Block.",
            },
            Self::FourierNeuralOperator { .. } | Self::DeepONet { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Learns resolution-invariant operators in function space by multiplying truncated Fourier modes in spectral domain and summing with a local residual bypass.",
                typical_usage: "Zero-shot super-resolution PDE surrogates (Navier-Stokes, weather, MHD plasma, elasticity).",
                typically_followed_by: "Additional FNO spectral blocks -> Linear projection head with CurlFreeVectorField or Sobolev H1 loss.",
            },
            Self::EGNN { .. } | Self::CellularSheafConv { .. } | Self::CliffordGeometricProduct { .. } | Self::GaussianSplatting3D { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Propagates messages across graph edges, cellular sheaves, or 3D coordinates while preserving E(3)/rotational equivariance.",
                typical_usage: "Molecular conformation generation, 3D point clouds, physical N-body simulations, and heterophilic graphs.",
                typically_followed_by: "Equivariant coordinate readout or invariant pooling -> Linear output head.",
            },
            Self::Linear { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Applies an affine matrix transformation y = x W^T + b across the feature dimension (optimized via Muon for internal 2D weights or AdamW for terminal heads).",
                typical_usage: "Input stem projection, attention output projection, classification/regression readout heads, and MLP layers.",
                typically_followed_by: "RMSNorm / Activation (if stem/hidden) or Output Port Loss / Structural Constraint (if terminal head).",
            },
            Self::Embedding { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Maps discrete token indices [B, T] into continuous dense vectors [B, T, D] via a learnable lookup table (partitioned to AdamW).",
                typical_usage: "Input stem for discrete text, code, genomic sequences, or quantized audio/visual tokens.",
                typically_followed_by: "RMSNorm -> Multi-Head Attention (MHA/GQA/MLA) or Mamba-2 SSD backbone.",
            },
            Self::KANLayer { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Replaces fixed node activations with learnable univariate spline functions (B-splines / Chebyshev / RBF) on every edge.",
                typical_usage: "Symbolic law discovery, scientific regression, and compact high-accuracy tabular/physics models.",
                typically_followed_by: "LayerNorm / RMSNorm -> next KANLayer or Linear readout head.",
            },
            Self::SwiGLU { .. } | Self::GeGLU { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Computes a multiplicative bilinear gated feedforward block: W_down(SiLU(W_gate x) * (W_up x)), expanding hidden dimension by ~8/3x.",
                typical_usage: "Primary channel-mixing FFN block following Attention or SSM layers in modern Transformers and Mamba hybrids.",
                typically_followed_by: "Residual stream merge -> next block's RMSNorm or final output Linear projection.",
            },
            Self::Snake { .. } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Shape-Preserving",
                what_it_does: "Applies the periodic activation x + (1/a) sin^2(a x), providing strong inductive bias for periodic and harmonic signals.",
                typical_usage: "Neural audio codecs (Descript/SoundStream), waveform synthesis, and periodic time series.",
                typically_followed_by: "Causal Mamba-2 SSD or 1D Conv / Linear projection.",
            },
            Self::RMSNorm { .. } | Self::LayerNorm { .. } | Self::QKNorm { .. } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Shape-Preserving",
                what_it_does: "Normalizes activation root-mean-square or variance (computed in FP32 reduction precision) to prevent gradient explosion and attention logit growth.",
                typical_usage: "Pre-Norm guard before every Attention, SSM, or FFN sub-block, and final normalization before the output head.",
                typically_followed_by: "Multi-Head Attention (MHA/GQA/MLA), Mamba-2 SSD, SwiGLU MLP, or final Linear readout.",
            },
            Self::LogitSoftCap { .. } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Shape-Preserving",
                what_it_does: "Smoothly bounds logits via C * tanh(x / C) to prevent attention entropy collapse and FP8/BF16 overflow.",
                typical_usage: "Placed immediately before final output CrossEntropy loss or inside attention score computation (Gemma-2 style).",
                typically_followed_by: "Output Port (CrossEntropy / GRPO) or Softmax/Entmax gate.",
            },
            Self::RevINLayer { .. } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Shape-Preserving",
                what_it_does: "Strips non-stationary instance mean and variance per time-series channel at input and restores exact scale/offset at output.",
                typical_usage: "Non-stationary financial, econometric, and sensor time-series forecasting.",
                typically_followed_by: "xLSTM / Mamba-2 SSD backbone -> symmetric RevIN denormalization.",
            },
            Self::DeepSeekMoE { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Routes each token to top-k fine-grained routed SwiGLU experts while always executing shared experts to isolate common knowledge.",
                typical_usage: "High-capacity frontier models where total parameters scale without increasing per-token FLOPs.",
                typically_followed_by: "Residual addition -> next block's RMSNorm + MLA/GQA or final output head.",
            },
            Self::SlotAttention { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Iteratively binds spatial/temporal features into K object-centric latent slots via competitive softmax normalization over the query/slot axis.",
                typical_usage: "Object-centric scene decomposition, multimodal fusion, and robotic perception.",
                typically_followed_by: "Cross-Attention policy decoder or per-slot MLP readout.",
            },
            Self::HardConcreteL0 { .. } | Self::SparsemaxGate | Self::Entmax15Gate | Self::SoftSortPermutahedron { .. } | Self::MixtureOfDepths { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Produces exact-zero sparse probabilities or top-k token routing masks with differentiable gradients, enabling true conditional compute skipping.",
                typical_usage: "Feature selection in tabular ML, dynamic layer skipping (MoD), and sparse attention/routing gates.",
                typically_followed_by: "Gated backbone block (MHA / SwiGLU) or Sparse Fenchel-Young output head.",
            },
        }
    }

    /// Returns 2-3 recommended operators that commonly follow `self` for one-click chaining in the Inspector.
    pub fn recommended_next_layers(&self) -> Vec<(&'static str, OperatorKind)> {
        match self {
            Self::RMSNorm { dim, .. } | Self::LayerNorm { dim, .. } => {
                let d = *dim;
                let heads = (d / 64).max(1);
                vec![
                    (
                        "Multi-Head Attention",
                        OperatorKind::MHA {
                            num_heads: heads,
                            d_k: 64,
                            d_v: 64,
                            dropout: 0.0,
                        },
                    ),
                    (
                        "Mamba-2 SSD",
                        OperatorKind::Mamba2SSD {
                            d_model: d,
                            d_state: 64,
                            d_conv: 4,
                            expand: 2,
                        },
                    ),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                ]
            }
            Self::MHA { num_heads, d_v, .. }
            | Self::MLA { num_heads, d_v, .. } => {
                let d = num_heads * d_v;
                vec![
                    ("RMSNorm Layer", OperatorKind::RMSNorm { dim: d, eps: 1e-5 }),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                    (
                        "DeepSeek Sparse MoE",
                        OperatorKind::DeepSeekMoE {
                            total_experts: 16,
                            routed_k: 4,
                            shared_experts: 2,
                            d_model: d,
                            d_ffn: d * 2,
                        },
                    ),
                ]
            }
            Self::Mamba2SSD { d_model, .. }
            | Self::MambaS6 { d_model, .. }
            | Self::xLSTM { d_model, .. } => {
                let d = *d_model;
                vec![
                    ("RMSNorm Layer", OperatorKind::RMSNorm { dim: d, eps: 1e-5 }),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                    (
                        "Dense Linear Head",
                        OperatorKind::Linear {
                            in_features: d,
                            out_features: d,
                            bias: false,
                        },
                    ),
                ]
            }
            Self::SwiGLU { in_features, .. } | Self::GeGLU { in_features, .. } => {
                let d = *in_features;
                vec![
                    ("RMSNorm Layer", OperatorKind::RMSNorm { dim: d, eps: 1e-5 }),
                    (
                        "Dense Linear Head",
                        OperatorKind::Linear {
                            in_features: d,
                            out_features: d,
                            bias: false,
                        },
                    ),
                ]
            }
            Self::Linear { out_features, .. } => {
                let d = *out_features;
                vec![
                    ("RMSNorm Layer", OperatorKind::RMSNorm { dim: d, eps: 1e-5 }),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                ]
            }
            _ => vec![
                ("RMSNorm Layer", OperatorKind::RMSNorm { dim: 512, eps: 1e-5 }),
                (
                    "SwiGLU Gated MLP",
                    OperatorKind::SwiGLU {
                        in_features: 512,
                        hidden_features: 1376,
                    },
                ),
            ],
        }
    }

    /// Nominal operational FLOPs per token / element.
    pub fn compute_flops(&self, seq_len: usize) -> usize {
        match self {
            Self::Linear {
                in_features,
                out_features,
                ..
            } => 2 * in_features * out_features,
            Self::MHA {
                num_heads,
                d_k,
                d_v,
                ..
            } => {
                let d_model = num_heads * d_k;
                // QKV projection + Attn dot products (2 * T^2 * D) + Output proj
                (2 * d_model * (2 * d_k * num_heads + d_v * num_heads))
                    + (2 * seq_len * d_model)
                    + (2 * d_model * d_model)
            }
            Self::MLA {
                num_heads,
                d_latent_kv,
                d_rope,
                d_v,
            } => {
                let d_model = num_heads * (*d_rope + *d_v);
                (2 * d_model * d_latent_kv) + (2 * seq_len * d_latent_kv)
            }
            Self::Mamba2SSD {
                d_model,
                d_state,
                expand,
                ..
            } => {
                let d_inner = d_model * expand;
                // 1-semiseparable block GEMMs: linear in seq_len
                2 * d_model * d_inner + 2 * d_inner * d_state
            }
            Self::ConvNeXtV2Block { dim, mlp_ratio, .. } => {
                let hidden = (*dim as f32 * mlp_ratio) as usize;
                2 * dim * dim + 2 * dim * hidden + 2 * hidden * dim
            }
            Self::RMSNorm { dim, .. } | Self::LayerNorm { dim, .. } => 2 * dim,
            _ => 1000, // fallback baseline
        }
    }

    /// Primary reference curriculum topic slug in AI-Examples.
    pub fn curriculum_slug(&self) -> &'static str {
        match self {
            Self::MHA { .. } => {
                "05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn"
            }
            Self::GQA { .. } | Self::MLA { .. } | Self::DiffAttn { .. } => {
                "05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn"
            }
            Self::MambaS6 { .. } | Self::Mamba2SSD { .. } => {
                "05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd"
            }
            Self::S4D { .. } | Self::LRU { .. } => {
                "05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru"
            }
            Self::xLSTM { .. } => {
                "05_architectures/03_recurrent_ssm_and_memory/05_xlstm_extended_long_short_term_memory_mlstm_and_slstm"
            }
            Self::LISTA { .. } | Self::ADMMNet { .. } | Self::CRATE { .. } => {
                "06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers"
            }
            Self::DEQ { .. } => {
                "06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation"
            }
            Self::FourierNeuralOperator { .. } => {
                "06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets"
            }
            Self::EGNN { .. } => {
                "05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn"
            }
            Self::KANLayer { .. } => {
                "05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan_bspline_chebyshev_rbf_and_wavkan"
            }
            Self::DeepSeekMoE { .. } => {
                "05_architectures/01_mlp_moe_soups_and_kan/03_deepseekmoe_fine_grained_experts_and_shared_isolation"
            }
            Self::SlotAttention { .. } => {
                "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_representations_slot_attention_and_global_workspace"
            }
            Self::WTConv { .. } => {
                "05_architectures/02_convolutional_and_spatial/04_wavelet_convolutions_wtconv_and_multiscale_wavelet_attention"
            }
            _ => "05_architectures",
        }
    }
}
