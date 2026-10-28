//! Morphological graph, wiring patterns, residual topologies, and node specifications.

use serde::{Deserialize, Serialize};

use super::operators::OperatorKind;
use super::precision::PrecisionFormat;

/// Normalization layer placement relative to attention and FFN blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormPlacement {
    #[default]
    #[serde(rename = "pre_ln", alias = "pre_l_n")]
    PreLN,
    #[serde(rename = "post_ln", alias = "post_l_n")]
    PostLN,
    #[serde(rename = "sandwich_ln", alias = "sandwich_l_n")]
    SandwichLN,
    #[serde(rename = "peri_ln", alias = "peri_l_n")]
    PeriLN,
}

/// Adaptive halting gate mechanism for recurrent depth deliberation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HaltingGate {
    #[default]
    None,
    /// PonderNet geometric KL halting hazard regularization.
    PonderNetKL { lambda_prior: f64 },
    /// Adaptive Computation Time (ACT) cumulative threshold halting.
    ACTThreshold { threshold: f64 },
}

/// Macro wiring topology across layers and residual streams.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WiringPattern {
    SerialStack,
    ResidualStream {
        #[serde(default)]
        norm_placement: NormPlacement,
        #[serde(default)]
        layer_scale_init: Option<f64>,
    },
    /// Reversible Split-Stream bijection y1 = x1 + F(x2), y2 = x2 + G(y1) achieving O(1) activation memory.
    ReversibleSplitStream,
    /// Multi-Lane Hyper-Connections with learnable depth/width mixing matrices.
    HyperConnections {
        num_lanes: usize,
    },
    /// Input-anchored recurrent depth deliberation s_{k+1} = T([s_k; e_x]) with dynamic halting.
    RecurrentDepthLoop {
        max_steps: usize,
        #[serde(default = "default_true")]
        input_anchored: bool,
        #[serde(default)]
        halting: HaltingGate,
    },
    /// Conditional sparse Mixture-of-Experts with routed and shared experts.
    ConditionalMoE {
        total_experts: usize,
        routed_k: usize,
        shared_experts: usize,
    },
}

impl Default for WiringPattern {
    fn default() -> Self {
        Self::ResidualStream {
            norm_placement: NormPlacement::PreLN,
            layer_scale_init: Some(1e-2),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Individual computation node within the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSpec {
    pub id: String,
    pub name: String,
    pub operator: OperatorKind,
    #[serde(default)]
    pub precision: Option<PrecisionFormat>,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

/// Directed connection channel between two node ports.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConnectionSpec {
    pub from_node: String,
    pub from_port: usize,
    pub to_node: String,
    pub to_port: usize,
}

/// Hierarchical computation graph combining topology, nodes, and interconnections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ComputeGraph {
    pub topology: WiringPattern,
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub connections: Vec<ConnectionSpec>,
}

impl ComputeGraph {
    pub fn find_node(&self, id: &str) -> Option<&NodeSpec> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn find_node_mut(&mut self, id: &str) -> Option<&mut NodeSpec> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn total_flops(&self, seq_len: usize) -> usize {
        self.nodes
            .iter()
            .map(|n| n.operator.compute_flops(seq_len))
            .sum()
    }
}
