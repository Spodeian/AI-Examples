//! Symbolic dimension propagation, tensor space tracking, and shape consistency checking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::spec::AiDesignSpec;
use crate::spec::io::SymbolicDim;
use crate::spec::operators::OperatorKind;

/// Represents an inferred or explicitly defined multi-dimensional tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InferredShape {
    pub dims: Vec<SymbolicDim>,
}

impl InferredShape {
    pub fn new(dims: Vec<SymbolicDim>) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn last_dim(&self) -> Option<&SymbolicDim> {
        self.dims.last()
    }

    pub fn is_concrete(&self) -> bool {
        self.dims.iter().all(|d| matches!(d, SymbolicDim::Known(_)))
    }

    pub fn display(&self) -> String {
        let parts: Vec<String> = self.dims.iter().map(|d| d.label()).collect();
        format!("[{}]", parts.join(", "))
    }
}

/// Shape inference results for an individual graph node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeShapeInference {
    pub node_id: String,
    pub input_shape: Option<InferredShape>,
    pub output_shape: Option<InferredShape>,
    pub is_valid: bool,
    pub error: Option<String>,
}

/// Severity level for shape analysis diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// A shape or dimension diagnostic report item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeDiagnostic {
    pub severity: DiagnosticSeverity,
    pub node_id: Option<String>,
    pub port_id: Option<String>,
    pub message: String,
    pub fix_suggestion: Option<String>,
}

/// Complete shape analysis report for an entire blueprint specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeReport {
    pub node_shapes: HashMap<String, NodeShapeInference>,
    pub diagnostics: Vec<ShapeDiagnostic>,
    pub input_shapes: HashMap<String, InferredShape>,
    pub output_shapes: HashMap<String, InferredShape>,
    pub is_valid: bool,
}

impl ShapeReport {
    pub fn new() -> Self {
        Self {
            node_shapes: HashMap::new(),
            diagnostics: Vec::new(),
            input_shapes: HashMap::new(),
            output_shapes: HashMap::new(),
            is_valid: true,
        }
    }
}

impl Default for ShapeReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzes an AiDesignSpec and computes full symbolic shape propagation through the network graph.
pub fn infer_shapes(spec: &AiDesignSpec) -> ShapeReport {
    let mut report = ShapeReport::new();

    // 1. Record input port shapes
    for input in &spec.inputs {
        let shape = InferredShape::new(input.symbolic_shape.clone());
        report.input_shapes.insert(input.id.clone(), shape);
    }

    // 2. Record target output port shapes
    for output in &spec.outputs {
        let shape = InferredShape::new(output.symbolic_shape.clone());
        report.output_shapes.insert(output.id.clone(), shape);
    }

    if spec.graph.nodes.is_empty() {
        report.diagnostics.push(ShapeDiagnostic {
            severity: DiagnosticSeverity::Warning,
            node_id: None,
            port_id: None,
            message: "Computation graph contains no nodes".to_string(),
            fix_suggestion: Some("Add at least one layer to the model architecture".to_string()),
        });
        report.is_valid = false;
        return report;
    }

    // 3. Sequential or directed propagation
    let mut current_shape = spec
        .inputs
        .first()
        .map(|i| InferredShape::new(i.symbolic_shape.clone()));

    for node in &spec.graph.nodes {
        let input_shape = current_shape.clone();
        let (output_shape, err) = match &input_shape {
            Some(in_shape) => infer_operator_output(&node.operator, in_shape),
            None => (
                None,
                Some("Node has no inbound tensor shape feeding into it".to_string()),
            ),
        };

        let is_node_valid = err.is_none();
        if let Some(ref e) = err {
            report.diagnostics.push(ShapeDiagnostic {
                severity: DiagnosticSeverity::Error,
                node_id: Some(node.id.clone()),
                port_id: None,
                message: format!("Shape mismatch at node '{}': {}", node.id, e),
                fix_suggestion: Some(
                    "Verify input features or channel projection dimensions".to_string(),
                ),
            });
            report.is_valid = false;
        }

        report.node_shapes.insert(
            node.id.clone(),
            NodeShapeInference {
                node_id: node.id.clone(),
                input_shape: input_shape.clone(),
                output_shape: output_shape.clone(),
                is_valid: is_node_valid,
                error: err,
            },
        );

        if output_shape.is_some() {
            current_shape = output_shape;
        }
    }

    // 4. Verify terminal shape matches desired output port
    if let (Some(last_shape), Some(target_output)) = (current_shape, spec.outputs.first()) {
        let target_shape = InferredShape::new(target_output.symbolic_shape.clone());
        if last_shape.rank() != target_shape.rank() {
            report.diagnostics.push(ShapeDiagnostic {
                severity: DiagnosticSeverity::Warning,
                node_id: spec.graph.nodes.last().map(|n| n.id.clone()),
                port_id: Some(target_output.id.clone()),
                message: format!(
                    "Final layer output rank {} ({}) does not match target output rank {} ({})",
                    last_shape.rank(),
                    last_shape.display(),
                    target_shape.rank(),
                    target_shape.display()
                ),
                fix_suggestion: Some(
                    "Add a projection head or pooling layer before output".to_string(),
                ),
            });
        }
    }

    report
}

/// Computes the output shape for a single operator given its inbound input shape.
fn infer_operator_output(
    op: &OperatorKind,
    in_shape: &InferredShape,
) -> (Option<InferredShape>, Option<String>) {
    match op {
        OperatorKind::Linear {
            in_features,
            out_features,
            ..
        } => {
            if in_shape.dims.is_empty() {
                return (
                    None,
                    Some("Empty input shape cannot pass through Linear layer".to_string()),
                );
            }
            if let Some(SymbolicDim::Known(k)) = in_shape.last_dim()
                && *k != *in_features
            {
                return (
                    None,
                    Some(format!(
                        "Linear layer expected in_features={}, but received tensor with last dimension={}",
                        in_features, k
                    )),
                );
            }
            let mut out_dims = in_shape.dims.clone();
            *out_dims.last_mut().unwrap() = SymbolicDim::Known(*out_features);
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::Embedding {
            num_embeddings: _,
            embedding_dim,
        } => {
            // e.g. [B, T] -> [B, T, D]
            let mut out_dims = in_shape.dims.clone();
            out_dims.push(SymbolicDim::Known(*embedding_dim));
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::MHA {
            num_heads,
            d_k: _,
            d_v,
            ..
        } => {
            let out_dim = num_heads * d_v;
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(out_dim);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::GQA {
            num_q_heads, d_k, ..
        } => {
            let out_dim = num_q_heads * d_k;
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(out_dim);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::MLA { num_heads, d_v, .. } => {
            let out_dim = num_heads * d_v;
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(out_dim);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::MambaS6 { d_model, .. } | OperatorKind::Mamba2SSD { d_model, .. } => {
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(*d_model);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::xLSTM { d_model, .. } => {
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(*d_model);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::KANLayer { out_features, .. } => {
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(*out_features);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::SwiGLU { in_features, .. } | OperatorKind::GeGLU { in_features, .. } => {
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(*in_features);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::DeepSeekMoE { d_model, .. } => {
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(*d_model);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::FourierNeuralOperator { width, .. } => {
            let mut out_dims = in_shape.dims.clone();
            if let Some(last) = out_dims.last_mut() {
                *last = SymbolicDim::Known(*width);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::WTConv { out_channels, .. } => {
            let mut out_dims = in_shape.dims.clone();
            // Conv shape is [B, C, H, W] -> channel dimension is index 1
            if out_dims.len() >= 2 {
                out_dims[1] = SymbolicDim::Known(*out_channels);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        OperatorKind::ConvNeXtV2Block { dim, .. } => {
            let mut out_dims = in_shape.dims.clone();
            if out_dims.len() >= 2 {
                out_dims[1] = SymbolicDim::Known(*dim);
            }
            (Some(InferredShape::new(out_dims)), None)
        }

        // Shape-preserving operators (Norms, Activations, Gating, Identity, etc.)
        OperatorKind::RMSNorm { .. }
        | OperatorKind::LayerNorm { .. }
        | OperatorKind::QKNorm { .. }
        | OperatorKind::LogitSoftCap { .. }
        | OperatorKind::RevINLayer { .. }
        | OperatorKind::Entmax15Gate
        | OperatorKind::SparsemaxGate
        | OperatorKind::Snake { .. }
        | OperatorKind::HardConcreteL0 { .. }
        | OperatorKind::DiffAttn { .. }
        | OperatorKind::LinearAttention { .. }
        | OperatorKind::SlidingWindowSink { .. }
        | OperatorKind::EvoformerTriangle { .. }
        | OperatorKind::InvariantPointAttention { .. }
        | OperatorKind::S4D { .. }
        | OperatorKind::LRU { .. }
        | OperatorKind::TitansMemory { .. }
        | OperatorKind::EchoStateNetwork { .. }
        | OperatorKind::NextGenReservoir { .. }
        | OperatorKind::LISTA { .. }
        | OperatorKind::ADMMNet { .. }
        | OperatorKind::CRATE { .. }
        | OperatorKind::DEQ { .. }
        | OperatorKind::TTT { .. }
        | OperatorKind::ModernHopfield { .. }
        | OperatorKind::DeepONet { .. }
        | OperatorKind::EGNN { .. }
        | OperatorKind::CellularSheafConv { .. }
        | OperatorKind::CliffordGeometricProduct { .. }
        | OperatorKind::GaussianSplatting3D { .. }
        | OperatorKind::SlotAttention { .. }
        | OperatorKind::SoftSortPermutahedron { .. }
        | OperatorKind::MixtureOfDepths { .. } => (Some(in_shape.clone()), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::io::{InputPortSpec, TensorSpace};

    #[test]
    fn test_shape_inference_linear_chain() {
        let mut spec = AiDesignSpec::new("test_chain", "Test Chain");
        spec.inputs.push(InputPortSpec {
            id: "x".to_string(),
            name: "Input".to_string(),
            space: TensorSpace::ContinuousVector {
                dimension: 64,
                range: None,
            },
            symbolic_shape: vec![SymbolicDim::Symbol("B".to_string()), SymbolicDim::Known(64)],
            normalizer: Default::default(),
            coordinate_encoding: Default::default(),
            patching: None,
        });

        spec.graph.nodes.push(crate::spec::graph::NodeSpec {
            id: "fc1".to_string(),
            name: "Linear 1".to_string(),
            operator: OperatorKind::Linear {
                in_features: 64,
                out_features: 128,
                bias: true,
            },
            precision: None,
            metadata: Default::default(),
        });

        spec.graph.nodes.push(crate::spec::graph::NodeSpec {
            id: "fc2".to_string(),
            name: "Linear 2".to_string(),
            operator: OperatorKind::Linear {
                in_features: 128,
                out_features: 10,
                bias: true,
            },
            precision: None,
            metadata: Default::default(),
        });

        let report = infer_shapes(&spec);
        assert!(report.is_valid);
        assert_eq!(
            report
                .node_shapes
                .get("fc1")
                .unwrap()
                .output_shape
                .as_ref()
                .unwrap()
                .display(),
            "[B, 128]"
        );
        assert_eq!(
            report
                .node_shapes
                .get("fc2")
                .unwrap()
                .output_shape
                .as_ref()
                .unwrap()
                .display(),
            "[B, 10]"
        );
    }
}
