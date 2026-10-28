//! Python AST Reverse-Engineering: Ingesting raw PyTorch `nn.Module` classes into valid `AiDesignSpec` blueprints.

use crate::spec::*;
use std::collections::HashMap;

/// Intermediate representation of an extracted PyTorch layer.
#[derive(Debug, Clone)]
pub struct ExtractedLayer {
    pub name: String,
    pub class_type: String,
    pub args_raw: String,
    pub operator: Option<OperatorKind>,
}

/// Extracted PyTorch class information.
#[derive(Debug, Clone)]
pub struct ExtractedModuleClass {
    pub class_name: String,
    pub docstring: Option<String>,
    pub layers: Vec<ExtractedLayer>,
    pub forward_params: Vec<String>,
    pub forward_body: Vec<String>,
    pub return_expr: Option<String>,
}

/// Reverse-engineers raw Python PyTorch code into a validated `AiDesignSpec`.
pub fn import_from_python_source(
    source: &str,
    target_class: Option<&str>,
) -> Result<AiDesignSpec, String> {
    let classes = extract_module_classes(source)?;
    if classes.is_empty() {
        return Err(
            "No PyTorch `nn.Module` classes found in the provided Python source.".to_string(),
        );
    }

    let target = match target_class {
        Some(name) => classes
            .iter()
            .find(|c| c.class_name == name)
            .ok_or_else(|| {
                format!(
                    "Class '{}' not found in Python source. Found: {:?}",
                    name,
                    classes.iter().map(|c| &c.class_name).collect::<Vec<_>>()
                )
            })?,
        None => {
            // Pick the last defined class, or the one that contains the most layers (often the top-level block/model)
            classes.iter().max_by_key(|c| c.layers.len()).unwrap()
        }
    };

    synthesize_spec_from_class(target)
}

/// Scans source lines and extracts all `nn.Module` class declarations with their layers and forward dataflow.
pub fn extract_module_classes(source: &str) -> Result<Vec<ExtractedModuleClass>, String> {
    let mut classes = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();
        if (line.starts_with("class ") && line.contains("Module"))
            || (line.starts_with("class ") && line.ends_with(':'))
        {
            // Extract class name
            if let Some(name) = extract_class_name(line) {
                let mut docstring = None;
                let mut layers = Vec::new();
                let mut forward_params = Vec::new();
                let mut forward_body = Vec::new();
                let mut return_expr = None;

                i += 1;
                // Parse class body until next class at root or end of file
                let mut in_init = false;
                let mut in_forward = false;

                while i < lines.len() {
                    let raw_line = lines[i];
                    let trimmed = raw_line.trim();

                    // Check if new class starts at 0 indentation
                    if !raw_line.starts_with(' ')
                        && !raw_line.starts_with('\t')
                        && trimmed.starts_with("class ")
                    {
                        break;
                    }

                    // Check for class docstring
                    if docstring.is_none()
                        && (trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''"))
                    {
                        let quote = if trimmed.starts_with("\"\"\"") {
                            "\"\"\""
                        } else {
                            "'''"
                        };
                        let rest = &trimmed[3..];
                        if rest.contains(quote) {
                            docstring = Some(rest.trim_end_matches(quote).trim().to_string());
                        } else {
                            let mut doc = rest.to_string();
                            i += 1;
                            while i < lines.len() && !lines[i].contains(quote) {
                                doc.push(' ');
                                doc.push_str(lines[i].trim());
                                i += 1;
                            }
                            docstring = Some(doc.trim().to_string());
                        }
                    }

                    // Detect __init__
                    if trimmed.starts_with("def __init__") {
                        in_init = true;
                        in_forward = false;
                    } else if trimmed.starts_with("def forward") {
                        in_init = false;
                        in_forward = true;
                        forward_params = parse_forward_params(trimmed);
                    } else if trimmed.starts_with("def ") {
                        in_init = false;
                        in_forward = false;
                    }

                    if in_init
                        && trimmed.starts_with("self.")
                        && trimmed.contains('=')
                        && let Some(layer) = parse_layer_assignment(trimmed)
                    {
                        layers.push(layer);
                    }

                    if in_forward {
                        if trimmed.starts_with("return ") {
                            return_expr =
                                Some(trimmed.trim_start_matches("return ").trim().to_string());
                        } else if !trimmed.starts_with("def forward") && !trimmed.is_empty() {
                            forward_body.push(trimmed.to_string());
                        }
                    }

                    i += 1;
                }

                classes.push(ExtractedModuleClass {
                    class_name: name,
                    docstring,
                    layers,
                    forward_params,
                    forward_body,
                    return_expr,
                });
                continue;
            }
        }
        i += 1;
    }

    Ok(classes)
}

fn extract_class_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let after_class = trimmed.strip_prefix("class ")?;
    let name_part = after_class.split('(').next()?.split(':').next()?.trim();
    if !name_part.is_empty() {
        Some(name_part.to_string())
    } else {
        None
    }
}

fn parse_forward_params(line: &str) -> Vec<String> {
    let mut params = Vec::new();
    if let Some(start) = line.find('(')
        && let Some(end) = line.rfind(')')
    {
        let inner = &line[start + 1..end];
        for p in inner.split(',') {
            let param = p.split(':').next().unwrap_or("").trim();
            if !param.is_empty() && param != "self" && param != "*args" && param != "**kwargs" {
                params.push(param.to_string());
            }
        }
    }
    if params.is_empty() {
        params.push("x".to_string());
    }
    params
}

fn parse_layer_assignment(line: &str) -> Option<ExtractedLayer> {
    let parts: Vec<&str> = line.split('=').collect();
    if parts.len() < 2 {
        return None;
    }

    let left = parts[0].trim();
    let right = parts[1..].join("=").trim().to_string();

    let layer_name = left.strip_prefix("self.")?.trim().to_string();

    // Look for nn.Linear, nn.LayerNorm, nn.RMSNorm, nn.Embedding, etc.
    if let Some(paren_pos) = right.find('(') {
        let type_part = right[..paren_pos].trim();
        let class_type = if let Some(stripped) = type_part.strip_prefix("nn.") {
            stripped.to_string()
        } else {
            type_part.to_string()
        };

        let args_raw = if right.ends_with(')') {
            right[paren_pos + 1..right.len() - 1].trim().to_string()
        } else {
            right[paren_pos + 1..].trim().to_string()
        };

        let operator = map_layer_to_operator(&class_type, &args_raw);

        Some(ExtractedLayer {
            name: layer_name,
            class_type,
            args_raw,
            operator,
        })
    } else {
        None
    }
}

fn extract_numeric_arg(args: &str, index: usize, kwarg: &str) -> Option<usize> {
    // Check keyword arg first: e.g. in_features=128
    for part in args.split(',') {
        let p = part.trim();
        if let Some(rest) = p.strip_prefix(&format!("{}=", kwarg)) {
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(val) = num_str.parse::<usize>() {
                return Some(val);
            }
        }
    }

    // Positional check
    let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
    if index < parts.len() {
        let p = parts[index];
        if !p.contains('=') {
            let num_str: String = p.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(val) = num_str.parse::<usize>() {
                return Some(val);
            }
        }
    }

    None
}

fn map_layer_to_operator(class_type: &str, args: &str) -> Option<OperatorKind> {
    match class_type {
        "Linear" => {
            let in_f = extract_numeric_arg(args, 0, "in_features").unwrap_or(256);
            let out_f = extract_numeric_arg(args, 1, "out_features").unwrap_or(in_f);
            let bias = !args.contains("bias=False") && !args.contains("bias = False");
            Some(OperatorKind::Linear {
                in_features: in_f,
                out_features: out_f,
                bias,
            })
        }
        "LayerNorm" => {
            let norm_shape = extract_numeric_arg(args, 0, "normalized_shape").unwrap_or(256);
            Some(OperatorKind::LayerNorm {
                dim: norm_shape,
                eps: 1e-5,
            })
        }
        "RMSNorm" => {
            let dim = extract_numeric_arg(args, 0, "dim").unwrap_or(256);
            Some(OperatorKind::RMSNorm { dim, eps: 1e-6 })
        }
        "Embedding" => {
            let num = extract_numeric_arg(args, 0, "num_embeddings").unwrap_or(32000);
            let dim = extract_numeric_arg(args, 1, "embedding_dim").unwrap_or(256);
            Some(OperatorKind::Embedding {
                num_embeddings: num,
                embedding_dim: dim,
            })
        }
        "MultiheadAttention" => {
            let embed_dim = extract_numeric_arg(args, 0, "embed_dim").unwrap_or(256);
            let num_heads = extract_numeric_arg(args, 1, "num_heads").unwrap_or(8);
            let d_k = embed_dim / num_heads.max(1);
            Some(OperatorKind::MHA {
                num_heads,
                d_k,
                d_v: d_k,
                dropout: 0.0,
            })
        }
        "SwiGLU" => {
            let in_f = extract_numeric_arg(args, 0, "in_features").unwrap_or(256);
            let h_f = extract_numeric_arg(args, 1, "hidden_features").unwrap_or(in_f * 4);
            Some(OperatorKind::SwiGLU {
                in_features: in_f,
                hidden_features: h_f,
            })
        }
        "GeGLU" => {
            let in_f = extract_numeric_arg(args, 0, "in_features").unwrap_or(256);
            let h_f = extract_numeric_arg(args, 1, "hidden_features").unwrap_or(in_f * 4);
            Some(OperatorKind::GeGLU {
                in_features: in_f,
                hidden_features: h_f,
            })
        }
        "KANLayer" => {
            let in_f = extract_numeric_arg(args, 0, "in_features").unwrap_or(256);
            let out_f = extract_numeric_arg(args, 1, "out_features").unwrap_or(in_f);
            Some(OperatorKind::KANLayer {
                in_features: in_f,
                out_features: out_f,
                grid_size: 5,
                spline_order: 3,
            })
        }
        _ => {
            // Default to linear projection if unknown
            Some(OperatorKind::Linear {
                in_features: 256,
                out_features: 256,
                bias: true,
            })
        }
    }
}

/// Synthesizes a validated `AiDesignSpec` from an extracted class.
fn synthesize_spec_from_class(module: &ExtractedModuleClass) -> Result<AiDesignSpec, String> {
    let model_id = to_snake_case(&module.class_name);
    let mut spec = AiDesignSpec::new(&model_id, &module.class_name);

    spec.metadata.domain = "ReverseEngineeredPyTorch".to_string();
    spec.metadata.tags = vec!["pytorch_imported".to_string(), module.class_name.clone()];

    // Determine primary dimension from first layer
    let (primary_in, primary_out) = if let Some(first) = module.layers.first() {
        match first.operator {
            Some(OperatorKind::Linear {
                in_features,
                out_features,
                ..
            }) => (in_features, out_features),
            Some(OperatorKind::LayerNorm { dim, .. }) => (dim, dim),
            Some(OperatorKind::RMSNorm { dim, .. }) => (dim, dim),
            Some(OperatorKind::MHA { num_heads, d_k, .. }) => (num_heads * d_k, num_heads * d_k),
            Some(OperatorKind::SwiGLU { in_features, .. }) => (in_features, in_features),
            _ => (256, 256),
        }
    } else {
        (256, 256)
    };

    let final_out = if let Some(last) = module.layers.last() {
        match last.operator {
            Some(OperatorKind::Linear { out_features, .. }) => out_features,
            _ => primary_out,
        }
    } else {
        primary_out
    };

    // Synthesize input ports
    let main_input = module
        .forward_params
        .first()
        .cloned()
        .unwrap_or_else(|| "x".to_string());
    spec.inputs.push(InputPortSpec {
        id: main_input.clone(),
        name: format!("Input ({})", main_input),
        space: TensorSpace::ContinuousVector {
            dimension: primary_in,
            range: None,
        },
        symbolic_shape: vec![
            SymbolicDim::from("B"),
            SymbolicDim::from("S"),
            SymbolicDim::from(primary_in),
        ],
        normalizer: NormalizerSpec::None,
        coordinate_encoding: CoordinateEncoding::None,
        patching: None,
    });

    // Synthesize computation graph
    if module.layers.is_empty() {
        // Create at least one fallback linear node to satisfy spec.validate()
        spec.graph.nodes.push(NodeSpec {
            id: "linear_head".to_string(),
            name: "Linear Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: primary_in,
                out_features: final_out,
                bias: true,
            },
            precision: Some(PrecisionFormat::BF16),
            metadata: HashMap::new(),
        });
    } else {
        for (idx, layer) in module.layers.iter().enumerate() {
            let op = layer.operator.clone().unwrap_or(OperatorKind::Linear {
                in_features: primary_in,
                out_features: primary_out,
                bias: true,
            });

            spec.graph.nodes.push(NodeSpec {
                id: format!("{}_{}", layer.name, idx),
                name: format!("{} ({})", layer.name, layer.class_type),
                operator: op,
                precision: Some(PrecisionFormat::BF16),
                metadata: HashMap::new(),
            });
        }
    }

    // Connect nodes linearly in the DAG
    for i in 0..spec.graph.nodes.len().saturating_sub(1) {
        let from_id = spec.graph.nodes[i].id.clone();
        let to_id = spec.graph.nodes[i + 1].id.clone();
        spec.graph.connections.push(ConnectionSpec {
            from_node: from_id,
            from_port: 0,
            to_node: to_id,
            to_port: 0,
        });
    }

    // Synthesize output ports
    let main_output = module
        .return_expr
        .clone()
        .unwrap_or_else(|| "out".to_string());
    let out_id = to_snake_case(&main_output.replace(['+', '(', ')', ' ', '.'], "_"));
    spec.outputs.push(OutputPortSpec {
        id: if out_id.is_empty() {
            "output".to_string()
        } else {
            out_id
        },
        name: format!("Output ({})", main_output),
        space: TensorSpace::ContinuousVector {
            dimension: final_out,
            range: None,
        },
        symbolic_shape: vec![
            SymbolicDim::from("B"),
            SymbolicDim::from("S"),
            SymbolicDim::from(final_out),
        ],
        constraint: StructuralConstraint::DirectLinear,
        loss: LossPairing::MSE,
    });

    // Validate synthesized spec
    spec.validate()?;

    Ok(spec)
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        if c.is_uppercase() {
            if i > 0 {
                let prev = chars[i - 1];
                let next = chars.get(i + 1);
                if prev.is_lowercase()
                    || (prev.is_uppercase() && next.map(|n| n.is_lowercase()).unwrap_or(false))
                {
                    result.push('_');
                }
            }
            for lc in c.to_lowercase() {
                result.push(lc);
            }
        } else if c.is_alphanumeric() {
            result.push(c);
        } else {
            result.push('_');
        }
    }
    result.trim_matches('_').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_engineer_pytorch_module() {
        let py_code = r#"
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedResMLP(nn.Module):
    """Gated Residual MLP with RMSNorm and Linear projections."""
    def __init__(self, in_features: int = 512, hidden_features: int = 2048, out_features: int = 512):
        super().__init__()
        self.norm = nn.RMSNorm(512)
        self.fc_in = nn.Linear(512, 2048, bias=False)
        self.fc_out = nn.Linear(2048, 512, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc_in(h)
        out = self.fc_out(h)
        return x + out
"#;

        let spec = import_from_python_source(py_code, None)
            .expect("Failed to reverse-engineer Python module");
        assert_eq!(spec.metadata.model_id, "gated_res_mlp");
        assert_eq!(spec.inputs.len(), 1);
        assert_eq!(spec.outputs.len(), 1);
        assert_eq!(spec.graph.nodes.len(), 3);

        // Verify layers
        assert!(matches!(
            spec.graph.nodes[0].operator,
            OperatorKind::RMSNorm { dim: 512, .. }
        ));
        assert!(matches!(
            spec.graph.nodes[1].operator,
            OperatorKind::Linear {
                in_features: 512,
                out_features: 2048,
                bias: false
            }
        ));
        assert!(matches!(
            spec.graph.nodes[2].operator,
            OperatorKind::Linear {
                in_features: 2048,
                out_features: 512,
                bias: true
            }
        ));
    }
}
