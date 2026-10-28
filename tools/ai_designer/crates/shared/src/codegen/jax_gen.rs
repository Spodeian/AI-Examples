//! JAX / Equinox project generator: emits functional model.py (eqx.Module), train.py (optax + filter_jit), and losses.py.

use crate::spec::AiDesignSpec;
use crate::spec::operators::OperatorKind;
use std::collections::HashMap;

/// Generates a complete JAX / Equinox project file map.
pub fn generate_jax_project(spec: &AiDesignSpec) -> HashMap<String, String> {
    let mut files = HashMap::new();

    files.insert("model.py".to_string(), generate_jax_model_py(spec));
    files.insert("train.py".to_string(), generate_jax_train_py(spec));
    files.insert(
        "requirements.txt".to_string(),
        generate_jax_requirements_txt(spec),
    );

    files
}

fn generate_jax_model_py(spec: &AiDesignSpec) -> String {
    let mut s = String::new();
    s.push_str("# Auto-generated JAX / Equinox Functional Model\n");
    s.push_str(
        "import jax\nimport jax.numpy as jnp\nimport equinox as eqx\nfrom typing import Tuple\n\n",
    );

    s.push_str(&format!(
        "class {}(eqx.Module):\n",
        sanitize_class_name(&spec.metadata.model_id)
    ));

    // Fields
    for node in &spec.graph.nodes {
        match &node.operator {
            OperatorKind::Linear { .. } => {
                s.push_str(&format!("    {}: eqx.nn.Linear\n", node.id));
            }
            OperatorKind::Embedding { .. } => {
                s.push_str(&format!("    {}: eqx.nn.Embedding\n", node.id));
            }
            OperatorKind::RMSNorm { .. } => {
                s.push_str(&format!("    {}: eqx.nn.RMSNorm\n", node.id));
            }
            OperatorKind::LayerNorm { .. } => {
                s.push_str(&format!("    {}: eqx.nn.LayerNorm\n", node.id));
            }
            _ => {
                s.push_str(&format!("    {}: eqx.nn.Identity\n", node.id));
            }
        }
    }

    s.push_str("\n    def __init__(self, key: jax.random.PRNGKey):\n");
    s.push_str("        keys = jax.random.split(key, 16)\n");

    for (idx, node) in spec.graph.nodes.iter().enumerate() {
        match &node.operator {
            OperatorKind::Linear {
                in_features,
                out_features,
                bias,
            } => {
                s.push_str(&format!(
                    "        self.{} = eqx.nn.Linear({}, {}, use_bias={}, key=keys[{}])\n",
                    node.id, in_features, out_features, bias, idx
                ));
            }
            OperatorKind::Embedding {
                num_embeddings,
                embedding_dim,
            } => {
                s.push_str(&format!(
                    "        self.{} = eqx.nn.Embedding({}, {}, key=keys[{}])\n",
                    node.id, num_embeddings, embedding_dim, idx
                ));
            }
            OperatorKind::RMSNorm { dim, eps } => {
                s.push_str(&format!(
                    "        self.{} = eqx.nn.RMSNorm({}, eps={})\n",
                    node.id, dim, eps
                ));
            }
            OperatorKind::LayerNorm { dim, eps } => {
                s.push_str(&format!(
                    "        self.{} = eqx.nn.LayerNorm({}, eps={})\n",
                    node.id, dim, eps
                ));
            }
            _ => {
                s.push_str(&format!("        self.{} = eqx.nn.Identity()\n", node.id));
            }
        }
    }

    s.push_str("\n    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:\n");
    for node in &spec.graph.nodes {
        s.push_str(&format!("        x = self.{}(x)\n", node.id));
    }
    s.push_str("        return x\n");

    s
}

fn generate_jax_train_py(spec: &AiDesignSpec) -> String {
    let mut s = String::new();
    s.push_str("# Auto-generated JAX / Optax Functional Training Pipeline\n");
    s.push_str("import jax\nimport jax.numpy as jnp\nimport equinox as eqx\nimport optax\nfrom model import ");
    s.push_str(&sanitize_class_name(&spec.metadata.model_id));
    s.push_str("\n\n");

    s.push_str(
        r#"def loss_fn(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y) ** 2)

@eqx.filter_jit
def make_step(model, opt_state, tx, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = tx.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def main():
    key = jax.random.PRNGKey(42)
    mkey, xkey, ykey = jax.random.split(key, 3)

    model = "#,
    );
    s.push_str(&sanitize_class_name(&spec.metadata.model_id));
    s.push_str("(mkey)\n\n");

    s.push_str(&format!("    # Optax Optimizer with AdamW\n    tx = optax.chain(\n        optax.clip_by_global_norm(1.0),\n        optax.adamw(learning_rate={}, weight_decay={})\n    )\n",
        spec.training.adamw.lr, spec.training.adamw.weight_decay));

    s.push_str(
        r#"    opt_state = tx.init(eqx.filter(model, eqx.is_array))

    print("Beginning JAX Training Loop...")
    for step in range(100):
        x = jax.random.normal(xkey, (8, 64))
        y = jax.random.normal(ykey, (8, 64))
        model, opt_state, loss = make_step(model, opt_state, tx, x, y)
        if step % 10 == 0:
            print(f"JAX Step {step:04d} | Loss: {loss:.6f}")

    print("JAX Training Complete.")

if __name__ == '__main__':
    main()
"#,
    );

    s
}

fn generate_jax_requirements_txt(_spec: &AiDesignSpec) -> String {
    r#"jax>=0.4.30
equinox>=0.11.4
optax>=0.2.2
"#
    .to_string()
}

fn sanitize_class_name(raw: &str) -> String {
    let parts: Vec<&str> = raw.split('_').collect();
    let mut s = String::new();
    for p in parts {
        if let Some(c) = p.chars().next() {
            s.push_str(&c.to_uppercase().to_string());
            s.push_str(&p[1..]);
        }
    }
    if s.is_empty() {
        "CustomAiModel".to_string()
    } else {
        s
    }
}
