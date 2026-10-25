"""Meta Learning Prototypical MAML and Reptile — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def protonet_prototypes(support_embeds: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(support_embeds, axis=1)


def protonet_logits(query_embeds: jnp.ndarray, prototypes: jnp.ndarray) -> jnp.ndarray:
    # query: [Q, D], prototypes: [N, D]
    diff = jnp.expand_dims(query_embeds, 1) - jnp.expand_dims(prototypes, 0)
    dists = jnp.sum(diff ** 2, axis=-1)
    return -dists


def maml_inner_linear_step(
    x_s: jnp.ndarray,
    y_s: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    alpha: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    pred_s = x_s @ w + b
    err = pred_s - y_s
    n_elements = err.size
    grad_w = (2.0 / n_elements) * (x_s.T @ err)
    grad_b = (2.0 / n_elements) * jnp.sum(err, axis=0)
    w_adapted = w - alpha * grad_w
    b_adapted = b - alpha * grad_b
    return w_adapted, b_adapted


def reptile_outer_step(
    theta: jnp.ndarray,
    theta_adapted: jnp.ndarray,
    epsilon: float = 0.1,
) -> jnp.ndarray:
    return theta + epsilon * (theta_adapted - theta)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. ProtoNet
    prototypes = protonet_prototypes(g["support_embeds"])
    err_proto = float(jnp.max(jnp.abs(prototypes - g["prototypes_ref"])))
    assert err_proto < 1e-5, f"prototypes error {err_proto:.2e}"

    logits = protonet_logits(g["query_embeds"], prototypes)
    err_logits = float(jnp.max(jnp.abs(logits - g["protonet_logits_ref"])))
    assert err_logits < 1e-5, f"protonet logits error {err_logits:.2e}"

    # 3. MAML
    w_ad, b_ad = maml_inner_linear_step(g["x_s"], g["y_s"], g["w_maml"], g["b_maml"], alpha=0.05)
    err_w = float(jnp.max(jnp.abs(w_ad - g["w_adapted_ref"])))
    assert err_w < 1e-5, f"maml w error {err_w:.2e}"
    err_b = float(jnp.max(jnp.abs(b_ad - g["b_adapted_ref"])))
    assert err_b < 1e-5, f"maml b error {err_b:.2e}"

    # 4. Reptile
    theta_new = reptile_outer_step(g["theta_base"], g["theta_adapted"], epsilon=0.1)
    err_rep = float(jnp.max(jnp.abs(theta_new - g["theta_new_ref"])))
    assert err_rep < 1e-5, f"reptile error {err_rep:.2e}"

    print("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
