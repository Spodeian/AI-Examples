"""Federated Learning Fedavg Fedprox Scaffold and Secagg — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def fedavg_aggregate(client_weights: jnp.ndarray, client_counts: jnp.ndarray) -> jnp.ndarray:
    weights_ratio = client_counts / jnp.sum(client_counts)
    return jnp.sum(jnp.expand_dims(weights_ratio, 1) * client_weights, axis=0)


def fedprox_gradient_step(
    w_local: jnp.ndarray,
    w_global: jnp.ndarray,
    grad_local: jnp.ndarray,
    eta: float = 0.05,
    mu: float = 0.1,
) -> jnp.ndarray:
    prox_grad = grad_local + mu * (w_local - w_global)
    return w_local - eta * prox_grad


def scaffold_correct_gradient(
    grad_i: jnp.ndarray,
    c_i: jnp.ndarray,
    c_global: jnp.ndarray,
) -> jnp.ndarray:
    return grad_i - c_i + c_global


def secagg_aggregate_sum(masked_models: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(masked_models, axis=0)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. FedAvg
    agg = fedavg_aggregate(g["client_weights"], g["client_counts"])
    err_avg = float(jnp.max(jnp.abs(agg - g["fedavg_aggregated_ref"])))
    assert err_avg < 1e-5, f"fedavg error {err_avg:.2e}"

    # 3. FedProx
    w_prox = fedprox_gradient_step(g["w_local"], g["w_global"], g["grad_local"], eta=0.05, mu=0.1)
    err_prox = float(jnp.max(jnp.abs(w_prox - g["w_prox_ref"])))
    assert err_prox < 1e-5, f"fedprox error {err_prox:.2e}"

    # 4. SCAFFOLD
    scaffold = scaffold_correct_gradient(g["grad_i"], g["c_i"], g["c_global"])
    err_scaf = float(jnp.max(jnp.abs(scaffold - g["scaffold_corrected_ref"])))
    assert err_scaf < 1e-5, f"scaffold error {err_scaf:.2e}"

    # 5. SecAgg
    secagg = secagg_aggregate_sum(g["masked_models"])
    err_sec = float(jnp.max(jnp.abs(secagg - g["secagg_unmasked_sum_ref"])))
    assert err_sec < 1e-5, f"secagg error {err_sec:.2e}"

    print("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
