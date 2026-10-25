"""Memory Optimization Selective Checkpointing Offloading and Grad Hooks — functional JAX implementation.

Topic: 12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks. Run ``python jax/12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def selective_checkpoint_silu_gate(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    gate: jnp.ndarray,
    grad_out: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    z = x @ w + b
    sig_z = 1.0 / (1.0 + jnp.exp(-z))
    a = z * sig_z
    y = a * gate

    g_a = grad_out * gate
    da_dz = sig_z * (1.0 + z * (1.0 - sig_z))
    g_z = g_a * da_dz

    grad_w = x.T @ g_z
    grad_b = jnp.sum(g_z, axis=0)
    grad_x = g_z @ w.T
    grad_gate = grad_out * a
    return y, grad_w, grad_b, grad_x, grad_gate


def post_accumulate_grad_hook_step(
    w: jnp.ndarray,
    grad_w: jnp.ndarray,
    running_m: jnp.ndarray,
    beta: float = 0.9,
    lr: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    m_new = float(beta) * running_m + (1.0 - float(beta)) * grad_w
    w_new = w - float(lr) * m_new
    return w_new, m_new


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. Selective Checkpoint SiLU Gate
    y, grad_w, grad_b, grad_x, grad_gate = selective_checkpoint_silu_gate(
        g["chk_x"], g["chk_w"], g["chk_b"], g["chk_gate"], g["chk_grad_out"]
    )
    assert float(jnp.max(jnp.abs(y - g["chk_y_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_w - g["chk_grad_w_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_b - g["chk_grad_b_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_x - g["chk_grad_x_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(grad_gate - g["chk_grad_gate_ref"]))) < 1e-5

    # 2. Post-Accumulate Grad Hook step
    w_new, m_new = post_accumulate_grad_hook_step(
        g["chk_w"], g["chk_grad_w_ref"], g["hook_running_m"], beta=0.9, lr=0.01
    )
    assert float(jnp.max(jnp.abs(w_new - g["hook_w_new_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(m_new - g["hook_m_new_ref"]))) < 1e-5

    print("12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
