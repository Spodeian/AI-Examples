"""Neuro-Symbolic, Differentiable Logic, VSA, and Combinatorial Solvers — JAX reference implementation.

Topic: 05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers
Golden weights: weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers.safetensors

``python jax/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/02_neuro_symbolic_differentiable_logic_vsa_and_combinatorial_solvers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.matmul(x, weight) + bias


def lukasiewicz_logic(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Differentiable Łukasiewicz fuzzy logic operators."""
    t_norm = jnp.clip(a + b - 1.0, 0.0, 1.0)
    t_conorm = jnp.clip(a + b, 0.0, 1.0)
    implication = jnp.clip(1.0 - a + b, 0.0, 1.0)
    return t_norm, t_conorm, implication


def vsa_bind_and_bundle(
    key: jax.Array,
    val: jax.Array,
    other: jax.Array,
    codebook: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Vector Symbolic Architecture (VSA / HDC) in JAX."""
    bound = key * val
    unnorm_bundled = bound + other
    denom_bundled = jnp.maximum(jnp.linalg.norm(unnorm_bundled, ord=2, axis=-1, keepdims=True), 1e-12)
    bundled = unnorm_bundled / denom_bundled

    denom_cb = jnp.maximum(jnp.linalg.norm(codebook, ord=2, axis=-1, keepdims=True), 1e-12)
    norm_cb = codebook / denom_cb
    sim = jnp.matmul(bundled, norm_cb.T)
    return bound, bundled, sim


def sinkhorn_knopp_solver(logits: jax.Array, num_iters: int = 5, tau: float = 0.5) -> jax.Array:
    """Differentiable Sinkhorn-Knopp assignment solver."""
    p = jnp.exp(logits / tau)
    for _ in range(num_iters):
        p = p / (jnp.sum(p, axis=-1, keepdims=True) + 1e-8)
        p = p / (jnp.sum(p, axis=-2, keepdims=True) + 1e-8)
    return p


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation reproduces the golden checkpoint."""
    golden = {k: jnp.asarray(v) for k, v in load_file(str(path)).items()}

    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear_anchor parity failed: max abs error {err:.2e}"

    t_norm, t_conorm, imp = lukasiewicz_logic(golden["logic_a"], golden["logic_b"])
    assert float(jnp.max(jnp.abs(t_norm - golden["t_norm"]))) < 1e-5
    assert float(jnp.max(jnp.abs(t_conorm - golden["t_conorm"]))) < 1e-5
    assert float(jnp.max(jnp.abs(imp - golden["implication"]))) < 1e-5

    bound, bundled, sim = vsa_bind_and_bundle(
        golden["vsa_key"], golden["vsa_val"], golden["vsa_other"], golden["vsa_codebook"]
    )
    assert float(jnp.max(jnp.abs(bound - golden["vsa_bound"]))) < 1e-5
    assert float(jnp.max(jnp.abs(bundled - golden["vsa_bundled"]))) < 1e-5
    assert float(jnp.max(jnp.abs(sim - golden["vsa_sim"]))) < 1e-5

    p = sinkhorn_knopp_solver(golden["sinkhorn_logits"], num_iters=5, tau=0.5)
    assert float(jnp.max(jnp.abs(p - golden["sinkhorn_p"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
