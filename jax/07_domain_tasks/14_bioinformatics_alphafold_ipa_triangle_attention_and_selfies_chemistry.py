"""Bioinformatics Alphafold Ipa Triangle Attention and Selfies Chemistry — JAX reference implementation.

Topic: 07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry
Golden weights: weights/07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "07_domain_tasks/14_bioinformatics_alphafold_ipa_triangle_attention_and_selfies_chemistry"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def triangle_multiplicative_update(
    z: jnp.ndarray,
    w_a: jnp.ndarray,
    b_a: jnp.ndarray,
    w_b: jnp.ndarray,
    b_b: jnp.ndarray,
    w_g: jnp.ndarray,
    b_g: jnp.ndarray,
    w_o: jnp.ndarray,
    b_o: jnp.ndarray,
) -> jnp.ndarray:
    """AlphaFold Triangle Multiplicative Update (Outgoing Edges) in JAX."""
    a = z @ w_a + b_a
    b = z @ w_b + b_b
    g = jax.nn.sigmoid(z @ w_g + b_g)
    s = jnp.einsum("bikc, bjkc -> bijc", a, b)
    out = (g * s) @ w_o + b_o
    return out


def fape_loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    r_x: jnp.ndarray,
    t_x: jnp.ndarray,
    r_y: jnp.ndarray,
    t_y: jnp.ndarray,
    d_clamp: float = 10.0,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """AlphaFold Frame Aligned Point Error (FAPE) Loss in JAX."""
    diff_x = jnp.expand_dims(x, 1) - jnp.expand_dims(t_x, 2)
    local_x = jnp.einsum("bijc, biac -> bija", diff_x, r_x)

    diff_y = jnp.expand_dims(y, 1) - jnp.expand_dims(t_y, 2)
    local_y = jnp.einsum("bijc, biac -> bija", diff_y, r_y)

    dist = jnp.sqrt(jnp.sum((local_x - local_y) ** 2, axis=-1) + eps)
    clamped_dist = jnp.clip(dist, a_max=d_clamp)
    return jnp.mean(clamped_dist)


def ipa_point_distance_penalty(
    q_pts: jnp.ndarray,
    k_pts: jnp.ndarray,
    gamma: jnp.ndarray,
) -> jnp.ndarray:
    """AlphaFold Invariant Point Attention (IPA) 3D Point Distance Penalty in JAX."""
    diff = jnp.expand_dims(q_pts, 2) - jnp.expand_dims(k_pts, 1)
    sq_dist = jnp.sum(diff ** 2, axis=(-2, -1))
    return -0.5 * gamma.reshape(1, 1, 1, -1) * sq_dist


def molecular_valence_loss(
    bond_logits: jnp.ndarray,
    max_valence: jnp.ndarray,
) -> jnp.ndarray:
    """Molecular chemistry valence penalty loss in JAX."""
    n = bond_logits.shape[1]
    probs = jax.nn.sigmoid(bond_logits)
    mask = 1.0 - jnp.eye(n).reshape(1, n, n)
    masked_probs = probs * mask
    valences = jnp.sum(masked_probs, axis=-1)
    excess = jax.nn.relu(valences - max_valence)
    return jnp.mean(jnp.sum(excess ** 2, axis=-1))


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    raw = load_file(str(path))
    tensors = {k: jnp.array(v) for k, v in raw.items()}

    # Anchor check
    out = linear_anchor(tensors["test_input"], tensors["weight"], tensors["bias"])
    err = float(jnp.max(jnp.abs(out - tensors["test_output"])))
    assert err < 1e-5, f"test_output parity error: {err}"

    # 1. Triangle Update
    tri_out = triangle_multiplicative_update(
        tensors["z"],
        tensors["w_a"],
        tensors["b_a"],
        tensors["w_b"],
        tensors["b_b"],
        tensors["w_g"],
        tensors["b_g"],
        tensors["w_o"],
        tensors["b_o"],
    )
    err_tri = float(jnp.max(jnp.abs(tri_out - tensors["tri_out_ref"])))
    assert err_tri < 1e-5, f"triangle update parity error: {err_tri}"

    # 2. FAPE Loss
    fape = fape_loss(
        tensors["x"],
        tensors["y"],
        tensors["r_x"],
        tensors["t_x"],
        tensors["r_y"],
        tensors["t_y"],
        d_clamp=10.0,
    )
    err_fape = float(jnp.abs(fape - tensors["fape_loss_ref"]))
    assert err_fape < 1e-5, f"fape loss parity error: {err_fape}"

    # 3. IPA Distance Penalty
    penalty = ipa_point_distance_penalty(tensors["q_pts"], tensors["k_pts"], tensors["gamma"])
    err_ipa = float(jnp.max(jnp.abs(penalty - tensors["ipa_penalty_ref"])))
    assert err_ipa < 1e-5, f"ipa penalty parity error: {err_ipa}"

    # 4. Valence Loss
    valence = molecular_valence_loss(tensors["bond_logits"], tensors["max_valence"])
    err_val = float(jnp.abs(valence - tensors["valence_loss_ref"]))
    assert err_val < 1e-5, f"valence loss parity error: {err_val}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
