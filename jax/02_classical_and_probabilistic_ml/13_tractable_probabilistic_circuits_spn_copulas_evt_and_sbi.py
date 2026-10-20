"""Tractable Probabilistic Circuits (SPN), Copulas, Extreme Value Theory (EVT), and SBI in JAX.

Features:
- Pure functional JIT-compiled Sum-Product Network (SPN) evaluation
- Vectorized Archimedean Clayton Copula
- Extreme Value Theory (EVT) Generalized Pareto Distribution (GPD)
- Simulation-Based Inference (SBI) neural density ratio estimation
"""

from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
import safetensors.torch
import numpy as np


@jax.jit
def evaluate_spn(
    leaves_p: jnp.ndarray,      # [4, 2]
    leaf_evidence: jnp.ndarray, # [4]
    prod_weights: jnp.ndarray,  # [2]
) -> jnp.ndarray:
    """Evaluates smooth and decomposable SPN over 4 variables."""
    # Compute leaf values based on evidence
    # evidence == 1.0 -> leaves_p
    # evidence == 0.0 -> 1.0 - leaves_p
    # evidence == -1.0 -> 1.0 (marginalized)
    ev = leaf_evidence[:, None] # [4, 1]
    leaf_vals = jnp.where(
        ev == 1.0,
        leaves_p,
        jnp.where(
            ev == 0.0,
            1.0 - leaves_p,
            jnp.ones_like(leaves_p)
        )
    )

    p0 = leaf_vals[0, 0] * leaf_vals[1, 0] * leaf_vals[2, 0] * leaf_vals[3, 0]
    p1 = leaf_vals[0, 1] * leaf_vals[1, 1] * leaf_vals[2, 1] * leaf_vals[3, 1]
    return prod_weights[0] * p0 + prod_weights[1] * p1


@jax.jit
def clayton_copula_cdf(
    u: jnp.ndarray,
    v: jnp.ndarray,
    theta: float = 2.0,
) -> jnp.ndarray:
    """Evaluates bivariate Clayton copula CDF."""
    inner = jnp.clip(u ** (-theta) + v ** (-theta) - 1.0, a_min=1e-7)
    return inner ** (-1.0 / theta)


@jax.jit
def gpd_tail_cdf(
    y: jnp.ndarray,
    xi: float = 0.25,
    sigma: float = 1.5,
) -> jnp.ndarray:
    """Evaluates Generalized Pareto Distribution (GPD) CDF for excesses y >= 0."""
    bracket = jnp.clip(1.0 + (xi * y) / sigma, a_min=1e-7)
    return 1.0 - bracket ** (-1.0 / xi)


@jax.jit
def sbi_neural_ratio(
    x: jnp.ndarray,
    theta: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluates neural density ratio r(x, theta) = exp(logit)."""
    features = jnp.concatenate([x, theta], axis=-1)
    logit = features @ w + b
    ratio = jnp.exp(logit)
    return ratio, logit


def verify_topic13_jax():
    ckpt_path = Path("weights/02_classical_and_probabilistic_ml/13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.safetensors")
    if not ckpt_path.exists():
        print("Checkpoint not found, skipping JAX verification.")
        return

    ckpt = safetensors.torch.load_file(ckpt_path)

    # 1. SPN Verification
    leaves_p = jnp.array(ckpt["leaves_p"].numpy())
    prod_weights = jnp.array(ckpt["prod_weights"].numpy())
    ev_full = jnp.array(ckpt["ev_full"].numpy())
    ev_marg = jnp.array(ckpt["ev_marg"].numpy())

    spn_full = evaluate_spn(leaves_p, ev_full, prod_weights)
    spn_marg = evaluate_spn(leaves_p, ev_marg, prod_weights)

    golden_full = ckpt["spn_prob_full"][0].item()
    golden_marg = ckpt["spn_prob_marg"][0].item()
    assert abs(float(spn_full) - golden_full) < 1e-5, f"SPN full diff {abs(float(spn_full) - golden_full)}"
    assert abs(float(spn_marg) - golden_marg) < 1e-5, f"SPN marg diff {abs(float(spn_marg) - golden_marg)}"

    # 2. Copula Verification
    u = jnp.array(ckpt["u_grid"].numpy())
    v = jnp.array(ckpt["v_grid"].numpy())
    c_vals = clayton_copula_cdf(u, v, theta=2.0)
    golden_c = ckpt["clayton_cdf"].numpy()
    assert np.allclose(np.array(c_vals), golden_c, atol=1e-5), "Clayton copula JAX mismatch"

    # 3. GPD EVT Verification
    y = jnp.array(ckpt["y_excesses"].numpy())
    gpd_vals = gpd_tail_cdf(y, xi=0.25, sigma=1.5)
    golden_gpd = ckpt["gpd_cdf"].numpy()
    assert np.allclose(np.array(gpd_vals), golden_gpd, atol=1e-5), "GPD CDF JAX mismatch"

    # 4. SBI Ratio Verification
    x_sbi = jnp.array(ckpt["x_sbi"].numpy())
    theta_sbi = jnp.array(ckpt["theta_sbi"].numpy())
    w_sbi = jnp.array(ckpt["w_sbi"].numpy())
    b_sbi = jnp.array(ckpt["b_sbi"].numpy())
    ratio, logit = sbi_neural_ratio(x_sbi, theta_sbi, w_sbi, b_sbi)
    golden_ratio = ckpt["sbi_ratio"].numpy()
    golden_logit = ckpt["sbi_logit"].numpy()
    assert np.allclose(np.array(ratio), golden_ratio, atol=1e-5), "SBI ratio JAX mismatch"
    assert np.allclose(np.array(logit), golden_logit, atol=1e-5), "SBI logit JAX mismatch"

    print("All Topic 13 JAX verifications passed!")


if __name__ == "__main__":
    verify_topic13_jax()
