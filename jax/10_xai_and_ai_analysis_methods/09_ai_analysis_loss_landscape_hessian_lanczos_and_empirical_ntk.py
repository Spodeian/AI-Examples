"""AI Analysis Loss Landscape Hessian Lanczos and Empirical NTK — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk. Run ``python jax/10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def filter_normalize_direction(theta: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    norm_theta = jnp.sqrt(jnp.sum(theta * theta, axis=-1, keepdims=True) + 1e-8)
    norm_delta = jnp.sqrt(jnp.sum(delta * delta, axis=-1, keepdims=True) + 1e-8)
    return delta * (norm_theta / norm_delta)


def loss_surface_coord(
    theta: jnp.ndarray,
    d1: jnp.ndarray,
    d2: jnp.ndarray,
    alpha: float,
    beta: float,
) -> jnp.ndarray:
    return theta + float(alpha) * d1 + float(beta) * d2


def lanczos_tridiagonal(
    h: jnp.ndarray,
    v0: jnp.ndarray,
    num_steps: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    p = h.shape[0]
    q_list = []
    v_norm = jnp.sqrt(jnp.sum(v0 * v0) + 1e-8)
    q_list.append(v0 / v_norm)

    alpha_diag = []
    beta_diag = []

    for j in range(num_steps):
        w = h @ q_list[j]
        if j > 0:
            w = w - beta_diag[j - 1] * q_list[j - 1]
        a_j = jnp.sum(q_list[j] * w)
        alpha_diag.append(a_j)
        w = w - a_j * q_list[j]
        if j < num_steps - 1:
            b_j = jnp.sqrt(jnp.sum(w * w) + 1e-8)
            beta_diag.append(b_j)
            q_list.append(w / b_j)

    return jnp.stack(alpha_diag), jnp.stack(beta_diag)


def empirical_ntk(jacobians: jnp.ndarray) -> jnp.ndarray:
    return jacobians @ jacobians.T


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Filter normalize direction & loss slice
    d1_norm = filter_normalize_direction(g["ls_theta"], g["ls_d1_raw"])
    assert float(jnp.max(jnp.abs(d1_norm - g["ls_d1_norm"]))) < 1e-5, "d1_norm mismatch"

    d2_norm = filter_normalize_direction(g["ls_theta"], g["ls_d2_raw"])
    assert float(jnp.max(jnp.abs(d2_norm - g["ls_d2_norm"]))) < 1e-5, "d2_norm mismatch"

    ls_slice = loss_surface_coord(g["ls_theta"], d1_norm, d2_norm, alpha=0.5, beta=-0.25)
    assert float(jnp.max(jnp.abs(ls_slice - g["ls_slice_ref"]))) < 1e-5, "ls_slice mismatch"

    # 3. Lanczos tridiagonal
    alpha_diag, beta_diag = lanczos_tridiagonal(g["h_mat"], g["v0"], num_steps=3)
    assert float(jnp.max(jnp.abs(alpha_diag - g["lanczos_alpha_ref"]))) < 1e-5, "lanczos_alpha mismatch"
    assert float(jnp.max(jnp.abs(beta_diag - g["lanczos_beta_ref"]))) < 1e-5, "lanczos_beta mismatch"

    # 4. Empirical NTK
    ntk = empirical_ntk(g["ntk_j"])
    assert float(jnp.max(jnp.abs(ntk - g["ntk_gram_ref"]))) < 1e-5, "empirical_ntk mismatch"

    print("10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
