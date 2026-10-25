"""XAI Data Attribution Influence Functions TracIn and TRAK — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak. Run ``python jax/10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def influence_score(
    hessian: jnp.ndarray,
    grad_test: jnp.ndarray,
    grad_train: jnp.ndarray,
    damping: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d = hessian.shape[0]
    h_damped = hessian + damping * jnp.eye(d)
    ihvp = jnp.linalg.solve(h_damped, grad_test)
    inf = -jnp.sum(grad_train * ihvp)
    return ihvp, inf


def tracin_trajectory_attribution(
    train_grads: jnp.ndarray,
    test_grads: jnp.ndarray,
    learning_rates: jnp.ndarray,
) -> jnp.ndarray:
    dot_per_checkpoint = jnp.sum(train_grads * test_grads, axis=-1)
    return jnp.sum(learning_rates * dot_per_checkpoint)


def trak_projected_attribution(
    train_grads: jnp.ndarray,
    test_grad: jnp.ndarray,
    proj_matrix: jnp.ndarray,
) -> jnp.ndarray:
    phi_train = train_grads @ proj_matrix
    phi_test = test_grad @ proj_matrix
    norm_train = jnp.sqrt(jnp.sum(phi_train * phi_train, axis=-1, keepdims=True) + 1e-8)
    norm_test = jnp.sqrt(jnp.sum(phi_test * phi_test) + 1e-8)
    scores = (phi_train @ jnp.expand_dims(phi_test, -1)) / (norm_train * norm_test)
    return jnp.squeeze(scores, -1)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Influence functions
    ihvp, inf_score = influence_score(g["inf_hessian"], g["inf_grad_test"], g["inf_grad_train"], damping=1e-3)
    assert float(jnp.max(jnp.abs(ihvp - g["inf_ihvp_ref"]))) < 1e-5, "Influence IHVP mismatch"
    assert float(jnp.max(jnp.abs(inf_score - g["inf_score_ref"]))) < 1e-5, "Influence score mismatch"

    # TracIn
    tracin = tracin_trajectory_attribution(g["tracin_train_g"], g["tracin_test_g"], g["tracin_lrs"])
    assert float(jnp.max(jnp.abs(tracin - g["tracin_score_ref"]))) < 1e-5, "TracIn mismatch"

    # TRAK
    trak = trak_projected_attribution(g["trak_train_g"], g["trak_test_g"], g["trak_proj"])
    assert float(jnp.max(jnp.abs(trak - g["trak_scores_ref"]))) < 1e-5, "TRAK mismatch"

    print("10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
