"""Discrete Diffusion D3PM, MDLM, and SEDD — functional JAX implementation.

Topic: 05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd. Run ``python jax/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def d3pm_transition_matrix(beta: float, num_classes: int, absorbing: bool = False, mask_idx: int = -1) -> jnp.ndarray:
    if mask_idx < 0:
        mask_idx = num_classes - 1

    eye = jnp.eye(num_classes)
    if not absorbing:
        q = (1.0 - beta) * eye + (beta / num_classes) * jnp.ones((num_classes, num_classes))
    else:
        q = (1.0 - beta) * eye
        q = q.at[:, mask_idx].add(beta)
    return q


def d3pm_forward_marginal(x_0_probs: jnp.ndarray, q_bar: jnp.ndarray) -> jnp.ndarray:
    return x_0_probs @ q_bar


def mdlm_continuous_marginal(x_0_probs: jnp.ndarray, alpha_t: jnp.ndarray, mask_idx: int = 4) -> jnp.ndarray:
    e_mask = jnp.zeros_like(x_0_probs)
    e_mask = e_mask.at[:, :, mask_idx].set(1.0)
    return alpha_t * x_0_probs + (1.0 - alpha_t) * e_mask


def discrete_denoiser(
    x_t: jnp.ndarray,
    t: jnp.ndarray,
    w_x: jnp.ndarray,
    b_x: jnp.ndarray,
    w_t: jnp.ndarray,
    b_t: jnp.ndarray,
    w_out: jnp.ndarray,
    b_out: jnp.ndarray,
) -> jnp.ndarray:
    t_emb = jnp.expand_dims(jax.nn.silu(t @ w_t + b_t), axis=1)
    h = jax.nn.silu(x_t @ w_x + b_x + t_emb)
    logits = h @ w_out + b_out
    return logits


def sedd_score_entropy(score_ratio: jnp.ndarray, target_ratio: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((score_ratio - target_ratio) ** 2).reshape((1,))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Parity checks
    q_uniform = d3pm_transition_matrix(beta=0.1, num_classes=5, absorbing=False)
    assert float(jnp.max(jnp.abs(q_uniform - g["q_uniform"]))) < 1e-5, "q_uniform parity mismatch"

    q_absorbing = d3pm_transition_matrix(beta=0.2, num_classes=5, absorbing=True, mask_idx=4)
    assert float(jnp.max(jnp.abs(q_absorbing - g["q_absorbing"]))) < 1e-5, "q_absorbing parity mismatch"

    x_t_uniform = d3pm_forward_marginal(g["x_0"], g["q_uniform"])
    assert float(jnp.max(jnp.abs(x_t_uniform - g["x_t_uniform"]))) < 1e-5, "x_t_uniform parity mismatch"

    x_t_mdlm = mdlm_continuous_marginal(g["x_0"], g["alpha_t"], mask_idx=4)
    assert float(jnp.max(jnp.abs(x_t_mdlm - g["x_t_mdlm"]))) < 1e-5, "x_t_mdlm parity mismatch"

    logits_pred = discrete_denoiser(x_t_mdlm, g["t"], g["w_x"], g["b_x"], g["w_t"], g["b_t"], g["w_out"], g["b_out"])
    assert float(jnp.max(jnp.abs(logits_pred - g["logits_pred"]))) < 1e-5, "logits_pred parity mismatch"

    score_ratio = jax.nn.softmax(logits_pred, axis=-1)
    assert float(jnp.max(jnp.abs(score_ratio - g["score_ratio"]))) < 1e-5, "score_ratio parity mismatch"

    sedd_loss = sedd_score_entropy(score_ratio, g["x_0"])
    assert float(jnp.max(jnp.abs(sedd_loss - g["sedd_loss"]))) < 1e-5, "sedd_loss parity mismatch"

    print("05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
