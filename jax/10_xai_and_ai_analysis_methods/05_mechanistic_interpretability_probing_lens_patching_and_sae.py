"""Mechanistic Interpretability Probing Lens Patching and SAE — functional JAX implementation.

Topic: 10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae. Run ``python jax/10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def logit_lens(
    hidden_states: jnp.ndarray,
    unembed_weight: jnp.ndarray,
    unembed_bias: jnp.ndarray,
) -> jnp.ndarray:
    return hidden_states @ unembed_weight + unembed_bias


def attribution_patching(
    clean_acts: jnp.ndarray,
    corrupt_acts: jnp.ndarray,
    clean_grads: jnp.ndarray,
) -> jnp.ndarray:
    diff = corrupt_acts - clean_acts
    atp_per_dim = diff * clean_grads
    return jnp.sum(atp_per_dim, axis=-1)


def topk_sparse_autoencoder(
    x: jnp.ndarray,
    w_enc: jnp.ndarray,
    b_enc: jnp.ndarray,
    w_dec: jnp.ndarray,
    b_dec: jnp.ndarray,
    k: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_cent = x - b_dec
    pre_act = x_cent @ w_enc + b_enc
    relu_act = jnp.maximum(pre_act, 0.0)
    
    vals, idxs = jax.lax.top_k(relu_act, k)
    row_idx = jnp.arange(x.shape[0])[:, None]
    f_z = jnp.zeros_like(relu_act).at[row_idx, idxs].set(vals)
    x_hat = f_z @ w_dec + b_dec
    return f_z, x_hat


def jumprelu_activation(
    z: jnp.ndarray,
    threshold: jnp.ndarray,
) -> jnp.ndarray:
    mask = jnp.where(z > threshold, 1.0, 0.0)
    return z * mask


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Logit Lens
    logits = logit_lens(g["ll_h"], g["ll_w"], g["ll_b"])
    assert float(jnp.max(jnp.abs(logits - g["ll_logits_ref"]))) < 1e-5, "Logit Lens mismatch"

    # Attribution Patching
    atp = attribution_patching(g["atp_clean"], g["atp_corrupt"], g["atp_grads"])
    assert float(jnp.max(jnp.abs(atp - g["atp_scores_ref"]))) < 1e-5, "Attribution Patching mismatch"

    # Top-K SAE
    f_z, x_hat = topk_sparse_autoencoder(g["sae_x"], g["sae_w_enc"], g["sae_b_enc"], g["sae_w_dec"], g["sae_b_dec"], k=2)
    assert float(jnp.max(jnp.abs(f_z - g["sae_feat_ref"]))) < 1e-5, "SAE features mismatch"
    assert float(jnp.max(jnp.abs(x_hat - g["sae_xhat_ref"]))) < 1e-5, "SAE reconstruction mismatch"

    # JumpReLU
    jump = jumprelu_activation(g["jump_z"], g["jump_thresh"])
    assert float(jnp.max(jnp.abs(jump - g["jump_out_ref"]))) < 1e-5, "JumpReLU mismatch"

    print("10_xai_and_ai_analysis_methods/05_mechanistic_interpretability_probing_lens_patching_and_sae: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
