"""Privacy DP SGD PATE Gradient Inversion DLG and MIA — functional JAX implementation.

Topic: 11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia. Run ``python jax/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def dp_sgd_clip_and_noise(
    per_sample_grads: jnp.ndarray,
    clip_norm: float,
    noise_vec: jnp.ndarray,
    noise_multiplier: float,
) -> jnp.ndarray:
    """DP-SGD per-sample gradient clipping and noise addition."""
    b = per_sample_grads.shape[0]
    norms = jnp.linalg.norm(per_sample_grads, axis=-1, keepdims=True) + 1e-8
    scales = jnp.clip(float(clip_norm) / norms, a_max=1.0)
    clipped = per_sample_grads * scales
    clipped_sum = jnp.sum(clipped, axis=0)
    
    noisy_sum = clipped_sum + float(noise_multiplier) * float(clip_norm) * noise_vec
    return noisy_sum / float(b)


def pate_noisy_aggregation(
    teacher_votes: jnp.ndarray,
    laplace_noise: jnp.ndarray,
    laplace_scale: float,
) -> jnp.ndarray:
    """PATE noisy aggregation with Laplace mechanism."""
    return teacher_votes + float(laplace_scale) * laplace_noise


def dlg_gradient_inversion_loss(
    dummy_grad: jnp.ndarray,
    target_grad: jnp.ndarray,
) -> jnp.ndarray:
    """Deep Leakage from Gradients matching loss."""
    diff = dummy_grad - target_grad
    return jnp.sum(diff ** 2)


def lira_membership_score(
    target_loss: jnp.ndarray,
    shadow_out_losses: jnp.ndarray,
) -> jnp.ndarray:
    """Likelihood-Ratio Membership Inference Attack."""
    mu_out = jnp.mean(shadow_out_losses, axis=-1)
    var_out = jnp.mean((shadow_out_losses - mu_out[:, None]) ** 2, axis=-1)
    sigma_out = jnp.sqrt(var_out + 1e-8)
    return (mu_out - target_loss) / sigma_out


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. DP-SGD
    dp_g = dp_sgd_clip_and_noise(g["per_sample_grads"], 1.0, g["noise_vec"], 0.5)
    assert float(jnp.max(jnp.abs(dp_g - g["dp_grad_ref"]))) < 1e-5
    
    # 3. PATE
    pate_v = pate_noisy_aggregation(g["teacher_votes"], g["laplace_noise"], 0.5)
    assert float(jnp.max(jnp.abs(pate_v - g["pate_votes_ref"]))) < 1e-5
    
    # 4. DLG
    dlg_l = dlg_gradient_inversion_loss(g["dummy_grad"], g["target_grad"])
    assert float(jnp.abs(dlg_l - g["dlg_loss_ref"])) < 1e-5
    
    # 5. LiRA
    lira_s = lira_membership_score(g["target_loss"], g["shadow_out_losses"])
    assert float(jnp.max(jnp.abs(lira_s - g["lira_scores_ref"]))) < 1e-5
    
    print("11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
