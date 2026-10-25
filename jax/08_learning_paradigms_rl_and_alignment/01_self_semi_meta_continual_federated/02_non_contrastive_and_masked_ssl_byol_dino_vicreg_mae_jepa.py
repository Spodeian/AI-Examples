"""Non Contrastive and Masked Ssl Byol Dino Vicreg MAE JEPA — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def byol_prediction_loss(p: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    p_norm = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + 1e-12)
    z_norm = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
    cos_sim = jnp.sum(p_norm * z_norm, axis=-1)
    return jnp.expand_dims(jnp.mean(2.0 - 2.0 * cos_sim), axis=0)


def vicreg_loss(
    za: jnp.ndarray,
    zb: jnp.ndarray,
    sim_coeff: float = 1.0,
    std_coeff: float = 1.0,
    cov_coeff: float = 0.04,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> jnp.ndarray:
    b, d = za.shape
    sim_loss = jnp.mean((za - zb) ** 2)

    def var_loss(z: jnp.ndarray) -> jnp.ndarray:
        std = jnp.sqrt(jnp.var(z, axis=0, ddof=0) + eps)
        return jnp.mean(jax.nn.relu(gamma - std))

    std_loss = var_loss(za) + var_loss(zb)

    def cov_loss(z: jnp.ndarray) -> jnp.ndarray:
        z_c = z - jnp.mean(z, axis=0, keepdims=True)
        cov = (z_c.T @ z_c) / (b - 1)
        cov_off = cov - jnp.diag(jnp.diag(cov))
        return jnp.sum(cov_off ** 2) / d

    cov_loss_val = cov_loss(za) + cov_loss(zb)
    return jnp.expand_dims(sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss_val, axis=0)


def dino_loss(
    s_logits: jnp.ndarray,
    t_logits: jnp.ndarray,
    t_center: jnp.ndarray,
    tau_s: float = 0.1,
    tau_t: float = 0.05,
) -> jnp.ndarray:
    p_t = jax.nn.softmax((t_logits - t_center) / tau_t, axis=-1)
    log_p_s = jax.nn.log_softmax(s_logits / tau_s, axis=-1)
    return jnp.expand_dims(jnp.mean(-jnp.sum(p_t * log_p_s, axis=-1)), axis=0)


def koleo_loss(z: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    b = z.shape[0]
    z_norm = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
    # pairwise euclidean distance
    diff = jnp.expand_dims(z_norm, 1) - jnp.expand_dims(z_norm, 0)
    dist_matrix = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)
    diag_mask = jnp.eye(b) * 1e9
    min_dist = jnp.min(dist_matrix + diag_mask, axis=-1)
    return jnp.expand_dims(jnp.mean(-jnp.log(min_dist + eps)), axis=0)


def mae_patch_gather(patches: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    b = patches.shape[0]
    return jnp.stack([patches[i, indices[i]] for i in range(b)], axis=0)


def jepa_feature_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.expand_dims(jnp.mean((pred - target) ** 2), axis=0)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. BYOL loss
    err_byol = float(jnp.max(jnp.abs(byol_prediction_loss(g["byol_p"], g["byol_z"]) - g["byol_loss_ref"])))
    assert err_byol < 1e-5, f"byol loss error {err_byol:.2e}"

    # 3. VICReg loss
    err_vicreg = float(jnp.max(jnp.abs(vicreg_loss(g["vicreg_za"], g["vicreg_zb"]) - g["vicreg_loss_ref"])))
    assert err_vicreg < 1e-5, f"vicreg loss error {err_vicreg:.2e}"

    # 4. DINO loss
    err_dino = float(jnp.max(jnp.abs(dino_loss(g["s_logits"], g["t_logits"], g["t_center"]) - g["dino_loss_ref"])))
    assert err_dino < 1e-5, f"dino loss error {err_dino:.2e}"

    # 5. KoLeo loss
    err_koleo = float(jnp.max(jnp.abs(koleo_loss(g["koleo_z"]) - g["koleo_loss_ref"])))
    assert err_koleo < 1e-5, f"koleo loss error {err_koleo:.2e}"

    # 6. MAE patch gather
    err_mae = float(jnp.max(jnp.abs(mae_patch_gather(g["mae_patches"], g["mae_indices"]) - g["mae_gathered_ref"])))
    assert err_mae < 1e-5, f"mae gather error {err_mae:.2e}"

    # 7. JEPA feature loss
    err_jepa = float(jnp.max(jnp.abs(jepa_feature_loss(g["jepa_pred"], g["jepa_target"]) - g["jepa_loss_ref"])))
    assert err_jepa < 1e-5, f"jepa loss error {err_jepa:.2e}"

    print("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
