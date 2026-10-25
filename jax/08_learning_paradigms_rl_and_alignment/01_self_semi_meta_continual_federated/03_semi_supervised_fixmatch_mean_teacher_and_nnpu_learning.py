"""Semi Supervised Fixmatch Mean Teacher and Nnpu Learning — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def mean_teacher_consistency_loss(student_logits: jnp.ndarray, teacher_logits: jnp.ndarray) -> jnp.ndarray:
    p_s = jax.nn.softmax(student_logits, axis=-1)
    p_t = jax.nn.softmax(teacher_logits, axis=-1)
    return jnp.expand_dims(jnp.mean((p_s - p_t) ** 2), axis=0)


def fixmatch_thresholded_loss(
    weak_logits: jnp.ndarray,
    strong_logits: jnp.ndarray,
    tau: float = 0.6,
) -> jnp.ndarray:
    q = jax.nn.softmax(weak_logits, axis=-1)
    max_prob = jnp.max(q, axis=-1)
    pseudo_label = jnp.argmax(q, axis=-1)
    mask = (max_prob >= tau).astype(jnp.float32)

    log_p = jax.nn.log_softmax(strong_logits, axis=-1)
    b = weak_logits.shape[0]
    ce = -log_p[jnp.arange(b), pseudo_label]
    return jnp.expand_dims(jnp.mean(mask * ce), axis=0)


def mixmatch_sharpen(probs: jnp.ndarray, temperature: float = 0.5) -> jnp.ndarray:
    sharpened = probs ** (1.0 / temperature)
    return sharpened / jnp.sum(sharpened, axis=-1, keepdims=True)


def nnpu_risk(
    g_pos: jnp.ndarray,
    g_unlabeled: jnp.ndarray,
    pi_p: float = 0.3,
) -> jnp.ndarray:
    ell_p_pos = jnp.mean(jax.nn.softplus(-g_pos))
    ell_p_neg = jnp.mean(jax.nn.softplus(g_pos))
    ell_u_neg = jnp.mean(jax.nn.softplus(g_unlabeled))

    neg_risk = ell_u_neg - pi_p * ell_p_neg
    return jnp.expand_dims(pi_p * ell_p_pos + jnp.maximum(0.0, neg_risk), axis=0)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Mean Teacher
    err_mt = float(jnp.max(jnp.abs(mean_teacher_consistency_loss(g["student_logits"], g["teacher_logits"]) - g["mt_loss_ref"])))
    assert err_mt < 1e-5, f"mean teacher error {err_mt:.2e}"

    # 3. FixMatch
    err_fm = float(jnp.max(jnp.abs(fixmatch_thresholded_loss(g["weak_logits"], g["strong_logits"], tau=0.6) - g["fixmatch_loss_ref"])))
    assert err_fm < 1e-5, f"fixmatch error {err_fm:.2e}"

    # 4. MixMatch Sharpen
    err_mm = float(jnp.max(jnp.abs(mixmatch_sharpen(g["p_unsharpened"], temperature=0.5) - g["mixmatch_sharpen_ref"])))
    assert err_mm < 1e-5, f"mixmatch error {err_mm:.2e}"

    # 5. nnPU Risk
    err_nnpu = float(jnp.max(jnp.abs(nnpu_risk(g["g_pos"], g["g_unlabeled"], pi_p=0.3) - g["nnpu_risk_ref"])))
    assert err_nnpu < 1e-5, f"nnpu error {err_nnpu:.2e}"

    print("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
