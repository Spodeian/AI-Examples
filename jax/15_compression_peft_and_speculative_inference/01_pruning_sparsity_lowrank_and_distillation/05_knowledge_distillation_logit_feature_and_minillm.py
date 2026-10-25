"""Knowledge Distillation Logit Feature and Minillm — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm.
Run ``python jax/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def hinton_kd_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    p_s_log = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    p_t = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    p_t_log = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)

    kl_div = jnp.mean(jnp.sum(p_t * (p_t_log - p_s_log), axis=-1))

    # Cross entropy: -log p_s(label)
    one_hot = jax.nn.one_hot(labels, student_logits.shape[-1])
    s_log = jax.nn.log_softmax(student_logits, axis=-1)
    ce_loss = -jnp.mean(jnp.sum(one_hot * s_log, axis=-1))

    total_loss = (1.0 - alpha) * ce_loss + alpha * (temperature**2) * kl_div
    return total_loss, kl_div


def feature_distillation_loss(
    student_feat: jnp.ndarray,
    teacher_feat: jnp.ndarray,
    proj_weight: jnp.ndarray,
) -> jnp.ndarray:
    projected = student_feat @ proj_weight
    return jnp.mean((projected - teacher_feat) ** 2)


def reverse_kl_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    temperature: float = 2.0,
) -> jnp.ndarray:
    p_s = jax.nn.softmax(student_logits / temperature, axis=-1)
    p_s_log = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    p_t_log = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)

    return jnp.mean(jnp.sum(p_s * (p_s_log - p_t_log), axis=-1))


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Hinton KD loss
    total_kd, fwd_kl = hinton_kd_loss(
        g["student_logits"], g["teacher_logits"], g["labels"], temperature=2.0, alpha=0.5
    )
    assert float(jnp.abs(total_kd - g["loss_metrics"][0])) < 1e-5
    assert float(jnp.abs(fwd_kl - g["loss_metrics"][1])) < 1e-5

    # 3. Reverse KL
    rev_kl = reverse_kl_loss(g["student_logits"], g["teacher_logits"], temperature=2.0)
    assert float(jnp.abs(rev_kl - g["loss_metrics"][2])) < 1e-5

    # 4. Feature loss
    feat_loss = feature_distillation_loss(g["student_feat"], g["teacher_feat"], g["proj_weight"])
    assert float(jnp.abs(feat_loss - g["loss_metrics"][3])) < 1e-5

    print("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
