"""Direct Preference DPO Ipo KTO SimPO DNO and Spo — functional JAX implementation.

Topic: 08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.
Run ``python jax/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def dpo_loss(
    pi_logps_w: jnp.ndarray,
    pi_logps_l: jnp.ndarray,
    ref_logps_w: jnp.ndarray,
    ref_logps_l: jnp.ndarray,
    beta: float = 0.1,
) -> jnp.ndarray:
    pi_diff = pi_logps_w - pi_logps_l
    ref_diff = ref_logps_w - ref_logps_l
    logits = beta * (pi_diff - ref_diff)
    return jnp.mean(jax.nn.softplus(-logits))


def ipo_loss(
    pi_logps_w: jnp.ndarray,
    pi_logps_l: jnp.ndarray,
    ref_logps_w: jnp.ndarray,
    ref_logps_l: jnp.ndarray,
    beta: float = 0.1,
    tau: float = 0.5,
) -> jnp.ndarray:
    pi_diff = pi_logps_w - pi_logps_l
    ref_diff = ref_logps_w - ref_logps_l
    logits = beta * (pi_diff - ref_diff)
    target = 1.0 / (2.0 * tau)
    return jnp.mean((logits - target) ** 2)


def simpo_loss(
    pi_logps_w: jnp.ndarray,
    pi_logps_l: jnp.ndarray,
    len_w: jnp.ndarray,
    len_l: jnp.ndarray,
    beta: float = 2.0,
    gamma_margin: float = 0.5,
) -> jnp.ndarray:
    norm_w = pi_logps_w / len_w
    norm_l = pi_logps_l / len_l
    logits = beta * (norm_w - norm_l) - gamma_margin
    return jnp.mean(jax.nn.softplus(-logits))


def kto_loss(
    pi_logps: jnp.ndarray,
    ref_logps: jnp.ndarray,
    desirable_flags: jnp.ndarray,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.33,
) -> jnp.ndarray:
    log_ratio = beta * (pi_logps - ref_logps)
    loss_desirable = jax.nn.sigmoid(-log_ratio) * desirable_weight
    loss_undesirable = jax.nn.sigmoid(log_ratio) * undesirable_weight
    per_sample_loss = jnp.where(desirable_flags > 0.5, loss_desirable, loss_undesirable)
    return jnp.mean(per_sample_loss)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"

    # 2. DPO Loss
    dpo_l = dpo_loss(g["pi_logps_w"], g["pi_logps_l"], g["ref_logps_w"], g["ref_logps_l"], beta=0.1)
    err_dpo = float(jnp.abs(dpo_l - g["dpo_loss_ref"]))
    assert err_dpo < 1e-5, f"dpo loss error {err_dpo:.2e}"

    # 3. IPO Loss
    ipo_l = ipo_loss(g["pi_logps_w"], g["pi_logps_l"], g["ref_logps_w"], g["ref_logps_l"], beta=0.1, tau=0.5)
    err_ipo = float(jnp.abs(ipo_l - g["ipo_loss_ref"]))
    assert err_ipo < 1e-5, f"ipo loss error {err_ipo:.2e}"

    # 4. SimPO Loss
    simpo_l = simpo_loss(g["pi_logps_w"], g["pi_logps_l"], g["len_w"], g["len_l"], beta=2.0, gamma_margin=0.5)
    err_simpo = float(jnp.abs(simpo_l - g["simpo_loss_ref"]))
    assert err_simpo < 1e-5, f"simpo loss error {err_simpo:.2e}"

    # 5. KTO Loss
    kto_l = kto_loss(g["pi_logps_unpaired"], g["ref_logps_unpaired"], g["desirable_flags"], beta=0.1)
    err_kto = float(jnp.abs(kto_l - g["kto_loss_ref"]))
    assert err_kto < 1e-5, f"kto loss error {err_kto:.2e}"

    print("08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
