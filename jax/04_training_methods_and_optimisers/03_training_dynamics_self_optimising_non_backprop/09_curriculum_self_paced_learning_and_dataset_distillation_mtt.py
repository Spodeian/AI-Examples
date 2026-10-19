"""JAX / Equinox implementation for Category 04 Topic 23:
Curriculum, Self-Paced Learning, and Dataset Distillation (MTT).

Provides functional, jit-compiled implementations of:
1. Self-Paced Learning (SPL) with hard and linear regularizers.
2. Competence-based automated curriculum scheduling.
3. Dataset distillation via gradient matching.
4. Trajectory matching loss (MTT).
5. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


@jax.jit
def compute_spl_weights_and_loss_jax(
    losses: jnp.ndarray,
    pace_lambda: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes hard and linear SPL weights and weighted losses in JAX."""
    spl_hard_weights = (losses < pace_lambda).astype(jnp.float32)
    spl_linear_weights = jnp.clip(1.0 - losses / pace_lambda, a_min=0.0)

    weighted_loss_hard = jnp.sum(spl_hard_weights * losses) / (jnp.sum(spl_hard_weights) + 1e-8)
    weighted_loss_linear = jnp.sum(spl_linear_weights * losses) / (jnp.sum(spl_linear_weights) + 1e-8)

    return spl_hard_weights, spl_linear_weights, weighted_loss_hard, weighted_loss_linear


@jax.jit
def compute_competence_and_curriculum_mask_jax(
    difficulties: jnp.ndarray,
    step_t: jnp.ndarray,
    t_warmup: jnp.ndarray,
    c0: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes competence function and active curriculum sample mask in JAX."""
    competence = jnp.clip(c0 + (1.0 - c0) * (step_t / t_warmup), a_min=0.0, a_max=1.0)
    active_mask = (difficulties <= competence).astype(jnp.float32)
    return competence, active_mask


@jax.jit
def compute_gradient_matching_loss_jax(
    grad_synth: jnp.ndarray,
    grad_real: jnp.ndarray,
) -> jnp.ndarray:
    """Computes cosine distance gradient matching loss in JAX."""
    dot_prod = jnp.sum(grad_synth * grad_real)
    norm_s = jnp.sqrt(jnp.sum(grad_synth ** 2))
    norm_r = jnp.sqrt(jnp.sum(grad_real ** 2))
    cos_sim = dot_prod / (norm_s * norm_r + 1e-8)
    return 1.0 - cos_sim


@jax.jit
def compute_trajectory_matching_loss_jax(
    theta_start: jnp.ndarray,
    theta_student: jnp.ndarray,
    theta_expert: jnp.ndarray,
) -> jnp.ndarray:
    """Computes trajectory matching loss (MTT) in JAX."""
    num = jnp.sum((theta_student - theta_expert) ** 2)
    denom = jnp.sum((theta_start - theta_expert) ** 2) + 1e-8
    return num / denom


@jax.jit
def anchor_forward_jax(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Forward pass for linear anchor module in JAX."""
    return jnp.matmul(x, w) + b


class CurriculumDistillationUnifiedModuleJAX:
    """JAX unified module for curriculum, self-paced learning, and dataset distillation."""

    def __init__(self, weights_path: Path):
        weights = load_file(str(weights_path))

        self.losses = jnp.array(weights["losses"])
        self.pace_lambda = jnp.array(weights["pace_lambda"])
        self.golden_spl_hard_weights = jnp.array(weights["golden_spl_hard_weights"])
        self.golden_spl_linear_weights = jnp.array(weights["golden_spl_linear_weights"])
        self.golden_weighted_loss_hard = jnp.array(weights["golden_weighted_loss_hard"])
        self.golden_weighted_loss_linear = jnp.array(weights["golden_weighted_loss_linear"])

        self.difficulties = jnp.array(weights["difficulties"])
        self.step_t = jnp.array(weights["step_t"])
        self.t_warmup = jnp.array(weights["t_warmup"])
        self.c0 = jnp.array(weights["c0"])
        self.golden_competence = jnp.array(weights["golden_competence"])
        self.golden_active_curriculum_mask = jnp.array(weights["golden_active_curriculum_mask"])

        self.grad_real = jnp.array(weights["grad_real"])
        self.grad_synth = jnp.array(weights["grad_synth"])
        self.golden_grad_matching_loss = jnp.array(weights["golden_grad_matching_loss"])

        self.theta_start = jnp.array(weights["theta_start"])
        self.theta_student = jnp.array(weights["theta_student"])
        self.theta_expert = jnp.array(weights["theta_expert"])
        self.golden_mtt_loss = jnp.array(weights["golden_mtt_loss"])

        self.anchor_w = jnp.array(weights["anchor_w"])
        self.anchor_b = jnp.array(weights["anchor_b"])
        self.anchor_x = jnp.array(weights["anchor_x"])
        self.golden_anchor_out = jnp.array(weights["golden_anchor_out"])

    def verify_invariants(self, tolerance: float = 1e-5) -> dict[str, float]:
        """Verifies parity against golden reference tensors."""
        h_w, l_w, l_h, l_l = compute_spl_weights_and_loss_jax(self.losses, self.pace_lambda)
        spl_err = max(
            float(jnp.max(jnp.abs(h_w - self.golden_spl_hard_weights))),
            float(jnp.max(jnp.abs(l_w - self.golden_spl_linear_weights))),
            float(jnp.abs(jnp.squeeze(l_h) - jnp.squeeze(self.golden_weighted_loss_hard))),
            float(jnp.abs(jnp.squeeze(l_l) - jnp.squeeze(self.golden_weighted_loss_linear))),
        )

        comp, mask = compute_competence_and_curriculum_mask_jax(
            self.difficulties, self.step_t, self.t_warmup, self.c0
        )
        curriculum_err = max(
            float(jnp.abs(jnp.squeeze(comp) - jnp.squeeze(self.golden_competence))),
            float(jnp.max(jnp.abs(mask - self.golden_active_curriculum_mask))),
        )

        gm_loss = compute_gradient_matching_loss_jax(self.grad_synth, self.grad_real)
        gm_err = float(jnp.abs(jnp.squeeze(gm_loss) - jnp.squeeze(self.golden_grad_matching_loss)))

        mtt = compute_trajectory_matching_loss_jax(self.theta_start, self.theta_student, self.theta_expert)
        mtt_err = float(jnp.abs(jnp.squeeze(mtt) - jnp.squeeze(self.golden_mtt_loss)))

        anchor_out = anchor_forward_jax(self.anchor_x, self.anchor_w, self.anchor_b)
        anchor_err = float(jnp.max(jnp.abs(anchor_out - self.golden_anchor_out)))

        assert spl_err < tolerance, f"SPL error {spl_err} exceeds tolerance {tolerance}"
        assert curriculum_err < tolerance, f"Curriculum error {curriculum_err} exceeds tolerance {tolerance}"
        assert gm_err < tolerance, f"Gradient matching error {gm_err} exceeds tolerance {tolerance}"
        assert mtt_err < tolerance, f"MTT error {mtt_err} exceeds tolerance {tolerance}"
        assert anchor_err < tolerance, f"Anchor error {anchor_err} exceeds tolerance {tolerance}"

        return {
            "spl_err": spl_err,
            "curriculum_err": curriculum_err,
            "gm_err": gm_err,
            "mtt_err": mtt_err,
            "anchor_err": anchor_err,
        }


if __name__ == "__main__":
    ckpt = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/09_curriculum_self_paced_learning_and_dataset_distillation_mtt.safetensors")
    module = CurriculumDistillationUnifiedModuleJAX(ckpt)
    errors = module.verify_invariants()
    print("JAX Topic 23 Invariants Verified:", errors)
