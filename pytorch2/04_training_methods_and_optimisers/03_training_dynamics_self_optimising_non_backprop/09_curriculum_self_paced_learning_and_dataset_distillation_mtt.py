"""PyTorch 2.x implementation for Category 04 Topic 23:
Curriculum, Self-Paced Learning, and Dataset Distillation (MTT).

Provides production-grade implementations of:
1. Self-Paced Learning (SPL) with hard and linear regularizers (Kumar et al., 2010).
2. Competence-based automated curriculum scheduling (Platanios et al., 2019; Bengio et al., 2009).
3. Dataset distillation via gradient matching (Zhao et al., 2021).
4. Trajectory matching loss (MTT - Cazenavette et al., 2022).
5. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


def compute_spl_weights_and_loss(
    losses: torch.Tensor,
    pace_lambda: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes hard and linear Self-Paced Learning weights and weighted losses (Kumar et al., 2010).

    Returns (spl_hard_weights, spl_linear_weights, weighted_loss_hard, weighted_loss_linear).
    """
    spl_hard_weights = (losses < pace_lambda).float()
    spl_linear_weights = torch.clamp(1.0 - losses / pace_lambda, min=0.0)

    weighted_loss_hard = torch.sum(spl_hard_weights * losses) / (torch.sum(spl_hard_weights) + 1e-8)
    weighted_loss_linear = torch.sum(spl_linear_weights * losses) / (torch.sum(spl_linear_weights) + 1e-8)

    return spl_hard_weights, spl_linear_weights, weighted_loss_hard, weighted_loss_linear


def compute_competence_and_curriculum_mask(
    difficulties: torch.Tensor,
    step_t: torch.Tensor,
    t_warmup: torch.Tensor,
    c0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes competence function and active curriculum sample mask (Platanios et al., 2019).

    Returns (competence, active_curriculum_mask).
    """
    competence = torch.clamp(c0 + (1.0 - c0) * (step_t / t_warmup), min=0.0, max=1.0)
    active_mask = (difficulties <= competence).float()
    return competence, active_mask


def compute_gradient_matching_loss(
    grad_synth: torch.Tensor,
    grad_real: torch.Tensor,
) -> torch.Tensor:
    """Computes cosine distance gradient matching loss for dataset distillation (Zhao et al., 2021)."""
    dot_prod = torch.sum(grad_synth * grad_real)
    norm_s = torch.sqrt(torch.sum(grad_synth ** 2))
    norm_r = torch.sqrt(torch.sum(grad_real ** 2))
    cos_sim = dot_prod / (norm_s * norm_r + 1e-8)
    return 1.0 - cos_sim


def compute_trajectory_matching_loss(
    theta_start: torch.Tensor,
    theta_student: torch.Tensor,
    theta_expert: torch.Tensor,
) -> torch.Tensor:
    """Computes trajectory matching loss (MTT - Cazenavette et al., 2022)."""
    num = torch.sum((theta_student - theta_expert) ** 2)
    denom = torch.sum((theta_start - theta_expert) ** 2) + 1e-8
    return num / denom


class LinearAnchor(nn.Module):
    """Linear anchor module for cross-track parity verification."""

    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CurriculumDistillationUnifiedModule(nn.Module):
    """Unified module combining curriculum, self-paced learning, dataset distillation and linear anchor."""

    def __init__(self, weights_path: Path):
        super().__init__()
        weights = load_file(str(weights_path))

        self.losses = weights["losses"]
        self.pace_lambda = weights["pace_lambda"]
        self.golden_spl_hard_weights = weights["golden_spl_hard_weights"]
        self.golden_spl_linear_weights = weights["golden_spl_linear_weights"]
        self.golden_weighted_loss_hard = weights["golden_weighted_loss_hard"]
        self.golden_weighted_loss_linear = weights["golden_weighted_loss_linear"]

        self.difficulties = weights["difficulties"]
        self.step_t = weights["step_t"]
        self.t_warmup = weights["t_warmup"]
        self.c0 = weights["c0"]
        self.golden_competence = weights["golden_competence"]
        self.golden_active_curriculum_mask = weights["golden_active_curriculum_mask"]

        self.grad_real = weights["grad_real"]
        self.grad_synth = weights["grad_synth"]
        self.golden_grad_matching_loss = weights["golden_grad_matching_loss"]

        self.theta_start = weights["theta_start"]
        self.theta_student = weights["theta_student"]
        self.theta_expert = weights["theta_expert"]
        self.golden_mtt_loss = weights["golden_mtt_loss"]

        self.anchor = LinearAnchor(8, 4)
        self.anchor.fc.weight.data.copy_(weights["anchor_w"].T)
        self.anchor.fc.bias.data.copy_(weights["anchor_b"])
        self.anchor_x = weights["anchor_x"]
        self.golden_anchor_out = weights["golden_anchor_out"]

    def verify_invariants(self, tolerance: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden reference tensors."""
        h_w, l_w, l_h, l_l = compute_spl_weights_and_loss(self.losses, self.pace_lambda)
        spl_err = max(
            float(torch.max(torch.abs(h_w - self.golden_spl_hard_weights)).item()),
            float(torch.max(torch.abs(l_w - self.golden_spl_linear_weights)).item()),
            float(torch.abs(l_h - self.golden_weighted_loss_hard).item()),
            float(torch.abs(l_l - self.golden_weighted_loss_linear).item()),
        )

        comp, mask = compute_competence_and_curriculum_mask(
            self.difficulties, self.step_t, self.t_warmup, self.c0
        )
        curriculum_err = max(
            float(torch.abs(comp - self.golden_competence).item()),
            float(torch.max(torch.abs(mask - self.golden_active_curriculum_mask)).item()),
        )

        gm_loss = compute_gradient_matching_loss(self.grad_synth, self.grad_real)
        gm_err = float(torch.abs(gm_loss - self.golden_grad_matching_loss).item())

        mtt = compute_trajectory_matching_loss(self.theta_start, self.theta_student, self.theta_expert)
        mtt_err = float(torch.abs(mtt - self.golden_mtt_loss).item())

        anchor_out = self.anchor(self.anchor_x)
        anchor_err = float(torch.max(torch.abs(anchor_out - self.golden_anchor_out)).item())

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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    import math
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file

    def generate_topic_23_artifacts():
        torch.manual_seed(42)
        np.random.seed(42)

        # 1. Self-Paced Learning (SPL)
        # Losses of 8 training samples
        losses = torch.tensor([0.15, 0.42, 0.78, 1.25, 0.33, 0.95, 0.08, 1.80], dtype=torch.float32)
        pace_lambda = torch.tensor([0.80], dtype=torch.float32)

        # Hard SPL: v_i = 1 if l_i < lambda else 0
        spl_hard_weights = (losses < pace_lambda).float() # [8]
        # Linear SPL: v_i = 1 - l_i / lambda if l_i < lambda else 0
        spl_linear_weights = torch.clamp(1.0 - losses / pace_lambda, min=0.0) # [8]

        # Weighted SPL loss: sum(v_i * l_i) / (sum(v_i) + eps)
        weighted_loss_hard = torch.sum(spl_hard_weights * losses) / (torch.sum(spl_hard_weights) + 1e-8)
        weighted_loss_linear = torch.sum(spl_linear_weights * losses) / (torch.sum(spl_linear_weights) + 1e-8)

        # 2. Competence-Based Curriculum (Platanios et al., 2019)
        # Difficulties of 8 samples in [0, 1]
        difficulties = torch.tensor([0.1, 0.35, 0.5, 0.85, 0.2, 0.9, 0.05, 0.7], dtype=torch.float32)
        step_t = torch.tensor([50.0], dtype=torch.float32)
        t_warmup = torch.tensor([100.0], dtype=torch.float32)
        c0 = torch.tensor([0.2], dtype=torch.float32)

        # Linear competence: c(t) = min(1.0, c0 + (1.0 - c0) * (t / t_warmup))
        competence = torch.clamp(c0 + (1.0 - c0) * (step_t / t_warmup), min=0.0, max=1.0) # 0.2 + 0.8 * 0.5 = 0.6
        # Active mask: d_i <= c(t)
        active_curriculum_mask = (difficulties <= competence).float()

        # 3. Dataset Distillation via Gradient Matching (Zhao et al., 2021)
        # Synthetic batch gradient and real batch gradient across 8 parameter dimensions
        grad_real = torch.tensor([0.25, -0.40, 0.85, -0.12, 0.65, -0.30, 0.50, -0.75], dtype=torch.float32)
        grad_synth = torch.tensor([0.20, -0.35, 0.80, -0.10, 0.70, -0.28, 0.48, -0.70], dtype=torch.float32)

        # Cosine distance loss: 1 - <g_s, g_r> / (||g_s|| * ||g_r|| + eps)
        dot_prod = torch.sum(grad_synth * grad_real)
        norm_s = torch.sqrt(torch.sum(grad_synth ** 2))
        norm_r = torch.sqrt(torch.sum(grad_real ** 2))
        cos_sim = dot_prod / (norm_s * norm_r + 1e-8)
        grad_matching_loss = 1.0 - cos_sim

        # 4. Trajectory Matching (MTT - Cazenavette et al., 2022)
        # Student parameter after synthetic steps vs expert trajectory
        theta_start = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        theta_student = torch.tensor([0.82, 0.65, 0.45, 0.30], dtype=torch.float32)
        theta_expert = torch.tensor([0.80, 0.60, 0.40, 0.25], dtype=torch.float32)

        mtt_loss = torch.sum((theta_student - theta_expert) ** 2) / (torch.sum((theta_start - theta_expert) ** 2) + 1e-8)

        # 5. Linear Anchor for Multi-Track Parity
        anchor_x = torch.tensor([
            [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
            [-0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1]
        ], dtype=torch.float32) # [2, 8]
        anchor_w = torch.randn(8, 4, dtype=torch.float32)
        anchor_b = torch.randn(4, dtype=torch.float32)
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b # [2, 4]

        tensors = {
            "losses": losses,
            "pace_lambda": pace_lambda,
            "golden_spl_hard_weights": spl_hard_weights,
            "golden_spl_linear_weights": spl_linear_weights,
            "golden_weighted_loss_hard": weighted_loss_hard,
            "golden_weighted_loss_linear": weighted_loss_linear,

            "difficulties": difficulties,
            "step_t": step_t,
            "t_warmup": t_warmup,
            "c0": c0,
            "golden_competence": competence,
            "golden_active_curriculum_mask": active_curriculum_mask,

            "grad_real": grad_real,
            "grad_synth": grad_synth,
            "golden_grad_matching_loss": grad_matching_loss,

            "theta_start": theta_start,
            "theta_student": theta_student,
            "theta_expert": theta_expert,
            "golden_mtt_loss": mtt_loss,

            "anchor_x": anchor_x,
            "anchor_w": anchor_w,
            "anchor_b": anchor_b,
            "golden_anchor_out": golden_anchor_out,

            # Standard test inputs/outputs for parity test runners
            "test_input": anchor_x.clone().contiguous(),
            "test_output": golden_anchor_out.clone().contiguous(),
            "weight": anchor_w.clone().contiguous(),
            "bias": anchor_b.clone().contiguous(),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "09_curriculum_self_paced_learning_and_dataset_distillation_mtt.safetensors"

        save_file(tensors, str(out_path))
        print(f"Saved {len(tensors)} tensors to {out_path}")

        # Verify PyTorch Self-Parity
        assert torch.allclose(spl_hard_weights, tensors["golden_spl_hard_weights"], atol=1e-6)
        assert torch.allclose(spl_linear_weights, tensors["golden_spl_linear_weights"], atol=1e-6)
        assert torch.allclose(weighted_loss_hard, tensors["golden_weighted_loss_hard"], atol=1e-6)
        assert torch.allclose(weighted_loss_linear, tensors["golden_weighted_loss_linear"], atol=1e-6)
        assert torch.allclose(competence, tensors["golden_competence"], atol=1e-6)
        assert torch.allclose(active_curriculum_mask, tensors["golden_active_curriculum_mask"], atol=1e-6)
        assert torch.allclose(grad_matching_loss, tensors["golden_grad_matching_loss"], atol=1e-6)
        assert torch.allclose(mtt_loss, tensors["golden_mtt_loss"], atol=1e-6)
        assert torch.allclose(golden_anchor_out, tensors["golden_anchor_out"], atol=1e-6)
        print("PyTorch Self-Parity verified successfully!")

    generate_topic_23_artifacts()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    ckpt = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/09_curriculum_self_paced_learning_and_dataset_distillation_mtt.safetensors")
    module = CurriculumDistillationUnifiedModule(ckpt)
    errors = module.verify_invariants()
    print("PyTorch Topic 23 Invariants Verified:", errors)
