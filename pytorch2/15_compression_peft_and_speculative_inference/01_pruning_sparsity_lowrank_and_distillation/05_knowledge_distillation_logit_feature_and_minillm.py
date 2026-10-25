"""Knowledge Distillation Logit Feature and Minillm — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm
Golden weights: weights/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def hinton_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hinton logit-based knowledge distillation combining cross-entropy and softened KL divergence.

    Args:
        student_logits: Student model logits of shape [batch, classes].
        teacher_logits: Teacher model logits of shape [batch, classes].
        labels: Ground truth class indices of shape [batch].
        temperature: Softening temperature T > 0.
        alpha: Weight balancing factor in [0, 1].

    Returns:
        total_loss: Combined scalar distillation loss.
        kl_div: Unscaled KL divergence D_KL(P_T || P_S).
    """
    # Softened log probabilities
    p_s_log = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)

    # KL(P_T || P_S) = sum P_T * (log P_T - log P_S)
    p_t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
    kl_div = torch.sum(p_t * (p_t_log - p_s_log), dim=-1).mean()

    # Hard cross entropy loss
    ce_loss = F.cross_entropy(student_logits, labels)

    total_loss = (1.0 - alpha) * ce_loss + alpha * (temperature**2) * kl_div
    return total_loss, kl_div


def feature_distillation_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Intermediate feature map hint distillation (FitNets style).

    Args:
        student_feat: Student hidden states of shape [batch, d_student].
        teacher_feat: Teacher hidden states of shape [batch, d_teacher].
        proj_weight: Projection matrix of shape [d_student, d_teacher].

    Returns:
        loss: Mean squared error ||student @ proj - teacher||_2^2.
    """
    projected = student_feat @ proj_weight
    return F.mse_loss(projected, teacher_feat)


def reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """MiniLLM reverse KL divergence for mode-seeking generative distribution matching.

    D_KL(P_S || P_T) = sum P_S * (log P_S - log P_T).

    Args:
        student_logits: Student logits of shape [batch, classes].
        teacher_logits: Teacher logits of shape [batch, classes].
        temperature: Softening temperature T.

    Returns:
        rev_kl: Scalar reverse KL divergence.
    """
    p_s = F.softmax(student_logits / temperature, dim=-1)
    p_s_log = F.log_softmax(student_logits / temperature, dim=-1)
    p_t_log = F.log_softmax(teacher_logits / temperature, dim=-1)

    rev_kl = torch.sum(p_s * (p_s_log - p_t_log), dim=-1).mean()
    return rev_kl


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Batch = 8, Vocab / Classes = 16
    student_logits = torch.randn(8, 16) * 1.5
    teacher_logits = torch.randn(8, 16) * 2.0
    labels = torch.randint(0, 16, (8,))

    # 1. Hinton KD loss
    total_kd, forward_kl = hinton_kd_loss(
        student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5
    )

    # 2. Reverse KL divergence (MiniLLM)
    reverse_kl = reverse_kl_loss(student_logits, teacher_logits, temperature=2.0)

    # 3. Feature map distillation (Student dim 8 -> Teacher dim 16)
    student_feat = torch.randn(8, 8)
    teacher_feat = torch.randn(8, 16)
    proj_weight = torch.randn(8, 16) * 0.2
    feat_loss = feature_distillation_loss(student_feat, teacher_feat, proj_weight)

    loss_metrics = torch.tensor(
        [float(total_kd), float(forward_kl), float(reverse_kl), float(feat_loss)],
        dtype=torch.float32,
    )

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "labels": labels,
        "student_feat": student_feat,
        "teacher_feat": teacher_feat,
        "proj_weight": proj_weight,
        "loss_metrics": loss_metrics,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-5, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
