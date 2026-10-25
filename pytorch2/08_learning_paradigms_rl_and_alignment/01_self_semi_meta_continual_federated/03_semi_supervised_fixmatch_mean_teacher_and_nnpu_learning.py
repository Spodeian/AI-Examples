"""Semi Supervised Fixmatch Mean Teacher and Nnpu Learning — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning
Golden weights: weights/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def mean_teacher_consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Mean Teacher MSE consistency loss between student and EMA teacher softmax probabilities."""
    p_s = F.softmax(student_logits, dim=-1)
    p_t = F.softmax(teacher_logits, dim=-1)
    return F.mse_loss(p_s, p_t).unsqueeze(0)


def fixmatch_thresholded_loss(
    weak_logits: torch.Tensor,
    strong_logits: torch.Tensor,
    tau: float = 0.6,
) -> torch.Tensor:
    """FixMatch pseudo-labeling consistency loss with confidence thresholding."""
    q = F.softmax(weak_logits, dim=-1)
    max_prob, pseudo_label = q.max(dim=-1)
    mask = (max_prob >= tau).float()
    ce = F.cross_entropy(strong_logits, pseudo_label, reduction="none")
    return (mask * ce).mean().unsqueeze(0)


def mixmatch_sharpen(probs: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """MixMatch temperature sharpening: p_k^(1/T) / sum_j p_j^(1/T)."""
    sharpened = probs ** (1.0 / temperature)
    return sharpened / sharpened.sum(dim=-1, keepdim=True)


def nnpu_risk(
    g_pos: torch.Tensor,
    g_unlabeled: torch.Tensor,
    pi_p: float = 0.3,
) -> torch.Tensor:
    """Non-Negative Positive-Unlabeled (nnPU) empirical risk: pi_p * R_p^+ + max(0, R_u^- - pi_p * R_p^-)."""
    ell_p_pos = F.softplus(-g_pos).mean()
    ell_p_neg = F.softplus(g_pos).mean()
    ell_u_neg = F.softplus(g_unlabeled).mean()

    neg_risk = ell_u_neg - pi_p * ell_p_neg
    return (pi_p * ell_p_pos + torch.clamp(neg_risk, min=0.0)).unsqueeze(0)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Linear anchor
    test_input = torch.randn(4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    test_output = linear_anchor(test_input, weight, bias)

    # 2. Mean Teacher
    student_logits = torch.randn(4, 4)
    teacher_logits = torch.randn(4, 4)
    mt_loss_ref = mean_teacher_consistency_loss(student_logits, teacher_logits)

    # 3. FixMatch
    weak_logits = torch.randn(4, 4) * 1.5
    strong_logits = torch.randn(4, 4)
    fixmatch_loss_ref = fixmatch_thresholded_loss(weak_logits, strong_logits, tau=0.6)

    # 4. MixMatch Sharpening
    p_unsharpened = F.softmax(torch.randn(4, 4), dim=-1)
    mixmatch_sharpen_ref = mixmatch_sharpen(p_unsharpened, temperature=0.5)

    # 5. nnPU Risk
    g_pos = torch.randn(4)
    g_unlabeled = torch.randn(6)
    nnpu_risk_ref = nnpu_risk(g_pos, g_unlabeled, pi_p=0.3)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "mt_loss_ref": mt_loss_ref,
        "weak_logits": weak_logits,
        "strong_logits": strong_logits,
        "fixmatch_loss_ref": fixmatch_loss_ref,
        "p_unsharpened": p_unsharpened,
        "mixmatch_sharpen_ref": mixmatch_sharpen_ref,
        "g_pos": g_pos,
        "g_unlabeled": g_unlabeled,
        "nnpu_risk_ref": nnpu_risk_ref,
    }


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
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
