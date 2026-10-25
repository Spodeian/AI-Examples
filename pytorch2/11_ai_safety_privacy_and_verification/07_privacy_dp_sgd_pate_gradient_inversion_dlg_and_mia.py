"""Privacy DP SGD PATE Gradient Inversion DLG and MIA — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia
Golden weights: weights/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def dp_sgd_clip_and_noise(
    per_sample_grads: torch.Tensor,
    clip_norm: float,
    noise_vec: torch.Tensor,
    noise_multiplier: float,
) -> torch.Tensor:
    """DP-SGD per-sample gradient clipping and Gaussian noise addition (Abadi et al. 2016).
    
    g_tilde = ( sum_i g_i * min(1, C / ||g_i||_2) + sigma * C * xi ) / B
    """
    b = per_sample_grads.shape[0]
    norms = torch.norm(per_sample_grads, p=2, dim=-1, keepdim=True) + 1e-8
    scales = torch.clamp(float(clip_norm) / norms, max=1.0)
    clipped = per_sample_grads * scales
    clipped_sum = torch.sum(clipped, dim=0)
    
    noisy_sum = clipped_sum + float(noise_multiplier) * float(clip_norm) * noise_vec
    return noisy_sum / float(b)


def pate_noisy_aggregation(
    teacher_votes: torch.Tensor,
    laplace_noise: torch.Tensor,
    laplace_scale: float,
) -> torch.Tensor:
    """PATE noisy aggregation with Laplace mechanism (Papernot et al. 2017).
    
    V_tilde = Votes + Laplace(0, b)
    """
    return teacher_votes + float(laplace_scale) * laplace_noise


def dlg_gradient_inversion_loss(
    dummy_grad: torch.Tensor,
    target_grad: torch.Tensor,
) -> torch.Tensor:
    """Deep Leakage from Gradients (DLG) matching loss (Zhu et al. 2019): ||g_dummy - g_target||_2^2."""
    diff = dummy_grad - target_grad
    return torch.sum(diff ** 2)


def lira_membership_score(
    target_loss: torch.Tensor,
    shadow_out_losses: torch.Tensor,
) -> torch.Tensor:
    """Likelihood-Ratio Membership Inference Attack (LiRA — Carlini et al. 2022).
    
    Lambda = (mu_out - loss_target) / (sigma_out + eps)
    High positive score indicates membership in private training set.
    """
    mu_out = torch.mean(shadow_out_losses, dim=-1)
    var_out = torch.mean((shadow_out_losses - mu_out.unsqueeze(-1)) ** 2, dim=-1)
    sigma_out = torch.sqrt(var_out + 1e-8)
    return (mu_out - target_loss) / sigma_out


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. DP-SGD inputs: B=8, D=6
    per_sample_grads = torch.randn(8, 6) * 2.0
    noise_vec = torch.randn(6)
    dp_grad_ref = dp_sgd_clip_and_noise(per_sample_grads, clip_norm=1.0, noise_vec=noise_vec, noise_multiplier=0.5)
    
    # 2. PATE inputs: B=4, C=3
    teacher_votes = torch.tensor([
        [15.0, 3.0, 2.0],
        [4.0, 14.0, 2.0],
        [1.0, 2.0, 17.0],
        [8.0, 9.0, 3.0],
    ])
    laplace_noise = torch.randn(4, 3) * 0.5
    pate_votes_ref = pate_noisy_aggregation(teacher_votes, laplace_noise, laplace_scale=0.5)
    
    # 3. DLG inputs: D=8
    dummy_grad = torch.randn(8)
    target_grad = torch.randn(8)
    dlg_loss_ref = dlg_gradient_inversion_loss(dummy_grad, target_grad)
    
    # 4. LiRA inputs: B=4, K=8 shadow models
    target_loss = torch.tensor([0.2, 1.5, 0.05, 0.8])
    shadow_out_losses = torch.randn(4, 8) * 0.3 + torch.tensor([[1.2], [1.3], [1.1], [0.9]])
    lira_scores_ref = lira_membership_score(target_loss, shadow_out_losses)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "per_sample_grads": per_sample_grads,
        "noise_vec": noise_vec,
        "dp_grad_ref": dp_grad_ref,
        "teacher_votes": teacher_votes,
        "laplace_noise": laplace_noise,
        "pate_votes_ref": pate_votes_ref,
        "dummy_grad": dummy_grad,
        "target_grad": target_grad,
        "dlg_loss_ref": dlg_loss_ref,
        "target_loss": target_loss,
        "shadow_out_losses": shadow_out_losses,
        "lira_scores_ref": lira_scores_ref,
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
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
