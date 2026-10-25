"""Machine Unlearning SISA Influence Scrub NPO and RMU — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu
Golden weights: weights/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def npo_loss(
    logits_theta: torch.Tensor,
    logits_ref: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Negative Preference Optimization (NPO) loss for machine unlearning (Zhang et al. 2024).
    
    L_NPO = (2 / beta) * softplus(beta * log(pi_theta(y|x) / pi_ref(y|x)))
    Saturates gradients when forget probability drops, preventing catastrophic model collapse.
    """
    log_p_theta = F.log_softmax(logits_theta, dim=-1)
    log_p_ref = F.log_softmax(logits_ref, dim=-1)
    
    b_idx = torch.arange(logits_theta.shape[0], device=logits_theta.device)
    log_pi_theta = log_p_theta[b_idx, labels.long()]
    log_pi_ref = log_p_ref[b_idx, labels.long()]
    
    log_ratio = log_pi_theta - log_pi_ref
    loss = (2.0 / beta) * F.softplus(beta * log_ratio)
    return loss.mean()


def rmu_loss(
    h_theta_forget: torch.Tensor,
    h_target_random: torch.Tensor,
    h_theta_retain: torch.Tensor,
    h_ref_retain: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Representation Misdirection for Unlearning (RMU — Li et al. 2024).
    
    Steers forget representations towards random targets while preserving retain representations.
    L_RMU = MSE(h_theta_forget, h_target) + alpha * MSE(h_theta_retain, h_ref_retain)
    """
    l_forget = torch.mean((h_theta_forget - h_target_random) ** 2)
    l_retain = torch.mean((h_theta_retain - h_ref_retain) ** 2)
    return l_forget + float(alpha) * l_retain


def scrub_loss(
    logits_f_theta: torch.Tensor,
    labels_f: torch.Tensor,
    logits_r_theta: torch.Tensor,
    logits_r_ref: torch.Tensor,
    gamma: float = 1.0,
    tau: float = 5.0,
) -> torch.Tensor:
    """SCRUB student-teacher unlearning loss (Kurmanji et al. 2023).
    
    L_SCRUB = -clamp(CE(f_theta, y_f), max=tau) + gamma * KL(pi_ref || pi_theta)_retain
    """
    ce_forget = F.cross_entropy(logits_f_theta, labels_f.long())
    l_unlearn = -torch.clamp(ce_forget, max=tau)
    
    p_ref = F.softmax(logits_r_ref, dim=-1)
    log_p_ref = F.log_softmax(logits_r_ref, dim=-1)
    log_p_theta = F.log_softmax(logits_r_theta, dim=-1)
    kl_retain = torch.sum(p_ref * (log_p_ref - log_p_theta), dim=-1).mean()
    
    return l_unlearn + float(gamma) * kl_retain


def sisa_ensemble_predict(shard_logits: torch.Tensor) -> torch.Tensor:
    """SISA multi-shard probability aggregation across S isolated models (Bourtoule et al. 2021).
    
    Input shape: [S, B, C] -> Output shape: [B, C]
    """
    probs = F.softmax(shard_logits, dim=-1)
    return torch.mean(probs, dim=0)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. NPO inputs: B=8, C=5
    logits_theta = torch.randn(8, 5)
    logits_ref = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    npo_ref = npo_loss(logits_theta, logits_ref, labels, beta=0.2)
    
    # 2. RMU inputs: B_f=4, B_r=4, D=8
    h_theta_forget = torch.randn(4, 8)
    h_target_random = torch.randn(4, 8)
    h_theta_retain = torch.randn(4, 8)
    h_ref_retain = torch.randn(4, 8)
    rmu_ref = rmu_loss(h_theta_forget, h_target_random, h_theta_retain, h_ref_retain, alpha=1.5)
    
    # 3. SCRUB inputs: B=6, C=4
    logits_f_theta = torch.randn(6, 4)
    labels_f = torch.randint(0, 4, (6,))
    logits_r_theta = torch.randn(6, 4)
    logits_r_ref = torch.randn(6, 4)
    scrub_ref = scrub_loss(logits_f_theta, labels_f, logits_r_theta, logits_r_ref, gamma=1.2, tau=4.0)
    
    # 4. SISA inputs: S=3 shards, B=4, C=3
    shard_logits = torch.randn(3, 4, 3)
    sisa_ref = sisa_ensemble_predict(shard_logits)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "logits_theta": logits_theta,
        "logits_ref": logits_ref,
        "labels": labels,
        "npo_ref": npo_ref,
        "h_theta_forget": h_theta_forget,
        "h_target_random": h_target_random,
        "h_theta_retain": h_theta_retain,
        "h_ref_retain": h_ref_retain,
        "rmu_ref": rmu_ref,
        "logits_f_theta": logits_f_theta,
        "labels_f": labels_f,
        "logits_r_theta": logits_r_theta,
        "logits_r_ref": logits_r_ref,
        "scrub_ref": scrub_ref,
        "shard_logits": shard_logits,
        "sisa_ref": sisa_ref,
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
