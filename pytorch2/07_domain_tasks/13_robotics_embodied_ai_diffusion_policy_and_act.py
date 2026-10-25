"""Robotics Embodied Ai Diffusion Policy and Act — PyTorch 2.x reference implementation.

Topic: 07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act
Golden weights: weights/07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act.safetensors

``python pytorch2/07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act.py`` verifies against the golden checkpoint;
``python pytorch2/07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def temporal_ensemble_actions(action_preds: torch.Tensor, decay: float = 0.1) -> torch.Tensor:
    """Action Chunking with Ensembling (ACT) temporal weighting:
    action_preds: [B, K, D_a] (K overlapping action predictions for current time step)
    w_k = exp(-decay * k)
    Returns: [B, D_a] blended action
    """
    k = action_preds.shape[1]
    steps = torch.arange(k, dtype=action_preds.dtype, device=action_preds.device)
    weights = torch.exp(-decay * steps)  # [K]
    weights = weights / torch.sum(weights)  # [K]
    # Weighted average along dimension 1: [B, K, D_a] * [1, K, 1] -> sum over K -> [B, D_a]
    return torch.sum(action_preds * weights.view(1, -1, 1), dim=1)


def diffusion_policy_ddpm_step(
    action: torch.Tensor,
    noise_pred: torch.Tensor,
    noise_z: torch.Tensor,
    alpha_t: float,
    alpha_bar_t: float,
    sigma_t: float,
) -> torch.Tensor:
    """Single reverse DDPM denoising step for continuous action diffusion:
    action: [B, H, D_a]
    noise_pred: [B, H, D_a]
    noise_z: [B, H, D_a]
    Returns: [B, H, D_a]
    """
    coeff = (1.0 - alpha_t) / ((1.0 - alpha_bar_t) ** 0.5)
    mu = (1.0 / (alpha_t ** 0.5)) * (action - coeff * noise_pred)
    return mu + sigma_t * noise_z


def act_cvae_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """ACT CVAE Gaussian latent regularization loss:
    D_KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    mu, logvar: [B, D_z]
    Returns: scalar mean loss
    """
    kl = -0.5 * torch.sum(1.0 + logvar - mu ** 2 - torch.exp(logvar), dim=-1)
    return kl.mean()


def robot_action_safety_clamp(
    actions: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
) -> torch.Tensor:
    """Clamps continuous multi-joint / 6-DoF delta actions within physical safety envelope.
    actions: [B, H, D_a]
    a_min, a_max: [D_a]
    Returns: [B, H, D_a]
    """
    return torch.clamp(actions, min=a_min.view(1, 1, -1), max=a_max.view(1, 1, -1))


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }

    # 1. Temporal Ensemble Actions
    b, k, d_a = 2, 4, 6
    action_preds = torch.randn(b, k, d_a)
    ensembled_ref = temporal_ensemble_actions(action_preds, decay=0.1)
    tensors["action_preds"] = action_preds
    tensors["ensembled_ref"] = ensembled_ref

    # 2. Diffusion Policy DDPM Step
    h = 8
    action_t = torch.randn(b, h, d_a)
    noise_pred = torch.randn(b, h, d_a) * 0.5
    noise_z = torch.randn(b, h, d_a)
    alpha_t = 0.99
    alpha_bar_t = 0.85
    sigma_t = 0.05
    denoised_ref = diffusion_policy_ddpm_step(
        action_t, noise_pred, noise_z, alpha_t, alpha_bar_t, sigma_t
    )
    tensors["action_t"] = action_t
    tensors["noise_pred"] = noise_pred
    tensors["noise_z"] = noise_z
    tensors["denoised_ref"] = denoised_ref

    # 3. ACT CVAE KL Loss
    d_z = 8
    mu = torch.randn(b, d_z) * 0.5
    logvar = torch.randn(b, d_z) * 0.1
    kl_loss_ref = act_cvae_kl_loss(mu, logvar)
    tensors["cvae_mu"] = mu
    tensors["cvae_logvar"] = logvar
    tensors["kl_loss_ref"] = kl_loss_ref

    # 4. Robot Action Safety Clamp
    raw_actions = torch.randn(b, h, d_a) * 2.0
    a_min = torch.tensor([-1.0, -1.0, -0.5, -2.0, -2.0, -2.0])
    a_max = torch.tensor([1.0, 1.0, 0.5, 2.0, 2.0, 2.0])
    clamped_ref = robot_action_safety_clamp(raw_actions, a_min, a_max)
    tensors["raw_actions"] = raw_actions
    tensors["a_min"] = a_min
    tensors["a_max"] = a_max
    tensors["clamped_ref"] = clamped_ref

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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
