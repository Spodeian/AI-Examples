"""Discrete Diffusion D3PM, MDLM, and SEDD — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd
Golden weights: weights/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.safetensors

``python pytorch2/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/08_discrete_diffusion_d3pm_mdlm_and_sedd"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def d3pm_transition_matrix(beta: float, num_classes: int, absorbing: bool = False, mask_idx: int = -1) -> torch.Tensor:
    """Constructs categorical transition matrix Q_t of shape [K, K].
    
    If absorbing=False, constructs uniform noise transition:
        Q_ij = (1 - beta) * delta_ij + beta / K
    If absorbing=True, constructs absorbing state (masking) transition:
        Q_ij = (1 - beta) * delta_ij + beta * delta_{j, mask_idx}
    """
    if mask_idx < 0:
        mask_idx = num_classes - 1

    eye = torch.eye(num_classes)
    if not absorbing:
        q = (1.0 - beta) * eye + (beta / num_classes) * torch.ones(num_classes, num_classes)
    else:
        q = (1.0 - beta) * eye
        q[:, mask_idx] += beta
    return q


def d3pm_forward_marginal(x_0_probs: torch.Tensor, q_bar: torch.Tensor) -> torch.Tensor:
    """Computes marginal categorical probabilities q(x_t | x_0) = x_0 * Q_bar.
    
    x_0_probs has shape [B, L, K], q_bar has shape [K, K].
    Returns x_t_probs of shape [B, L, K].
    """
    return x_0_probs @ q_bar


def mdlm_continuous_marginal(x_0_probs: torch.Tensor, alpha_t: torch.Tensor, mask_idx: int = 4) -> torch.Tensor:
    """Continuous-time masked diffusion marginal: q(x_t | x_0) = alpha_t * x_0 + (1 - alpha_t) * e_mask.
    
    alpha_t has shape [B, 1, 1], x_0_probs has shape [B, L, K].
    """
    e_mask = torch.zeros_like(x_0_probs)
    e_mask[:, :, mask_idx] = 1.0
    return alpha_t * x_0_probs + (1.0 - alpha_t) * e_mask


def discrete_denoiser(
    x_t: torch.Tensor,
    t: torch.Tensor,
    w_x: torch.Tensor,
    b_x: torch.Tensor,
    w_t: torch.Tensor,
    b_t: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> torch.Tensor:
    """Scores categorical tokens x_t conditioned on continuous time t.
    
    x_t has shape [B, L, K], t has shape [B, 1].
    Returns predicted logits of shape [B, L, K].
    """
    t_emb = F.silu(t @ w_t + b_t).unsqueeze(1)  # [B, 1, D_h]
    h = F.silu(x_t @ w_x + b_x + t_emb)         # [B, L, D_h]
    logits = h @ w_out + b_out                  # [B, L, K]
    return logits


def sedd_score_entropy(score_ratio: torch.Tensor, target_ratio: torch.Tensor) -> torch.Tensor:
    """Score entropy discrete loss evaluating ratio matching: mean((s - s*)^2)."""
    return torch.mean((score_ratio - target_ratio) ** 2).reshape(1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Discrete diffusion setup: B=2, L=4, K=5 (4 tokens + 1 mask token at index 4)
    B, L, K = 2, 4, 5
    token_indices = torch.randint(0, 4, (B, L))
    x_0 = F.one_hot(token_indices, num_classes=K).float()
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float32)
    alpha_t = torch.tensor([[[0.75]], [[0.25]]], dtype=torch.float32)

    # Transition matrices
    q_uniform = d3pm_transition_matrix(beta=0.1, num_classes=K, absorbing=False)
    q_absorbing = d3pm_transition_matrix(beta=0.2, num_classes=K, absorbing=True, mask_idx=4)

    # Forward marginals
    x_t_uniform = d3pm_forward_marginal(x_0, q_uniform)
    x_t_mdlm = mdlm_continuous_marginal(x_0, alpha_t, mask_idx=4)

    # Denoising network
    D_h = 8
    w_x = torch.randn(K, D_h) * 0.2
    b_x = torch.randn(D_h) * 0.1
    w_t = torch.randn(1, D_h) * 0.2
    b_t = torch.randn(D_h) * 0.1
    w_out = torch.randn(D_h, K) * 0.2
    b_out = torch.randn(K) * 0.1

    logits_pred = discrete_denoiser(x_t_mdlm, t, w_x, b_x, w_t, b_t, w_out, b_out)

    # SEDD score entropy
    score_ratio = F.softmax(logits_pred, dim=-1)
    target_ratio = x_0
    sedd_loss = sedd_score_entropy(score_ratio, target_ratio)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_0": x_0,
        "t": t,
        "alpha_t": alpha_t,
        "q_uniform": q_uniform,
        "q_absorbing": q_absorbing,
        "x_t_uniform": x_t_uniform,
        "x_t_mdlm": x_t_mdlm,
        "w_x": w_x,
        "b_x": b_x,
        "w_t": w_t,
        "b_t": b_t,
        "w_out": w_out,
        "b_out": b_out,
        "logits_pred": logits_pred,
        "score_ratio": score_ratio,
        "sedd_loss": sedd_loss,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous().cpu() for k, v in build_golden_tensors().items()}
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
