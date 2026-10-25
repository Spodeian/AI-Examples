"""Losses: Divergences, Information Bottleneck, and Coding Rate (C03 Topic 09).

Implements fundamental information-theoretic, divergence, and representation objectives:
1. Forward & Reverse KL Divergence + Jensen-Shannon Divergence (Kullback & Leibler 1951).
2. alpha-Renyi Divergence (Renyi 1961).
3. Itakura-Saito Distance (Bregman 1967).
4. Variational Information Bottleneck (VIB) Gaussian Prior KL (Alemi et al. 2016).
5. Maximal Coding Rate Reduction (MCR^2) Total Rate (Yu et al. 2020).
6. Linear Anchor: Standard linear baseline.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file


def compute_kl_forward(
    p: Float[torch.Tensor, "batch dim"],
    q: Float[torch.Tensor, "batch dim"],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Forward KL Divergence: D_KL(P || Q) = sum P * log(P / Q)."""
    p_safe = p.clamp(min=eps)
    q_safe = q.clamp(min=eps)
    return (p_safe * (torch.log(p_safe) - torch.log(q_safe))).sum(dim=-1).mean()


def compute_kl_reverse(
    p: Float[torch.Tensor, "batch dim"],
    q: Float[torch.Tensor, "batch dim"],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reverse KL Divergence: D_KL(Q || P) = sum Q * log(Q / P)."""
    p_safe = p.clamp(min=eps)
    q_safe = q.clamp(min=eps)
    return (q_safe * (torch.log(q_safe) - torch.log(p_safe))).sum(dim=-1).mean()


def compute_jsd(
    p: Float[torch.Tensor, "batch dim"],
    q: Float[torch.Tensor, "batch dim"],
) -> torch.Tensor:
    """Jensen-Shannon Divergence: 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M), where M = 0.5 * (P + Q)."""
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_forward(p, m) + 0.5 * compute_kl_forward(q, m)


def compute_renyi_divergence(
    p: Float[torch.Tensor, "batch dim"],
    q: Float[torch.Tensor, "batch dim"],
    alpha: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """alpha-Renyi Divergence: 1/(alpha - 1) * log sum P^alpha * Q^(1-alpha)."""
    p_safe = p.clamp(min=eps)
    q_safe = q.clamp(min=eps)
    inner = (p_safe.pow(alpha) * q_safe.pow(1.0 - alpha)).sum(dim=-1)
    return (torch.log(inner.clamp(min=eps)) / (alpha - 1.0)).mean()


def compute_itakura_saito(
    x: Float[torch.Tensor, "batch dim"],
    y: Float[torch.Tensor, "batch dim"],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Itakura-Saito Distance: mean(x/y - log(x/y) - 1)."""
    x_safe = x.clamp(min=eps)
    y_safe = y.clamp(min=eps)
    ratio = x_safe / y_safe
    return (ratio - torch.log(ratio) - 1.0).mean()


def compute_vib_kl(
    mu: Float[torch.Tensor, "batch dim"],
    logvar: Float[torch.Tensor, "batch dim"],
) -> torch.Tensor:
    """VIB Gaussian Prior KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))."""
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
    return kl_per_sample.mean()


def compute_mcr2_coding_rate(
    z: Float[torch.Tensor, "batch dim"],
    eps: float = 0.5,
) -> torch.Tensor:
    """MCR^2 Total Coding Rate: 0.5 * log det(I + D / (N * eps^2) * Z^T Z)."""
    n, d = z.shape
    scale = float(d) / (float(n) * (eps ** 2))
    gram = torch.matmul(z.t(), z)
    c = torch.eye(d, device=z.device) + scale * gram
    return 0.5 * torch.logdet(c)


class DivergenceUnifiedModule(nn.Module):
    """Unified module combining divergences and coding rates with a linear anchor."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[DivergenceUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = DivergenceUnifiedModule(in_dim=8, out_dim=4)

    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])

    return module, weights


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/09_losses_divergences_information_bottleneck_and_coding_rate.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    module, weights = load_checkpoint(str(weights_path))
    module.eval()

    test_in = weights["test_input"]
    p_dist = weights["p_dist"]
    q_dist = weights["q_dist"]
    r_alpha = float(weights["renyi_alpha"][0])

    is_x = weights["is_x"]
    is_y = weights["is_y"]

    vib_mu = weights["vib_mu"]
    vib_logvar = weights["vib_logvar"]

    mcr2_z = weights["mcr2_z"]
    mcr2_eps = float(weights["mcr2_eps"][0])

    with torch.no_grad():
        # 1. Linear Anchor Parity
        y_anchor = module(test_in)
        diff_anchor = (y_anchor - weights["test_output"]).abs().max().item()
        assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

        # 2. Forward KL Parity
        loss_fwd_kl = compute_kl_forward(p_dist, q_dist)
        diff_fwd_kl = (loss_fwd_kl - weights["loss_fwd_kl"][0]).abs().item()
        assert diff_fwd_kl < 1e-5, f"Forward KL mismatch: {diff_fwd_kl:.2e}"

        # 3. Reverse KL Parity
        loss_rev_kl = compute_kl_reverse(p_dist, q_dist)
        diff_rev_kl = (loss_rev_kl - weights["loss_rev_kl"][0]).abs().item()
        assert diff_rev_kl < 1e-5, f"Reverse KL mismatch: {diff_rev_kl:.2e}"

        # 4. JSD Parity
        loss_jsd = compute_jsd(p_dist, q_dist)
        diff_jsd = (loss_jsd - weights["loss_jsd"][0]).abs().item()
        assert diff_jsd < 1e-5, f"JSD mismatch: {diff_jsd:.2e}"

        # 5. Renyi Parity
        loss_renyi = compute_renyi_divergence(p_dist, q_dist, r_alpha)
        diff_renyi = (loss_renyi - weights["loss_renyi"][0]).abs().item()
        assert diff_renyi < 1e-5, f"Renyi mismatch: {diff_renyi:.2e}"

        # 6. Itakura-Saito Parity
        loss_is = compute_itakura_saito(is_x, is_y)
        diff_is = (loss_is - weights["loss_is"][0]).abs().item()
        assert diff_is < 1e-5, f"Itakura-Saito mismatch: {diff_is:.2e}"

        # 7. VIB KL Parity
        loss_vib = compute_vib_kl(vib_mu, vib_logvar)
        diff_vib = (loss_vib - weights["loss_vib"][0]).abs().item()
        assert diff_vib < 1e-5, f"VIB mismatch: {diff_vib:.2e}"

        # 8. MCR^2 Coding Rate Parity
        loss_mcr2 = compute_mcr2_coding_rate(mcr2_z, mcr2_eps)
        diff_mcr2 = (loss_mcr2 - weights["loss_mcr2"][0]).abs().item()
        assert diff_mcr2 < 1e-5, f"MCR^2 mismatch: {diff_mcr2:.2e}"

    print(
        f"All PyTorch assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"FwdKL: {diff_fwd_kl:.2e}, RevKL: {diff_rev_kl:.2e}, JSD: {diff_jsd:.2e}, "
        f"Renyi: {diff_renyi:.2e}, IS: {diff_is:.2e}, VIB: {diff_vib:.2e}, MCR2: {diff_mcr2:.2e})"
    )


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/03_building_blocks_and_losses/09_losses_divergences_information_bottleneck_and_coding_rate.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "03_building_blocks_and_losses/09_losses_divergences_information_bottleneck_and_coding_rate"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
