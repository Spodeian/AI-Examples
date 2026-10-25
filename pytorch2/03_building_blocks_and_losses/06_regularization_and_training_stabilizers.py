"""Regularization and Training Stabilizers: Dropout, DropPath, Soft-Capping, Label Smoothing, and Router Z-Loss (C03 Topic 06).

Implements fundamental neural network regularization and modern training stabilizers:
1. Inverted Dropout: Preserves activation expectation E[y] = E[x] during training (Srivastava et al. 2014).
2. DropPath (Stochastic Depth): Randomly drops entire residual branches sample-wise (Huang et al. 2016).
3. Logit Soft-Capping: Enforces hyperbolic tangent bounds C * tanh(x / C) preventing attention logit explosion (Gemma / Grok).
4. Label Smoothing Cross-Entropy: Distributes target probability mass over uniform distribution to prevent overconfidence (Szegedy et al. 2016).
5. Router Z-Loss: Penalizes large routing logits in Mixture-of-Experts to stabilize FP16/BF16 training (Fedus et al. 2022).
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file


class InvertedDropout(nn.Module):
    """Inverted Dropout with deterministic mask support for verification."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(
        self,
        x: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        return (x * mask) / (1.0 - self.p)


class DropPath(nn.Module):
    """DropPath / Stochastic Depth (Huang et al. 2016)."""

    def __init__(self, drop_prob: float = 0.25) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(
        self,
        x_identity: Float[torch.Tensor, "batch ..."],
        x_branch: Float[torch.Tensor, "batch ..."],
        mask: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch ..."]:
        scale = 1.0 / (1.0 - self.drop_prob)
        return x_identity + (x_branch * mask) * scale


class LogitSoftCapping(nn.Module):
    """Logit Soft-Capping: C * tanh(x / C)."""

    def __init__(self, cap: float = 30.0) -> None:
        super().__init__()
        self.cap = cap

    def forward(
        self, x: Float[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "..."]:
        return self.cap * torch.tanh(x / self.cap)


def compute_label_smoothing_loss(
    logits: Float[torch.Tensor, "batch num_classes"],
    targets: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Label Smoothing Cross-Entropy (Szegedy et al. 2016)."""
    return F.cross_entropy(logits, targets, label_smoothing=alpha, reduction="mean")


def compute_router_z_loss(
    router_logits: Float[torch.Tensor, "batch num_experts"],
    z_coef: float = 1e-4,
) -> torch.Tensor:
    """Router Z-Loss for MoE stability: (z_coef / B) * sum( (log sum exp(z))^2 ) (Fedus et al. 2022)."""
    logsumexp_z = torch.logsumexp(router_logits, dim=-1)
    return z_coef * torch.mean(logsumexp_z.pow(2))


class RegularizationUnifiedModule(nn.Module):
    """Unified module combining stabilizers with a linear anchor."""

    def __init__(
        self,
        in_dim: int = 8,
        out_dim: int = 4,
        drop_p: float = 0.5,
        droppath_p: float = 0.25,
        softcap: float = 30.0,
    ) -> None:
        super().__init__()
        self.dropout = InvertedDropout(drop_p)
        self.droppath = DropPath(droppath_p)
        self.softcapping = LogitSoftCapping(softcap)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[RegularizationUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = RegularizationUnifiedModule(
        in_dim=8,
        out_dim=4,
        drop_p=float(weights["drop_rate"][0]),
        droppath_p=float(weights["droppath_rate"][0]),
        softcap=float(weights["softcap_val"][0]),
    )

    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])

    return module, weights


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/06_regularization_and_training_stabilizers.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    module, weights = load_checkpoint(str(weights_path))
    module.eval()

    test_in = weights["test_input"]
    x_drop = weights["x_drop"]
    drop_mask = weights["drop_mask"]
    x_id = weights["x_identity"]
    x_br = weights["x_branch"]
    dp_mask = weights["droppath_mask"]
    x_log = weights["x_logits"]
    logits_ls = weights["logits_ls"]
    targets_ls = weights["targets_ls"].to(torch.int64)
    alpha_ls = float(weights["smooth_alpha"][0])
    router_logits = weights["router_logits"]
    z_coef = float(weights["z_coef"][0])

    with torch.no_grad():
        # 1. Linear Anchor Parity
        y_anchor = module(test_in)
        diff_anchor = (y_anchor - weights["test_output"]).abs().max().item()
        assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

        # 2. Inverted Dropout Parity
        y_dp = module.dropout(x_drop, drop_mask)
        diff_dp = (y_dp - weights["y_drop"]).abs().max().item()
        assert diff_dp < 1e-5, f"Dropout mismatch: {diff_dp:.2e}"

        # 3. DropPath Parity
        y_dpath = module.droppath(x_id, x_br, dp_mask)
        diff_dpath = (y_dpath - weights["y_droppath"]).abs().max().item()
        assert diff_dpath < 1e-5, f"DropPath mismatch: {diff_dpath:.2e}"

        # 4. Logit Soft-Capping Parity
        y_sc = module.softcapping(x_log)
        diff_sc = (y_sc - weights["y_softcap"]).abs().max().item()
        assert diff_sc < 1e-5, f"Soft-Capping mismatch: {diff_sc:.2e}"

        # 5. Label Smoothing Loss Parity
        loss_ls = compute_label_smoothing_loss(logits_ls, targets_ls, alpha_ls)
        diff_ls = (loss_ls - weights["loss_ls"][0]).abs().item()
        assert diff_ls < 1e-5, f"Label Smoothing mismatch: {diff_ls:.2e}"

        # 6. Router Z-Loss Parity
        loss_z = compute_router_z_loss(router_logits, z_coef)
        diff_z = (loss_z - weights["loss_z"][0]).abs().item()
        assert diff_z < 1e-5, f"Router Z-Loss mismatch: {diff_z:.2e}"

    print(
        f"All PyTorch assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"Dropout: {diff_dp:.2e}, DropPath: {diff_dpath:.2e}, SoftCap: {diff_sc:.2e}, "
        f"LabelSmooth: {diff_ls:.2e}, RouterZ: {diff_z:.2e})"
    )


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/03_building_blocks_and_losses/06_regularization_and_training_stabilizers.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "03_building_blocks_and_losses/06_regularization_and_training_stabilizers"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
