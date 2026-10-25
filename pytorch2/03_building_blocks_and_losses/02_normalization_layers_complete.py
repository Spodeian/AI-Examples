"""Normalization Layers Complete: BatchNorm, LayerNorm, RMSNorm, DyT, adaLN-Zero, and Weight Standardization (C03 Topic 02).

Implements fundamental and modern neural network normalization operators:
1. LayerNorm: Mean-centering and variance scaling across hidden dimensions (Ba et al. 2016).
2. RMSNorm: Root Mean Square Normalization eliminating mean-centering for ~30% throughput speedup (Zhang & Sennrich 2019).
3. Dynamic Tanh (DyT): Reduction-free normalization replacing cross-channel barriers with elementwise nonlinear scaling (Zhu et al. 2025).
4. adaLN-Zero: Adaptive LayerNorm with conditioning vector modulation initialized to zero for stable diffusion transformers (Peebles & Xie 2023).
5. QK-Norm: Query-Key head-wise RMS normalization bounding attention logit growth (Dehghani et al. 2023).
6. Scaled Weight Standardization: Standardizing linear weight matrices across fan-in dimensions for normalizer-free architectures (Brock et al. 2021).
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file


class LayerNorm(nn.Module):
    """Standard Layer Normalization over the innermost feature dimension (Ba et al. 2016)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

    Eliminates mean-centering, reducing memory traffic and GPU reduction overhead.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        # RMS(x) = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma


class DynamicTanh(nn.Module):
    """Dynamic Tanh (DyT) reduction-free normalization (Zhu et al. 2025).

    DyT(x) = gamma * tanh(alpha * x) + beta.
    Eliminates cross-channel reduction barriers entirely, operating strictly elementwise.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with Zero-Initialized modulation (Peebles & Xie 2023)."""

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.proj = nn.Linear(cond_dim, 2 * dim)
        # Initialize projection to zero so modulation starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: Float[torch.Tensor, "batch dim"],
        c: Float[torch.Tensor, "batch cond_dim"],
    ) -> Float[torch.Tensor, "batch dim"]:
        adaln_mod = self.proj(c)
        gamma_mod, beta_mod = torch.split(adaln_mod, self.dim, dim=-1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        base_ln = (x - mean) / torch.sqrt(var + self.eps)
        return (1.0 + gamma_mod) * base_ln + beta_mod


class QKNorm(nn.Module):
    """Query-Key Normalization using RMSNorm (Dehghani et al. 2023)."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self, x: Float[torch.Tensor, "... head_dim"]
    ) -> Float[torch.Tensor, "... head_dim"]:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms


def scaled_weight_standardization(
    weight: Float[torch.Tensor, "out_features in_features"],
    eps: float = 1e-5,
) -> Float[torch.Tensor, "out_features in_features"]:
    """Scales and standardizes weight matrix along fan-in dimension (Brock et al. 2021)."""
    fan_in = weight.shape[1]
    mean = weight.mean(dim=1, keepdim=True)
    var = weight.var(dim=1, keepdim=True, unbiased=False)
    return (weight - mean) / (torch.sqrt(var + eps) * (fan_in ** 0.5))


class NormalizationLayersUnifiedModule(nn.Module):
    """Unified module combining normalization layers with a linear anchor."""

    def __init__(
        self,
        dim: int = 8,
        cond_dim: int = 4,
        out_dim: int = 4,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layernorm = LayerNorm(dim, eps)
        self.rmsnorm = RMSNorm(dim, eps)
        self.dyt = DynamicTanh(dim)
        self.adaln = AdaLNZero(dim, cond_dim, eps)
        self.qknorm = QKNorm(eps)
        self.linear = nn.Linear(dim, out_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[NormalizationLayersUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    eps = float(weights["eps"][0])
    module = NormalizationLayersUnifiedModule(dim=8, cond_dim=4, out_dim=4, eps=eps)

    with torch.no_grad():
        module.layernorm.gamma.copy_(weights["gamma"])
        module.layernorm.beta.copy_(weights["beta"])
        module.rmsnorm.gamma.copy_(weights["gamma"])
        module.dyt.alpha.copy_(weights["dyt_alpha"])
        module.dyt.gamma.copy_(weights["dyt_gamma"])
        module.dyt.beta.copy_(weights["dyt_beta"])
        module.adaln.proj.weight.copy_(weights["w_adaln"].t())
        module.adaln.proj.bias.copy_(weights["b_adaln"])
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])

    return module, weights


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/02_normalization_layers_complete.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} does not exist.")
        return

    module, weights = load_checkpoint(str(weights_path))
    module.eval()

    test_in = weights["test_input"]
    cond = weights["cond_vector"]
    w_mat = weights["w_matrix"]

    with torch.no_grad():
        # 1. Linear Anchor Parity
        y_anchor = module(test_in)
        diff_anchor = (y_anchor - weights["test_output"]).abs().max().item()
        assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

        # 2. LayerNorm Parity
        y_ln = module.layernorm(test_in)
        diff_ln = (y_ln - weights["y_layernorm"]).abs().max().item()
        assert diff_ln < 1e-5, f"LayerNorm mismatch: {diff_ln:.2e}"

        # 3. RMSNorm Parity
        y_rms = module.rmsnorm(test_in)
        diff_rms = (y_rms - weights["y_rmsnorm"]).abs().max().item()
        assert diff_rms < 1e-5, f"RMSNorm mismatch: {diff_rms:.2e}"

        # 4. Dynamic Tanh (DyT) Parity
        y_dyt = module.dyt(test_in)
        diff_dyt = (y_dyt - weights["y_dyt"]).abs().max().item()
        assert diff_dyt < 1e-5, f"DyT mismatch: {diff_dyt:.2e}"

        # 5. adaLN-Zero Parity
        y_adaln = module.adaln(test_in, cond)
        diff_adaln = (y_adaln - weights["y_adaln_zero"]).abs().max().item()
        assert diff_adaln < 1e-5, f"adaLN mismatch: {diff_adaln:.2e}"

        # 6. QK-Norm Parity
        y_qk = module.qknorm(test_in)
        diff_qk = (y_qk - weights["y_qknorm"]).abs().max().item()
        assert diff_qk < 1e-5, f"QKNorm mismatch: {diff_qk:.2e}"

        # 7. Scaled Weight Standardization Parity
        w_std = scaled_weight_standardization(w_mat, eps=float(weights["eps"][0]))
        diff_ws = (w_std - weights["w_standardized"]).abs().max().item()
        assert diff_ws < 1e-5, f"Weight standardization mismatch: {diff_ws:.2e}"

    print(
        f"All PyTorch assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"LN: {diff_ln:.2e}, RMS: {diff_rms:.2e}, DyT: {diff_dyt:.2e}, adaLN: {diff_adaln:.2e}, "
        f"QK: {diff_qk:.2e}, WS: {diff_ws:.2e})"
    )


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/03_building_blocks_and_losses/02_normalization_layers_complete.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "03_building_blocks_and_losses/02_normalization_layers_complete"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
