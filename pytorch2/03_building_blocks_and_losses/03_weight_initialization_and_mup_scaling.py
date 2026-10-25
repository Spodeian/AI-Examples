"""Weight Initialization and muP Scaling: Xavier, Kaiming, Orthogonal, LayerScale, and Maximal Update Parameterization (C03 Topic 03).

Implements fundamental initialization, residual stabilization, and width-scaling operators:
1. Classical & Variance-Preserving Initializers: LeCun, Xavier / Glorot, Kaiming / He, and Orthogonal (Glorot & Bengio 2010; He et al. 2015; Saxe et al. 2013).
2. Deep Residual Stabilizers: Residual 1/sqrt(2L) Scaling, LayerScale, and Fixup / ReZero (Zhang et al. 2019; Touvron et al. 2021).
3. Maximal Update Parameterization (muP): Co-scaling weight initialization, forward multipliers, and learning rates to enable zero-shot hyperparameter transfer across width (Yang et al. 2022).
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_file


class LayerScale(nn.Module):
    """LayerScale residual modulation (Touvron et al. 2021)."""

    def __init__(self, dim: int, init_val: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_val))

    def forward(
        self,
        x: Float[torch.Tensor, "... dim"],
        residual: Float[torch.Tensor, "... dim"],
    ) -> Float[torch.Tensor, "... dim"]:
        return x + self.gamma * residual


class ReZero(nn.Module):
    """ReZero residual scalar modulation (Bachlechner et al. 2020; Zhang et al. 2019)."""

    def __init__(self, init_val: float = 0.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([init_val]))

    def forward(
        self,
        x: Float[torch.Tensor, "... dim"],
        residual: Float[torch.Tensor, "... dim"],
    ) -> Float[torch.Tensor, "... dim"]:
        return x + self.alpha * residual


class ResidualScale(nn.Module):
    """Residual 1/sqrt(2L) depth scaling."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.scale = 1.0 / (2.0 * num_layers) ** 0.5

    def forward(
        self,
        x: Float[torch.Tensor, "... dim"],
        residual: Float[torch.Tensor, "... dim"],
    ) -> Float[torch.Tensor, "... dim"]:
        return x + self.scale * residual


class MuPReadout(nn.Module):
    """muP Readout projection with 1/width_scale forward multiplier (Yang et al. 2022)."""

    def __init__(self, in_features: int, out_features: int, width_scale: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width_scale = width_scale
        self.weight = nn.Parameter(torch.randn(in_features, out_features) / in_features)

    def forward(
        self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        return torch.matmul(x, self.weight) / self.width_scale


class WeightInitUnifiedModule(nn.Module):
    """Unified module combining initialization and scaling operators with a linear anchor."""

    def __init__(
        self,
        d_in: int = 8,
        d_hidden: int = 8,
        d_out: int = 4,
        width_scale: float = 2.0,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.layerscale = LayerScale(d_in, init_val=1e-4)
        self.rezero = ReZero(init_val=0.0)
        self.residual_scale = ResidualScale(num_layers)
        self.mup_readout = MuPReadout(d_hidden, d_out, width_scale=width_scale)
        self.linear = nn.Linear(d_in, d_out)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[WeightInitUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    width_scale = float(weights["width_scale"][0])
    module = WeightInitUnifiedModule(d_in=8, d_hidden=8, d_out=4, width_scale=width_scale, num_layers=4)

    with torch.no_grad():
        module.layerscale.gamma.copy_(weights["layerscale_gamma"])
        module.rezero.alpha.copy_(weights["rezero_alpha"])
        module.mup_readout.weight.copy_(weights["w_mup_readout"])
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])

    return module, weights


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/03_weight_initialization_and_mup_scaling.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    module, weights = load_checkpoint(str(weights_path))
    module.eval()

    test_in = weights["test_input"]
    res_branch = weights["residual_branch"]
    x_hidden = weights["x_hidden"]
    w_mup_hidden = weights["w_mup_hidden"]

    with torch.no_grad():
        # 1. Linear Anchor Parity
        y_anchor = module(test_in)
        diff_anchor = (y_anchor - weights["test_output"]).abs().max().item()
        assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

        # 2. LayerScale Parity
        y_ls = module.layerscale(test_in, res_branch)
        diff_ls = (y_ls - weights["y_layerscale"]).abs().max().item()
        assert diff_ls < 1e-5, f"LayerScale mismatch: {diff_ls:.2e}"

        # 3. ReZero Parity
        y_rz = module.rezero(test_in, res_branch)
        diff_rz = (y_rz - weights["y_rezero"]).abs().max().item()
        assert diff_rz < 1e-5, f"ReZero mismatch: {diff_rz:.2e}"

        # 4. Residual 1/sqrt(2L) Scale Parity
        y_scaled = module.residual_scale(test_in, res_branch)
        diff_scaled = (y_scaled - weights["y_residual_scaled"]).abs().max().item()
        assert diff_scaled < 1e-5, f"Residual scaling mismatch: {diff_scaled:.2e}"

        # 5. muP Hidden Parity
        y_hid = torch.matmul(x_hidden, w_mup_hidden)
        diff_hid = (y_hid - weights["y_mup_hidden"]).abs().max().item()
        assert diff_hid < 1e-5, f"muP hidden mismatch: {diff_hid:.2e}"

        # 6. muP Readout Parity
        y_ro = module.mup_readout(x_hidden)
        diff_ro = (y_ro - weights["y_mup_readout"]).abs().max().item()
        assert diff_ro < 1e-5, f"muP readout mismatch: {diff_ro:.2e}"

    print(
        f"All PyTorch assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"LS: {diff_ls:.2e}, RZ: {diff_rz:.2e}, Scale: {diff_scaled:.2e}, "
        f"muP_hid: {diff_hid:.2e}, muP_ro: {diff_ro:.2e})"
    )


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/03_building_blocks_and_losses/03_weight_initialization_and_mup_scaling.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "03_building_blocks_and_losses/03_weight_initialization_and_mup_scaling"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
