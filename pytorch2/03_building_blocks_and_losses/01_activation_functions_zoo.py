"""Activation Functions Zoo: Rectified, Bounded, Gated, Periodic, and Self-Normalizing (C03 Topic 01).

Implements fundamental modern activation operators:
1. Gated Linear Units: SwiGLU and GeGLU with 8/3 hidden scaling (Shazeer 2020).
2. Periodic Extrapolator: Snake activation x + (1 - cos(2ax)) / (2a) (Ziyin et al. 2020).
3. Self-Normalizing Dynamics: SELU with fixed-point (0, 1) variance preservation (Klambauer et al. 2017).
4. Rectified Linearity & Smooth Approximations: ReLU, LeakyReLU, SiLU / Swish (Nair & Hinton 2010; Hendrycks & Gimpel 2016).
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from safetensors.torch import load_file


class SwiGLUFFN(nn.Module):
    """Gated feed-forward network using SwiGLU activation (Shazeer 2020)."""

    def __init__(
        self,
        w_gate: Float[torch.Tensor, "in_dim hidden_dim"],
        w_up: Float[torch.Tensor, "in_dim hidden_dim"],
        w_down: Float[torch.Tensor, "hidden_dim out_dim"],
    ) -> None:
        super().__init__()
        self.register_buffer("w_gate", w_gate)
        self.register_buffer("w_up", w_up)
        self.register_buffer("w_down", w_down)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        gate = F.silu(torch.matmul(x, self.w_gate))
        up = torch.matmul(x, self.w_up)
        return torch.matmul(gate * up, self.w_down)


class GeGLUFFN(nn.Module):
    """Gated feed-forward network using GeGLU activation (Shazeer 2020)."""

    def __init__(
        self,
        w_gate: Float[torch.Tensor, "in_dim hidden_dim"],
        w_up: Float[torch.Tensor, "in_dim hidden_dim"],
        w_down: Float[torch.Tensor, "hidden_dim out_dim"],
    ) -> None:
        super().__init__()
        self.register_buffer("w_gate", w_gate)
        self.register_buffer("w_up", w_up)
        self.register_buffer("w_down", w_down)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        gate = F.gelu(torch.matmul(x, self.w_gate))
        up = torch.matmul(x, self.w_up)
        return torch.matmul(gate * up, self.w_down)


class SnakeActivation(nn.Module):
    """Periodic activation preserving frequency inductive bias (Ziyin et al. 2020)."""

    def __init__(self, a: float = 2.0) -> None:
        super().__init__()
        self.a = a

    def forward(
        self, x: Float[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "..."]:
        # Snake(x) = x + (1 - cos(2*a*x)) / (2*a)
        return x + (1.0 - torch.cos(2.0 * self.a * x)) / (2.0 * self.a)


class SELUActivation(nn.Module):
    """Self-normalizing scaled exponential linear unit (Klambauer et al. 2017)."""

    def __init__(
        self,
        scale: float = 1.0507009873554804934193349852946,
        alpha: float = 1.6732632423543772848170429916717,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.alpha = alpha

    def forward(
        self, x: Float[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "..."]:
        return torch.where(
            x > 0.0,
            self.scale * x,
            self.scale * self.alpha * (torch.exp(x) - 1.0),
        )


class ActivationZooUnifiedModule(nn.Module):
    """Unified module combining activation operators with linear anchor."""

    def __init__(
        self,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        snake_a: float = 2.0,
        selu_scale: float = 1.0507009873554805,
        selu_alpha: float = 1.6732632423543772,
    ) -> None:
        super().__init__()
        self.swiglu = SwiGLUFFN(w_gate, w_up, w_down)
        self.geglu = GeGLUFFN(w_gate, w_up, w_down)
        self.snake = SnakeActivation(snake_a)
        self.selu = SELUActivation(selu_scale, selu_alpha)
        self.linear = nn.Linear(8, 4)

    def forward(
        self, x: Float[torch.Tensor, "batch in_dim"]
    ) -> Float[torch.Tensor, "batch out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[ActivationZooUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = ActivationZooUnifiedModule(
        w_gate=weights["w_gate"],
        w_up=weights["w_up"],
        w_down=weights["w_down"],
        snake_a=float(weights["snake_a"][0]),
        selu_scale=float(weights["selu_scale"][0]),
        selu_alpha=float(weights["selu_alpha"][0]),
    )
    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])
    return module, weights


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/03_building_blocks_and_losses/01_activation_functions_zoo.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "03_building_blocks_and_losses/01_activation_functions_zoo"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "03_building_blocks_and_losses"
        / "01_activation_functions_zoo.safetensors"
    )
    model, tensors = load_checkpoint(str(ckpt))

    # 1. Linear Anchor Parity
    out_anchor = model(tensors["x_anchor"])
    assert torch.allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5), "Linear anchor mismatch"

    # 2. SwiGLU & GeGLU Parity
    swiglu_res = model.swiglu(tensors["x_gate"])
    assert torch.allclose(swiglu_res, tensors["swiglu_out"], atol=1e-5), "SwiGLU mismatch"
    geglu_res = model.geglu(tensors["x_gate"])
    assert torch.allclose(geglu_res, tensors["geglu_out"], atol=1e-5), "GeGLU mismatch"

    # 3. Snake Activation Parity
    snake_res = model.snake(tensors["x_snake"])
    assert torch.allclose(snake_res, tensors["snake_out"], atol=1e-5), "Snake mismatch"

    # 4. SELU Activation Parity
    selu_res = model.selu(tensors["x_selu"])
    assert torch.allclose(selu_res, tensors["selu_out"], atol=1e-5), "SELU mismatch"

    # 5. Rectified & Smooth Parity
    assert torch.allclose(F.relu(tensors["x_dead"]), tensors["relu_out"], atol=1e-5), "ReLU mismatch"
    assert torch.allclose(F.leaky_relu(tensors["x_dead"], negative_slope=0.1), tensors["leaky_relu_out"], atol=1e-5), "LeakyReLU mismatch"
    assert torch.allclose(F.silu(tensors["x_dead"]), tensors["silu_out"], atol=1e-5), "SiLU mismatch"

    print("PyTorch 2 Activation Functions Zoo successfully verified with zero errors!")
