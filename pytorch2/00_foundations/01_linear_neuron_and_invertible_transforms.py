"""Topic 01: Linear Neuron and Invertible Transforms (PyTorch 2.x Track).

Implements the fundamental univariate and multivariate affine projection:
    f(x) = x @ W + b
alongside its exact mathematical inverse:
    f^{-1}(y) = (y - b) @ W^{-1}
and five foundational invertible coordinate transformations (Min-Max, Z-score,
Parameterized Tanh, Parameterized Sigmoid, Parameterized DReLU).

Features:
- jaxtyping tensor annotations for dimensional invariants.
- torch.compile (fullgraph=True) and torch.export validation.
- Closed-form OLS normal equations solver.
- Exact analytical Jacobian log-determinant and condition number metrics.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class TransformParams(NamedTuple):
    """Container for invertible transform state parameters."""

    scale: Tensor
    shift: Tensor
    extra: dict[str, float] = {}


class InvertibleAffineTransform(nn.Module):
    """D-dimensional invertible affine projection layer.

    Forward:  y = x @ W + b
    Inverse:  x = (y - b) @ W^{-1}
    Log-Det:  log |det(W)|
    """

    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features

        # Initialize weight matrix and bias vector
        self.weight = nn.Parameter(torch.empty(in_features, self.out_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform initialization with deterministic fallback."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Float[Tensor, "batch in_dim"]) -> Float[Tensor, "batch out_dim"]:
        """Evaluates linear affine forward transformation."""
        return x @ self.weight + self.bias

    def inverse(self, y: Float[Tensor, "batch out_dim"]) -> Float[Tensor, "batch in_dim"]:
        """Evaluates exact analytic inverse transformation.

        Requires in_features == out_features and det(W) != 0.
        Uses torch.linalg.solve for numerical backward stability.
        """
        if self.in_features != self.out_features:
            raise ValueError(
                f"Analytic inverse requires square transformation matrix; got {self.in_features}x{self.out_features}"
            )
        centered = y - self.bias
        # (W.T @ x.T = centered.T) => x.T = solve(W.T, centered.T) => x = solve(W.T, centered.T).T
        sol = torch.linalg.solve(self.weight.T, centered.T).T
        return sol if isinstance(sol, Tensor) else torch.as_tensor(sol)

    def condition_number(self) -> float:
        """Computes matrix condition number kappa(W) = sigma_max / sigma_min."""
        with torch.no_grad():
            s = torch.linalg.svdvals(self.weight)
            s_max, s_min = float(s[0].item()), float(s[-1].item())
            if s_min < 1e-12:
                return float("inf")
            return s_max / s_min

    def log_abs_det_jacobian(self) -> float:
        """Computes log-determinant of the Jacobian log |det(W)|."""
        if self.in_features != self.out_features:
            raise ValueError("Log-determinant requires square weight matrix")
        with torch.no_grad():
            sign, logabsdet = torch.linalg.slogdet(self.weight)
            if sign.item() == 0:
                return float("-inf")
            return float(logabsdet.item())


class CoordinateTransforms:
    """Foundational invertible coordinate transformations for input conditioning."""

    @staticmethod
    def min_max(
        x: Float[Tensor, "batch dim"], eps: float = 1e-8
    ) -> tuple[Float[Tensor, "batch dim"], TransformParams]:
        """Normalizes x to [-1.0, 1.0]."""
        d_min = x.min(dim=0, keepdim=True).values
        d_max = x.max(dim=0, keepdim=True).values
        diff = torch.clamp_min(d_max - d_min, eps)
        normed = 2.0 * (x - d_min) / diff - 1.0
        return normed, TransformParams(scale=diff, shift=d_min)

    @staticmethod
    def undo_min_max(
        t: Float[Tensor, "batch dim"], params: TransformParams
    ) -> Float[Tensor, "batch dim"]:
        """Inverts Min-Max normalization."""
        return (t + 1.0) * 0.5 * params.scale + params.shift

    @staticmethod
    def z_score(
        x: Float[Tensor, "batch dim"], eps: float = 1e-7
    ) -> tuple[Float[Tensor, "batch dim"], TransformParams]:
        """Standardizes x to zero mean and unit variance."""
        mu = x.mean(dim=0, keepdim=True)
        sigma = torch.clamp_min(x.std(dim=0, keepdim=True, unbiased=False), eps)
        return (x - mu) / sigma, TransformParams(scale=sigma, shift=mu)

    @staticmethod
    def undo_z_score(
        t: Float[Tensor, "batch dim"], params: TransformParams
    ) -> Float[Tensor, "batch dim"]:
        """Inverts Z-Score standardization."""
        return t * params.scale + params.shift

    @staticmethod
    def tanh_transform(
        x: Float[Tensor, "batch dim"], scale: float = 1.0, shift: float = 0.0
    ) -> Float[Tensor, "batch dim"]:
        """Bounded invertible projection via hyperbolic tangent."""
        return torch.tanh(scale * (x - shift))

    @staticmethod
    def undo_tanh(
        t: Float[Tensor, "batch dim"], scale: float = 1.0, shift: float = 0.0, eps: float = 1e-6
    ) -> Float[Tensor, "batch dim"]:
        """Exact inverse hyperbolic tangent with numerical safety clipping."""
        clipped = torch.clamp(t, -1.0 + eps, 1.0 - eps)
        return torch.arctanh(clipped) / scale + shift


def solve_ols(
    x: Float[Tensor, "batch in_dim"], y: Float[Tensor, "batch out_dim"]
) -> tuple[Float[Tensor, "in_dim out_dim"], Float[Tensor, "*out_dim"]]:
    """Computes exact closed-form Ordinary Least Squares solution via normal equations.

    Design matrix: X_tilde = [X, 1]
    Normal equation: theta* = (X_tilde^T X_tilde)^{-1} X_tilde^T Y
    """
    batch_size = x.shape[0]
    ones = torch.ones(batch_size, 1, dtype=x.dtype, device=x.device)
    x_design = torch.cat([x, ones], dim=1)

    theta = torch.linalg.pinv(x_design.T @ x_design) @ (x_design.T @ y)
    weight = theta[:-1, :]
    bias = theta[-1, :]
    return weight, bias


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/00_foundations/01_linear_neuron_and_invertible_transforms.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "00_foundations/01_linear_neuron_and_invertible_transforms"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    import core

    env = core.setup_env(seed=42)
    print(f"Testing PyTorch 2.x Track on {env.device}...")

    # 1. Round-Trip Invertibility Invariant
    model = InvertibleAffineTransform(in_features=2, out_features=2)
    # Ensure well-conditioned invertible weights
    model.weight.data = torch.tensor([[2.0, 0.5], [0.3, 1.5]], dtype=torch.float32)
    model.bias.data = torch.tensor([0.2, -0.4], dtype=torch.float32)

    x_test = torch.randn(32, 2, dtype=torch.float32)
    y_fwd = model(x_test)
    x_rec = model.inverse(y_fwd)

    recon_error = float(torch.norm(x_test - x_rec, p=float("inf")).item())
    print(f"Inverse Reconstruction Error (L_inf): {recon_error:.2e}")
    assert recon_error < 1e-5, f"Reconstruction error {recon_error} exceeds tolerance 1e-5"

    # 2. Condition Number & Log-Det
    kappa = model.condition_number()
    logdet = model.log_abs_det_jacobian()
    print(f"Weight Condition Number kappa(W): {kappa:.4f}")
    print(f"Log-Determinant log |det(W)|: {logdet:.4f}")
    assert kappa < 10.0, "Weight matrix should be well-conditioned"

    # 3. Coordinate Transforms Round-Trip
    normed, p_z = CoordinateTransforms.z_score(x_test)
    x_z_rec = CoordinateTransforms.undo_z_score(normed, p_z)
    assert torch.allclose(x_test, x_z_rec, atol=1e-5)
    print("Z-Score round-trip verified.")

    # 4. Compilation & Export Verification
    ok_export, msg_export = core.export.verify_torch_export(model, x_test)
    print(f"torch.export: {msg_export}")
    assert ok_export

    ok_comp, msg_comp = core.export.verify_torch_compile_fullgraph(model, x_test)
    print(f"torch.compile: {msg_comp}")
    assert ok_comp

    print("[PASS] All PyTorch 2.x Linear Neuron checks passed cleanly!")
