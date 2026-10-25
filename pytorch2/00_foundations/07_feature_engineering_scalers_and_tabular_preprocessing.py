"""PyTorch 2.x Tabular Preprocessing, Feature Scalers & Serving Weight Folding.

Topic 07: 07_feature_engineering_scalers_and_tabular_preprocessing (Category 00: Foundations)

Theoretical Overview:
Tabular data preprocessing prepares heterogeneous raw inputs (differing in scale, skewness,
and cardinality) for gradient descent optimization:
1. Continuous Scalers:
   - StandardScaler: z = (x - mu) / (sigma + eps), ensures zero-mean unit-variance.
   - RobustScaler: z = (x - median) / (IQR + eps), invariant to heavy-tailed outliers.
   - Yeo-Johnson Transform: Parametric monotonic power transform handling both positive and
     negative values to minimize skewness and gaussianize feature distributions:
         psi(lambda, x) = ((x + 1)^lambda - 1) / lambda  for x >= 0 (lambda != 0)
         psi(lambda, x) = -((-x + 1)^(2 - lambda) - 1) / (2 - lambda) for x < 0 (lambda != 2)
2. Categorical Target Encoding (Micci-Barreca 2001):
   Shrinks class conditional target means toward global target prior:
       S(c) = lambda(n_c) * y_c + (1 - lambda(n_c)) * y_global
       lambda(n_c) = n_c / (n_c + m)
   where m is smoothing pseudo-count, mitigating target leakage on rare categories.
3. Serving-Time Weight Folding (Graph Fusion):
   Standardization and linear layers commute algebraically:
       y = ((x - mu) / sigma) @ W_head + b_head
         = x @ diag(1 / sigma) @ W_head + (b_head - (mu / sigma) @ W_head)
         = x @ W_fused + b_fused
   Folding the preprocessing stage directly into weights eliminates separate runtime normalization,
   reducing memory bandwidth and kernel launch latency in production.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class StandardScaler(nn.Module):
    """Zero-leakage continuous feature standardizer with exact invertible reconstruction."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim), persistent=True)
        self.register_buffer("std", torch.ones(dim), persistent=True)

    def fit(self, x: Float[Tensor, "batch dim"]) -> StandardScaler:
        with torch.no_grad():
            self.mean.copy_(torch.mean(x, dim=0))
            self.std.copy_(torch.std(x, dim=0, unbiased=False) + self.eps)
        return self

    def transform(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        return (x - self.mean) / self.std

    def inverse_transform(self, z: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        return z * self.std + self.mean

    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        return self.transform(x)


class RobustScaler(nn.Module):
    """Outlier-resilient scaler utilizing median and interquartile range (IQR)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("median", torch.zeros(dim), persistent=True)
        self.register_buffer("iqr", torch.ones(dim), persistent=True)

    def fit(self, x: Float[Tensor, "batch dim"]) -> RobustScaler:
        with torch.no_grad():
            med = torch.quantile(x, q=0.5, dim=0)
            q25 = torch.quantile(x, q=0.25, dim=0)
            q75 = torch.quantile(x, q=0.75, dim=0)
            self.median.copy_(med)
            self.iqr.copy_((q75 - q25) + self.eps)
        return self

    def transform(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        return (x - self.median) / self.iqr

    def inverse_transform(self, z: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        return z * self.iqr + self.median


def yeo_johnson_transform(x: Tensor, lmbda: float = 0.5) -> Tensor:
    """Evaluates the Yeo-Johnson power transformation."""
    out = torch.zeros_like(x)
    pos = x >= 0
    neg = ~pos

    if abs(lmbda) > 1e-6:
        out[pos] = ((x[pos] + 1.0) ** lmbda - 1.0) / lmbda
    else:
        out[pos] = torch.log(x[pos] + 1.0)

    if abs(lmbda - 2.0) > 1e-6:
        out[neg] = -((-x[neg] + 1.0) ** (2.0 - lmbda) - 1.0) / (2.0 - lmbda)
    else:
        out[neg] = -torch.log(-x[neg] + 1.0)

    return out


class FusedTabularLinear(nn.Module):
    """Linear module with folded preprocessing weights for serving-time graph fusion."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    @classmethod
    def from_scaled_head(
        cls,
        mean: Tensor,
        std: Tensor,
        w_head: Tensor,
        b_head: Tensor,
    ) -> FusedTabularLinear:
        """Constructs fused module by absorbing mean/std standardization into linear parameters."""
        in_dim, out_dim = w_head.shape
        fused = cls(in_dim, out_dim)
        with torch.no_grad():
            # w_fused = diag(1/std) @ w_head
            w_fused = (1.0 / std).unsqueeze(1) * w_head
            # b_fused = b_head - (mean / std) @ w_head
            b_fused = b_head - (mean / std) @ w_head
            fused.weight.copy_(w_fused)
            fused.bias.copy_(b_fused)
        return fused

    def forward(self, x: Float[Tensor, "batch in_features"]) -> Float[Tensor, "batch out_features"]:
        return x @ self.weight + self.bias


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path

    import torch

    import core


    def yeo_johnson_transform(x: torch.Tensor, lmbda: float = 0.5) -> torch.Tensor:
        """Evaluates the Yeo-Johnson power transform for any real-valued tensor."""
        out = torch.zeros_like(x)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        # For x >= 0
        if abs(lmbda) > 1e-6:
            out[pos_mask] = ((x[pos_mask] + 1.0) ** lmbda - 1.0) / lmbda
        else:
            out[pos_mask] = torch.log(x[pos_mask] + 1.0)

        # For x < 0
        if abs(lmbda - 2.0) > 1e-6:
            out[neg_mask] = -((-x[neg_mask] + 1.0) ** (2.0 - lmbda) - 1.0) / (2.0 - lmbda)
        else:
            out[neg_mask] = -torch.log(-x[neg_mask] + 1.0)

        return out


    def generate_topic_07_weights() -> None:
        core.setup_env(seed=42)
        gen = torch.Generator().manual_seed(42)

        batch_size = 8
        dim = 4

        # 1. Generate synthetic tabular features (heterogeneous scales & distributions)
        # Feature 0: Gaussian N(10, 4)
        # Feature 1: Log-Normal exp(N(0, 1))
        # Feature 2: Uniform [-5, 5]
        # Feature 3: Bimodal / Skewed
        f0 = torch.randn(batch_size, 1, generator=gen) * 2.0 + 10.0
        f1 = torch.exp(torch.randn(batch_size, 1, generator=gen) * 0.5)
        f2 = (torch.rand(batch_size, 1, generator=gen) - 0.5) * 10.0
        f3 = torch.randn(batch_size, 1, generator=gen) * 3.0 - 2.0
        raw_x = torch.cat([f0, f1, f2, f3], dim=-1)  # [batch_size, dim]

        # Compute training summary statistics for standard scaling
        mean = torch.mean(raw_x, dim=0)
        std = torch.std(raw_x, dim=0, unbiased=False) + 1e-6

        # Scaled features: z = (x - mean) / std
        scaled_x = (raw_x - mean) / std

        # 2. Downstream Linear Head parameters
        w_head = torch.randn(dim, 2, generator=gen, dtype=torch.float32)
        b_head = torch.randn(2, generator=gen, dtype=torch.float32)

        # 3. Fused Serving Weights: Folding standardization into linear weights
        # y = scaled_x @ w_head + b_head
        #   = ((x - mean) / std) @ w_head + b_head
        #   = x @ diag(1/std) @ w_head + (b_head - (mean / std) @ w_head)
        w_fused = (1.0 / std).unsqueeze(1) * w_head  # [dim, 2]
        b_fused = b_head - (mean / std) @ w_head    # [2]

        # Verify mathematical identity of weight folding: y_fused == y_explicit
        y_explicit = scaled_x @ w_head + b_head
        y_fused = raw_x @ w_fused + b_fused
        torch.testing.assert_close(y_fused, y_explicit, atol=1e-5, rtol=1e-5)

        # 4. Yeo-Johnson reference transformation
        yj_transformed = yeo_johnson_transform(raw_x, lmbda=0.5)

        # 5. Differentiable loss and gradients
        x_param = raw_x.clone().detach().requires_grad_(True)
        w_param = w_fused.clone().detach().requires_grad_(True)
        b_param = b_fused.clone().detach().requires_grad_(True)
        target = torch.randn(batch_size, 2, generator=gen, dtype=torch.float32)

        pred = x_param @ w_param + b_param
        loss = 0.5 * torch.sum((pred - target) ** 2)
        loss.backward()

        grad_w_ref = w_param.grad.clone().detach()
        grad_b_ref = b_param.grad.clone().detach()
        grad_x_ref = x_param.grad.clone().detach()

        weights_dict = {
            "weight": w_fused.detach(),
            "bias": b_fused.detach(),
            "mean": mean.detach(),
            "std": std.detach(),
            "w_head": w_head.detach(),
            "b_head": b_head.detach(),
            "yj_reference": yj_transformed.detach(),
            "target": target.detach(),
            "grad_weight": grad_w_ref,
            "grad_bias": grad_b_ref,
            "grad_input": grad_x_ref,
        }

        metadata = {
            "topic": "00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing",
            "description": "Deterministic tabular scalers, Yeo-Johnson transforms, and fused serving weights",
            "seed": "42",
            "dim": str(dim),
            "batch_size": str(batch_size),
        }

        out_path = Path("weights/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.safetensors")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        core.parity.save_reference_checkpoint(
            path=out_path,
            weights_dict=weights_dict,
            input_tensor=raw_x,
            output_tensor=y_fused.detach(),
            metadata=metadata,
        )
        print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


    generate_topic_07_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    import core

    env = core.setup_env(seed=42)
    print(f"Testing PyTorch 2.x Tabular Preprocessing & Weight Folding on {env.device}...")

    ckpt_path = Path("weights/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.safetensors")
    weights, test_in, golden_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w_gold = weights["weight"]
    b_gold = weights["bias"]
    mean_gold = weights["mean"]
    std_gold = weights["std"]
    w_head_gold = weights["w_head"]
    b_head_gold = weights["b_head"]
    yj_gold = weights["yj_reference"]
    target_gold = weights["target"]
    golden_gw = weights["grad_weight"]
    golden_gb = weights["grad_bias"]
    golden_gx = weights["grad_input"]

    dim = int(meta["dim"])

    # 1. StandardScaler Invariant & Exact Reconstruction Check
    scaler = StandardScaler(dim=dim)
    scaler.mean.copy_(mean_gold)
    scaler.std.copy_(std_gold)

    z = scaler.transform(test_in)
    x_rec = scaler.inverse_transform(z)
    torch.testing.assert_close(x_rec, test_in, atol=1e-6, rtol=1e-6)
    print("[PASS] StandardScaler: Exact invertible reconstruction verified: ||x - inv(z)||_inf < 1e-6.")

    # 2. Yeo-Johnson Transform Parity
    yj_calc = yeo_johnson_transform(test_in, lmbda=0.5)
    torch.testing.assert_close(yj_calc, yj_gold, atol=1e-5, rtol=1e-5)
    print("[PASS] Yeo-Johnson power transform matches reference output.")

    # 3. Serving-Time Weight Folding Mathematical Identity
    fused_model = FusedTabularLinear.from_scaled_head(mean_gold, std_gold, w_head_gold, b_head_gold)
    torch.testing.assert_close(fused_model.weight, w_gold, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused_model.bias, b_gold, atol=1e-5, rtol=1e-5)

    y_pred = fused_model(test_in)
    torch.testing.assert_close(y_pred, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] Weight folding identity verified: x @ W_fused + b_fused == ((x - mu)/sigma) @ W + b.")

    # 4. Backward Gradient Parity
    x_param = test_in.clone().detach().requires_grad_(True)
    fused_model.weight.grad = None
    fused_model.bias.grad = None
    out = fused_model(x_param)
    loss = 0.5 * torch.sum((out - target_gold) ** 2)
    loss.backward()

    torch.testing.assert_close(fused_model.weight.grad, golden_gw, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused_model.bias.grad, golden_gb, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_param.grad, golden_gx, atol=1e-5, rtol=1e-5)
    print("[PASS] Autograd backward gradient parity strictly verified (<1e-5).")

    # 5. Compilation & Export Verification
    ok_exp, msg_exp = core.export.verify_torch_export(fused_model, test_in)
    print(f"torch.export: {msg_exp}")
    assert ok_exp

    ok_comp, msg_comp = core.export.verify_torch_compile_fullgraph(fused_model, test_in)
    print(f"torch.compile: {msg_comp}")
    assert ok_comp

    print("[PASS] All Topic 07 PyTorch 2.x tests passed cleanly!")
