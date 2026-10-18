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


if __name__ == "__main__":
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
