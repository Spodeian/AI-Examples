"""Production PyTorch 2.x implementation for Topic 10:
10_floating_point_arithmetic_conditioning_and_numerical_stability
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float


def welford_online_mean_variance(
    x: Float[torch.Tensor, "batch dim"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes single-pass numerically stable mean and population variance via Welford's algorithm."""
    n, d = x.shape
    mean = torch.zeros(d, dtype=x.dtype, device=x.device)
    m2 = torch.zeros(d, dtype=x.dtype, device=x.device)

    for i in range(n):
        xi = x[i]
        delta = xi - mean
        mean = mean + delta / float(i + 1)
        delta2 = xi - mean
        m2 = m2 + delta * delta2

    var = m2 / float(n)
    return mean, var


def log_sum_exp_stable(
    x: Float[torch.Tensor, "... dim"],
    dim: int = -1,
) -> torch.Tensor:
    """Computes numerically stable Log-Sum-Exp avoiding intermediate float overflow or underflow."""
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m), dim=dim))


def matrix_condition_number(
    a: Float[torch.Tensor, "m n"],
) -> float:
    """Computes the 2-norm condition number kappa(A) = sigma_max / sigma_min via SVD."""
    s = torch.linalg.svdvals(a)
    s_max = float(s[0].item())
    s_min = float(s[-1].item())
    if s_min == 0.0:
        return float("inf")
    return s_max / s_min


def neumaier_sum(
    x: Float[torch.Tensor, "batch dim"],
    dim: int = 0,
) -> torch.Tensor:
    """Evaluates Neumaier compensated summation along the specified dimension."""
    slices = [s.squeeze(dim) for s in torch.split(x, 1, dim=dim)]
    s = torch.zeros_like(slices[0])
    c = torch.zeros_like(s)

    for xi in slices:
        t = s + xi
        mask = torch.abs(s) >= torch.abs(xi)
        comp = torch.where(mask, (s - t) + xi, (xi - t) + s)
        c = c + comp
        s = t

    return s + c


class StableLinear(nn.Module):
    """Linear projection layer equipped with numerically stable forward operators."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / (self.in_features**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_features"],
    ) -> Float[torch.Tensor, "batch out_features"]:
        return x @ self.weight + self.bias

    def forward_lse(
        self,
        x: Float[torch.Tensor, "batch in_features"],
    ) -> torch.Tensor:
        logits = self.forward(x)
        return log_sum_exp_stable(logits, dim=-1)


def create_stable_linear(
    in_features: int,
    out_features: int,
) -> StableLinear:
    """Factory creating an initialized StableLinear module."""
    return StableLinear(in_features, out_features)
