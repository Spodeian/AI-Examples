"""Generate deterministic reference safetensors checkpoint for Topic 10:
10_floating_point_arithmetic_conditioning_and_numerical_stability
"""

from __future__ import annotations

from pathlib import Path

import torch

import core


def welford_mean_var(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass numerically stable Welford online mean and population variance."""
    n, d = x.shape
    mean = torch.zeros(d, dtype=torch.float32)
    m2 = torch.zeros(d, dtype=torch.float32)

    for i in range(n):
        xi = x[i]
        delta = xi - mean
        mean += delta / float(i + 1)
        delta2 = xi - mean
        m2 += delta * delta2

    var = m2 / float(n)
    return mean, var


def log_sum_exp_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-sum-exp with max subtraction."""
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m), dim=dim))


def generate_topic_10_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    dim = 4
    out_dim = 2

    # Deterministic input tensor with large base offset to stress-test numerical variance stability
    raw_x = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)

    # Calculate golden Welford statistics
    welford_mean, welford_var = welford_mean_var(raw_x)

    # Linear projection parameters
    weight = torch.randn(dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)

    # Forward projection
    y_pred = raw_x @ weight + bias

    # Stable Log-Sum-Exp across channels of y_pred
    lse_ref = log_sum_exp_stable(y_pred, dim=-1)

    # Differentiable backward pass for autograd parity
    x_param = raw_x.clone().detach().requires_grad_(True)
    w_param = weight.clone().detach().requires_grad_(True)
    b_param = bias.clone().detach().requires_grad_(True)
    target = torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)

    pred = x_param @ w_param + b_param
    loss = 0.5 * torch.sum((pred - target) ** 2)
    loss.backward()

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "welford_mean": welford_mean.detach(),
        "welford_var": welford_var.detach(),
        "lse_ref": lse_ref.detach(),
        "target": target.detach(),
        "grad_weight": w_param.grad.clone().detach(),
        "grad_bias": b_param.grad.clone().detach(),
        "grad_input": x_param.grad.clone().detach(),
    }

    metadata = {
        "topic": "00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability",
        "description": "Welford online variance, stable LogSumExp, condition numbers and FP stability",
        "batch_size": str(batch_size),
        "dim": str(dim),
        "out_dim": str(out_dim),
    }

    out_path = Path("weights/00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core.parity.save_reference_checkpoint(
        path=out_path,
        weights_dict=weights_dict,
        input_tensor=raw_x,
        output_tensor=y_pred.detach(),
        metadata=metadata,
    )
    print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_topic_10_weights()
