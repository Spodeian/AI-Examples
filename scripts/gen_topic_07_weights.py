"""Generate deterministic reference safetensors checkpoint for Topic 07:
07_feature_engineering_scalers_and_tabular_preprocessing
"""

from __future__ import annotations

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


if __name__ == "__main__":
    generate_topic_07_weights()
