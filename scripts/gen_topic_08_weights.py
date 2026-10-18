"""Generate deterministic reference safetensors checkpoint for Topic 08:
08_reproducible_preprocessing_pipelines_and_determinism
"""

from __future__ import annotations

from pathlib import Path

import torch

import core


def kahan_sum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Evaluates Kahan compensated summation along the specified dimension."""
    # Transpose target dim to front for iteration
    x_t = x.transpose(0, dim)
    s = torch.zeros_like(x_t[0])
    c = torch.zeros_like(s)

    for i in range(x_t.shape[0]):
        y = x_t[i] - c
        t = s + y
        c = (t - s) - y
        s = t

    return s


def generate_topic_08_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    dim = 4
    out_dim = 2

    # 1. Deterministic synthetic features
    raw_x = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)

    # 2. Linear projection parameters
    weight = torch.randn(dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)

    # 3. Kahan compensated sum & mean reference
    kahan_mean_ref = kahan_sum(raw_x, dim=0) / float(batch_size)

    # 4. Forward output
    y_pred = raw_x @ weight + bias

    # 5. Differentiable backward gradients
    x_param = raw_x.clone().detach().requires_grad_(True)
    w_param = weight.clone().detach().requires_grad_(True)
    b_param = bias.clone().detach().requires_grad_(True)
    target = torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)

    pred = x_param @ w_param + b_param
    loss = 0.5 * torch.sum((pred - target) ** 2)
    loss.backward()

    grad_w_ref = w_param.grad.clone().detach()
    grad_b_ref = b_param.grad.clone().detach()
    grad_x_ref = x_param.grad.clone().detach()

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "kahan_mean": kahan_mean_ref.detach(),
        "target": target.detach(),
        "grad_weight": grad_w_ref,
        "grad_bias": grad_b_ref,
        "grad_input": grad_x_ref,
    }

    metadata = {
        "topic": "00_foundations/08_reproducible_preprocessing_pipelines_and_determinism",
        "description": "Deterministic pipeline reference, Kahan compensated summation, and seed invariants",
        "seed": "42",
        "dim": str(dim),
        "batch_size": str(batch_size),
    }

    out_path = Path("weights/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.safetensors")
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
    generate_topic_08_weights()
