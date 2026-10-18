"""Generate deterministic reference safetensors checkpoint for Topic 05:
05_hooks_and_saved_tensor_hooks
"""

from __future__ import annotations

from pathlib import Path

import torch

import core


def generate_topic_05_weights() -> None:
    # Deterministic seed 42
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    # Invertible linear system dimensions
    # Layer: y = GeLU(x @ W + b) with batch_size=4, in_dim=3, out_dim=2
    # Activation x @ W + b is saved for backward by GeLU.
    in_dim, out_dim = 3, 2
    batch_size = 4

    weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    test_in = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
    target = torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)

    # Standard forward and backward pass
    w_std = weight.clone().detach().requires_grad_(True)
    b_std = bias.clone().detach().requires_grad_(True)
    x_std = test_in.clone().detach().requires_grad_(True)

    # Affine linear layer: y = x @ W + b
    # Autograd engine saves x for backward to compute grad_W = x^T @ grad_out
    y_std = x_std @ w_std + b_std
    loss = 0.5 * torch.sum((y_std - target) ** 2)
    loss.backward()

    grad_w_ref = w_std.grad.clone().detach()
    grad_b_ref = b_std.grad.clone().detach()
    grad_x_ref = x_std.grad.clone().detach()

    weights_dict = {
        "weight": weight,
        "bias": bias,
        "target": target,
        "grad_weight": grad_w_ref,
        "grad_bias": grad_b_ref,
        "grad_input": grad_x_ref,
    }

    metadata = {
        "topic": "00_foundations/05_hooks_and_saved_tensor_hooks",
        "description": "Deterministic weights, activations, and reference gradients for hooks and saved_tensors_hooks",
        "seed": "42",
        "framework": "torch",
    }

    out_path = Path("weights/00_foundations/05_hooks_and_saved_tensor_hooks.safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core.parity.save_reference_checkpoint(
        path=out_path,
        weights_dict=weights_dict,
        input_tensor=test_in,
        output_tensor=y_std.detach(),
        metadata=metadata,
    )
    print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_topic_05_weights()
