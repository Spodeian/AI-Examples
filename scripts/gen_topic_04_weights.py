"""Generate deterministic reference safetensors checkpoint for Topic 04:
04_forward_ad_and_functional_transforms_torch_func
"""

from __future__ import annotations

from pathlib import Path

import torch

import core


def generate_topic_04_weights() -> None:
    # Deterministic seed 42
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    # Invertible linear system dimensions
    # For forward AD (JVP), we evaluate a parameterized MLP / projection:
    # y = x @ W + b
    # With batch_size=4, in_dim=3, out_dim=2
    in_dim, out_dim = 3, 2
    weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)

    test_input = torch.randn(4, in_dim, generator=gen, dtype=torch.float32)

    # Reference forward pass:
    test_output = torch.matmul(test_input, weight) + bias

    # Dual numbers / Forward AD JVP tangent vector test
    # Tangent v for x: Jv = v @ W
    tangent_in = torch.randn(4, in_dim, generator=gen, dtype=torch.float32)
    tangent_out = torch.matmul(tangent_in, weight)

    weights_dict = {
        "weight": weight,
        "bias": bias,
        "tangent_in": tangent_in,
        "tangent_out": tangent_out,
    }

    metadata = {
        "topic": "00_foundations/04_forward_ad_and_functional_transforms_torch_func",
        "description": "Deterministic weights, test I/O, and JVP tangents for forward AD & functorch",
        "seed": "42",
        "framework": "torch",
    }

    out_path = Path("weights/00_foundations/04_forward_ad_and_functional_transforms_torch_func.safetensors")
    core.parity.save_reference_checkpoint(
        path=out_path,
        weights_dict=weights_dict,
        input_tensor=test_input,
        output_tensor=test_output,
        metadata=metadata,
    )
    print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_topic_04_weights()
