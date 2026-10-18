"""Generate deterministic reference safetensors checkpoint for Topic 06:
06_nn_module_internals_parametrizations_and_meta_device
"""

from __future__ import annotations

from pathlib import Path

import torch

import core


def cayley_transform(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[0]
    tri_u = torch.triu(x, diagonal=1)
    skew_a = tri_u - tri_u.T
    eye = torch.eye(dim, dtype=x.dtype, device=x.device)
    inv_term = torch.linalg.inv(eye + skew_a)
    return (eye - skew_a) @ inv_term


def generate_topic_06_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    dim = 3
    batch_size = 4

    # 1. Generate unconstrained matrix parameter M
    m_raw = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(dim, generator=gen, dtype=torch.float32)

    # 2. Cayley transform: W = (I - A)(I + A)^{-1}
    w_ortho = cayley_transform(m_raw)

    # Verify strict orthogonality: W^T @ W = I
    eye = torch.eye(dim, dtype=torch.float32)
    ortho_identity = w_ortho.T @ w_ortho
    torch.testing.assert_close(ortho_identity, eye, atol=1e-6, rtol=1e-6)

    test_in = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    target = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)

    # 3. Differentiable forward & backward pass through Cayley parameterization
    m_param = m_raw.clone().detach().requires_grad_(True)
    b_param = bias.clone().detach().requires_grad_(True)
    x_param = test_in.clone().detach().requires_grad_(True)

    w_t = cayley_transform(m_param)
    y_pred = x_param @ w_t + b_param
    loss = 0.5 * torch.sum((y_pred - target) ** 2)
    loss.backward()

    grad_m_ref = m_param.grad.clone().detach()
    grad_b_ref = b_param.grad.clone().detach()
    grad_x_ref = x_param.grad.clone().detach()

    weights_dict = {
        "weight": w_ortho.detach(),
        "bias": bias,
        "skew_param": m_raw,
        "target": target,
        "grad_skew": grad_m_ref,
        "grad_bias": grad_b_ref,
        "grad_input": grad_x_ref,
    }

    metadata = {
        "topic": "00_foundations/06_nn_module_internals_parametrizations_and_meta_device",
        "description": "Deterministic weights, Cayley orthogonal manifold parameterization, and reference gradients",
        "seed": "42",
        "framework": "torch",
        "dim": str(dim),
        "batch_size": str(batch_size),
    }

    out_path = Path("weights/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core.parity.save_reference_checkpoint(
        path=out_path,
        weights_dict=weights_dict,
        input_tensor=test_in,
        output_tensor=y_pred.detach(),
        metadata=metadata,
    )
    print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_topic_06_weights()
