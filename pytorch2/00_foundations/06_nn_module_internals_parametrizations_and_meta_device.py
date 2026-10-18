"""PyTorch 2.x nn.Module Internals, Weight Parametrizations & Meta Devices.

Topic 06: 06_nn_module_internals_parametrizations_and_meta_device (Category 00: Foundations)

Theoretical Overview:
1. PyTorch Module Data Model:
   - Parameter: Differentiable leaf tensor with requires_grad=True, returned by .parameters().
   - Buffer: Non-differentiable state tensor (e.g. running mean/var), tracked in state_dict
     with persistent=True (default) or excluded from checkpoints with persistent=False.
   - LazyModule / UninitializedParameter: Defers parameter allocation until the first forward pass.
2. Weight Parametrizations (torch.nn.utils.parametrize):
   Constrains parameter manifolds during optimization (e.g., orthogonal, symmetric, positive-definite).
   - Cayley Orthogonal Transform:
     Given skew-symmetric matrix A = -A^T, the Cayley transform:
         W = (I - A)(I + A)^{-1}
     is strictly orthogonal: W^T W = I. This guarantees isometry ||W x||_2 = ||x||_2, provably
     preventing gradient vanishing or explosion across arbitrary layer depth.
3. Meta Device (torch.device("meta")) & Fake Tensors:
   Represents tensor metadata (shape, stride, dtype) with zero physical storage allocation.
   Enables dry-run shape propagation, FLOP verification, and initialization staging for 70B+ LLMs.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from jaxtyping import Float
from torch import Tensor


class CayleyOrthogonal(nn.Module):
    """Parametrization that transforms unconstrained matrices into strictly orthogonal matrices."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("eye", torch.eye(dim, dtype=torch.float32), persistent=False)

    def forward(self, x: Float[Tensor, "dim dim"]) -> Float[Tensor, "dim dim"]:
        """Evaluates Cayley transform: W = (I - A)(I + A)^{-1} where A is skew-symmetric."""
        # Project x onto the Lie algebra of skew-symmetric matrices so(n)
        tri_u = torch.triu(x, diagonal=1)
        skew_a = tri_u - tri_u.T

        eye: Tensor = self.eye  # type: ignore[assignment]
        inv_term = torch.linalg.inv(eye + skew_a)
        w_ortho = (eye - skew_a) @ inv_term
        return w_ortho


class OrthogonalLinear(nn.Module):
    """Linear layer constrained to the Stiefel/orthogonal manifold via Cayley transform."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim, bias=True)
        # Register Cayley parametrization over the weight tensor
        parametrize.register_parametrization(self.linear, "weight", CayleyOrthogonal(dim))

    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        return x @ self.linear.weight + self.linear.bias


def instantiate_meta_model(dim: int, num_layers: int = 100) -> nn.Sequential:
    """Instantiates a deep network on the 'meta' device consuming 0 bytes of physical RAM."""
    layers = []
    with torch.device("meta"):
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, dim))
    return nn.Sequential(*layers)


if __name__ == "__main__":
    import core

    env = core.setup_env(seed=42)
    print(f"Testing PyTorch 2.x Module Internals & Meta Device on {env.device}...")

    # Load golden reference checkpoint
    ckpt_path = Path("weights/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.safetensors")
    weights, test_in, golden_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w_gold = weights["weight"]
    b_gold = weights["bias"]
    skew_gold = weights["skew_param"]
    target = weights["target"]
    golden_ga = weights["grad_skew"]
    golden_gb = weights["grad_bias"]
    golden_gx = weights["grad_input"]

    dim = int(meta.get("dim", 3))

    # 1. Forward Parity & Orthogonality Verification
    model = OrthogonalLinear(dim=dim)
    # Set unconstrained parameter to skew_gold
    with torch.no_grad():
        model.linear.parametrizations.weight.original.copy_(skew_gold)
        model.linear.bias.copy_(b_gold)

    # Invariant 1: Orthogonality W^T @ W = I
    w_effective = model.linear.weight
    torch.testing.assert_close(w_effective, w_gold, atol=1e-5, rtol=1e-5)

    identity = torch.eye(dim)
    ortho_check = w_effective.T @ w_effective
    torch.testing.assert_close(ortho_check, identity, atol=1e-6, rtol=1e-6)
    print("[PASS] Cayley transform strictly orthogonal: ||W^T W - I||_inf < 1e-6.")

    # Invariant 2: Isometry ||W x||_2 == ||x||_2
    x_test = torch.randn(4, dim)
    norm_before = torch.norm(x_test, dim=-1)
    norm_after = torch.norm(x_test @ w_effective, dim=-1)
    torch.testing.assert_close(norm_before, norm_after, atol=1e-5, rtol=1e-5)
    print("[PASS] Isometric norm preservation verified: ||W x||_2 == ||x||_2.")

    # 2. Forward & Backward Output Parity
    x_active = test_in.clone().detach().requires_grad_(True)
    out_pred = model(x_active)
    torch.testing.assert_close(out_pred, golden_out, atol=1e-5, rtol=1e-5)

    loss = 0.5 * torch.sum((out_pred - target) ** 2)
    loss.backward()

    # Compare gradients
    orig_grad = model.linear.parametrizations.weight.original.grad
    assert orig_grad is not None and model.linear.bias.grad is not None and x_active.grad is not None
    torch.testing.assert_close(orig_grad, golden_ga, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.linear.bias.grad, golden_gb, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_active.grad, golden_gx, atol=1e-5, rtol=1e-5)
    print("[PASS] Gradient parity verified through Cayley parametrization backprop.")

    # 3. Meta Device Zero-Memory Allocation Verification
    meta_model = instantiate_meta_model(dim=1024, num_layers=50)
    meta_x = torch.randn(8, 1024, device="meta")
    meta_out = meta_model(meta_x)
    assert meta_out.is_meta
    assert meta_out.shape == (8, 1024)
    print("[PASS] Meta device dry-run: 50-layer 1024-dim model initialized with 0 bytes RAM.")

    # 4. Compilation & Export Verification
    plain_linear = nn.Linear(dim, dim)
    plain_linear.weight.data.copy_(w_gold)
    plain_linear.bias.data.copy_(b_gold)

    ok_exp, msg_exp = core.export.verify_torch_export(plain_linear, test_in)
    print(f"torch.export: {msg_exp}")
    assert ok_exp

    ok_comp, msg_comp = core.export.verify_torch_compile_fullgraph(plain_linear, test_in)
    print(f"torch.compile: {msg_comp}")
    assert ok_comp

    print("[PASS] All Topic 06 PyTorch 2.x tests passed cleanly!")
