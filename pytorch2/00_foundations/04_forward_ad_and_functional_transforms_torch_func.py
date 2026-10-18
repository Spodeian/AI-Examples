"""Topic 04: Forward-Mode AD & Functional Transforms (PyTorch 2.x Track).

Implements modern functional automatic differentiation and tensor transforms using `torch.func`:
1. Dual Numbers & Forward-Mode AD: Vector-Jacobian Product (VJP, Reverse AD) vs.
   Jacobian-Vector Product (JVP, Forward AD).
2. Composable vectorization (`torch.func.vmap`) for loop-free batching, per-sample gradients,
   and model ensembling.
3. Higher-order derivatives: exact full Hessians (`torch.func.hessian`) and mixed-mode
   directional curvature ($H v$ via forward-over-reverse `jvp(grad)`).
4. Stateless execution of stateful modules: `torch.func.functional_call`.
5. Production validation via `torch.compile(fullgraph=True)` and `torch.export`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.func as func
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class LinearMLP(nn.Module):
    """Two-layer linear network for testing functional autodiff transforms."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch out_dim"]:
        return x @ self.weight + self.bias


def forward_jvp(
    fn: Callable[[Tensor], Tensor],
    primals: Tensor,
    tangents: Tensor,
) -> tuple[Tensor, Tensor]:
    """Computes primal output and forward-mode directional derivative (JVP: J @ v).

    Args:
        fn: Differentiable function taking a primal tensor and returning a tensor.
        primals: Evaluation point x.
        tangents: Perturbation tangent vector v.

    Returns:
        tuple of (f(x), J(x) @ v).
    """
    jvp_result = func.jvp(fn, (primals,), (tangents,))
    return cast(Tensor, jvp_result[0]), cast(Tensor, jvp_result[1])


def compute_per_sample_grads(
    model: nn.Module,
    batch_x: Float[Tensor, "batch in_dim"],
    batch_y: Float[Tensor, "batch out_dim"],
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, Tensor]:
    """Computes exact per-sample gradients (dL_i / dW) without a Python for-loop using vmap.

    Standard backprop computes the batch-averaged gradient: (1/B) sum_i grad_i.
    Differential privacy (DP-SGD) and sample influence diagnostics require individual grad_i.
    """
    if loss_fn is None:
        def default_loss_fn(pred: Tensor, target: Tensor) -> Tensor:
            return 0.5 * torch.sum((pred - target) ** 2)

        loss_fn = default_loss_fn

    params = dict(model.named_parameters())

    def compute_single_loss(p: dict[str, Tensor], x_single: Tensor, y_single: Tensor) -> Tensor:
        pred_single = func.functional_call(model, p, (x_single.unsqueeze(0),)).squeeze(0)
        return loss_fn(pred_single, y_single)

    # Vectorize gradient computation across batch dimension
    per_sample_grad_fn = func.vmap(
        func.grad(compute_single_loss),
        in_dims=(None, 0, 0),
    )
    result = per_sample_grad_fn(params, batch_x, batch_y)
    return cast(dict[str, Tensor], result)


def compute_exact_hessian(
    fn: Callable[[Tensor], Tensor],
    x: Float[Tensor, "dim ..."],
) -> Float[Tensor, "dim dim"]:
    """Computes exact full Hessian matrix H_ij = d^2 f / dx_i dx_j via func.hessian."""
    result = func.hessian(fn)(x)
    return cast(Tensor, result)


def compute_hessian_vector_product(
    scalar_fn: Callable[[Tensor], Tensor],
    x: Float[Tensor, "dim ..."],
    v: Float[Tensor, "dim ..."],
) -> Float[Tensor, "dim ..."]:
    """Computes exact Hessian-Vector Product (HVP: H @ v) in O(D) time using forward-over-reverse AD.

    Avoids materializing the D x D Hessian matrix:
        H @ v = jvp(grad(f), (x,), (v,))[1]
    """
    grad_fn = func.grad(scalar_fn)
    hvp_result = func.jvp(grad_fn, (x,), (v,))
    return cast(Tensor, hvp_result[1])


if __name__ == "__main__":
    import core

    env = core.setup_env(seed=42)
    print(f"Testing PyTorch 2.x torch.func Track on {env.device}...")

    # Load deterministic reference checkpoint
    ckpt_path = "weights/00_foundations/04_forward_ad_and_functional_transforms_torch_func.safetensors"
    weights, test_in, test_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w = weights["weight"]
    b = weights["bias"]
    tangent_in = weights["tangent_in"]
    golden_tangent_out = weights["tangent_out"]

    # 1. Forward-Mode AD (JVP) Parity Check
    def linear_fn(x: Tensor) -> Tensor:
        return x @ w + b

    primals_out, jvp_out = forward_jvp(linear_fn, test_in, tangent_in)

    torch.testing.assert_close(primals_out, test_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(jvp_out, golden_tangent_out, atol=1e-5, rtol=1e-5)
    print("Forward AD (JVP): Verified against golden reference tangents.")

    # 2. Per-Sample Gradients via vmap
    model = LinearMLP(in_dim=3, out_dim=2)
    model.weight.data.copy_(w)
    model.bias.data.copy_(b)

    per_sample_grads = compute_per_sample_grads(model, test_in, test_out)
    weight_grads = per_sample_grads["weight"]
    assert weight_grads.shape == (test_in.shape[0], w.shape[0], w.shape[1]), (
        f"Expected shape (4, 3, 2), got {weight_grads.shape}"
    )
    print(f"torch.func.vmap per-sample grads: verified shape {weight_grads.shape}")

    # 3. Hessian-Vector Product (HVP) vs Full Hessian
    def scalar_loss(x: Tensor) -> Tensor:
        return 0.5 * torch.sum((x @ w) ** 2)

    x_single = test_in[0]
    v_single = tangent_in[0]

    full_hessian = compute_exact_hessian(scalar_loss, x_single)
    hvp_exact = compute_hessian_vector_product(scalar_loss, x_single, v_single)
    hvp_from_mat = full_hessian @ v_single

    torch.testing.assert_close(hvp_exact, hvp_from_mat, atol=1e-5, rtol=1e-5)
    print("Hessian-Vector Product (jvp over grad): Verified identical to H @ v.")

    # 4. Compilation & Export Verification
    ok_exp, msg_exp = core.export.verify_torch_export(model, test_in)
    print(f"torch.export: {msg_exp}")
    assert ok_exp

    ok_comp, msg_comp = core.export.verify_torch_compile_fullgraph(model, test_in)
    print(f"torch.compile: {msg_comp}")
    assert ok_comp

    print("[PASS] All PyTorch 2.x torch.func tests passed cleanly!")
