"""Topic 03: Autograd Mechanics, Complex Wirtinger & Custom Functions (PyTorch 2.x Track).

Implements reverse-mode automatic differentiation internals:
1. Modern PyTorch 2.x custom `autograd.Function` with `setup_context` and double-backward support.
2. Vector-Jacobian Product (VJP) mechanics and tape memory conservation.
3. Complex-valued Wirtinger calculus for non-holomorphic real-valued loss objectives.
4. Validation via `torch.autograd.gradcheck` and `torch.compile(fullgraph=True)`.

Mathematical Foundations:
- Reverse-mode AD evaluates v^T J via chain rule accumulation.
- For complex variables z = x + iy, the Wirtinger derivatives are:
    d/dz   = 0.5 * (d/dx - i * d/dy)
    d/dz*  = 0.5 * (d/dx + i * d/dy)
  PyTorch autograd defines `z.grad` as dL/dz* so that the update z <- z - lr * z.grad
  corresponds to the steepest descent direction in the complex plane.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Complex, Float
from torch import Tensor
from torch.autograd import Function


class ScaledLinearFunction(Function):
    """Custom autograd Function using PyTorch 2.x setup_context API."""

    @staticmethod
    def forward(
        input_tensor: Float[Tensor, "batch in_dim"],
        weight: Float[Tensor, "in_dim out_dim"],
        bias: Float[Tensor, "*out_dim"],
        scale: float = 1.0,
    ) -> Float[Tensor, "batch out_dim"]:
        """Evaluates scaled linear projection: out = scale * (input @ weight + bias)."""
        return scale * (input_tensor @ weight + bias)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        """Saves only necessary tensors to minimize activation memory footprint."""
        input_tensor, weight, bias, scale = inputs
        ctx.save_for_backward(input_tensor, weight)
        ctx.has_bias = bias is not None
        ctx.scale = scale

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        """Evaluates exact analytical vector-Jacobian products (VJPs)."""
        input_tensor, weight = ctx.saved_tensors
        scale = ctx.scale

        grad_input: Tensor | None = None
        grad_weight: Tensor | None = None
        grad_bias: Tensor | None = None

        scaled_grad = grad_output * scale

        if ctx.needs_input_grad[0]:
            # dL/dX = scaled_grad @ W^T
            grad_input = scaled_grad @ weight.T

        if ctx.needs_input_grad[1]:
            # dL/dW = X^T @ scaled_grad
            grad_weight = input_tensor.T @ scaled_grad

        if ctx.needs_input_grad[2] and ctx.has_bias:
            # dL/db = sum_batch(scaled_grad)
            grad_bias = scaled_grad.sum(dim=0)

        # scale is a non-tensor hyperparameter
        return grad_input, grad_weight, grad_bias, None


def scaled_linear(
    x: Float[Tensor, "batch in_dim"],
    w: Float[Tensor, "in_dim out_dim"],
    b: Float[Tensor, "*out_dim"],
    scale: float = 1.0,
) -> Float[Tensor, "batch out_dim"]:
    """Functional wrapper for ScaledLinearFunction."""
    out = ScaledLinearFunction.apply(x, w, b, scale)
    return out if isinstance(out, Tensor) else torch.as_tensor(out)


class ComplexWirtingerLinear(nn.Module):
    """Complex-valued linear transformation demonstrating Wirtinger calculus.

    Forward: Z = X @ W + b where X, W, b in C.
    Real Loss: L = 0.5 * ||Z - Y||_F^2
    Wirtinger gradient: dL/dZ* = 0.5 * (Z - Y)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize complex parameters
        self.weight = nn.Parameter(
            torch.complex(
                torch.randn(in_features, out_features) * 0.1,
                torch.randn(in_features, out_features) * 0.1,
            )
        )
        self.bias = nn.Parameter(
            torch.complex(
                torch.zeros(out_features),
                torch.zeros(out_features),
            )
        )

    def forward(self, x: Complex[Tensor, "batch in_dim"]) -> Complex[Tensor, "batch out_dim"]:
        """Evaluates complex affine mapping."""
        return x @ self.weight + self.bias

    @staticmethod
    def analytical_wirtinger_grad(
        z: Complex[Tensor, "batch out_dim"], y: Complex[Tensor, "batch out_dim"]
    ) -> Complex[Tensor, "batch out_dim"]:
        """Computes PyTorch complex gradient for real loss L = 0.5 * ||z - y||^2.
        PyTorch defines z.grad = dL/dx + i*dL/dy = 2 * dL/dz* = (z - y).
        """
        return z - y


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "00_foundations/03_autograd_mechanics_complex_wirtinger_and_custom_functions"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    import core

    env = core.setup_env(seed=42)
    print(f"Testing PyTorch 2.x Autograd & Wirtinger Track on {env.device}...")

    # 1. Custom Autograd Function Correctness & Gradcheck
    b_dim, in_dim, out_dim = 4, 3, 2
    x_test = torch.randn(b_dim, in_dim, dtype=torch.float64, requires_grad=True)
    w_test = torch.randn(in_dim, out_dim, dtype=torch.float64, requires_grad=True)
    b_test = torch.randn(out_dim, dtype=torch.float64, requires_grad=True)

    # Validate against finite difference approximations
    test_passed = torch.autograd.gradcheck(
        lambda x, w, b: scaled_linear(x, w, b, 1.5),
        (x_test, w_test, b_test),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )
    assert test_passed, "Gradcheck failed on ScaledLinearFunction!"
    print("torch.autograd.gradcheck: PASSED (analytical VJP matches finite differences).")

    # 2. Complex Wirtinger Calculus Invariant
    c_model = ComplexWirtingerLinear(in_features=3, out_features=2)
    x_c = torch.complex(torch.randn(4, 3), torch.randn(4, 3))
    y_c_target = torch.complex(torch.randn(4, 2), torch.randn(4, 2))

    z_pred = c_model(x_c)
    # Real-valued loss: L = 0.5 * sum |z - y|^2
    diff = z_pred - y_c_target
    loss = 0.5 * torch.sum(diff.real**2 + diff.imag**2)
    loss.backward()

    # Analytical Wirtinger gradient with respect to z: dL/dz* = 0.5 * (z - y)
    grad_z_analytic = ComplexWirtingerLinear.analytical_wirtinger_grad(z_pred, y_c_target)
    # Check that manual VJP for weights: W.grad = X^H @ dL/dZ*
    grad_w_analytic = x_c.mH @ grad_z_analytic
    torch.testing.assert_close(c_model.weight.grad, grad_w_analytic, atol=1e-5, rtol=1e-5)
    print("Complex Wirtinger gradient verified: dL/dW matches X^H @ (dL/dZ*) analytically.")

    # 3. Compilation & Export Verification
    class ScaledLinearModule(nn.Module):
        def __init__(self, w: Tensor, b: Tensor) -> None:
            super().__init__()
            self.w = nn.Parameter(w)
            self.b = nn.Parameter(b)

        def forward(self, x: Tensor) -> Tensor:
            return scaled_linear(x, self.w, self.b, scale=2.0)

    mod = ScaledLinearModule(w_test.float(), b_test.float())
    sample_x = x_test.float()

    ok_exp, msg_exp = core.export.verify_torch_export(mod, sample_x)
    print(f"torch.export: {msg_exp}")
    assert ok_exp

    ok_comp, msg_comp = core.export.verify_torch_compile_fullgraph(mod, sample_x)
    print(f"torch.compile: {msg_comp}")
    assert ok_comp

    print("[PASS] All PyTorch 2.x Autograd & Wirtinger checks passed cleanly!")
