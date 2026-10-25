"""Physics Informed PINNs Deep Ritz Hard Bc and NTK Balancing — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing
Golden weights: weights/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
    }

    # PINN domain setup: 1D Poisson -u''(x) = pi^2 * sin(pi * x) on [0, 1] with u(0)=0, u(1)=0
    x_colloc = torch.linspace(0.1, 0.9, 8).unsqueeze(1)
    w1 = torch.randn(1, 16) * 0.5
    b1 = torch.randn(16) * 0.1
    w2 = torch.randn(16, 1) * 0.5
    b2 = torch.randn(1) * 0.1

    # Forward network evaluation: u(x) = tanh(x @ w1 + b1) @ w2 + b2
    z = x_colloc @ w1 + b1
    a = torch.tanh(z)
    u_raw = a @ w2 + b2

    # First and second spatial derivatives
    psi = (1.0 - a**2) * w1  # [8, 16]
    u_x = psi @ w2  # [8, 1]
    phi = (-2.0 * a * (1.0 - a**2)) * (w1**2)  # [8, 16]
    u_xx = phi @ w2  # [8, 1]

    # Strong-form PDE residual: r(x) = -u_xx - f(x)
    f_source = (math.pi**2) * torch.sin(math.pi * x_colloc)
    pde_residual = -u_xx - f_source
    loss_pde = torch.mean(pde_residual**2).unsqueeze(0)

    # Hard Boundary Conditions: u_hard(x) = d(x) * u(x) where d(x) = x * (1 - x)
    d = x_colloc * (1.0 - x_colloc)
    d_x = 1.0 - 2.0 * x_colloc
    u_hard = d * u_raw
    u_hard_x = d_x * u_raw + d * u_x

    # Variational / Weak-Form Deep Ritz Energy: E(u) = (1/N) * sum(0.5 * (u_hard_x)^2 - f * u_hard)
    ritz_energy = torch.mean(0.5 * (u_hard_x**2) - f_source * u_hard).unsqueeze(0)

    # Boundary evaluation for soft PINN & NTK balancing
    x_bc = torch.tensor([[0.0], [1.0]])
    z_bc = x_bc @ w1 + b1
    a_bc = torch.tanh(z_bc)
    u_bc = a_bc @ w2 + b2
    g_bc_target = torch.tensor([[0.0], [0.0]])
    loss_bc = torch.mean((u_bc - g_bc_target) ** 2).unsqueeze(0)

    # Analytical gradients with respect to w2
    # grad_w2(loss_pde) = (-2 / N) * phi^T @ pde_residual
    g_pde = (-2.0 / x_colloc.shape[0]) * (phi.t() @ pde_residual)
    # grad_w2(loss_bc) = (2 / N_bc) * a_bc^T @ (u_bc - g_bc_target)
    g_bc = (2.0 / x_bc.shape[0]) * (a_bc.t() @ (u_bc - g_bc_target))

    norm_pde = torch.norm(g_pde)
    norm_bc = torch.norm(g_bc)
    lambda_ntk = (norm_pde / (norm_bc + 1e-8)).unsqueeze(0)
    g_balanced = g_pde + lambda_ntk * g_bc

    tensors.update({
        "x_colloc": x_colloc,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "u_raw": u_raw,
        "u_x": u_x,
        "u_xx": u_xx,
        "pde_residual": pde_residual,
        "loss_pde": loss_pde,
        "u_hard": u_hard,
        "u_hard_x": u_hard_x,
        "ritz_energy": ritz_energy,
        "loss_bc": loss_bc,
        "g_pde": g_pde,
        "g_bc": g_bc,
        "lambda_ntk": lambda_ntk,
        "g_balanced": g_balanced,
    })
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
