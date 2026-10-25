"""Input Convex Neural Networks (ICNN), Brenier Optimal Transport, and Certified Monotonic Networks — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic
Golden weights: weights/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def icnn_potential(
    x: torch.Tensor,
    w_x0: torch.Tensor,
    b0: torch.Tensor,
    w_z1: torch.Tensor,
    w_x1: torch.Tensor,
    b1: torch.Tensor,
) -> torch.Tensor:
    """Evaluates convex potential psi(x) using a 2-layer Input Convex Neural Network.
    
    x: [B, D]
    w_x0: [D, D_h]
    b0: [D_h]
    w_z1: [D_h, 1]
    w_x1: [D, 1]
    b1: [1]
    Returns scalar convex potential psi(x) of shape [B, 1].
    """
    w_z1_pos = F.softplus(w_z1)
    z0 = F.relu(x @ w_x0 + b0)
    psi = z0 @ w_z1_pos + x @ w_x1 + b1
    return psi


def brenier_ot_gradient_map(
    x: torch.Tensor,
    w_x0: torch.Tensor,
    b0: torch.Tensor,
    w_z1: torch.Tensor,
    w_x1: torch.Tensor,
) -> torch.Tensor:
    """Computes the Brenier optimal transport map T(x) = x + nabla_x psi(x).
    
    nabla_x psi(x) = (step(x @ w_x0 + b0) * softplus(w_z1)^T) @ w_x0^T + w_x1^T.
    T(x) is certified strictly monotone: (T(x1) - T(x2)) . (x1 - x2) >= ||x1 - x2||^2.
    """
    w_z1_pos = F.softplus(w_z1)  # [D_h, 1]
    pre_act = x @ w_x0 + b0     # [B, D_h]
    mask = (pre_act > 0.0).to(x.dtype)  # [B, D_h]
    effective_z = mask * w_z1_pos.t()   # [B, D_h]
    grad_psi = effective_z @ w_x0.t() + w_x1.t()  # [B, D]
    return x + grad_psi


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # ICNN setup: B=3, D=4, D_h=8
    B, D, D_h = 3, 4, 8
    x_input = torch.randn(B, D)
    w_x0 = torch.randn(D, D_h) * 0.2
    b0 = torch.randn(D_h) * 0.1
    w_z1 = torch.randn(D_h, 1) * 0.2
    w_x1 = torch.randn(D, 1) * 0.2
    b1 = torch.randn(1) * 0.1

    psi = icnn_potential(x_input, w_x0, b0, w_z1, w_x1, b1)
    t_map = brenier_ot_gradient_map(x_input, w_x0, b0, w_z1, w_x1)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_input": x_input,
        "w_x0": w_x0,
        "b0": b0,
        "w_z1": w_z1,
        "w_x1": w_x1,
        "b1": b1,
        "icnn_psi": psi,
        "brenier_t_map": t_map,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous().cpu() for k, v in build_golden_tensors().items()}
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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
