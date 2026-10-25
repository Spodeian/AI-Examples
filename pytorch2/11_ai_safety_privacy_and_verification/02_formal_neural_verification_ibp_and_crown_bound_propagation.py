"""Formal Neural Verification IBP and CROWN Bound Propagation — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation
Golden weights: weights/11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ibp_linear(
    l_in: torch.Tensor,
    u_in: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval Bound Propagation for linear layer y = x @ W + b.
    
    Uses center-radius formulation:
    mu = (u + l)/2, r = (u - l)/2
    mu_out = mu @ W + b, r_out = r @ |W|
    l_out = mu_out - r_out, u_out = mu_out + r_out
    """
    mu = (u_in + l_in) * 0.5
    r = (u_in - l_in) * 0.5
    mu_out = mu @ weight + bias
    r_out = r @ torch.abs(weight)
    return mu_out - r_out, mu_out + r_out


def ibp_relu(l_in: torch.Tensor, u_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval Bound Propagation for ReLU activation."""
    return torch.clamp(l_in, min=0.0), torch.clamp(u_in, min=0.0)


def crown_relu_relaxation(
    l_in: torch.Tensor,
    u_in: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes upper and lower linear bounding parameters for ReLU on [l, u]:
    s_L * x + b_L <= ReLU(x) <= s_U * x + b_U
    """
    diff = torch.clamp(u_in - l_in, min=1e-8)
    
    # Upper bound secant line between (l, 0) and (u, u) for crossing neurons
    s_u_crossing = u_in / diff
    b_u_crossing = -(u_in * l_in) / diff
    
    # Lower bound line slope for crossing neurons
    if alpha is None:
        s_l_crossing = torch.where(u_in > torch.abs(l_in), 1.0, 0.0)
    else:
        s_l_crossing = torch.clamp(alpha, 0.0, 1.0)
    b_l_crossing = torch.zeros_like(l_in)
    
    is_active = (l_in >= 0.0).float()
    is_inactive = (u_in <= 0.0).float()
    is_crossing = (1.0 - is_active) * (1.0 - is_inactive)
    
    s_L = is_active * 1.0 + is_crossing * s_l_crossing
    b_L = is_crossing * b_l_crossing
    s_U = is_active * 1.0 + is_crossing * s_u_crossing
    b_U = is_crossing * b_u_crossing
    
    return s_L, b_L, s_U, b_U


def crown_2layer_bounds(
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CROWN backward linear bound propagation for 2-layer MLP f(x) = ReLU(x @ W1 + b1) @ W2 + b2.
    
    Computes verified lower and upper bounds on scalar margin specification f(x) @ c for x in [x_l, x_u].
    """
    # 1. Forward IBP for hidden layer pre-activations
    z1_l, z1_u = ibp_linear(x_l, x_u, w1, b1)
    
    # 2. CROWN linear relaxation on ReLU
    s_L, b_L, s_U, b_U = crown_relu_relaxation(z1_l, z1_u)
    
    # 3. Backward propagation of linear specification vector v = W2 @ c
    v = w2 @ c  # [H]
    
    # Lower bound coefficients
    lambda_l = torch.where(v >= 0.0, s_L, s_U) * v  # [B, H]
    bias_relu_l = torch.sum(torch.where(v >= 0.0, b_L, b_U) * v, dim=-1)  # [B]
    
    a_l = lambda_l @ w1.t()  # [B, D]
    gamma_l = torch.sum(lambda_l * b1, dim=-1) + bias_relu_l + torch.dot(b2, c)  # [B]
    
    # Input box optimization
    mu_x = (x_u + x_l) * 0.5
    r_x = (x_u - x_l) * 0.5
    crown_lower = torch.sum(a_l * mu_x, dim=-1) - torch.sum(torch.abs(a_l) * r_x, dim=-1) + gamma_l
    
    # Upper bound coefficients
    lambda_u = torch.where(v >= 0.0, s_U, s_L) * v  # [B, H]
    bias_relu_u = torch.sum(torch.where(v >= 0.0, b_U, b_L) * v, dim=-1)  # [B]
    
    a_u = lambda_u @ w1.t()  # [B, D]
    gamma_u = torch.sum(lambda_u * b1, dim=-1) + bias_relu_u + torch.dot(b2, c)  # [B]
    
    crown_upper = torch.sum(a_u * mu_x, dim=-1) + torch.sum(torch.abs(a_u) * r_x, dim=-1) + gamma_u
    
    return crown_lower, crown_upper


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # Setup 2-layer network: D_in=4, H=6, D_out=3
    w1 = torch.randn(4, 6) * 0.5
    b1 = torch.randn(6) * 0.2
    w2 = torch.randn(6, 3) * 0.5
    b2 = torch.randn(3) * 0.1
    
    # Input bounds with perturbation epsilon=0.1
    x_center = torch.randn(8, 4) * 0.5
    eps = 0.1
    x_l = x_center - eps
    x_u = x_center + eps
    
    # Specification vector: verify class 0 > class 1 (margin = y[0] - y[1])
    c = torch.tensor([1.0, -1.0, 0.0])
    
    # Reference calculations
    z1_l, z1_u = ibp_linear(x_l, x_u, w1, b1)
    a1_l, a1_u = ibp_relu(z1_l, z1_u)
    z2_l, z2_u = ibp_linear(a1_l, a1_u, w2, b2)
    ibp_margin_l = (z2_l @ torch.clamp(c, min=0.0) + z2_u @ torch.clamp(c, max=0.0))
    
    s_L, b_L, s_U, b_U = crown_relu_relaxation(z1_l, z1_u)
    crown_l, crown_u = crown_2layer_bounds(x_l, x_u, w1, b1, w2, b2, c)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_l": x_l,
        "x_u": x_u,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "c": c,
        "z1_l_ref": z1_l,
        "z1_u_ref": z1_u,
        "a1_l_ref": a1_l,
        "a1_u_ref": a1_u,
        "s_L_ref": s_L,
        "b_L_ref": b_L,
        "s_U_ref": s_U,
        "b_U_ref": b_U,
        "crown_lower_ref": crown_l,
        "crown_upper_ref": crown_u,
    }
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
