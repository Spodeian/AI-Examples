"""Numerical Integration PCEC RK45 Neural ODE LTC and SDE — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde
Golden weights: weights/06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def eval_vector_field(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Evaluates neural ODE vector field f(x) = tanh(x W1 + b1) W2 + b2."""
    z = x @ w1 + b1
    a = torch.tanh(z)
    return a @ w2 + b2


def eval_jacobian(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Computes analytical Jacobian J_{ij} = df_j / dx_i of shape [B, d_in, d_out]."""
    z = x @ w1 + b1
    a = torch.tanh(z)
    d_a = 1.0 - a**2  # [B, 4]
    # Einsum: sum_k W1_{ik} (1 - a_{bk}^2) W2_{kj}
    return torch.einsum("ik, bk, kj -> bij", w1, d_a, w2)


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

    # Initial state x0: B=2, D=2
    x0 = torch.randn(2, 2)
    w1 = torch.randn(2, 4) * 0.5
    b1 = torch.randn(4) * 0.1
    w2 = torch.randn(4, 2) * 0.5
    b2 = torch.randn(2) * 0.1

    f = lambda x: eval_vector_field(x, w1, b1, w2, b2)
    h = 0.1
    f0 = f(x0)

    # 1. Explicit Euler: x_1 = x_0 + h * f0
    x_euler = x0 + h * f0

    # 2. Heun's 2nd-Order Method (RK2):
    k1 = f0
    k2 = f(x0 + h * k1)
    x_heun = x0 + 0.5 * h * (k1 + k2)

    # 3. Predictor-Corrector Scheme (PCEC Adams-Bashforth-Moulton):
    # Predict: x_p = x0 + h * f0
    # Evaluate: f_p = f(x_p)
    # Correct 1: x_c1 = x0 + 0.5 * h * (f0 + f_p)
    # Evaluate 2: f_c = f(x_c1)
    # Correct 2: x_pcec = x0 + 0.5 * h * (f0 + f_c)
    x_p = x0 + h * f0
    f_p = f(x_p)
    x_c1 = x0 + 0.5 * h * (f0 + f_p)
    f_c = f(x_c1)
    x_pcec = x0 + 0.5 * h * (f0 + f_c)

    # 4. Classical Runge-Kutta 4th-Order (RK4):
    rk_k1 = f0
    rk_k2 = f(x0 + 0.5 * h * rk_k1)
    rk_k3 = f(x0 + 0.5 * h * rk_k2)
    rk_k4 = f(x0 + h * rk_k3)
    x_rk4 = x0 + (h / 6.0) * (rk_k1 + 2.0 * rk_k2 + 2.0 * rk_k3 + rk_k4)

    # 5. Continuous Adjoint Sensitivity Backward Step:
    # a(0) = a(T) + h * (a(T) @ J_f)
    adj_T = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    j0 = eval_jacobian(x0, w1, b1, w2)  # [B, 2, 2]
    # a(0)_i = a(T)_i + h * sum_j a(T)_j * df_j/dx_i
    adj_0 = adj_T + h * torch.einsum("bj, bij -> bi", adj_T, j0)

    # 6. Closed-Form Continuous-Time (CfC) Network:
    w_cfc1 = torch.randn(2, 2) * 0.5
    w_cfc2 = torch.randn(2, 2) * 0.5
    gate = torch.sigmoid(-torch.abs(f0) * h)
    x_cfc = gate * (x0 @ w_cfc1) + (1.0 - gate) * (x0 @ w_cfc2)

    # 7. Neural SDE (Euler-Maruyama vs Milstein):
    sigma = 0.2
    dW = torch.randn(2, 2) * (h**0.5)
    x_em = x0 + h * f0 + sigma * x0 * dW
    x_milstein = x_em + 0.5 * (sigma**2) * x0 * (dW**2 - h)

    tensors.update({
        "x0": x0,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "f0": f0,
        "x_euler": x_euler,
        "x_heun": x_heun,
        "x_pcec": x_pcec,
        "x_rk4": x_rk4,
        "adj_T": adj_T,
        "j0": j0,
        "adj_0": adj_0,
        "w_cfc1": w_cfc1,
        "w_cfc2": w_cfc2,
        "x_cfc": x_cfc,
        "dW": dW,
        "x_em": x_em,
        "x_milstein": x_milstein,
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
