"""Flow Matching Rectified Flow and Consistency Models — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models
Golden weights: weights/05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models.safetensors

``python pytorch2/05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def conditional_flow_matching(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes straight-path interpolation x_t and target velocity u_t(x_1 | x_0) = x_1 - x_0.
    
    t has shape [B, 1], x_0 and x_1 have shape [B, D].
    """
    x_t = (1.0 - t) * x_0 + t * x_1
    v_target = x_1 - x_0
    return x_t, v_target


def velocity_network(
    x: torch.Tensor,
    t: torch.Tensor,
    w_x: torch.Tensor,
    b_x: torch.Tensor,
    w_t: torch.Tensor,
    b_t: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> torch.Tensor:
    """Predicts velocity v_theta(x, t) given position x and time t."""
    t_emb = F.silu(t @ w_t + b_t)
    h = F.silu(x @ w_x + b_x + t_emb)
    v_pred = h @ w_out + b_out
    return v_pred


def rectified_flow_step(x: torch.Tensor, v_pred: torch.Tensor, dt: float) -> torch.Tensor:
    """Simulates one Euler ODE step along the rectified vector field."""
    return x + dt * v_pred


def consistency_coefficients(
    t: torch.Tensor,
    epsilon: float = 0.002,
    sigma_data: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes skip and output scaling coefficients for boundary condition enforcement."""
    dt = t - epsilon
    c_skip = (sigma_data ** 2) / (dt ** 2 + sigma_data ** 2)
    c_out = (sigma_data * dt) / torch.sqrt(sigma_data ** 2 + t ** 2)
    return c_skip, c_out


def consistency_forward(
    x: torch.Tensor,
    t: torch.Tensor,
    f_x: torch.Tensor,
    epsilon: float = 0.002,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """Evaluates consistency function f_theta(x, t) = c_skip(t)*x + c_out(t)*F_theta(x, t)."""
    c_skip, c_out = consistency_coefficients(t, epsilon, sigma_data)
    return c_skip * x + c_out * f_x


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Topic-specific tensors
    x_0 = torch.randn(4, 4)  # Noise source p_0
    x_1 = torch.randn(4, 4)  # Target data p_1
    t = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.float32)

    w_x = torch.randn(4, 8) * 0.2
    b_x = torch.randn(8) * 0.1
    w_t = torch.randn(1, 8) * 0.2
    b_t = torch.randn(8) * 0.1
    w_out = torch.randn(8, 4) * 0.2
    b_out = torch.randn(4) * 0.1

    x_t, v_target = conditional_flow_matching(x_0, x_1, t)
    v_pred = velocity_network(x_t, t, w_x, b_x, w_t, b_t, w_out, b_out)
    cfm_loss = torch.mean((v_pred - v_target) ** 2).reshape(1)
    x_next = rectified_flow_step(x_t, v_pred, dt=0.1)

    c_skip, c_out = consistency_coefficients(t, epsilon=0.002, sigma_data=0.5)
    f_x = v_pred  # using v_pred as neural network output F_theta
    cons_out = consistency_forward(x_t, t, f_x, epsilon=0.002, sigma_data=0.5)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_0": x_0,
        "x_1": x_1,
        "t": t,
        "w_x": w_x,
        "b_x": b_x,
        "w_t": w_t,
        "b_t": b_t,
        "w_out": w_out,
        "b_out": b_out,
        "x_t": x_t,
        "v_target": v_target,
        "v_pred": v_pred,
        "cfm_loss": cfm_loss,
        "x_next": x_next,
        "c_skip": c_skip,
        "c_out": c_out,
        "cons_out": cons_out,
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
