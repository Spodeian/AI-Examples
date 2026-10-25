"""Capstone Physics Smooth and Certified AI Pipeline — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline
Golden weights: weights/17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def compute_hamiltonian_energy(
    q: torch.Tensor,
    p: torch.Tensor,
    w_v: torch.Tensor,
) -> torch.Tensor:
    """Computes total Hamiltonian mechanical energy: H = T(p) + V(q) = 0.5 * ||p||^2 + 0.5 * q W_V q^T."""
    kinetic = 0.5 * torch.sum(p * p, dim=-1)
    potential = 0.5 * torch.sum(q * (q @ w_v), dim=-1)
    return kinetic + potential


def symplectic_leapfrog_step(
    q: torch.Tensor,
    p: torch.Tensor,
    w_v: torch.Tensor,
    dt: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Executes a single symplectic leapfrog integration step preserving Hamiltonian phase space.
    
    p_{1/2} = p - (dt / 2) * grad_V(q)
    q_{next} = q + dt * p_{1/2}
    p_{next} = p_{1/2} - (dt / 2) * grad_V(q_{next})
    """
    grad_v_0 = q @ w_v
    p_half = p - 0.5 * dt * grad_v_0
    q_next = q + dt * p_half
    grad_v_1 = q_next @ w_v
    p_next = p_half - 0.5 * dt * grad_v_1
    return q_next, p_next


def certified_smooth_forward(
    q_next: torch.Tensor,
    w_l: torch.Tensor,
    b_l: torch.Tensor,
) -> torch.Tensor:
    """Applies contractive 1-Lipschitz readout layer with SiLU activation."""
    z = q_next @ w_l + b_l
    return F.silu(z)


def physics_certified_pipeline(
    q: torch.Tensor,
    p: torch.Tensor,
    w_v: torch.Tensor,
    w_l: torch.Tensor,
    b_l: torch.Tensor,
    dt: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end physics-informed and certified pipeline execution."""
    energy_init = compute_hamiltonian_energy(q, p, w_v)
    q_next, p_next = symplectic_leapfrog_step(q, p, w_v, dt=dt)
    energy_next = compute_hamiltonian_energy(q_next, p_next, w_v)
    y_smooth = certified_smooth_forward(q_next, w_l, b_l)
    return y_smooth, q_next, p_next, energy_init, energy_next


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    b, d = 2, 4

    q = torch.randn(b, d)
    p = torch.randn(b, d)

    # Symmetric positive definite matrix for potential energy V(q) = 0.5 * q W_V q^T
    raw_v = torch.randn(d, d) * 0.2
    w_v = (raw_v.T @ raw_v) + 0.1 * torch.eye(d)

    w_l = torch.randn(d, d) * 0.2
    b_l = torch.randn(d) * 0.1

    y_smooth, q_next, p_next, e_init, e_next = physics_certified_pipeline(
        q, p, w_v, w_l, b_l, dt=0.05
    )

    # Energy conservation check: Leapfrog preserves energy to high precision
    energy_diff = torch.max(torch.abs(e_next - e_init))
    assert energy_diff < 0.05

    linear_w = torch.randn(d, d) * 0.1
    linear_b = torch.randn(d) * 0.1
    test_out = linear_anchor(y_smooth, linear_w, linear_b)

    tensors = {
        "q": q,
        "p": p,
        "w_v": w_v,
        "w_l": w_l,
        "b_l": b_l,
        "q_next": q_next,
        "p_next": p_next,
        "energy_init": e_init,
        "energy_next": e_next,
        "y_smooth": y_smooth,
        "linear_w": linear_w,
        "linear_b": linear_b,
        "test_input": y_smooth.clone(),
        "weight": linear_w.clone(),
        "bias": linear_b.clone(),
        "test_output": test_out,
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
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
