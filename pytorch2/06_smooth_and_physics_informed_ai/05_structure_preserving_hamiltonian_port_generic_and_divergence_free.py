"""Structure Preserving Hamiltonian Port Generic and Divergence Free — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free
Golden weights: weights/06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def compute_potential_and_grad(
    q: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates potential V(q) and analytical spatial gradient grad_q V."""
    z = q @ w1 + b1
    a = torch.tanh(z)
    v = a @ w2 + b2
    # Analytical gradient: dV/dq = ((1 - a^2) * w2^T) @ w1^T
    temp = (1.0 - a**2) * w2.t()
    grad_v = temp @ w1.t()
    return v, grad_v


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

    # State z = [q, p] with q in R^2, p in R^2 -> z in R^4, batch B = 4
    z_state = torch.randn(4, 4)
    q, p = z_state[:, :2], z_state[:, 2:]

    # Potential V(q) parameters
    w_v1 = torch.randn(2, 8) * 0.5
    b_v1 = torch.randn(8) * 0.1
    w_v2 = torch.randn(8, 1) * 0.5
    b_v2 = torch.randn(1) * 0.1

    v_potential, grad_v = compute_potential_and_grad(q, w_v1, b_v1, w_v2, b_v2)
    t_kinetic = 0.5 * torch.sum(p**2, dim=-1, keepdim=True)
    h_total = t_kinetic + v_potential

    # 1. Hamiltonian Neural Network: dot_q = p, dot_p = -grad_v
    z_dot_hnn = torch.cat([p, -grad_v], dim=-1)

    # 2. Port-Hamiltonian with dissipation gamma = 0.2: dot_p = -grad_v - gamma * p
    gamma = 0.2
    z_dot_phnn = torch.cat([p, -grad_v - gamma * p], dim=-1)

    # 3. Symplectic Leapfrog Integration step with h = 0.1
    h = 0.1
    p_half = p - 0.5 * h * grad_v
    q_next = q + h * p_half
    _, grad_v_next = compute_potential_and_grad(q_next, w_v1, b_v1, w_v2, b_v2)
    p_next = p_half - 0.5 * h * grad_v_next
    z_leapfrog = torch.cat([q_next, p_next], dim=-1)

    # 4. Divergence-Free Solenoidal Field via Stream Function psi(x, y)
    xy = q
    w_psi1 = torch.randn(2, 8) * 0.5
    b_psi1 = torch.randn(8) * 0.1
    w_psi2 = torch.randn(8, 1) * 0.5
    b_psi2 = torch.randn(1) * 0.1

    psi_stream, grad_psi = compute_potential_and_grad(xy, w_psi1, b_psi1, w_psi2, b_psi2)
    # v_solenoidal = [dpsi/dy, -dpsi/dx]
    v_solenoidal = torch.stack([grad_psi[:, 1], -grad_psi[:, 0]], dim=-1)

    # 5. Thermodynamic GENERIC Network: z_dot = L * grad_E + M * grad_S
    # Energy E = H_total, Entropy S = -0.5 * ||z||^2
    grad_E = torch.cat([grad_v, p], dim=-1)
    grad_S = -z_state

    l0_raw = torch.randn(4, 4) * 0.5
    l0_skew = l0_raw - l0_raw.t()

    m0_raw = torch.randn(4, 4) * 0.3
    m0_psd = m0_raw @ m0_raw.t()

    eye = torch.eye(4)
    norm_s_sq = torch.sum(grad_S**2, dim=-1, keepdim=True).unsqueeze(-1) + 1e-8
    p_s = eye.unsqueeze(0) - (grad_S.unsqueeze(-1) @ grad_S.unsqueeze(-2)) / norm_s_sq
    l_proj = p_s @ l0_skew.unsqueeze(0) @ p_s

    norm_e_sq = torch.sum(grad_E**2, dim=-1, keepdim=True).unsqueeze(-1) + 1e-8
    p_e = eye.unsqueeze(0) - (grad_E.unsqueeze(-1) @ grad_E.unsqueeze(-2)) / norm_e_sq
    m_proj = p_e @ m0_psd.unsqueeze(0) @ p_e

    z_dot_generic = (l_proj @ grad_E.unsqueeze(-1) + m_proj @ grad_S.unsqueeze(-1)).squeeze(-1)
    de_dt = torch.sum(grad_E * z_dot_generic, dim=-1)
    ds_dt = torch.sum(grad_S * z_dot_generic, dim=-1)

    tensors.update({
        "z_state": z_state,
        "w_v1": w_v1,
        "b_v1": b_v1,
        "w_v2": w_v2,
        "b_v2": b_v2,
        "v_potential": v_potential,
        "grad_v": grad_v,
        "h_total": h_total,
        "z_dot_hnn": z_dot_hnn,
        "z_dot_phnn": z_dot_phnn,
        "z_leapfrog": z_leapfrog,
        "w_psi1": w_psi1,
        "b_psi1": b_psi1,
        "w_psi2": w_psi2,
        "b_psi2": b_psi2,
        "psi_stream": psi_stream,
        "v_solenoidal": v_solenoidal,
        "l0_skew": l0_skew,
        "m0_psd": m0_psd,
        "z_dot_generic": z_dot_generic,
        "de_dt": de_dt,
        "ds_dt": ds_dt,
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
