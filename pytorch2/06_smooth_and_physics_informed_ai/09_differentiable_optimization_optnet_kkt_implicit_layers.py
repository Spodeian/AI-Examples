"""Differentiable Optimization OptNet KKT Implicit Layers — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers
Golden weights: weights/06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def solve_equality_qp(
    Q: torch.Tensor,
    p: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Solve strictly convex equality-constrained QP per batch:
    
    min_z (1/2) z^T Q z + p^T z  s.t.  A z = b.
    
    KKT Matrix:
    [ Q   A^T ] [ z^\star ] = [ -p ]
    [ A    0  ] [ \nu^\star] = [  b ]
    
    Q: [D, D] SPD
    p: [B, D]
    A: [M, D] (full row rank)
    b: [B, M]
    """
    B, D = p.shape
    M = b.shape[1]
    
    # Construct KKT matrix [D + M, D + M]
    KKT = torch.zeros(D + M, D + M, dtype=Q.dtype, device=Q.device)
    KKT[:D, :D] = Q
    KKT[:D, D:] = A.t()
    KKT[D:, :D] = A
    
    # RHS for batch: [B, D + M]
    rhs = torch.cat([-p, b], dim=-1)
    
    # Solve KKT @ sol.t() = rhs.t()
    sol = torch.linalg.solve(KKT, rhs.t()).t() # [B, D + M]
    z_star = sol[:, :D]
    nu_star = sol[:, D:]
    return z_star, nu_star


def optnet_backward_step(
    Q: torch.Tensor,
    A: torch.Tensor,
    grad_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Differentiate through the KKT conditions (Implicit Function Theorem):
    
    [ Q   A^T ] [ d_z  ] = [ - grad_z ]
    [ A    0  ] [ d_nu ] = [     0     ]
    
    Returns:
    d_p = -d_z
    d_b = -d_nu
    d_Q = 0.5 * (d_z^T z^\star + z^{\star T} d_z) (per batch summed)
    """
    B, D = grad_z.shape
    M = A.shape[0]
    
    KKT = torch.zeros(D + M, D + M, dtype=Q.dtype, device=Q.device)
    KKT[:D, :D] = Q
    KKT[:D, D:] = A.t()
    KKT[D:, :D] = A
    
    rhs_adj = torch.cat([-grad_z, torch.zeros(B, M, dtype=Q.dtype, device=Q.device)], dim=-1)
    sol_adj = torch.linalg.solve(KKT, rhs_adj.t()).t()
    d_z = sol_adj[:, :D]
    d_nu = sol_adj[:, D:]
    
    grad_p = -d_z
    grad_b = -d_nu
    return d_z, grad_p, grad_b


def barrier_qp_solve(
    Q: torch.Tensor,
    p: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    num_steps: int = 15,
    eta: float = 0.05,
    t_barrier: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-barrier smoothed interior point QP:
    
    min_z (1/2) z^T Q z + p^T z - (1/t) sum_i log(h_i - g_i^T z).
    Gradient:
    grad = Q z + p + (1/t) sum_i (g_i / (h_i - g_i^T z)).
    Dual variable estimate: lambda_i = 1 / (t * (h_i - g_i^T z)).
    Gradient descent step: z_{k+1} = z_k - eta * grad.
    """
    B, D = p.shape
    z = torch.zeros(B, D, dtype=Q.dtype, device=Q.device)
    
    for _ in range(num_steps):
        # slacks: h - z @ G.t()  [B, K]
        slacks = h - z @ G.t()
        # clamp slack strictly positive for barrier stability
        safe_slacks = torch.clamp(slacks, min=1e-3)
        dual_lambda = 1.0 / (t_barrier * safe_slacks) # [B, K]
        # barrier grad: dual_lambda @ G  [B, D]
        grad_barrier = dual_lambda @ G
        grad_total = z @ Q + p + grad_barrier
        z = z - eta * grad_total
        
    final_slacks = torch.clamp(h - z @ G.t(), min=1e-4)
    final_lambda = 1.0 / (t_barrier * final_slacks)
    return z, final_lambda


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B, D, M, K = 4, 3, 1, 2
    
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. Equality-Constrained QP Parameters
    # Positive definite Q: M_q^T M_q + 0.2 I
    m_q = torch.randn(D, D) * 0.4
    Q = m_q.t() @ m_q + 0.2 * torch.eye(D)
    p = torch.randn(B, D) * 0.5
    A = torch.randn(M, D) * 0.5
    b = torch.randn(B, M) * 0.2
    
    z_eq, nu_eq = solve_equality_qp(Q, p, A, b)
    
    # 2. Implicit KKT Backward Gradients
    grad_loss_z = torch.randn(B, D) * 0.3
    d_z, grad_p, grad_b = optnet_backward_step(Q, A, grad_loss_z)
    
    # 3. Log-Barrier Inequality QP
    G = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # z_0 <= h_0, z_1 <= h_1
    h_bound = torch.tensor([[1.5, 1.5]]).expand(B, K)
    z_barrier, lambda_barrier = barrier_qp_solve(Q, p, G, h_bound, num_steps=20, eta=0.08, t_barrier=5.0)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        
        "Q": Q,
        "p": p,
        "A": A,
        "b": b,
        "z_eq": z_eq,
        "nu_eq": nu_eq,
        
        "grad_loss_z": grad_loss_z,
        "d_z": d_z,
        "grad_p": grad_p,
        "grad_b": grad_b,
        
        "G": G,
        "h_bound": h_bound,
        "z_barrier": z_barrier,
        "lambda_barrier": lambda_barrier,
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
