"""Deep Equilibrium Models DEQ Implicit Differentiation — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation
Golden weights: weights/06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def f_theta(z: torch.Tensor, x: torch.Tensor, w_z: torch.Tensor, w_x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Nonlinear equilibrium step: f_theta(z, x) = tanh(z @ w_z + x @ w_x + b)."""
    return torch.tanh(z @ w_z + x @ w_x + b)


def fixed_point_iteration(
    x: torch.Tensor,
    w_z: torch.Tensor,
    w_x: torch.Tensor,
    b: torch.Tensor,
    num_iters: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Picard iteration: z_{k+1} = f_theta(z_k, x) starting from z_0 = 0."""
    z = torch.zeros(x.shape[0], w_z.shape[0], dtype=x.dtype, device=x.device)
    for _ in range(num_iters):
        z = f_theta(z, x, w_z, w_x, b)
    res = z - f_theta(z, x, w_z, w_x, b)
    return z, res


def anderson_acceleration(
    x: torch.Tensor,
    w_z: torch.Tensor,
    w_x: torch.Tensor,
    b: torch.Tensor,
    m: int = 3,
    num_iters: int = 15,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anderson acceleration (type II) with history depth m."""
    B, D = x.shape[0], w_z.shape[0]
    z = torch.zeros(B, D, dtype=x.dtype, device=x.device)
    
    # Store history of iterates and residuals per batch item
    # To maintain strict deterministic tensor arithmetic, run for each item or vectorized
    # Given small B=4, D=4, m=3, vectorizing over batch:
    z_hist = []
    g_hist = [] # g(z) = f(z)
    
    for k in range(num_iters):
        g = f_theta(z, x, w_z, w_x, b)
        res = g - z
        z_hist.append(z)
        g_hist.append(g)
        if len(z_hist) > m + 1:
            z_hist.pop(0)
            g_hist.pop(0)
            
        k_hist = len(z_hist) - 1
        if k_hist == 0:
            z = g
        else:
            # Solve min || Delta R * gamma ||_2^2 s.t. sum(gamma) = 1
            # Or unconstrained on alpha with gamma_m = 1 - sum(alpha):
            # Delta R_i = res_i - res_{i-1}
            # For exact reproducibility and simplicity across all backends:
            # Solve standard Anderson formulation per batch element:
            z_next = torch.empty_like(z)
            for b_idx in range(B):
                # Residuals for last k_hist+1 iterates:
                R_mat = torch.stack([g_hist[j][b_idx] - z_hist[j][b_idx] for j in range(k_hist + 1)], dim=1) # [D, k_hist+1]
                # Delta R = R_j - R_{j-1}
                d_R = R_mat[:, 1:] - R_mat[:, :-1] # [D, k_hist]
                # Least squares solve for (d_R^T d_R) alpha = d_R^T R_last
                # Regularize with eps * I for stability
                reg = 1e-5 * torch.eye(k_hist, dtype=x.dtype, device=x.device)
                rhs = d_R.t() @ R_mat[:, -1:] # [k_hist, 1]
                lhs = d_R.t() @ d_R + reg # [k_hist, k_hist]
                alpha = torch.linalg.solve(lhs, rhs).squeeze(1) # [k_hist]
                
                # gamma:
                # gamma_0 = alpha_0
                # gamma_j = alpha_j - alpha_{j-1}
                # gamma_k = 1 - alpha_{k-1}
                # Equivalently: z_next = g_last - sum_{j=0}^{k_hist-1} alpha_j * (g_{j+1} - g_j)
                g_diff = torch.stack([g_hist[j+1][b_idx] - g_hist[j][b_idx] for j in range(k_hist)], dim=1) # [D, k_hist]
                z_b = g_hist[-1][b_idx] - (g_diff @ alpha)
                z_next[b_idx] = (1.0 - beta) * (z_hist[-1][b_idx] - ( (torch.stack([z_hist[j+1][b_idx] - z_hist[j][b_idx] for j in range(k_hist)], dim=1)) @ alpha )) + beta * z_b
            z = z_next
            
    final_res = z - f_theta(z, x, w_z, w_x, b)
    return z, final_res


def implicit_jacobian_transpose_solve(
    z_star: torch.Tensor,
    x: torch.Tensor,
    w_z: torch.Tensor,
    w_x: torch.Tensor,
    b: torch.Tensor,
    grad_z: torch.Tensor,
    num_steps: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve (I - J_f^T) v = grad_z via Neumann series / fixed point:
    
    v_{t+1} = grad_z + J_f^T v_t.
    Then compute grad_x = v^T (df/dx) and grad_wz = v^T (df/dwz).
    
    For f(z) = tanh(u), u = z @ w_z + x @ w_x + b,
    df/dz = diag(1 - tanh^2(u)) @ w_z^T.
    J_f^T v = w_z @ ( (1 - tanh^2(u)) * v ).
    """
    u = z_star @ w_z + x @ w_x + b
    tanh_u = torch.tanh(u)
    d_act = 1.0 - tanh_u**2  # [B, D]
    
    # Vector-Jacobian product v @ J_f:
    # J_f^T v = (v * d_act) @ w_z.t()
    v = torch.zeros_like(grad_z)
    for _ in range(num_steps):
        v = grad_z + (v * d_act) @ w_z.t()
        
    # Now compute gradients:
    # delta = v * d_act  [B, D]
    delta = v * d_act
    # df/dx is delta @ w_x.t()
    grad_x = delta @ w_x.t()
    # df/dw_z is z_star.t() @ delta
    grad_wz = z_star.t() @ delta
    return v, grad_x, grad_wz


def mon_operator_splitting(
    x: torch.Tensor,
    w_sym: torch.Tensor,
    w_skew: torch.Tensor,
    w_x: torch.Tensor,
    b: torch.Tensor,
    alpha_step: float = 0.5,
    num_iters: int = 15,
) -> torch.Tensor:
    """Monotone Operator DEQ (Peaceman-Rachford / Cayley operator splitting step):
    
    f(z) = (I - alpha*W)^{-1} [ (I + alpha*W_skew) z - alpha * (W_sym z + act(z) + x @ w_x + b) ].
    For a simplified contractive monotone operator:
    W = - (W_sym + W_skew), where W_sym is PSD (W_sym = M^T M + gamma I).
    Forward update with resolvent step:
    z_{k+1} = Prox_{alpha * phi} ( z_k - alpha * (W z_k + x @ w_x + b) )
    where Prox is elementwise clip / tanh / soft-threshold.
    Using clipped / smooth monotone activation:
    z_{k+1} = tanh( z_k - alpha_step * (z_k @ (w_sym + w_skew) + x @ w_x + b) ).
    """
    W = w_sym + w_skew
    z = torch.zeros(x.shape[0], W.shape[0], dtype=x.dtype, device=x.device)
    for _ in range(num_iters):
        lin = z @ W + x @ w_x + b
        z = torch.tanh(z - alpha_step * lin)
    return z


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B, D, In_D = 4, 4, 3
    
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # DEQ parameters
    # Stable weight matrix with spectral radius < 1:
    raw_wz = torch.randn(D, D) * 0.3
    # enforce contractivity: normalize by spectral norm / trace
    w_z = raw_wz / (torch.linalg.matrix_norm(raw_wz, ord=2) * 1.5)
    w_x = torch.randn(In_D, D) * 0.4
    b_z = torch.randn(D) * 0.1
    x_input = torch.randn(B, In_D)
    
    # 1. Picard Fixed-Point Iteration
    z_picard, res_picard = fixed_point_iteration(x_input, w_z, w_x, b_z, num_iters=20)
    
    # 2. Anderson Acceleration (Type II)
    z_anderson, res_anderson = anderson_acceleration(x_input, w_z, w_x, b_z, m=2, num_iters=15, beta=1.0)
    
    # 3. Implicit Differentiation via Adjoint Fixed-Point / Neumann solve
    grad_loss_z = torch.randn(B, D) * 0.5
    adj_v, grad_x, grad_wz = implicit_jacobian_transpose_solve(
        z_picard, x_input, w_z, w_x, b_z, grad_loss_z, num_steps=20
    )
    
    # 4. Monotone Operator Network parameters
    m_mat = torch.randn(D, D) * 0.3
    w_sym = m_mat.t() @ m_mat + 0.1 * torch.eye(D)
    raw_skew = torch.randn(D, D) * 0.3
    w_skew = raw_skew - raw_skew.t()
    w_mon_x = torch.randn(In_D, D) * 0.3
    b_mon = torch.randn(D) * 0.1
    
    z_mon = mon_operator_splitting(x_input, w_sym, w_skew, w_mon_x, b_mon, alpha_step=0.2, num_iters=15)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        
        "x_input": x_input,
        "w_z": w_z,
        "w_x": w_x,
        "b_z": b_z,
        "z_picard": z_picard,
        "res_picard": res_picard,
        
        "z_anderson": z_anderson,
        "res_anderson": res_anderson,
        
        "grad_loss_z": grad_loss_z,
        "adj_v": adj_v,
        "grad_x": grad_x,
        "grad_wz": grad_wz,
        
        "w_sym": w_sym,
        "w_skew": w_skew,
        "w_mon_x": w_mon_x,
        "b_mon": b_mon,
        "z_mon": z_mon,
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
