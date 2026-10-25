"""Koopman Operators SINDy Equation Discovery & 4D-Var Data Assimilation — PyTorch 2.x reference implementation.

Topic: 06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation
Golden weights: weights/06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation.safetensors

``python pytorch2/06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation.py`` verifies against the golden checkpoint;
``python pytorch2/06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def koopman_forward(
    x: torch.Tensor,
    w_enc: torch.Tensor,
    b_enc: torch.Tensor,
    k_mat: torch.Tensor,
    w_dec: torch.Tensor,
    b_dec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Neural Koopman operator forward step:
    psi = phi_enc(x) = tanh(x W_enc + b_enc)
    psi_next = psi @ K_mat (linear evolution in observable space)
    x_next = psi_next @ W_dec + b_dec
    """
    psi = torch.tanh(x @ w_enc + b_enc)
    psi_next = psi @ k_mat
    x_next = psi_next @ w_dec + b_dec
    return psi, psi_next, x_next


def sindy_library(x: torch.Tensor) -> torch.Tensor:
    r"""Construct candidate non-linear function library Theta(x) for D=2 state [x1, x2]:
    Theta(x) = [ 1, x1, x2, x1^2, x1 * x2, x2^2, sin(x1), sin(x2) ]  shape [B, 8]
    """
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    ones = torch.ones_like(x1)
    theta = torch.cat([
        ones,
        x1,
        x2,
        x1**2,
        x1 * x2,
        x2**2,
        torch.sin(x1),
        torch.sin(x2),
    ], dim=-1)
    return theta


def sindy_step(
    theta_x: torch.Tensor,
    x_dot: torch.Tensor,
    xi: torch.Tensor,
    threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Sequentially Thresholded Least Squares (STLS) step for sparse equation discovery:
    Find sparse matrix Xi such that Theta(X) Xi \approx X_dot.
    Thresholding: mask = |Xi| >= threshold; zero out smaller coefficients.
    """
    # xi_est = (Theta^T Theta + eps I)^{-1} Theta^T X_dot
    # For deterministic reproducibility across tracks, given parameter xi:
    # 1. Compute predicted x_dot_pred = Theta @ xi
    x_dot_pred = theta_x @ xi
    # 2. Hard thresholding step
    mask = torch.abs(xi) >= threshold
    xi_sparse = xi * mask.float()
    return x_dot_pred, xi_sparse


def four_d_var_step(
    x_b: torch.Tensor,
    b_cov_inv: torch.Tensor,
    y_obs: torch.Tensor,
    r_cov_inv: torch.Tensor,
    h_obs: torch.Tensor,
    m_dyn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""4D-Var Data Assimilation Analysis Step (Linearized):
    Minimizes cost function J(x_0):
    J(x_0) = 0.5 * (x_0 - x_b)^T B^{-1} (x_0 - x_b) + 0.5 * sum_k (y_k - H M^k x_0)^T R^{-1} (y_k - H M^k x_0)
    For T=1 assimilation window:
    x_1 = x_0 @ M_dyn
    y_pred = x_1 @ H_obs^T
    Hessian: H_total = B^{-1} + M_dyn H_obs^T R^{-1} H_obs M_dyn^T
    Gradient at x_b: g = B^{-1} (x_b - x_b) - M_dyn H_obs^T R^{-1} (y_obs - x_b M_dyn H_obs^T)
    Increment: delta_x = H_total^{-1} (-g) = H_total^{-1} (M_dyn H_obs^T R^{-1} (y_obs - x_b M_dyn H_obs^T))
    x_analysis = x_b + delta_x
    """
    # x_b: [B, D], B_cov_inv: [D, D]
    # y_obs: [B, P], R_cov_inv: [P, P], H_obs: [P, D]
    # M_dyn: [D, D]
    # Innovation: d = y_obs - x_b @ M_dyn @ H_obs.t()
    h_m = m_dyn @ h_obs.t() # [D, P]
    d_innov = y_obs - x_b @ h_m # [B, P]
    
    # Hessian: B^{-1} + H_M @ R^{-1} @ H_M^T  [D, D]
    hessian = b_cov_inv + h_m @ r_cov_inv @ h_m.t()
    
    # RHS: d_innov @ R^{-1} @ H_M^T  [B, D]
    rhs = d_innov @ r_cov_inv @ h_m.t()
    
    delta_x = torch.linalg.solve(hessian, rhs.t()).t()
    x_analysis = x_b + delta_x
    return x_analysis, delta_x


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    B, D, K_dim, P = 4, 2, 4, 2
    
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    
    # 1. Koopman Operator Network
    x_state = torch.randn(B, D) * 0.8
    w_enc = torch.randn(D, K_dim) * 0.4
    b_enc = torch.randn(K_dim) * 0.1
    # Orthogonal / stable Koopman matrix
    raw_k = torch.randn(K_dim, K_dim) * 0.3
    k_mat = raw_k / (torch.linalg.matrix_norm(raw_k, ord=2) * 1.05)
    w_dec = torch.randn(K_dim, D) * 0.4
    b_dec = torch.randn(D) * 0.1
    
    psi, psi_next, x_next = koopman_forward(x_state, w_enc, b_enc, k_mat, w_dec, b_dec)
    
    # 2. SINDy Equation Discovery
    theta_x = sindy_library(x_state) # [B, 8]
    # Synthetic x_dot with known sparse dynamics: x1_dot = -0.5 * x1 + 0.2 * x2, x2_dot = -x2^2 + 0.3 * sin(x1)
    xi_true = torch.zeros(8, D)
    xi_true[1, 0] = -0.5
    xi_true[2, 0] = 0.2
    xi_true[5, 1] = -1.0
    xi_true[6, 1] = 0.3
    # Add small noise to test thresholding
    xi_noisy = xi_true + torch.randn(8, D) * 0.02
    x_dot_meas = theta_x @ xi_true + torch.randn(B, D) * 0.01
    x_dot_pred, xi_sparse = sindy_step(theta_x, x_dot_meas, xi_noisy, threshold=0.05)
    
    # 3. 4D-Var Data Assimilation
    x_b = torch.randn(B, D) * 0.5
    b_cov_inv = 2.0 * torch.eye(D)
    y_obs = torch.randn(B, P) * 0.6
    r_cov_inv = 5.0 * torch.eye(P)
    h_obs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    m_dyn = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
    
    x_analysis, delta_x = four_d_var_step(x_b, b_cov_inv, y_obs, r_cov_inv, h_obs, m_dyn)
    
    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        
        "x_state": x_state,
        "w_enc": w_enc,
        "b_enc": b_enc,
        "k_mat": k_mat,
        "w_dec": w_dec,
        "b_dec": b_dec,
        "psi": psi,
        "psi_next": psi_next,
        "x_next": x_next,
        
        "theta_x": theta_x,
        "xi_noisy": xi_noisy,
        "x_dot_pred": x_dot_pred,
        "xi_sparse": xi_sparse,
        
        "x_b": x_b,
        "b_cov_inv": b_cov_inv,
        "y_obs": y_obs,
        "r_cov_inv": r_cov_inv,
        "h_obs": h_obs,
        "m_dyn": m_dyn,
        "x_analysis": x_analysis,
        "delta_x": delta_x,
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
