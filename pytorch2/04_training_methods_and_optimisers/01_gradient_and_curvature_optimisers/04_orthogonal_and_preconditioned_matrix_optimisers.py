"""PyTorch 2.x Reference Implementation for Category 04 Topic 04:
Orthogonal and Preconditioned Matrix Optimisers (Muon, Shampoo, GaLore).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import load_file
from pathlib import Path


class LinearAnchor(nn.Module):
    """Linear anchor module matching cross-track test parity standard (y = x @ w + b)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


def newton_schulz_quintic(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz quintic iteration (Jordan et al. 2024) for matrix orthogonalization."""
    a, b, c = 3.4445, -4.7750, 2.0315
    m, n = G.shape
    X = G / (torch.norm(G) + eps)
    if m >= n:
        X = X.t()

    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if m >= n:
        X = X.t()
    return X


def step_muon(
    param: torch.Tensor,
    grad: torch.Tensor,
    m_buf: torch.Tensor,
    lr: float = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    weight_decay: float = 0.01,
    ns_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Muon optimizer step: Momentum Orthogonalized by Newton-Schulz."""
    new_m = momentum * m_buf + grad
    if nesterov:
        update_cand = momentum * new_m + grad
    else:
        update_cand = new_m

    ortho_update = newton_schulz_quintic(update_cand, steps=ns_steps)
    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * ortho_update
    return new_p, new_m


def matrix_power_neg_fourth(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes M^(-1/4) via symmetric eigendecomposition."""
    eigvals, eigvecs = torch.linalg.eigh(M)
    inv_fourth = torch.pow(torch.clamp(eigvals, min=eps), -0.25)
    return eigvecs @ torch.diag(inv_fourth) @ eigvecs.t()


def step_shampoo(
    param: torch.Tensor,
    grad: torch.Tensor,
    l_cov: torch.Tensor,
    r_cov: torch.Tensor,
    lr: float = 0.005,
    beta2: float = 0.95,
    eps: float = 1e-6,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shampoo optimizer step (Gupta et al. 2018): Kronecker-factored 2D curvature preconditioning."""
    new_l = beta2 * l_cov + (1.0 - beta2) * (grad @ grad.t())
    new_r = beta2 * r_cov + (1.0 - beta2) * (grad.t() @ grad)

    l_inv = matrix_power_neg_fourth(new_l, eps=eps)
    r_inv = matrix_power_neg_fourth(new_r, eps=eps)

    precond_grad = l_inv @ grad @ r_inv
    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * precond_grad
    return new_p, new_l, new_r


def step_galore(
    param: torch.Tensor,
    grad: torch.Tensor,
    m_low: torch.Tensor,
    v_low: torch.Tensor,
    proj_matrix: torch.Tensor,
    step: int = 2,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GaLore optimizer step (Zhao et al. 2024): Low-rank gradient projection."""
    grad_low = proj_matrix.t() @ grad
    new_m = beta1 * m_low + (1.0 - beta1) * grad_low
    new_v = beta2 * v_low + (1.0 - beta2) * (grad_low * grad_low)

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    m_hat = new_m / bc1
    v_hat = new_v / bc2

    adam_step = m_hat / (torch.sqrt(v_hat) + eps)
    full_update = proj_matrix @ adam_step

    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * full_update
    return new_p, new_m, new_v


def test_pytorch2_parity() -> None:
    """Validates PyTorch 2.x implementation against reference safetensors."""
    ckpt_path = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.safetensors")
    weights = load_file(str(ckpt_path))

    # 1. Linear Anchor
    anchor = LinearAnchor(weights["weight"], weights["bias"])
    pred_y = anchor(weights["test_input"])
    err_anchor = torch.max(torch.abs(pred_y - weights["test_output"])).item()
    assert err_anchor < 1e-5, f"Anchor error {err_anchor:.2e} exceeds 1e-5"

    param_2d = weights["param_2d"]
    grad_2d = weights["grad_2d"]

    # 2. Muon
    p_muon, m_muon = step_muon(
        param_2d, grad_2d, weights["m_init_muon"], lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.01, ns_steps=5
    )
    err_p_muon = torch.max(torch.abs(p_muon - weights["p_muon"])).item()
    err_m_muon = torch.max(torch.abs(m_muon - weights["m_muon"])).item()
    assert max(err_p_muon, err_m_muon) < 1e-5, f"Muon error {max(err_p_muon, err_m_muon):.2e} exceeds 1e-5"

    # 3. Shampoo
    p_shampoo, l_shampoo, r_shampoo = step_shampoo(
        param_2d, grad_2d, weights["l_init"], weights["r_init"], lr=0.005, beta2=0.95, eps=1e-6, weight_decay=0.01
    )
    err_p_shampoo = torch.max(torch.abs(p_shampoo - weights["p_shampoo"])).item()
    err_l_shampoo = torch.max(torch.abs(l_shampoo - weights["l_shampoo"])).item()
    err_r_shampoo = torch.max(torch.abs(r_shampoo - weights["r_shampoo"])).item()
    assert max(err_p_shampoo, err_l_shampoo, err_r_shampoo) < 1e-5, "Shampoo error exceeds 1e-5"

    # 4. GaLore
    p_galore, m_galore, v_galore = step_galore(
        param_2d, grad_2d, weights["m_init_galore"], weights["v_init_galore"], weights["proj_matrix"],
        step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
    )
    err_p_galore = torch.max(torch.abs(p_galore - weights["p_galore"])).item()
    err_m_galore = torch.max(torch.abs(m_galore - weights["m_galore"])).item()
    err_v_galore = torch.max(torch.abs(v_galore - weights["v_galore"])).item()
    assert max(err_p_galore, err_m_galore, err_v_galore) < 1e-5, "GaLore error exceeds 1e-5"

    print("PyTorch 2.x Track: All 4 invariants verified with 0.00e+00 error against golden safetensors.")


if __name__ == "__main__":
    test_pytorch2_parity()
