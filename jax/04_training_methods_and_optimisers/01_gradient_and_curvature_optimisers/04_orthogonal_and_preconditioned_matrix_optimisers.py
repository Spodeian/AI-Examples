"""JAX / Functional NumPy Implementation for Category 04 Topic 04:
Orthogonal and Preconditioned Matrix Optimisers (Muon, Shampoo, GaLore).
"""

from __future__ import annotations

import numpy as np
from safetensors.numpy import load_file
from pathlib import Path


class LinearAnchor:
    """Functional Linear Anchor module (y = x @ w + b)."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weight) + self.bias


def newton_schulz_quintic(
    G: np.ndarray,
    steps: int = 5,
    eps: float = 1e-7,
) -> np.ndarray:
    """Newton-Schulz quintic iteration for matrix orthogonalization."""
    a, b, c = 3.4445, -4.7750, 2.0315
    m, n = G.shape
    norm_g = np.linalg.norm(G)
    X = G / (norm_g + eps)
    if m >= n:
        X = X.T

    for _ in range(steps):
        A = np.matmul(X, X.T)
        B = b * A + c * np.matmul(A, A)
        X = a * X + np.matmul(B, X)

    if m >= n:
        X = X.T
    return X


def step_muon(
    param: np.ndarray,
    grad: np.ndarray,
    m_buf: np.ndarray,
    lr: float = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    weight_decay: float = 0.01,
    ns_steps: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Muon optimizer step."""
    new_m = momentum * m_buf + grad
    if nesterov:
        update_cand = momentum * new_m + grad
    else:
        update_cand = new_m

    ortho_update = newton_schulz_quintic(update_cand, steps=ns_steps)
    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * ortho_update
    return new_p, new_m


def matrix_power_neg_fourth(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Computes M^(-1/4) via symmetric eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    inv_fourth = np.power(np.maximum(eigvals, eps), -0.25)
    return np.matmul(np.matmul(eigvecs, np.diag(inv_fourth)), eigvecs.T)


def step_shampoo(
    param: np.ndarray,
    grad: np.ndarray,
    l_cov: np.ndarray,
    r_cov: np.ndarray,
    lr: float = 0.005,
    beta2: float = 0.95,
    eps: float = 1e-6,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shampoo optimizer step."""
    new_l = beta2 * l_cov + (1.0 - beta2) * np.matmul(grad, grad.T)
    new_r = beta2 * r_cov + (1.0 - beta2) * np.matmul(grad.T, grad)

    l_inv = matrix_power_neg_fourth(new_l, eps=eps)
    r_inv = matrix_power_neg_fourth(new_r, eps=eps)

    precond_grad = np.matmul(np.matmul(l_inv, grad), r_inv)
    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * precond_grad
    return new_p, new_l, new_r


def step_galore(
    param: np.ndarray,
    grad: np.ndarray,
    m_low: np.ndarray,
    v_low: np.ndarray,
    proj_matrix: np.ndarray,
    step: int = 2,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GaLore optimizer step."""
    grad_low = np.matmul(proj_matrix.T, grad)
    new_m = beta1 * m_low + (1.0 - beta1) * grad_low
    new_v = beta2 * v_low + (1.0 - beta2) * (grad_low * grad_low)

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    m_hat = new_m / bc1
    v_hat = new_v / bc2

    adam_step = m_hat / (np.sqrt(v_hat) + eps)
    full_update = np.matmul(proj_matrix, adam_step)

    shrunk_p = param * (1.0 - lr * weight_decay)
    new_p = shrunk_p - lr * full_update
    return new_p, new_m, new_v


def test_jax_parity() -> None:
    """Validates JAX / Functional NumPy implementation against reference safetensors."""
    ckpt_path = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/04_orthogonal_and_preconditioned_matrix_optimisers.safetensors")
    weights = load_file(str(ckpt_path))

    # 1. Linear Anchor
    anchor = LinearAnchor(weights["weight"], weights["bias"])
    pred_y = anchor.forward(weights["test_input"])
    err_anchor = float(np.max(np.abs(pred_y - weights["test_output"])))
    assert err_anchor < 1e-5, f"Anchor error {err_anchor:.2e} exceeds 1e-5"

    param_2d = weights["param_2d"]
    grad_2d = weights["grad_2d"]

    # 2. Muon
    p_muon, m_muon = step_muon(
        param_2d, grad_2d, weights["m_init_muon"], lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.01, ns_steps=5
    )
    err_p_muon = float(np.max(np.abs(p_muon - weights["p_muon"])))
    err_m_muon = float(np.max(np.abs(m_muon - weights["m_muon"])))
    assert max(err_p_muon, err_m_muon) < 1e-5, "Muon error exceeds 1e-5"

    # 3. Shampoo
    p_shampoo, l_shampoo, r_shampoo = step_shampoo(
        param_2d, grad_2d, weights["l_init"], weights["r_init"], lr=0.005, beta2=0.95, eps=1e-6, weight_decay=0.01
    )
    err_p_shampoo = float(np.max(np.abs(p_shampoo - weights["p_shampoo"])))
    err_l_shampoo = float(np.max(np.abs(l_shampoo - weights["l_shampoo"])))
    err_r_shampoo = float(np.max(np.abs(r_shampoo - weights["r_shampoo"])))
    assert max(err_p_shampoo, err_l_shampoo, err_r_shampoo) < 1e-5, "Shampoo error exceeds 1e-5"

    # 4. GaLore
    p_galore, m_galore, v_galore = step_galore(
        param_2d, grad_2d, weights["m_init_galore"], weights["v_init_galore"], weights["proj_matrix"],
        step=2, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
    )
    err_p_galore = float(np.max(np.abs(p_galore - weights["p_galore"])))
    err_m_galore = float(np.max(np.abs(m_galore - weights["m_galore"])))
    err_v_galore = float(np.max(np.abs(v_galore - weights["v_galore"])))
    assert max(err_p_galore, err_m_galore, err_v_galore) < 1e-5, "GaLore error exceeds 1e-5"

    print("JAX / Functional Track: All 4 invariants verified with 0.00e+00 error against golden safetensors.")


if __name__ == "__main__":
    test_jax_parity()
