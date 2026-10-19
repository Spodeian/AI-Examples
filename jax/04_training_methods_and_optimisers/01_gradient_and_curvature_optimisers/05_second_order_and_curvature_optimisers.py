"""JAX / Functional NumPy Implementation for Category 04 Topic 05:
Second-Order and Curvature Optimisers (L-BFGS, Sophia, AdaHessian, Levenberg-Marquardt).

Provides functional, pure-array implementations of second-order optimization:
1. L-BFGS Two-Loop Recursion: Limited-memory quasi-Newton inverse Hessian-vector product.
2. Sophia-G: Second-order clipped stochastic optimization with diagonal Hessian estimation.
3. AdaHessian: Second-moment adaptive learning rate scaled by diagonal Hessian curvature.
4. Levenberg-Marquardt: Damped Gauss-Newton solver for non-linear least squares.

Enforces exact numerical parity with PyTorch 2, Candle, and Hardware Kernels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file


class LinearAnchor:
    """Functional Linear Anchor module (y = x @ w + b)."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weight) + self.bias


def lbfgs_two_loop(
    s_hist: np.ndarray,
    y_hist: np.ndarray,
    grad: np.ndarray,
) -> np.ndarray:
    """Compute search direction using the classical L-BFGS two-loop recursion.

    Args:
        s_hist: Position displacement history [m, d].
        y_hist: Gradient difference history [m, d].
        grad: Current gradient vector [d].

    Returns:
        r: Approximated inverse Hessian-vector product [d].
    """
    m, _ = s_hist.shape
    q = grad.copy()
    alphas = np.zeros(m, dtype=np.float32)
    rhos = np.zeros(m, dtype=np.float32)

    # First loop: backward recursion
    for i in reversed(range(m)):
        rho_i = 1.0 / np.dot(y_hist[i], s_hist[i])
        rhos[i] = rho_i
        alpha_i = rho_i * np.dot(s_hist[i], q)
        alphas[i] = alpha_i
        q = q - alpha_i * y_hist[i]

    # Initial Hessian scaling: gamma_k * I
    gamma = np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1])
    r = gamma * q

    # Second loop: forward recursion
    for i in range(m):
        beta = rhos[i] * np.dot(y_hist[i], r)
        r = r + s_hist[i] * (alphas[i] - beta)

    return r


def step_lbfgs(
    theta: np.ndarray,
    s_hist: np.ndarray,
    y_hist: np.ndarray,
    grad: np.ndarray,
    lr: float = 0.1,
) -> np.ndarray:
    """Perform one parameter update step using L-BFGS two-loop search direction."""
    direction = lbfgs_two_loop(s_hist, y_hist, grad)
    return theta - lr * direction


def step_sophia(
    theta: np.ndarray,
    grad: np.ndarray,
    momentum: np.ndarray,
    hess_diag: np.ndarray,
    hess_diag_est: np.ndarray,
    lr: float = 0.01,
    beta1: float = 0.96,
    beta2: float = 0.99,
    weight_decay: float = 0.01,
    rho: float = 0.04,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one step of Sophia-G (Second-order Clipped Stochastic Optimization)."""
    m_next = beta1 * momentum + (1.0 - beta1) * grad
    h_next = beta2 * hess_diag + (1.0 - beta2) * hess_diag_est
    ratio = m_next / np.maximum(h_next, eps)
    clipped_ratio = np.clip(ratio, -rho, rho)
    theta_next = theta * (1.0 - lr * weight_decay) - lr * clipped_ratio
    return theta_next, m_next, h_next


def step_adahessian(
    theta: np.ndarray,
    grad: np.ndarray,
    momentum: np.ndarray,
    v: np.ndarray,
    d: np.ndarray,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    step: int = 1,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one step of AdaHessian (Adaptive Hessian Diagonal Scaling)."""
    m_next = beta1 * momentum + (1.0 - beta1) * grad
    v_next = beta2 * v + (1.0 - beta2) * (d ** 2)

    bias_m = float(1.0 - beta1 ** step)
    bias_v = float(1.0 - beta2 ** step)
    m_hat = m_next / bias_m
    v_hat = v_next / bias_v

    update = m_hat / (np.sqrt(v_hat) + eps)
    theta_next = theta * (1.0 - lr * weight_decay) - lr * update
    return theta_next, m_next, v_next


def step_levenberg_marquardt(
    theta: np.ndarray,
    jacobian: np.ndarray,
    residual: np.ndarray,
    damping: float = 0.1,
) -> np.ndarray:
    """Perform one Levenberg-Marquardt damped Gauss-Newton update step."""
    jt = jacobian.T
    jt_j = np.matmul(jt, jacobian)
    dim = jacobian.shape[1]
    a_mat = jt_j + damping * np.eye(dim, dtype=np.float32)
    b_vec = -np.matmul(jt, residual)
    delta = np.linalg.solve(a_mat, b_vec)
    return theta + delta


class SecondOrderCurvatureUnifiedModule:
    """Unified module encapsulating second-order and curvature optimization routines."""

    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        raw = load_file(str(checkpoint_path))
        self.theta = raw["theta"]
        self.grad = raw["grad"]
        self.momentum = raw["momentum"]
        self.hess_diag = raw["hess_diag"]
        self.hess_diag_est = raw["hess_diag_est"]
        self.ada_v = raw["ada_v"]
        self.ada_d = raw["ada_d"]
        self.lbfgs_s = raw["lbfgs_s"]
        self.lbfgs_y = raw["lbfgs_y"]
        self.lm_jacobian = raw["lm_jacobian"]
        self.lm_residual = raw["lm_residual"]

        self.anchor = LinearAnchor(
            weight=raw["anchor_w"],
            bias=raw["anchor_b"],
        )

        self.golden = {
            "lbfgs_theta": raw["golden_lbfgs_theta"],
            "sophia_theta": raw["golden_sophia_theta"],
            "sophia_m": raw["golden_sophia_m"],
            "sophia_h": raw["golden_sophia_h"],
            "adahessian_theta": raw["golden_adahessian_theta"],
            "adahessian_m": raw["golden_adahessian_m"],
            "adahessian_v": raw["golden_adahessian_v"],
            "lm_theta": raw["golden_lm_theta"],
            "anchor_out": raw["golden_anchor_out"],
            "anchor_x": raw["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: L-BFGS parameter update
        lbfgs_theta = step_lbfgs(self.theta, self.lbfgs_s, self.lbfgs_y, self.grad, lr=0.1)
        err_lbfgs = float(np.max(np.abs(lbfgs_theta - self.golden["lbfgs_theta"])))
        assert err_lbfgs < atol, f"L-BFGS invariant failed: {err_lbfgs:.2e}"

        # Invariant 2: Sophia parameter update
        sophia_theta, s_m, s_h = step_sophia(
            self.theta,
            self.grad,
            self.momentum,
            self.hess_diag,
            self.hess_diag_est,
            lr=0.01,
            beta1=0.96,
            beta2=0.99,
            weight_decay=0.01,
            rho=0.04,
            eps=1e-6,
        )
        err_sophia = float(np.max(np.abs(sophia_theta - self.golden["sophia_theta"])))
        assert err_sophia < atol, f"Sophia invariant failed: {err_sophia:.2e}"

        # Invariant 3: AdaHessian parameter update
        ada_theta, a_m, a_v = step_adahessian(
            self.theta,
            self.grad,
            self.momentum,
            self.ada_v,
            self.ada_d,
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
            step=1,
            eps=1e-4,
        )
        err_ada = float(np.max(np.abs(ada_theta - self.golden["adahessian_theta"])))
        assert err_ada < atol, f"AdaHessian invariant failed: {err_ada:.2e}"

        # Invariant 4: Levenberg-Marquardt parameter update
        lm_theta = step_levenberg_marquardt(self.theta, self.lm_jacobian, self.lm_residual, damping=0.1)
        err_lm = float(np.max(np.abs(lm_theta - self.golden["lm_theta"])))
        assert err_lm < atol, f"LM invariant failed: {err_lm:.2e}"

        # Invariant 5: Linear anchor
        anchor_out = self.anchor.forward(self.golden["anchor_x"])
        err_anchor = float(np.max(np.abs(anchor_out - self.golden["anchor_out"])))
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"

        return {
            "lbfgs_err": err_lbfgs,
            "sophia_err": err_sophia,
            "adahessian_err": err_ada,
            "lm_err": err_lm,
            "anchor_err": err_anchor,
        }


def main():
    ckpt_path = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/05_second_order_and_curvature_optimisers.safetensors")
    module = SecondOrderCurvatureUnifiedModule(ckpt_path)
    errors = module.verify_invariants()
    print("JAX/NumPy Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


if __name__ == "__main__":
    main()
