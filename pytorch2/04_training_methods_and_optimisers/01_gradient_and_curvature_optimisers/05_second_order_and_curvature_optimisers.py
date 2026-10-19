"""Second-Order and Curvature Optimisers: L-BFGS, Sophia, AdaHessian, and Levenberg-Marquardt.

Provides mathematically rigorous, PyTorch 2.x implementations of curvature-informed optimization:
1. L-BFGS Two-Loop Recursion: Limited-memory quasi-Newton approximation of inverse Hessian-vector product.
2. Sophia-G: Second-order clipped stochastic optimization with Gauss-Newton-Bartlett diagonal Hessian estimation.
3. AdaHessian: Second-moment adaptive learning rate scaled by diagonal Hessian curvature.
4. Levenberg-Marquardt: Damped Gauss-Newton solver for non-linear least squares.

All implementations strictly conform to numerical parity with JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file


def lbfgs_two_loop(
    s_hist: torch.Tensor,
    y_hist: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    """Compute search direction using the classical L-BFGS two-loop recursion.

    Args:
        s_hist: Position displacement history [m, d], where s_k = x_{k+1} - x_k.
        y_hist: Gradient difference history [m, d], where y_k = g_{k+1} - g_k.
        grad: Current gradient vector [d].

    Returns:
        r: Approximated inverse Hessian-vector product H_k * grad [d].
    """
    m, _ = s_hist.shape
    q = grad.clone()
    alphas = torch.zeros(m, dtype=torch.float32, device=grad.device)
    rhos = torch.zeros(m, dtype=torch.float32, device=grad.device)

    # First loop: backward recursion
    for i in reversed(range(m)):
        rho_i = 1.0 / torch.dot(y_hist[i], s_hist[i])
        rhos[i] = rho_i
        alpha_i = rho_i * torch.dot(s_hist[i], q)
        alphas[i] = alpha_i
        q = q - alpha_i * y_hist[i]

    # Initial Hessian scaling: gamma_k * I
    gamma = torch.dot(s_hist[-1], y_hist[-1]) / torch.dot(y_hist[-1], y_hist[-1])
    r = gamma * q

    # Second loop: forward recursion
    for i in range(m):
        beta = rhos[i] * torch.dot(y_hist[i], r)
        r = r + s_hist[i] * (alphas[i] - beta)

    return r


def step_lbfgs(
    theta: torch.Tensor,
    s_hist: torch.Tensor,
    y_hist: torch.Tensor,
    grad: torch.Tensor,
    lr: float = 0.1,
) -> torch.Tensor:
    """Perform one parameter update step using L-BFGS two-loop search direction."""
    direction = lbfgs_two_loop(s_hist, y_hist, grad)
    return theta - lr * direction


def step_sophia(
    theta: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    hess_diag: torch.Tensor,
    hess_diag_est: torch.Tensor,
    lr: float = 0.01,
    beta1: float = 0.96,
    beta2: float = 0.99,
    weight_decay: float = 0.01,
    rho: float = 0.04,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one step of Sophia-G (Second-order Clipped Stochastic Optimization).

    Updates:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        h_t = beta2 * h_{t-1} + (1 - beta2) * diag(H)_t
        theta_{t+1} = theta_t * (1 - lr * wd) - lr * clamp(m_t / max(h_t, eps), -rho, rho)
    """
    m_next = beta1 * momentum + (1.0 - beta1) * grad
    h_next = beta2 * hess_diag + (1.0 - beta2) * hess_diag_est
    ratio = m_next / torch.clamp(h_next, min=eps)
    clipped_ratio = torch.clamp(ratio, -rho, rho)
    theta_next = theta * (1.0 - lr * weight_decay) - lr * clipped_ratio
    return theta_next, m_next, h_next


def step_adahessian(
    theta: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    v: torch.Tensor,
    d: torch.Tensor,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    step: int = 1,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one step of AdaHessian (Adaptive Hessian Diagonal Scaling).

    Updates:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * d_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_{t+1} = theta_t * (1 - lr * wd) - lr * (m_hat / (sqrt(v_hat) + eps))
    """
    m_next = beta1 * momentum + (1.0 - beta1) * grad
    v_next = beta2 * v + (1.0 - beta2) * (d ** 2)

    bias_m = float(1.0 - beta1 ** step)
    bias_v = float(1.0 - beta2 ** step)
    m_hat = m_next / bias_m
    v_hat = v_next / bias_v

    update = m_hat / (torch.sqrt(v_hat) + eps)
    theta_next = theta * (1.0 - lr * weight_decay) - lr * update
    return theta_next, m_next, v_next


def step_levenberg_marquardt(
    theta: torch.Tensor,
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    damping: float = 0.1,
) -> torch.Tensor:
    """Perform one Levenberg-Marquardt damped Gauss-Newton update step.

    Solves:
        (J^T J + damping * I) delta = - J^T R
        theta_{t+1} = theta_t + delta
    """
    jt = jacobian.T
    jt_j = torch.matmul(jt, jacobian)
    dim = jacobian.shape[1]
    a_mat = jt_j + damping * torch.eye(dim, dtype=torch.float32, device=theta.device)
    b_vec = -torch.matmul(jt, residual)
    delta = torch.linalg.solve(a_mat, b_vec)
    return theta + delta


class LinearAnchor(nn.Module):
    """Linear projection layer serving as deterministic anchor across all tracks."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class SecondOrderCurvatureUnifiedModule(nn.Module):
    """Unified module encapsulating second-order and curvature optimization routines."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        tensors = load_file(str(checkpoint_path))
        self.theta = tensors["theta"]
        self.grad = tensors["grad"]
        self.momentum = tensors["momentum"]
        self.hess_diag = tensors["hess_diag"]
        self.hess_diag_est = tensors["hess_diag_est"]
        self.ada_v = tensors["ada_v"]
        self.ada_d = tensors["ada_d"]
        self.lbfgs_s = tensors["lbfgs_s"]
        self.lbfgs_y = tensors["lbfgs_y"]
        self.lm_jacobian = tensors["lm_jacobian"]
        self.lm_residual = tensors["lm_residual"]

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(tensors["anchor_w"].T)
        self.anchor.bias.data.copy_(tensors["anchor_b"])

        self.golden = {
            "lbfgs_theta": tensors["golden_lbfgs_theta"],
            "sophia_theta": tensors["golden_sophia_theta"],
            "sophia_m": tensors["golden_sophia_m"],
            "sophia_h": tensors["golden_sophia_h"],
            "adahessian_theta": tensors["golden_adahessian_theta"],
            "adahessian_m": tensors["golden_adahessian_m"],
            "adahessian_v": tensors["golden_adahessian_v"],
            "lm_theta": tensors["golden_lm_theta"],
            "anchor_out": tensors["golden_anchor_out"],
            "anchor_x": tensors["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: L-BFGS parameter update
        lbfgs_theta = step_lbfgs(self.theta, self.lbfgs_s, self.lbfgs_y, self.grad, lr=0.1)
        err_lbfgs = float(torch.max(torch.abs(lbfgs_theta - self.golden["lbfgs_theta"])))
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
        err_sophia = float(torch.max(torch.abs(sophia_theta - self.golden["sophia_theta"])))
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
        err_ada = float(torch.max(torch.abs(ada_theta - self.golden["adahessian_theta"])))
        assert err_ada < atol, f"AdaHessian invariant failed: {err_ada:.2e}"

        # Invariant 4: Levenberg-Marquardt parameter update
        lm_theta = step_levenberg_marquardt(self.theta, self.lm_jacobian, self.lm_residual, damping=0.1)
        err_lm = float(torch.max(torch.abs(lm_theta - self.golden["lm_theta"])))
        assert err_lm < atol, f"LM invariant failed: {err_lm:.2e}"

        # Invariant 5: Linear anchor
        anchor_out = self.anchor(self.golden["anchor_x"]).detach()
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden["anchor_out"])))
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
    print("PyTorch 2.x Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


if __name__ == "__main__":
    main()
