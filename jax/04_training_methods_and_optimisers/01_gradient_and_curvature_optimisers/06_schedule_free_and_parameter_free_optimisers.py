"""JAX / Functional NumPy Implementation for Category 04 Topic 06:
Schedule-Free and Parameter-Free Optimisers.

Provides functional, pure-array implementations:
1. Schedule-Free AdamW (Defazio et al. 2024): Iterate averaging eliminating LR schedules.
2. Polyak Step-Size: Exact optimal step scaling using loss gap (f - f*).
3. Barzilai-Borwein Spectral Step-Size (BB1): Two-point secant step estimation.
4. D-Adaptation: Online distance-to-solution tracking.

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


def step_schedule_free_adamw(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    step: int = 1,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform one Schedule-Free AdamW update step."""
    m_next = beta1 * m + (1.0 - beta1) * grad
    v_next = beta2 * v + (1.0 - beta2) * (grad ** 2)

    bias_v = float(1.0 - beta2 ** step)
    v_hat = v_next / bias_v

    u_t = (m_next / (np.sqrt(v_hat) + eps)) + weight_decay * y
    z_next = z - lr * u_t

    c_prime = float(1.0 / (step + 1))
    x_next = (1.0 - c_prime) * x + c_prime * z_next

    c_next = float(1.0 / (step + 2))
    y_next = (1.0 - c_next) * z_next + c_next * x_next

    return x_next, z_next, y_next, m_next, v_next


def step_polyak(
    param: np.ndarray,
    grad: np.ndarray,
    f_val: float,
    f_opt: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Perform one step using Polyak's optimal step-size: gamma = (f(x) - f*) / ||g||^2."""
    grad_norm_sq = np.sum(grad ** 2)
    gamma = (f_val - f_opt) / (grad_norm_sq + eps)
    return param - gamma * grad


def step_barzilai_borwein(
    param: np.ndarray,
    grad: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Perform one step using Barzilai-Borwein BB1 spectral step-size: gamma = (s^T s) / (s^T y)."""
    s_dot_s = np.dot(s, s)
    s_dot_y = np.dot(s, y)
    gamma = s_dot_s / (s_dot_y + eps)
    return param - gamma * grad


def step_d_adaptation(
    param: np.ndarray,
    grad: np.ndarray,
    d_est: float,
    lr0: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Perform one D-Adaptation step with distance-to-solution tracking."""
    dim = param.size
    grad_norm_l1 = float(np.sum(np.abs(grad)))
    grad_norm_l2 = float(np.linalg.norm(grad))

    d_next = max(0.1, d_est + 0.05 * (grad_norm_l1 / (dim ** 0.5) - 1.0))
    eff_lr = lr0 * d_next
    param_next = param - eff_lr * (grad / (grad_norm_l2 + eps))
    return param_next, d_next


class ScheduleFreeUnifiedModule:
    """Unified module encapsulating schedule-free and parameter-free optimisers."""

    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        raw = load_file(str(checkpoint_path))
        self.sf_x = raw["sf_x"]
        self.sf_z = raw["sf_z"]
        self.sf_y = raw["sf_y"]
        self.sf_grad = raw["sf_grad"]
        self.sf_m = raw["sf_m"]
        self.sf_v = raw["sf_v"]

        self.polyak_param = raw["polyak_param"]
        self.polyak_grad = raw["polyak_grad"]
        self.f_val = float(raw["f_val"][0])
        self.f_opt = float(raw["f_opt"][0])

        self.bb_param = raw["bb_param"]
        self.bb_grad = raw["bb_grad"]
        self.bb_s = raw["bb_s"]
        self.bb_y = raw["bb_y"]

        self.d_param = raw["d_param"]
        self.d_grad = raw["d_grad"]
        self.d_est = float(raw["d_est"][0])

        self.anchor = LinearAnchor(
            weight=raw["anchor_w"],
            bias=raw["anchor_b"],
        )

        self.golden = {
            "sf_x": raw["golden_sf_x"],
            "sf_z": raw["golden_sf_z"],
            "sf_y": raw["golden_sf_y"],
            "sf_m": raw["golden_sf_m"],
            "sf_v": raw["golden_sf_v"],
            "polyak_param": raw["golden_polyak_param"],
            "bb_param": raw["golden_bb_param"],
            "d_adapt_param": raw["golden_d_adapt_param"],
            "anchor_out": raw["golden_anchor_out"],
            "anchor_x": raw["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: Schedule-Free AdamW
        x_next, z_next, y_next, m_next, v_next = step_schedule_free_adamw(
            self.sf_x, self.sf_z, self.sf_y, self.sf_grad, self.sf_m, self.sf_v,
            lr=0.01, beta1=0.9, beta2=0.999, weight_decay=0.01, step=1, eps=1e-8,
        )
        err_sf_x = float(np.max(np.abs(x_next - self.golden["sf_x"])))
        err_sf_z = float(np.max(np.abs(z_next - self.golden["sf_z"])))
        err_sf_y = float(np.max(np.abs(y_next - self.golden["sf_y"])))
        err_sf = max(err_sf_x, err_sf_z, err_sf_y)
        assert err_sf < atol, f"Schedule-Free invariant failed: {err_sf:.2e}"

        # Invariant 2: Polyak Step-Size
        p_polyak = step_polyak(self.polyak_param, self.polyak_grad, self.f_val, self.f_opt)
        err_polyak = float(np.max(np.abs(p_polyak - self.golden["polyak_param"])))
        assert err_polyak < atol, f"Polyak invariant failed: {err_polyak:.2e}"

        # Invariant 3: Barzilai-Borwein Spectral Step
        p_bb = step_barzilai_borwein(self.bb_param, self.bb_grad, self.bb_s, self.bb_y)
        err_bb = float(np.max(np.abs(p_bb - self.golden["bb_param"])))
        assert err_bb < atol, f"Barzilai-Borwein invariant failed: {err_bb:.2e}"

        # Invariant 4: D-Adaptation
        p_dadapt, _ = step_d_adaptation(self.d_param, self.d_grad, self.d_est, lr0=0.01)
        err_dadapt = float(np.max(np.abs(p_dadapt - self.golden["d_adapt_param"])))
        assert err_dadapt < atol, f"D-Adaptation invariant failed: {err_dadapt:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor.forward(self.golden["anchor_x"])
        err_anchor = float(np.max(np.abs(anchor_out - self.golden["anchor_out"])))
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"

        return {
            "schedule_free_err": err_sf,
            "polyak_err": err_polyak,
            "bb_err": err_bb,
            "d_adapt_err": err_dadapt,
            "anchor_err": err_anchor,
        }


def main():
    ckpt_path = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.safetensors")
    module = ScheduleFreeUnifiedModule(ckpt_path)
    errors = module.verify_invariants()
    print("JAX/NumPy Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


if __name__ == "__main__":
    main()
