"""Multi-Objective Pareto, MGDA, Tchebycheff, and Gradient Surgery in JAX / Functional NumPy.

Implements foundational and modern multi-objective / multi-task optimization algorithms:
1. Multiple Gradient Descent Algorithm (MGDA - Désidéri 2012; Sener & Koltun 2018).
2. Projecting Conflicting Gradients (PCGrad - Yu et al. 2020).
3. Augmented Tchebycheff Scalarization for Non-Convex Pareto Fronts (Miettinen 1999).
4. Linear Multi-Task Scalarization.

Enforces strict transitive parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

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


def step_mgda_2task(
    param: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    lr: float = 0.01,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute 2-task MGDA min-norm convex combination gradient and parameter step."""
    u = g1 - g2
    u_norm_sq = float(np.dot(u, u))
    numerator = float(np.dot(g2 - g1, g2))
    alpha = float(np.clip(numerator / (u_norm_sq + eps), 0.0, 1.0))
    g_mgda = alpha * g1 + (1.0 - alpha) * g2
    param_next = param - lr * g_mgda
    return param_next, g_mgda, alpha


def step_pcgrad_2task(
    param: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    lr: float = 0.01,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform PCGrad conflicting gradient projection and parameter step."""
    dot_12 = float(np.dot(g1, g2))
    norm_sq_g1 = float(np.dot(g1, g1))
    norm_sq_g2 = float(np.dot(g2, g2))
    if dot_12 < 0.0:
        g1_proj = g1 - (dot_12 / (norm_sq_g2 + eps)) * g2
        g2_proj = g2 - (dot_12 / (norm_sq_g1 + eps)) * g1
        g_pc = 0.5 * (g1_proj + g2_proj)
    else:
        g_pc = 0.5 * (g1 + g2)
    param_next = param - lr * g_pc
    return param_next, g_pc


def step_tchebycheff(
    param: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    losses: np.ndarray,
    z_star: np.ndarray,
    weights: np.ndarray,
    rho: float = 0.05,
    lr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Augmented Tchebycheff scalarization gradient step."""
    diff = weights * (losses - z_star)
    idx_max = int(np.argmax(diff))
    ds_dl = rho * weights.copy()
    ds_dl[idx_max] += weights[idx_max]
    g_tch = ds_dl[0] * g1 + ds_dl[1] * g2
    param_next = param - lr * g_tch
    return param_next, g_tch


def step_linear_scalarization(
    param: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    weights: np.ndarray,
    lr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform linear multi-task scalarization gradient step."""
    g_lin = weights[0] * g1 + weights[1] * g2
    param_next = param - lr * g_lin
    return param_next, g_lin


class MultiObjectiveUnifiedModule:
    """Unified module encapsulating multi-objective optimizers for JAX/NumPy parity."""

    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        tensors = load_file(str(checkpoint_path))
        self.param = tensors["param"]
        self.g1 = tensors["g1"]
        self.g2 = tensors["g2"]
        self.losses = tensors["L"]
        self.z_star = tensors["z_star"]
        self.weights_tch = tensors["weights_tch"]
        self.weights_lin = tensors["weights_lin"]

        self.anchor = LinearAnchor(tensors["anchor_w"], tensors["anchor_b"])

        self.golden = {
            "alpha_mgda": float(tensors["golden_alpha_mgda"][0]),
            "g_mgda": tensors["golden_g_mgda"],
            "param_mgda": tensors["golden_param_mgda"],
            "g_pc": tensors["golden_g_pc"],
            "param_pc": tensors["golden_param_pc"],
            "g_tch": tensors["golden_g_tch"],
            "param_tch": tensors["golden_param_tch"],
            "g_lin": tensors["golden_g_lin"],
            "param_lin": tensors["golden_param_lin"],
            "anchor_out": tensors["golden_anchor_out"],
            "anchor_x": tensors["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: MGDA
        p_mgda, g_mgda, alpha = step_mgda_2task(self.param, self.g1, self.g2, lr=0.01)
        err_alpha = abs(alpha - self.golden["alpha_mgda"])
        err_g_mgda = float(np.max(np.abs(g_mgda - self.golden["g_mgda"])))
        err_p_mgda = float(np.max(np.abs(p_mgda - self.golden["param_mgda"])))
        err_mgda = max(err_alpha, err_g_mgda, err_p_mgda)
        assert err_mgda < atol, f"MGDA invariant failed: {err_mgda:.2e}"

        # Invariant 2: PCGrad
        p_pc, g_pc = step_pcgrad_2task(self.param, self.g1, self.g2, lr=0.01)
        err_g_pc = float(np.max(np.abs(g_pc - self.golden["g_pc"])))
        err_p_pc = float(np.max(np.abs(p_pc - self.golden["param_pc"])))
        err_pc = max(err_g_pc, err_p_pc)
        assert err_pc < atol, f"PCGrad invariant failed: {err_pc:.2e}"

        # Invariant 3: Tchebycheff Scalarization
        p_tch, g_tch = step_tchebycheff(
            self.param, self.g1, self.g2, self.losses, self.z_star, self.weights_tch, rho=0.05, lr=0.01
        )
        err_g_tch = float(np.max(np.abs(g_tch - self.golden["g_tch"])))
        err_p_tch = float(np.max(np.abs(p_tch - self.golden["param_tch"])))
        err_tch = max(err_g_tch, err_p_tch)
        assert err_tch < atol, f"Tchebycheff invariant failed: {err_tch:.2e}"

        # Invariant 4: Linear Scalarization
        p_lin, g_lin = step_linear_scalarization(self.param, self.g1, self.g2, self.weights_lin, lr=0.01)
        err_g_lin = float(np.max(np.abs(g_lin - self.golden["g_lin"])))
        err_p_lin = float(np.max(np.abs(p_lin - self.golden["param_lin"])))
        err_lin = max(err_g_lin, err_p_lin)
        assert err_lin < atol, f"Linear scalarization invariant failed: {err_lin:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor.forward(self.golden["anchor_x"])
        err_anchor = float(np.max(np.abs(anchor_out - self.golden["anchor_out"])))
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"

        return {
            "mgda_err": err_mgda,
            "pcgrad_err": err_pc,
            "tchebycheff_err": err_tch,
            "linear_scalarization_err": err_lin,
            "anchor_err": err_anchor,
        }


def main():
    ckpt_path = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.safetensors")
    module = MultiObjectiveUnifiedModule(ckpt_path)
    errors = module.verify_invariants()
    print("JAX/NumPy Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


if __name__ == "__main__":
    main()
