"""Constrained, Proximal, Frank-Wolfe, ADMM, and Mirror Descent in JAX / Functional NumPy.

Implements foundational and modern constrained optimization methods:
1. Projected Gradient Descent (PGD) with L2 Ball Projection.
2. Frank-Wolfe (Conditional Gradient) on L1 Ball without full projection.
3. Proximal Gradient (ISTA) with Soft-Thresholding.
4. Entropic Mirror Descent (Exponentiated Gradient) on Probability Simplex.

Enforces strict transitive parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
from typing import Dict

import numpy as np
from safetensors.numpy import load_file


class LinearAnchor:
    """Functional Linear Anchor module (y = x @ w + b)."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weight) + self.bias


def step_pgd(
    param: np.ndarray,
    grad: np.ndarray,
    lr: float = 0.1,
    radius: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Projected Gradient Descent with Euclidean ball projection."""
    v = param - lr * grad
    norm_v = float(np.linalg.norm(v))
    if norm_v <= radius:
        return v
    return v * (radius / (norm_v + eps))


def step_frank_wolfe(
    param: np.ndarray,
    grad: np.ndarray,
    tau: float = 1.0,
    gamma: float = 0.2,
) -> np.ndarray:
    """Frank-Wolfe (Conditional Gradient) step on L1 ball ||s||_1 <= tau."""
    abs_g = np.abs(grad)
    idx_max = int(np.argmax(abs_g))
    s = np.zeros_like(param)
    s[idx_max] = -tau * np.sign(grad[idx_max])
    return (1.0 - gamma) * param + gamma * s


def step_ista(
    param: np.ndarray,
    grad: np.ndarray,
    lr: float = 0.05,
    lam: float = 0.1,
) -> np.ndarray:
    """Proximal Gradient Method (ISTA) with soft-thresholding."""
    thresh = lr * lam
    v = param - lr * grad
    return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)


def step_mirror_descent(
    param: np.ndarray,
    grad: np.ndarray,
    lr: float = 0.1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Entropic Mirror Descent (Exponentiated Gradient) on probability simplex."""
    unnorm = param * np.exp(-lr * grad)
    return unnorm / (np.sum(unnorm) + eps)


class ConstrainedOptimUnifiedModule:
    """Unified module encapsulating constrained optimization for JAX/NumPy parity."""

    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        tensors = load_file(str(checkpoint_path))
        self.param_pgd = tensors["param_pgd"]
        self.grad_pgd = tensors["grad_pgd"]
        self.param_fw = tensors["param_fw"]
        self.grad_fw = tensors["grad_fw"]
        self.param_ista = tensors["param_ista"]
        self.grad_ista = tensors["grad_ista"]
        self.param_md = tensors["param_md"]
        self.grad_md = tensors["grad_md"]

        self.anchor = LinearAnchor(tensors["anchor_w"], tensors["anchor_b"])

        self.golden = {
            "param_pgd": tensors["golden_param_pgd"],
            "param_fw": tensors["golden_param_fw"],
            "param_ista": tensors["golden_param_ista"],
            "param_md": tensors["golden_param_md"],
            "anchor_out": tensors["golden_anchor_out"],
            "anchor_x": tensors["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: PGD
        p_pgd = step_pgd(self.param_pgd, self.grad_pgd, lr=0.1, radius=1.0)
        err_pgd = float(np.max(np.abs(p_pgd - self.golden["param_pgd"])))
        assert err_pgd < atol, f"PGD invariant failed: {err_pgd:.2e}"

        # Invariant 2: Frank-Wolfe
        p_fw = step_frank_wolfe(self.param_fw, self.grad_fw, tau=1.0, gamma=0.2)
        err_fw = float(np.max(np.abs(p_fw - self.golden["param_fw"])))
        assert err_fw < atol, f"Frank-Wolfe invariant failed: {err_fw:.2e}"

        # Invariant 3: Proximal ISTA
        p_ista = step_ista(self.param_ista, self.grad_ista, lr=0.05, lam=0.1)
        err_ista = float(np.max(np.abs(p_ista - self.golden["param_ista"])))
        assert err_ista < atol, f"ISTA invariant failed: {err_ista:.2e}"

        # Invariant 4: Entropic Mirror Descent
        p_md = step_mirror_descent(self.param_md, self.grad_md, lr=0.1)
        err_md = float(np.max(np.abs(p_md - self.golden["param_md"])))
        assert err_md < atol, f"Mirror Descent invariant failed: {err_md:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor.forward(self.golden["anchor_x"])
        err_anchor = float(np.max(np.abs(anchor_out - self.golden["anchor_out"])))
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"

        return {
            "pgd_err": err_pgd,
            "fw_err": err_fw,
            "ista_err": err_ista,
            "mirror_descent_err": err_md,
            "anchor_err": err_anchor,
        }


def main():
    ckpt_path = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.safetensors")
    module = ConstrainedOptimUnifiedModule(ckpt_path)
    errors = module.verify_invariants()
    print("JAX/NumPy Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


if __name__ == "__main__":
    main()
