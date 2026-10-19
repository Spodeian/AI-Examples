"""Constrained, Proximal, Frank-Wolfe, ADMM, and Mirror Descent in PyTorch 2.x.

Implements foundational and modern constrained optimization methods:
1. Projected Gradient Descent (PGD) with L2 Ball Projection.
2. Frank-Wolfe (Conditional Gradient) on L1 Ball without full projection.
3. Proximal Gradient (ISTA) with Soft-Thresholding.
4. Entropic Mirror Descent (Exponentiated Gradient) on Probability Simplex.

Enforces strict transitive parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file


def step_pgd(
    param: torch.Tensor,
    grad: torch.Tensor,
    lr: float = 0.1,
    radius: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Projected Gradient Descent with Euclidean ball projection."""
    v = param - lr * grad
    norm_v = torch.norm(v)
    if norm_v <= radius:
        return v
    return v * (radius / (norm_v + eps))


def step_frank_wolfe(
    param: torch.Tensor,
    grad: torch.Tensor,
    tau: float = 1.0,
    gamma: float = 0.2,
) -> torch.Tensor:
    """Frank-Wolfe (Conditional Gradient) step on L1 ball ||s||_1 <= tau."""
    abs_g = torch.abs(grad)
    idx_max = torch.argmax(abs_g)
    s = torch.zeros_like(param)
    s[idx_max] = -tau * torch.sign(grad[idx_max])
    return (1.0 - gamma) * param + gamma * s


def step_ista(
    param: torch.Tensor,
    grad: torch.Tensor,
    lr: float = 0.05,
    lam: float = 0.1,
) -> torch.Tensor:
    """Proximal Gradient Method (ISTA) with soft-thresholding."""
    thresh = lr * lam
    v = param - lr * grad
    return torch.sign(v) * torch.clamp(torch.abs(v) - thresh, min=0.0)


def step_mirror_descent(
    param: torch.Tensor,
    grad: torch.Tensor,
    lr: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Entropic Mirror Descent (Exponentiated Gradient) on probability simplex."""
    unnorm = param * torch.exp(-lr * grad)
    return unnorm / (torch.sum(unnorm) + eps)


class LinearAnchor(nn.Module):
    """Linear projection layer serving as deterministic anchor across all tracks."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class ConstrainedOptimUnifiedModule(nn.Module):
    """Unified module encapsulating constrained and proximal optimization methods."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
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

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(tensors["anchor_w"].T)
        self.anchor.bias.data.copy_(tensors["anchor_b"])

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
        err_pgd = float(torch.max(torch.abs(p_pgd - self.golden["param_pgd"])))
        assert err_pgd < atol, f"PGD invariant failed: {err_pgd:.2e}"

        # Invariant 2: Frank-Wolfe
        p_fw = step_frank_wolfe(self.param_fw, self.grad_fw, tau=1.0, gamma=0.2)
        err_fw = float(torch.max(torch.abs(p_fw - self.golden["param_fw"])))
        assert err_fw < atol, f"Frank-Wolfe invariant failed: {err_fw:.2e}"

        # Invariant 3: Proximal ISTA
        p_ista = step_ista(self.param_ista, self.grad_ista, lr=0.05, lam=0.1)
        err_ista = float(torch.max(torch.abs(p_ista - self.golden["param_ista"])))
        assert err_ista < atol, f"ISTA invariant failed: {err_ista:.2e}"

        # Invariant 4: Entropic Mirror Descent
        p_md = step_mirror_descent(self.param_md, self.grad_md, lr=0.1)
        err_md = float(torch.max(torch.abs(p_md - self.golden["param_md"])))
        assert err_md < atol, f"Mirror Descent invariant failed: {err_md:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor(self.golden["anchor_x"]).detach()
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden["anchor_out"])))
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
    print("PyTorch 2.x Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


if __name__ == "__main__":
    main()
