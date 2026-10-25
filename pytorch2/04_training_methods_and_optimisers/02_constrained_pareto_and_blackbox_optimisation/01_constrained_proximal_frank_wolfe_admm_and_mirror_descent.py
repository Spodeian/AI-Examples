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


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    from safetensors.torch import save_file


    def main():
        torch.manual_seed(42)
        device = torch.device("cpu")
        d = 8

        # State parameters
        # 1. PGD: L2 ball projection (radius R = 1.0)
        param_pgd = torch.randn(d, dtype=torch.float32)
        grad_pgd = torch.randn(d, dtype=torch.float32) * 2.0
        lr_pgd = 0.1
        R_pgd = 1.0
        v_pgd = param_pgd - lr_pgd * grad_pgd
        norm_v_pgd = torch.norm(v_pgd)
        golden_param_pgd = v_pgd if norm_v_pgd <= R_pgd else v_pgd * (R_pgd / norm_v_pgd)

        # 2. Frank-Wolfe: L1 ball (radius tau = 1.0)
        # Linear subproblem: s = argmin_{||s||_1 <= tau} <s, grad>
        # -> s = -tau * sign(grad_i) * e_i where i = argmax |grad_i|
        param_fw = torch.randn(d, dtype=torch.float32)
        param_fw = param_fw / torch.norm(param_fw, p=1) * 0.5  # start feasible inside L1 ball
        grad_fw = torch.randn(d, dtype=torch.float32)
        tau_fw = 1.0
        gamma_fw = 0.2  # step size in [0, 1]
        abs_g = torch.abs(grad_fw)
        idx_max = torch.argmax(abs_g)
        s_fw = torch.zeros(d, dtype=torch.float32)
        s_fw[idx_max] = -tau_fw * torch.sign(grad_fw[idx_max])
        golden_param_fw = (1.0 - gamma_fw) * param_fw + gamma_fw * s_fw

        # 3. Proximal ISTA: Soft-Thresholding for L1 regularized loss f(x) + lambda * ||x||_1
        param_ista = torch.randn(d, dtype=torch.float32)
        grad_ista = torch.randn(d, dtype=torch.float32)
        lr_ista = 0.05
        lambda_ista = 0.1
        thresh = lr_ista * lambda_ista
        v_ista = param_ista - lr_ista * grad_ista
        golden_param_ista = torch.sign(v_ista) * torch.clamp(torch.abs(v_ista) - thresh, min=0.0)

        # 4. Entropic Mirror Descent on Simplex Delta^d
        # p_{t+1} = p_t * exp(-lr * grad) / sum(p_t * exp(-lr * grad))
        param_md = torch.softmax(torch.randn(d, dtype=torch.float32), dim=0)
        grad_md = torch.randn(d, dtype=torch.float32)
        lr_md = 0.1
        unnorm_md = param_md * torch.exp(-lr_md * grad_md)
        golden_param_md = unnorm_md / torch.sum(unnorm_md)

        # 5. Linear Anchor
        anchor_x = torch.randn(2, d, dtype=torch.float32)
        anchor_w = torch.randn(d, 4, dtype=torch.float32)
        anchor_b = torch.randn(4, dtype=torch.float32)
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        tensors = {
            # PGD
            "param_pgd": param_pgd,
            "grad_pgd": grad_pgd,
            "golden_param_pgd": golden_param_pgd,
            # Frank-Wolfe
            "param_fw": param_fw,
            "grad_fw": grad_fw,
            "golden_param_fw": golden_param_fw,
            # ISTA
            "param_ista": param_ista,
            "grad_ista": grad_ista,
            "golden_param_ista": golden_param_ista,
            # Mirror Descent
            "param_md": param_md,
            "grad_md": grad_md,
            "golden_param_md": golden_param_md,
            # Linear Anchor
            "anchor_x": anchor_x,
            "anchor_w": anchor_w,
            "anchor_b": anchor_b,
            "golden_anchor_out": golden_anchor_out,
            # Standard Parity keys
            "weight": anchor_w.clone(),
            "bias": anchor_b.clone(),
            "test_input": anchor_x.clone(),
            "test_output": golden_anchor_out.clone(),
        }

        out_dir = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.safetensors"
        save_file(tensors, str(out_file))
        print(f"Saved {len(tensors)} tensors to {out_file}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
