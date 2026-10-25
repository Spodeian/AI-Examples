"""Multi-Objective Pareto, MGDA, Tchebycheff, and Gradient Surgery in PyTorch 2.x.

Implements foundational and modern multi-objective / multi-task optimization algorithms:
1. Multiple Gradient Descent Algorithm (MGDA - Désidéri 2012; Sener & Koltun 2018).
2. Projecting Conflicting Gradients (PCGrad - Yu et al. 2020).
3. Augmented Tchebycheff Scalarization for Non-Convex Pareto Fronts (Miettinen 1999).
4. Linear Multi-Task Scalarization.

Enforces strict transitive parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file


def step_mgda_2task(
    param: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    lr: float = 0.01,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute 2-task MGDA min-norm convex combination gradient and parameter step."""
    u = g1 - g2
    u_norm_sq = torch.dot(u, u)
    alpha = float(torch.clamp(torch.dot(g2 - g1, g2) / (u_norm_sq + eps), min=0.0, max=1.0).item())
    g_mgda = alpha * g1 + (1.0 - alpha) * g2
    param_next = param - lr * g_mgda
    return param_next, g_mgda, alpha


def step_pcgrad_2task(
    param: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    lr: float = 0.01,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform PCGrad conflicting gradient projection and parameter step."""
    dot_12 = torch.dot(g1, g2)
    norm_sq_g1 = torch.dot(g1, g1)
    norm_sq_g2 = torch.dot(g2, g2)
    if dot_12 < 0.0:
        g1_proj = g1 - (dot_12 / (norm_sq_g2 + eps)) * g2
        g2_proj = g2 - (dot_12 / (norm_sq_g1 + eps)) * g1
        g_pc = 0.5 * (g1_proj + g2_proj)
    else:
        g_pc = 0.5 * (g1 + g2)
    param_next = param - lr * g_pc
    return param_next, g_pc


def step_tchebycheff(
    param: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    losses: torch.Tensor,
    z_star: torch.Tensor,
    weights: torch.Tensor,
    rho: float = 0.05,
    lr: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform Augmented Tchebycheff scalarization gradient step."""
    diff = weights * (losses - z_star)
    idx_max = torch.argmax(diff)
    ds_dl = rho * weights
    ds_dl[idx_max] += weights[idx_max]
    g_tch = ds_dl[0] * g1 + ds_dl[1] * g2
    param_next = param - lr * g_tch
    return param_next, g_tch


def step_linear_scalarization(
    param: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    weights: torch.Tensor,
    lr: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform linear multi-task scalarization gradient step."""
    g_lin = weights[0] * g1 + weights[1] * g2
    param_next = param - lr * g_lin
    return param_next, g_lin


class LinearAnchor(nn.Module):
    """Linear projection layer serving as deterministic anchor across all tracks."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class MultiObjectiveUnifiedModule(nn.Module):
    """Unified module encapsulating multi-objective optimizers for parity verification."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
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

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(tensors["anchor_w"].T)
        self.anchor.bias.data.copy_(tensors["anchor_b"])

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
        err_g_mgda = float(torch.max(torch.abs(g_mgda - self.golden["g_mgda"])))
        err_p_mgda = float(torch.max(torch.abs(p_mgda - self.golden["param_mgda"])))
        err_mgda = max(err_alpha, err_g_mgda, err_p_mgda)
        assert err_mgda < atol, f"MGDA invariant failed: {err_mgda:.2e}"

        # Invariant 2: PCGrad
        p_pc, g_pc = step_pcgrad_2task(self.param, self.g1, self.g2, lr=0.01)
        err_g_pc = float(torch.max(torch.abs(g_pc - self.golden["g_pc"])))
        err_p_pc = float(torch.max(torch.abs(p_pc - self.golden["param_pc"])))
        err_pc = max(err_g_pc, err_p_pc)
        assert err_pc < atol, f"PCGrad invariant failed: {err_pc:.2e}"

        # Invariant 3: Tchebycheff Scalarization
        p_tch, g_tch = step_tchebycheff(
            self.param, self.g1, self.g2, self.losses, self.z_star, self.weights_tch, rho=0.05, lr=0.01
        )
        err_g_tch = float(torch.max(torch.abs(g_tch - self.golden["g_tch"])))
        err_p_tch = float(torch.max(torch.abs(p_tch - self.golden["param_tch"])))
        err_tch = max(err_g_tch, err_p_tch)
        assert err_tch < atol, f"Tchebycheff invariant failed: {err_tch:.2e}"

        # Invariant 4: Linear Scalarization
        p_lin, g_lin = step_linear_scalarization(self.param, self.g1, self.g2, self.weights_lin, lr=0.01)
        err_g_lin = float(torch.max(torch.abs(g_lin - self.golden["g_lin"])))
        err_p_lin = float(torch.max(torch.abs(p_lin - self.golden["param_lin"])))
        err_lin = max(err_g_lin, err_p_lin)
        assert err_lin < atol, f"Linear scalarization invariant failed: {err_lin:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor(self.golden["anchor_x"]).detach()
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden["anchor_out"])))
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

        param = torch.randn(d, dtype=torch.float32)

        # Two conflicting task gradients (ensure negative inner product to trigger PCGrad)
        g1 = torch.randn(d, dtype=torch.float32)
        g2 = -0.8 * g1 + 0.3 * torch.randn(d, dtype=torch.float32)  # negative dot product

        lr = 0.01

        # 1. MGDA (Multiple Gradient Descent Algorithm) for 2 tasks
        # Solve min_{alpha in [0, 1]} ||alpha * g1 + (1 - alpha) * g2||^2
        # Analytical solution:
        # d_vec = g1 - g2
        # alpha = clamp( (g2 - g1)^T g2 / ||g1 - g2||^2, 0.0, 1.0 ) = clamp( - (g1 - g2)^T g2 / ||d_vec||^2, 0.0, 1.0 )
        # Let u = g1 - g2:
        # ||alpha u + g2||^2 -> alpha = - (u^T g2) / ||u||^2 = (g2 - g1)^T g2 / ||g1 - g2||^2
        u = g1 - g2
        u_norm_sq = torch.dot(u, u)
        alpha = torch.clamp(torch.dot(g2 - g1, g2) / (u_norm_sq + 1e-12), min=0.0, max=1.0)
        golden_g_mgda = alpha * g1 + (1.0 - alpha) * g2
        golden_param_mgda = param - lr * golden_g_mgda

        # 2. PCGrad (Projecting Conflicting Gradients)
        # If g1 . g2 < 0:
        #   g1_proj = g1 - (g1 . g2 / ||g2||^2) * g2
        #   g2_proj = g2 - (g2 . g1 / ||g1||^2) * g1
        #   g_pc = 0.5 * (g1_proj + g2_proj)
        # Else:
        #   g_pc = 0.5 * (g1 + g2)
        dot_12 = torch.dot(g1, g2)
        norm_sq_g1 = torch.dot(g1, g1)
        norm_sq_g2 = torch.dot(g2, g2)
        if dot_12 < 0.0:
            g1_proj = g1 - (dot_12 / (norm_sq_g2 + 1e-12)) * g2
            g2_proj = g2 - (dot_12 / (norm_sq_g1 + 1e-12)) * g1
            golden_g_pc = 0.5 * (g1_proj + g2_proj)
        else:
            golden_g_pc = 0.5 * (g1 + g2)
        golden_param_pc = param - lr * golden_g_pc

        # 3. Augmented Tchebycheff Scalarization
        # Task losses L1, L2, weights w1=0.6, w2=0.4, ideal reference z1*=0.1, z2*=0.2, rho=0.05
        # S(L) = max_i w_i * (L_i - z_i*) + rho * sum_i w_i * (L_i - z_i*)
        # Let L1 = 1.5, L2 = 2.0
        L = torch.tensor([1.5, 2.0], dtype=torch.float32)
        z_star = torch.tensor([0.1, 0.2], dtype=torch.float32)
        weights_tch = torch.tensor([0.6, 0.4], dtype=torch.float32)
        rho_tch = 0.05

        diff = weights_tch * (L - z_star)  # [0.6 * 1.4, 0.4 * 1.8] = [0.84, 0.72]
        idx_max = torch.argmax(diff)  # 0
        # Gradient wrt parameters:
        # dS/dL = e_{idx_max} * w_{idx_max} + rho * weights_tch
        # g_tch = (dS/dL)_1 * g1 + (dS/dL)_2 * g2
        ds_dl = rho_tch * weights_tch
        ds_dl[idx_max] += weights_tch[idx_max]
        golden_g_tch = ds_dl[0] * g1 + ds_dl[1] * g2
        golden_param_tch = param - lr * golden_g_tch

        # 4. Linear Scalarization
        # g_lin = w1 * g1 + w2 * g2
        w_lin = torch.tensor([0.6, 0.4], dtype=torch.float32)
        golden_g_lin = w_lin[0] * g1 + w_lin[1] * g2
        golden_param_lin = param - lr * golden_g_lin

        # 5. Linear Anchor
        anchor_x = torch.randn(2, d, dtype=torch.float32)
        anchor_w = torch.randn(d, 4, dtype=torch.float32)
        anchor_b = torch.randn(4, dtype=torch.float32)
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        tensors = {
            # Base parameters and task gradients
            "param": param,
            "g1": g1,
            "g2": g2,
            # MGDA
            "golden_alpha_mgda": alpha.unsqueeze(0),
            "golden_g_mgda": golden_g_mgda,
            "golden_param_mgda": golden_param_mgda,
            # PCGrad
            "golden_g_pc": golden_g_pc,
            "golden_param_pc": golden_param_pc,
            # Tchebycheff
            "L": L,
            "z_star": z_star,
            "weights_tch": weights_tch,
            "golden_g_tch": golden_g_tch,
            "golden_param_tch": golden_param_tch,
            # Linear Scalarization
            "weights_lin": w_lin,
            "golden_g_lin": golden_g_lin,
            "golden_param_lin": golden_param_lin,
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
        out_file = out_dir / "02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.safetensors"
        save_file(tensors, str(out_file))
        print(f"Saved {len(tensors)} tensors to {out_file}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
