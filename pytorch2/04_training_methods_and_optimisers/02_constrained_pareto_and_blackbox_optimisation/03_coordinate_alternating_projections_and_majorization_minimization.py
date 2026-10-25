"""Coordinate, Alternating Projections, and Majorization-Minimization in PyTorch 2.x.

Implements foundational and modern coordinate and projection algorithms:
1. Dykstra's Alternating Projections algorithm onto intersection of convex sets.
2. Block Coordinate Descent (BCD) with cyclic coordinate updates.
3. Iteratively Reweighted Least Squares (IRLS) for robust L1/Lp norms.
4. Concave-Convex Procedure (CCCP) and Majorization-Minimization (MM).

Enforces strict transitive parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file


def proj_halfspace(x: torch.Tensor, normal: torch.Tensor, offset: float) -> torch.Tensor:
    """Project onto halfspace <normal, x> <= offset."""
    dot = torch.dot(x, normal)
    if dot <= offset:
        return x
    return x - ((dot - offset) / torch.dot(normal, normal)) * normal


def proj_box(x: torch.Tensor, lower: float = 0.0, upper: float = 2.0) -> torch.Tensor:
    """Project onto box [lower, upper]^d."""
    return torch.clamp(x, min=lower, max=upper)


def step_dykstra(
    y_prev: torch.Tensor,
    p_prev: torch.Tensor,
    q_prev: torch.Tensor,
    normal: torch.Tensor,
    offset: float = 1.0,
    lower: float = 0.0,
    upper: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one full step of Dykstra's Alternating Projections algorithm."""
    x_curr = proj_halfspace(y_prev + p_prev, normal, offset)
    p_curr = y_prev + p_prev - x_curr
    y_curr = proj_box(x_curr + q_prev, lower=lower, upper=upper)
    q_curr = x_curr + q_prev - y_curr
    return x_curr, y_curr, p_curr, q_curr


def step_bcd(
    param: torch.Tensor,
    b: torch.Tensor,
    a_diag: torch.Tensor,
    coupling: float = 0.2,
) -> torch.Tensor:
    """Perform Block Coordinate Descent update step."""
    return (b - coupling * param) / a_diag


def step_irls(
    param: torch.Tensor,
    target: torch.Tensor,
    mu: float = 0.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Perform Iteratively Reweighted Least Squares (IRLS) L1 minimization step."""
    r = torch.abs(param - target)
    w = 1.0 / (r + eps)
    return (w * target + mu * param) / (w + mu)


def step_cccp(param: torch.Tensor) -> torch.Tensor:
    """Perform Concave-Convex Procedure (CCCP) majorization step for f(x) = 0.5*x^2 - ln(1+x^2)."""
    return (2.0 * param) / (1.0 + param ** 2)


class LinearAnchor(nn.Module):
    """Linear projection layer serving as deterministic anchor across all tracks."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


class CoordinateMMUnifiedModule(nn.Module):
    """Unified module encapsulating coordinate and MM optimization for parity verification."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        tensors = load_file(str(checkpoint_path))
        self.x0 = tensors["x0"]
        self.normal = tensors["normal"]
        self.offset = float(tensors["offset"][0])

        self.param_bcd = tensors["param_bcd"]
        self.b_bcd = tensors["b_bcd"]
        self.a_diag = tensors["A_diag"]

        self.param_irls = tensors["param_irls"]
        self.target_irls = tensors["target_irls"]

        self.param_cccp = tensors["param_cccp"]

        self.anchor = LinearAnchor(8, 4)
        self.anchor.weight.data.copy_(tensors["anchor_w"].T)
        self.anchor.bias.data.copy_(tensors["anchor_b"])

        self.golden = {
            "dykstra_x": tensors["golden_dykstra_x"],
            "dykstra_y": tensors["golden_dykstra_y"],
            "dykstra_p": tensors["golden_dykstra_p"],
            "dykstra_q": tensors["golden_dykstra_q"],
            "param_bcd": tensors["golden_param_bcd"],
            "param_irls": tensors["golden_param_irls"],
            "param_cccp": tensors["golden_param_cccp"],
            "anchor_out": tensors["golden_anchor_out"],
            "anchor_x": tensors["anchor_x"],
        }

    def verify_invariants(self, atol: float = 1e-5) -> Dict[str, float]:
        """Verify all 5 invariants against golden reference tensors."""
        # Invariant 1: Dykstra's Alternating Projections
        p0 = torch.zeros_like(self.x0)
        q0 = torch.zeros_like(self.x0)
        x1, y1, p1, q1 = step_dykstra(self.x0, p0, q0, self.normal, offset=self.offset)
        err_x = float(torch.max(torch.abs(x1 - self.golden["dykstra_x"])))
        err_y = float(torch.max(torch.abs(y1 - self.golden["dykstra_y"])))
        err_p = float(torch.max(torch.abs(p1 - self.golden["dykstra_p"])))
        err_q = float(torch.max(torch.abs(q1 - self.golden["dykstra_q"])))
        err_dykstra = max(err_x, err_y, err_p, err_q)
        assert err_dykstra < atol, f"Dykstra invariant failed: {err_dykstra:.2e}"

        # Invariant 2: Block Coordinate Descent (BCD)
        p_bcd = step_bcd(self.param_bcd, self.b_bcd, self.a_diag, coupling=0.2)
        err_bcd = float(torch.max(torch.abs(p_bcd - self.golden["param_bcd"])))
        assert err_bcd < atol, f"BCD invariant failed: {err_bcd:.2e}"

        # Invariant 3: Iteratively Reweighted Least Squares (IRLS)
        p_irls = step_irls(self.param_irls, self.target_irls, mu=0.5, eps=1e-4)
        err_irls = float(torch.max(torch.abs(p_irls - self.golden["param_irls"])))
        assert err_irls < atol, f"IRLS invariant failed: {err_irls:.2e}"

        # Invariant 4: Concave-Convex Procedure (CCCP)
        p_cccp = step_cccp(self.param_cccp)
        err_cccp = float(torch.max(torch.abs(p_cccp - self.golden["param_cccp"])))
        assert err_cccp < atol, f"CCCP invariant failed: {err_cccp:.2e}"

        # Invariant 5: Linear Anchor
        anchor_out = self.anchor(self.golden["anchor_x"]).detach()
        err_anchor = float(torch.max(torch.abs(anchor_out - self.golden["anchor_out"])))
        assert err_anchor < atol, f"Anchor invariant failed: {err_anchor:.2e}"

        return {
            "dykstra_err": err_dykstra,
            "bcd_err": err_bcd,
            "irls_err": err_irls,
            "cccp_err": err_cccp,
            "anchor_err": err_anchor,
        }


def main():
    ckpt_path = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization.safetensors")
    module = CoordinateMMUnifiedModule(ckpt_path)
    errors = module.verify_invariants()
    print("PyTorch 2.x Invariant Parity:")
    for k, v in errors.items():
        print(f"  {k}: {v:.2e}")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    from safetensors.torch import save_file


    def proj_halfspace(x: torch.Tensor, normal: torch.Tensor, offset: float) -> torch.Tensor:
        """Project onto halfspace <normal, x> <= offset."""
        dot = torch.dot(x, normal)
        if dot <= offset:
            return x
        return x - ((dot - offset) / torch.dot(normal, normal)) * normal


    def proj_box(x: torch.Tensor, lower: float = 0.0, upper: float = 2.0) -> torch.Tensor:
        """Project onto box [lower, upper]^d."""
        return torch.clamp(x, min=lower, max=upper)


    def main():
        torch.manual_seed(42)
        device = torch.device("cpu")
        d = 8

        # 1. Dykstra's Alternating Projections onto Intersection of Two Convex Sets:
        # C1: Halfspace sum(x) <= 1.0 (normal = 1, offset = 1.0)
        # C2: Box [0.0, 2.0]^d
        # Target point x0 (outside both sets)
        x0 = torch.randn(d, dtype=torch.float32) + 1.0
        normal = torch.ones(d, dtype=torch.float32)
        offset = 1.0

        # One full Dykstra step:
        # State: y0 = x0, p0 = 0, q0 = 0
        p0 = torch.zeros(d, dtype=torch.float32)
        q0 = torch.zeros(d, dtype=torch.float32)
        y0 = x0.clone()

        # Step to C1
        x1 = proj_halfspace(y0 + p0, normal, offset)
        p1 = y0 + p0 - x1
        # Step to C2
        y1 = proj_box(x1 + q0, lower=0.0, upper=2.0)
        q1 = x1 + q0 - y1

        golden_dykstra_x = x1
        golden_dykstra_y = y1
        golden_dykstra_p = p1
        golden_dykstra_q = q1

        # 2. Block Coordinate Descent (BCD) Step on Quadratic
        # f(x) = 0.5 * x^T A x - b^T x where A is block-diagonal with off-diagonal coupling
        # Split x into Block 1 (indices 0..3) and Block 2 (indices 4..7)
        param_bcd = torch.randn(d, dtype=torch.float32)
        b_bcd = torch.randn(d, dtype=torch.float32)
        # Diagonal dominant A
        A_diag = torch.tensor([2.0, 2.5, 3.0, 2.0, 2.5, 3.0, 2.0, 2.5], dtype=torch.float32)
        # Update coordinate-wise: x_i* = (b_i - sum_{j!=i} A_ij x_j) / A_ii
        # For independent block coordinate update with damped coupling:
        # x_new = (b_bcd - 0.2 * param_bcd) / A_diag
        golden_param_bcd = (b_bcd - 0.2 * param_bcd) / A_diag

        # 3. Iteratively Reweighted Least Squares (IRLS) Step
        # Minimize ||x - target||_1 via IRLS: w_i = 1.0 / (|x_i - target_i| + eps)
        # x_next = (w * target + mu * x) / (w + mu)
        param_irls = torch.randn(d, dtype=torch.float32)
        target_irls = torch.randn(d, dtype=torch.float32)
        mu_irls = 0.5
        eps_irls = 1e-4
        r = torch.abs(param_irls - target_irls)
        weights_irls = 1.0 / (r + eps_irls)
        golden_param_irls = (weights_irls * target_irls + mu_irls * param_irls) / (weights_irls + mu_irls)

        # 4. Majorization-Minimization / CCCP Step
        # Non-convex objective: f(x) = 0.5 * x^2 - log(1 + x^2)
        # u(x) = 0.5 * x^2 (convex), v(x) = log(1 + x^2) (convex)
        # CCCP update: grad u(x_{t+1}) = grad v(x_t) -> x_{t+1} = 2 * x_t / (1 + x_t^2)
        param_cccp = torch.randn(d, dtype=torch.float32)
        golden_param_cccp = (2.0 * param_cccp) / (1.0 + param_cccp ** 2)

        # 5. Linear Anchor
        anchor_x = torch.randn(2, d, dtype=torch.float32)
        anchor_w = torch.randn(d, 4, dtype=torch.float32)
        anchor_b = torch.randn(4, dtype=torch.float32)
        golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

        tensors = {
            # Dykstra
            "x0": x0,
            "normal": normal,
            "offset": torch.tensor([offset], dtype=torch.float32),
            "golden_dykstra_x": golden_dykstra_x,
            "golden_dykstra_y": golden_dykstra_y,
            "golden_dykstra_p": golden_dykstra_p,
            "golden_dykstra_q": golden_dykstra_q,
            # BCD
            "param_bcd": param_bcd,
            "b_bcd": b_bcd,
            "A_diag": A_diag,
            "golden_param_bcd": golden_param_bcd,
            # IRLS
            "param_irls": param_irls,
            "target_irls": target_irls,
            "golden_param_irls": golden_param_irls,
            # CCCP
            "param_cccp": param_cccp,
            "golden_param_cccp": golden_param_cccp,
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
        out_file = out_dir / "03_coordinate_alternating_projections_and_majorization_minimization.safetensors"
        save_file(tensors, str(out_file))
        print(f"Saved {len(tensors)} tensors to {out_file}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
