"""Generate reference weights and golden outputs for Category 04, Topic 10.

Topic 10: Coordinate, Alternating Projections, and Majorization-Minimization
(04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization)
Enforces transitive numerical parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

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


if __name__ == "__main__":
    main()
