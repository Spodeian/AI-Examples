"""Generate reference weights and golden outputs for Category 04, Topic 09.

Topic 09: Multi-Objective Pareto, MGDA, Tchebycheff, and Gradient Surgery
(04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery)
Enforces transitive numerical parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

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


if __name__ == "__main__":
    main()
