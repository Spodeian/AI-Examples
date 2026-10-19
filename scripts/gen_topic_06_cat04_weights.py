"""Generate reference weights and golden outputs for Category 04, Topic 06.

Topic 06: Schedule-Free and Parameter-Free Optimisers
(Schedule-Free AdamW, Polyak Step-Size, Barzilai-Borwein Spectral Steps, D-Adaptation)
Enforces transitive numerical parity across PyTorch 2, JAX, Candle, and Hardware Kernels.
"""

from pathlib import Path
import torch
from safetensors.torch import save_file


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")
    d = 8

    # Schedule-Free state tensors
    sf_x = torch.randn(d, dtype=torch.float32)
    sf_z = torch.randn(d, dtype=torch.float32)
    c_0 = 0.5  # c_t for t=1
    sf_y = (1.0 - c_0) * sf_z + c_0 * sf_x
    sf_grad = torch.randn(d, dtype=torch.float32)
    sf_m = torch.randn(d, dtype=torch.float32) * 0.1
    sf_v = torch.abs(torch.randn(d, dtype=torch.float32)) * 0.1 + 0.01

    # Polyak state tensors
    polyak_param = torch.randn(d, dtype=torch.float32)
    polyak_grad = torch.randn(d, dtype=torch.float32)
    f_val = torch.tensor(3.5, dtype=torch.float32)
    f_opt = torch.tensor(1.0, dtype=torch.float32)

    # Barzilai-Borwein state tensors
    bb_param = torch.randn(d, dtype=torch.float32)
    bb_grad = torch.randn(d, dtype=torch.float32)
    bb_s = torch.randn(d, dtype=torch.float32) * 0.5
    bb_y = torch.randn(d, dtype=torch.float32) * 0.5
    # Ensure s^T y > 0
    dot_sy = torch.dot(bb_s, bb_y).item()
    if dot_sy <= 0.1:
        bb_y = bb_y + bb_s * (0.5 - dot_sy / (torch.norm(bb_s)**2 + 1e-6))

    # D-Adaptation state tensors
    d_param = torch.randn(d, dtype=torch.float32)
    d_grad = torch.randn(d, dtype=torch.float32)
    d_est = torch.tensor(1.2, dtype=torch.float32)

    # Linear Anchor
    anchor_x = torch.randn(2, d, dtype=torch.float32)
    anchor_w = torch.randn(d, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    # 1. Schedule-Free AdamW Golden Step
    sf_lr = 0.01
    sf_beta1 = 0.9
    sf_beta2 = 0.999
    sf_wd = 0.01
    sf_step = 1
    sf_eps = 1e-8

    golden_sf_m = sf_beta1 * sf_m + (1.0 - sf_beta1) * sf_grad
    golden_sf_v = sf_beta2 * sf_v + (1.0 - sf_beta2) * (sf_grad ** 2)
    bias_v = float(1.0 - sf_beta2 ** sf_step)
    v_hat = golden_sf_v / bias_v

    u_t = (golden_sf_m / (torch.sqrt(v_hat) + sf_eps)) + sf_wd * sf_y
    golden_sf_z = sf_z - sf_lr * u_t

    c_prime = float(1.0 / (sf_step + 1))  # 0.5
    golden_sf_x = (1.0 - c_prime) * sf_x + c_prime * golden_sf_z

    c_next = float(1.0 / (sf_step + 2))  # 0.33333334
    golden_sf_y = (1.0 - c_next) * golden_sf_z + c_next * golden_sf_x

    # 2. Polyak Step-Size Golden Computation
    grad_norm_sq = torch.sum(polyak_grad ** 2)
    polyak_gamma = (f_val - f_opt) / (grad_norm_sq + 1e-8)
    golden_polyak_param = polyak_param - polyak_gamma * polyak_grad

    # 3. Barzilai-Borwein BB1 Golden Computation
    s_dot_s = torch.dot(bb_s, bb_s)
    s_dot_y = torch.dot(bb_s, bb_y)
    bb_gamma = s_dot_s / (s_dot_y + 1e-8)
    golden_bb_param = bb_param - bb_gamma * bb_grad

    # 4. D-Adaptation Golden Computation
    d_norm_l1 = torch.sum(torch.abs(d_grad))
    d_norm_l2 = torch.norm(d_grad)
    d_next = torch.clamp(d_est + 0.05 * (d_norm_l1 / (d**0.5) - 1.0), min=0.1)
    d_eff_lr = 0.01 * d_next
    golden_d_adapt_param = d_param - d_eff_lr * (d_grad / (d_norm_l2 + 1e-8))

    tensors = {
        # Schedule-Free
        "sf_x": sf_x,
        "sf_z": sf_z,
        "sf_y": sf_y,
        "sf_grad": sf_grad,
        "sf_m": sf_m,
        "sf_v": sf_v,
        "golden_sf_x": golden_sf_x,
        "golden_sf_z": golden_sf_z,
        "golden_sf_y": golden_sf_y,
        "golden_sf_m": golden_sf_m,
        "golden_sf_v": golden_sf_v,
        # Polyak
        "polyak_param": polyak_param,
        "polyak_grad": polyak_grad,
        "f_val": f_val.unsqueeze(0),
        "f_opt": f_opt.unsqueeze(0),
        "golden_polyak_param": golden_polyak_param,
        # Barzilai-Borwein
        "bb_param": bb_param,
        "bb_grad": bb_grad,
        "bb_s": bb_s,
        "bb_y": bb_y,
        "golden_bb_param": golden_bb_param,
        # D-Adaptation
        "d_param": d_param,
        "d_grad": d_grad,
        "d_est": d_est.unsqueeze(0),
        "golden_d_adapt_param": golden_d_adapt_param,
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

    out_dir = Path("weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "06_schedule_free_and_parameter_free_optimisers.safetensors"
    save_file(tensors, str(out_file))
    print(f"Saved {len(tensors)} tensors to {out_file}")


if __name__ == "__main__":
    main()
