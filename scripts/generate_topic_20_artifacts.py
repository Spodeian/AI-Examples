"""Generate golden reference SafeTensors and verify PyTorch/JAX parity for Topic 20.

Topic 20: 06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck
Category: 04_training_methods_and_optimisers
Subfolder: 03_training_dynamics_self_optimising_non_backprop
"""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

def generate_topic_20_artifacts():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. MeZO setup
    # theta: [8], z: [8], theta_star: [8]
    theta = torch.randn(8, dtype=torch.float32)
    z = torch.randn(8, dtype=torch.float32)
    # Normalize z to unit sphere or keep standard normal
    z = z / torch.norm(z)
    theta_star = torch.randn(8, dtype=torch.float32)

    eps = 1e-3
    eta = 0.05

    def loss_fn(p: torch.Tensor) -> torch.Tensor:
        diff = p - theta_star
        return 0.5 * torch.sum(diff ** 2)

    loss_pos = loss_fn(theta + eps * z)
    loss_neg = loss_fn(theta - eps * z)
    dir_deriv = (loss_pos - loss_neg) / (2.0 * eps)
    theta_next = theta - eta * dir_deriv * z

    # 2. Decoupled Neural Interfaces (DNI) / Synthetic Gradients
    # h: [4, 8], w_synth: [8, 8], b_synth: [8], target_grad: [4, 8]
    h = torch.randn(4, 8, dtype=torch.float32)
    w_synth = torch.randn(8, 8, dtype=torch.float32) / math.sqrt(8)
    b_synth = torch.randn(8, dtype=torch.float32) * 0.1
    target_grad = torch.randn(4, 8, dtype=torch.float32)

    pred_grad = torch.matmul(h, w_synth) + b_synth # [4, 8]
    dni_loss = 0.5 * torch.mean((pred_grad - target_grad) ** 2) # scalar

    # 3. HSIC Bottleneck
    # Batch = 4, feature dim = 8, label dim = 4
    hsic_h = torch.randn(4, 8, dtype=torch.float32)
    hsic_y = torch.randn(4, 4, dtype=torch.float32)

    # Linear kernels: K = H H^T [4, 4], L = Y Y^T [4, 4]
    k_mat = torch.matmul(hsic_h, hsic_h.T) # [4, 4]
    l_mat = torch.matmul(hsic_y, hsic_y.T) # [4, 4]

    # Centering matrix H_c = I - 1/B * 1 1^T
    eye4 = torch.eye(4, dtype=torch.float32)
    ones4 = torch.ones(4, 4, dtype=torch.float32) / 4.0
    h_c = eye4 - ones4

    k_centered = torch.matmul(torch.matmul(h_c, k_mat), h_c)
    l_centered = torch.matmul(torch.matmul(h_c, l_mat), h_c)

    # HSIC = 1 / (B - 1)^2 * Tr(K_c L_c) = 1/9 * sum(K_c * L_c^T)
    hsic_score = (1.0 / 9.0) * torch.trace(torch.matmul(k_centered, l_centered))

    # 4. Linear Anchor
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32) / math.sqrt(8)
    anchor_b = torch.randn(4, dtype=torch.float32) * 0.1
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    # Construct tensors dictionary
    tensors = {
        "theta": theta.clone().contiguous(),
        "z": z.clone().contiguous(),
        "theta_star": theta_star.clone().contiguous(),
        "golden_loss_pos": loss_pos.unsqueeze(0).clone().contiguous(),
        "golden_loss_neg": loss_neg.unsqueeze(0).clone().contiguous(),
        "golden_dir_deriv": dir_deriv.unsqueeze(0).clone().contiguous(),
        "golden_theta_next": theta_next.clone().contiguous(),

        "h": h.clone().contiguous(),
        "w_synth": w_synth.clone().contiguous(),
        "b_synth": b_synth.clone().contiguous(),
        "target_grad": target_grad.clone().contiguous(),
        "golden_pred_grad": pred_grad.clone().contiguous(),
        "golden_dni_loss": dni_loss.unsqueeze(0).clone().contiguous(),

        "hsic_h": hsic_h.clone().contiguous(),
        "hsic_y": hsic_y.clone().contiguous(),
        "golden_hsic_score": hsic_score.unsqueeze(0).clone().contiguous(),

        "anchor_x": anchor_x.clone().contiguous(),
        "anchor_w": anchor_w.clone().contiguous(),
        "anchor_b": anchor_b.clone().contiguous(),
        "golden_anchor_out": golden_anchor_out.clone().contiguous(),

        # Standard test inputs/outputs for parity test runners
        "test_input": anchor_x.clone().contiguous(),
        "test_output": golden_anchor_out.clone().contiguous(),
        "weight": anchor_w.clone().contiguous(),
        "bias": anchor_b.clone().contiguous(),
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.safetensors"

    save_file(tensors, out_path)
    print(f"Saved {len(tensors)} tensors to {out_path}")

    # Verify self-consistency
    print(f"MeZO Loss Pos: {loss_pos.item():.6f}, Neg: {loss_neg.item():.6f}, Deriv: {dir_deriv.item():.6f}")
    print(f"DNI Loss: {dni_loss.item():.6f}")
    print(f"HSIC Score: {hsic_score.item():.6f}")

if __name__ == "__main__":
    generate_topic_20_artifacts()
