#!/usr/bin/env python3
"""Generate golden weights and inputs/outputs for Category 05 Topic 04:
Soup of Experts, SMEAR, Rewarded Soups, BTX, and Sparse Upcycling.
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_model = 16
    d_ff = 32
    num_experts = 4

    # Input tensor: (B, S, D)
    x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    # 1. Base pre-trained model (for Sparse Upcycling)
    base_w1 = torch.randn(d_model, d_ff, dtype=torch.float32) * 0.2
    base_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
    base_w2 = torch.randn(d_ff, d_model, dtype=torch.float32) * 0.2
    base_b2 = torch.randn(d_model, dtype=torch.float32) * 0.1

    # 2. Upcycled & branched experts (shared pre-trained basin + task fine-tuning delta)
    experts_w1 = torch.zeros(num_experts, d_model, d_ff, dtype=torch.float32)
    experts_b1 = torch.zeros(num_experts, d_ff, dtype=torch.float32)
    experts_w2 = torch.zeros(num_experts, d_ff, d_model, dtype=torch.float32)
    experts_b2 = torch.zeros(num_experts, d_model, dtype=torch.float32)

    for e in range(num_experts):
        # Small fine-tuning perturbation from common base
        delta_w1 = torch.randn(d_model, d_ff) * 0.05
        delta_b1 = torch.randn(d_ff) * 0.02
        delta_w2 = torch.randn(d_ff, d_model) * 0.05
        delta_b2 = torch.randn(d_model) * 0.02

        experts_w1[e] = base_w1 + delta_w1
        experts_b1[e] = base_b1 + delta_b1
        experts_w2[e] = base_w2 + delta_w2
        experts_b2[e] = base_b2 + delta_b2

    # 3. Sequence-level router weights
    router_w = torch.randn(d_model, num_experts, dtype=torch.float32) * 0.2
    router_b = torch.randn(num_experts, dtype=torch.float32) * 0.1

    # ==============================================================
    # A. Soup of Experts (SMEAR) Dynamic Parameter Merging
    # ==============================================================
    # For each sequence b: pool sequence representations and compute routing weights
    x_pool = x.mean(dim=1)  # (B, D)
    routing_logits = x_pool @ router_w + router_b  # (B, E)
    routing_weights = F.softmax(routing_logits, dim=-1)  # (B, E)

    smear_output = torch.zeros_like(x)
    for b in range(batch_size):
        g = routing_weights[b]  # (E,)
        # Dynamically merge weights in parameter space
        merged_w1 = torch.sum(g.view(num_experts, 1, 1) * experts_w1, dim=0)  # (D, D_ff)
        merged_b1 = torch.sum(g.view(num_experts, 1) * experts_b1, dim=0)        # (D_ff,)
        merged_w2 = torch.sum(g.view(num_experts, 1, 1) * experts_w2, dim=0)  # (D_ff, D)
        merged_b2 = torch.sum(g.view(num_experts, 1) * experts_b2, dim=0)        # (D,)

        # Single dense forward pass for sequence b
        h = F.gelu(x[b] @ merged_w1 + merged_b1)
        smear_output[b] = h @ merged_w2 + merged_b2

    # ==============================================================
    # B. Contrast: Activation-Space MoE Forward
    # ==============================================================
    act_output = torch.zeros_like(x)
    for b in range(batch_size):
        g = routing_weights[b]
        for e in range(num_experts):
            h_e = F.gelu(x[b] @ experts_w1[e] + experts_b1[e])
            out_e = h_e @ experts_w2[e] + experts_b2[e]
            act_output[b] += g[e] * out_e

    # ==============================================================
    # C. Rewarded Pareto Soup (Static Weight Interpolation)
    # ==============================================================
    # Interpolating between Expert 0 and Expert 1 with alpha = 0.6
    alpha = 0.6
    pareto_w1 = alpha * experts_w1[0] + (1.0 - alpha) * experts_w1[1]
    pareto_b1 = alpha * experts_b1[0] + (1.0 - alpha) * experts_b1[1]
    pareto_w2 = alpha * experts_w2[0] + (1.0 - alpha) * experts_w2[1]
    pareto_b2 = alpha * experts_b2[0] + (1.0 - alpha) * experts_b2[1]

    h_p = F.gelu(x @ pareto_w1 + pareto_b1)
    pareto_soup_output = h_p @ pareto_w2 + pareto_b2

    # Standard linear parity tensors for test_cross_track_parity.py
    D_core = 4
    test_input = torch.randn(8, D_core)
    weight = torch.randn(D_core, D_core)
    bias = torch.randn(D_core)
    test_output = test_input @ weight + bias

    # Output directory
    out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors"

    tensors = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "input": x,
        "base_w1": base_w1,
        "base_b1": base_b1,
        "base_w2": base_w2,
        "base_b2": base_b2,
        "experts_w1": experts_w1,
        "experts_b1": experts_b1,
        "experts_w2": experts_w2,
        "experts_b2": experts_b2,
        "router_weight": router_w,
        "router_bias": router_b,
        "smear_output": smear_output,
        "act_output": act_output,
        "pareto_soup_output": pareto_soup_output,
    }

    save_file(tensors, str(out_file))
    print(f"Generated {out_file} successfully.")
    print(f"smear_output shape: {smear_output.shape}, norm: {smear_output.norm():.6f}")
    print(f"act_output shape: {act_output.shape}, norm: {act_output.norm():.6f}")
    print(f"pareto_soup_output shape: {pareto_soup_output.shape}, norm: {pareto_soup_output.norm():.6f}")


if __name__ == "__main__":
    main()
