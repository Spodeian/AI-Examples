#!/usr/bin/env python3
"""Generate golden weights and inputs/outputs for Category 05 Topic 06:
Lossless Deliberation: Reversible RevNet, Hyper-Connections, and Recurrent Looped Depth.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_model = 8       # Total hidden dimension (split into d_sub = 4 for revnet)
    d_sub = d_model // 2
    d_ff = 16

    # Input tensor: (B, S, D)
    x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    # ==============================================================
    # 1. Reversible RevNet Parameters (F and G blocks)
    # ==============================================================
    f_w1 = torch.randn(d_sub, d_ff, dtype=torch.float32) * 0.2
    f_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
    f_w2 = torch.randn(d_ff, d_sub, dtype=torch.float32) * 0.2
    f_b2 = torch.randn(d_sub, dtype=torch.float32) * 0.1

    g_w1 = torch.randn(d_sub, d_ff, dtype=torch.float32) * 0.2
    g_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
    g_w2 = torch.randn(d_ff, d_sub, dtype=torch.float32) * 0.2
    g_b2 = torch.randn(d_sub, dtype=torch.float32) * 0.1

    def eval_f(v):
        return F.gelu(v @ f_w1 + f_b1) @ f_w2 + f_b2

    def eval_g(v):
        return F.gelu(v @ g_w1 + g_b1) @ g_w2 + g_b2

    # RevNet Forward
    x1 = x[:, :, :d_sub]
    x2 = x[:, :, d_sub:]
    y1 = x1 + eval_f(x2)
    y2 = x2 + eval_g(y1)
    revnet_output = torch.cat([y1, y2], dim=-1)

    # Exact Reconstruction Check (Invariance)
    x2_rec = y2 - eval_g(y1)
    x1_rec = y1 - eval_f(x2_rec)
    rec_err = torch.max(torch.abs(torch.cat([x1_rec, x2_rec], dim=-1) - x)).item()
    assert rec_err < 1e-6, f"RevNet reconstruction failed: {rec_err}"

    # ==============================================================
    # 2. Hyper-Connections (HC) Parameters (2 residual lanes)
    # ==============================================================
    # Lane 1 and Lane 2 initial states
    h1 = x.clone()
    h2 = x * 0.5

    alpha = torch.tensor([0.6, 0.4], dtype=torch.float32)  # depth-mixing read
    beta = torch.tensor([0.5, 0.5], dtype=torch.float32)   # depth-mixing write

    hc_w1 = torch.randn(d_model, d_ff, dtype=torch.float32) * 0.2
    hc_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
    hc_w2 = torch.randn(d_ff, d_model, dtype=torch.float32) * 0.2
    hc_b2 = torch.randn(d_model, dtype=torch.float32) * 0.1

    # Read from lanes
    u_hc = alpha[0] * h1 + alpha[1] * h2
    v_hc = F.gelu(u_hc @ hc_w1 + hc_b1) @ hc_w2 + hc_b2

    # Write to lanes
    h1_out = h1 + beta[0] * v_hc
    h2_out = h2 + beta[1] * v_hc
    # Combined output: average of updated lanes
    hc_output = 0.5 * (h1_out + h2_out)

    # ==============================================================
    # 3. Looped Recurrent Depth with Input Anchoring
    # ==============================================================
    loop_steps = 4
    gamma = 0.3  # Persistent input injection strength

    loop_w1 = torch.randn(d_model, d_ff, dtype=torch.float32) * 0.2
    loop_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
    loop_w2 = torch.randn(d_ff, d_model, dtype=torch.float32) * 0.2
    loop_b2 = torch.randn(d_model, dtype=torch.float32) * 0.1

    z = x.clone()
    for k in range(loop_steps):
        # Anchor with initial representation x
        z_in = z + gamma * x
        delta = F.gelu(z_in @ loop_w1 + loop_b1) @ loop_w2 + loop_b2
        z = F.layer_norm(z + delta, normalized_shape=(d_model,))
    looped_output = z

    # Standard linear anchor for cross-track parity verification
    D_core = 4
    test_input = torch.randn(8, D_core)
    weight = torch.randn(D_core, D_core)
    bias = torch.randn(D_core)
    test_output = test_input @ weight + bias

    # Output directory
    out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors"

    tensors = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "input": x,
        "f_w1": f_w1,
        "f_b1": f_b1,
        "f_w2": f_w2,
        "f_b2": f_b2,
        "g_w1": g_w1,
        "g_b1": g_b1,
        "g_w2": g_w2,
        "g_b2": g_b2,
        "revnet_output": revnet_output,
        "alpha": alpha,
        "beta": beta,
        "hc_w1": hc_w1,
        "hc_b1": hc_b1,
        "hc_w2": hc_w2,
        "hc_b2": hc_b2,
        "hc_output": hc_output,
        "loop_w1": loop_w1,
        "loop_b1": loop_b1,
        "loop_w2": loop_w2,
        "loop_b2": loop_b2,
        "gamma": torch.tensor(gamma, dtype=torch.float32),
        "looped_output": looped_output,
    }

    save_file(tensors, str(out_file))
    print(f"Generated {out_file} successfully.")
    print(f"revnet_output shape: {revnet_output.shape}, norm: {revnet_output.norm():.6f}")
    print(f"hc_output shape: {hc_output.shape}, norm: {hc_output.norm():.6f}")
    print(f"looped_output shape: {looped_output.shape}, norm: {looped_output.norm():.6f}")


if __name__ == "__main__":
    main()
