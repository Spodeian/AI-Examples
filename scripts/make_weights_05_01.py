"""Generates reference SafeTensors checkpoint for Category 05 Topic 01:
05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch


def mlp_mixer_block(
    x: torch.Tensor,                # [B, S, C]
    w1_token: torch.Tensor,         # [D_token, S]
    b1_token: torch.Tensor,         # [D_token]
    w2_token: torch.Tensor,         # [S, D_token]
    b2_token: torch.Tensor,         # [S]
    w1_chan: torch.Tensor,          # [D_chan, C]
    b1_chan: torch.Tensor,          # [D_chan]
    w2_chan: torch.Tensor,          # [C, D_chan]
    b2_chan: torch.Tensor,          # [C]
    ln1_w: torch.Tensor,            # [C]
    ln1_b: torch.Tensor,            # [C]
    ln2_w: torch.Tensor,            # [C]
    ln2_b: torch.Tensor,            # [C]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates 1 complete MLP-Mixer block with Token-mixing and Channel-mixing."""
    # 1. Token-mixing: LayerNorm over channels
    mean1 = x.mean(dim=-1, keepdim=True)
    var1 = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm1 = (x - mean1) / torch.sqrt(var1 + 1e-5) * ln1_w + ln1_b

    # Transpose to [B, C, S] for token mixing
    x_t = x_norm1.transpose(1, 2)
    h_tok = torch.nn.functional.gelu(x_t @ w1_token.T + b1_token)
    out_tok = (h_tok @ w2_token.T + b2_token).transpose(1, 2)
    y = x + out_tok

    # 2. Channel-mixing: LayerNorm over channels
    mean2 = y.mean(dim=-1, keepdim=True)
    var2 = y.var(dim=-1, keepdim=True, unbiased=False)
    y_norm = (y - mean2) / torch.sqrt(var2 + 1e-5) * ln2_w + ln2_b

    h_chan = torch.nn.functional.gelu(y_norm @ w1_chan.T + b1_chan)
    out_chan = h_chan @ w2_chan.T + b2_chan
    z = y + out_chan
    return z, y


def gmlp_block(
    x: torch.Tensor,                # [B, S, C]
    w_in: torch.Tensor,             # [2 * D, C]
    b_in: torch.Tensor,             # [2 * D]
    w_sgu: torch.Tensor,            # [S, S]
    b_sgu: torch.Tensor,            # [S, 1]
    w_out: torch.Tensor,            # [C, D]
    b_out: torch.Tensor,            # [C]
    ln_w: torch.Tensor,             # [C]
    ln_b: torch.Tensor,             # [C]
) -> torch.Tensor:
    """Evaluates 1 gMLP block with Spatial Gating Unit (SGU)."""
    # Pre-LayerNorm
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b

    # Channel projection: [B, S, 2D]
    z = torch.nn.functional.gelu(x_norm @ w_in.T + b_in)
    z1, z2 = torch.chunk(z, 2, dim=-1) # Each [B, S, D]

    # Spatial Gating Unit across tokens: w_sgu @ z2 + b_sgu
    # z2: [B, S, D], w_sgu: [S, S]
    # SGU operates on token dimension S
    z2_gate = w_sgu @ z2 + b_sgu
    sgu_out = z1 * z2_gate

    # Channel projection out: [B, S, C]
    out = sgu_out @ w_out.T + b_out
    return x + out


def generate_checkpoint():
    torch.manual_seed(42)

    B, S, C = 2, 4, 8
    D_tok = 16
    D_chan = 16
    D_gmlp = 8

    x = torch.randn(B, S, C)

    # MLP-Mixer weights
    w1_token = torch.randn(D_tok, S) * 0.1
    b1_token = torch.randn(D_tok) * 0.01
    w2_token = torch.randn(S, D_tok) * 0.1
    b2_token = torch.randn(S) * 0.01

    w1_chan = torch.randn(D_chan, C) * 0.1
    b1_chan = torch.randn(D_chan) * 0.01
    w2_chan = torch.randn(C, D_chan) * 0.1
    b2_chan = torch.randn(C) * 0.01

    ln1_w = torch.ones(C)
    ln1_b = torch.zeros(C)
    ln2_w = torch.ones(C)
    ln2_b = torch.zeros(C)

    mixer_out, mixer_tok_out = mlp_mixer_block(
        x, w1_token, b1_token, w2_token, b2_token,
        w1_chan, b1_chan, w2_chan, b2_chan,
        ln1_w, ln1_b, ln2_w, ln2_b
    )

    # gMLP weights
    w_in = torch.randn(2 * D_gmlp, C) * 0.1
    b_in = torch.randn(2 * D_gmlp) * 0.01
    w_sgu = torch.randn(S, S) * 0.05
    b_sgu = torch.ones(S, 1) # Initialized to 1.0 per Liu et al. 2021
    w_out = torch.randn(C, D_gmlp) * 0.1
    b_out = torch.randn(C) * 0.01
    gmlp_ln_w = torch.ones(C)
    gmlp_ln_b = torch.zeros(C)

    gmlp_out = gmlp_block(x, w_in, b_in, w_sgu, b_sgu, w_out, b_out, gmlp_ln_w, gmlp_ln_b)

    # Standard linear parity tensors for test_cross_track_parity.py
    D_core = 4
    test_input = torch.randn(8, D_core)
    weight = torch.randn(D_core, D_core)
    bias = torch.randn(D_core)
    test_output = test_input @ weight + bias

    checkpoint_data = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "x": x.clone().contiguous(),
        "w1_token": w1_token.clone().contiguous(),
        "b1_token": b1_token.clone().contiguous(),
        "w2_token": w2_token.clone().contiguous(),
        "b2_token": b2_token.clone().contiguous(),
        "w1_chan": w1_chan.clone().contiguous(),
        "b1_chan": b1_chan.clone().contiguous(),
        "w2_chan": w2_chan.clone().contiguous(),
        "b2_chan": b2_chan.clone().contiguous(),
        "ln1_w": ln1_w.clone().contiguous(),
        "ln1_b": ln1_b.clone().contiguous(),
        "ln2_w": ln2_w.clone().contiguous(),
        "ln2_b": ln2_b.clone().contiguous(),
        "mixer_out": mixer_out.clone().contiguous(),
        "mixer_tok_out": mixer_tok_out.clone().contiguous(),
        "w_in": w_in.clone().contiguous(),
        "b_in": b_in.clone().contiguous(),
        "w_sgu": w_sgu.clone().contiguous(),
        "b_sgu": b_sgu.clone().contiguous(),
        "w_out": w_out.clone().contiguous(),
        "b_out": b_out.clone().contiguous(),
        "gmlp_ln_w": gmlp_ln_w.clone().contiguous(),
        "gmlp_ln_b": gmlp_ln_b.clone().contiguous(),
        "gmlp_out": gmlp_out.clone().contiguous(),
    }

    out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "01_residual_mlp_mlp_mixer_and_gmlp.safetensors"
    safetensors.torch.save_file(checkpoint_data, out_path)
    print("Saved Category 05 Topic 01 reference checkpoint to", out_path)


if __name__ == "__main__":
    generate_checkpoint()
