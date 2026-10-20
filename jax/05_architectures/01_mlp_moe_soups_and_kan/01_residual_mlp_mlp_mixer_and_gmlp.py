"""JAX / NumPy implementation for Category 05 Topic 01:
01_residual_mlp_mlp_mixer_and_gmlp in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def gelu_np(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation using erf."""
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Pre-LayerNorm over innermost channel dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def mlp_mixer_block_np(
    x: np.ndarray,
    w1_token: np.ndarray,
    b1_token: np.ndarray,
    w2_token: np.ndarray,
    b2_token: np.ndarray,
    w1_chan: np.ndarray,
    b1_chan: np.ndarray,
    w2_chan: np.ndarray,
    b2_chan: np.ndarray,
    ln1_w: np.ndarray,
    ln1_b: np.ndarray,
    ln2_w: np.ndarray,
    ln2_b: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Evaluates 1 complete MLP-Mixer block with Token-mixing and Channel-mixing."""
    # 1. Token-mixing: LayerNorm across channels, transpose to [B, C, S], Linear on tokens, transpose back
    norm1 = layer_norm_np(x, ln1_w, ln1_b, eps=eps)
    x_t = np.swapaxes(norm1, 1, 2)
    h_tok = gelu_np(np.matmul(x_t, w1_token.T) + b1_token)
    out_tok = np.swapaxes(np.matmul(h_tok, w2_token.T) + b2_token, 1, 2)
    y = x + out_tok

    # 2. Channel-mixing: LayerNorm across channels, Linear on channels
    norm2 = layer_norm_np(y, ln2_w, ln2_b, eps=eps)
    h_chan = gelu_np(np.matmul(norm2, w1_chan.T) + b1_chan)
    out_chan = np.matmul(h_chan, w2_chan.T) + b2_chan
    z = y + out_chan
    return z


def spatial_gating_unit_np(z: np.ndarray, w_sgu: np.ndarray, b_sgu: np.ndarray) -> np.ndarray:
    """Spatial Gating Unit (SGU) for gMLP."""
    # Split across channel dimension into 2 halves
    d = z.shape[-1] // 2
    z1 = z[..., :d]
    z2 = z[..., d:]
    # w_sgu: [S, S], z2: [B, S, D], b_sgu: [S, 1]
    # Gate applies linear projection across sequence dimension S
    gate = np.matmul(w_sgu, z2) + b_sgu
    return z1 * gate


def gmlp_block_np(
    x: np.ndarray,
    w_in: np.ndarray,
    b_in: np.ndarray,
    w_sgu: np.ndarray,
    b_sgu: np.ndarray,
    w_out: np.ndarray,
    b_out: np.ndarray,
    ln_w: np.ndarray,
    ln_b: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Evaluates 1 gMLP block with Spatial Gating Unit."""
    norm = layer_norm_np(x, ln_w, ln_b, eps=eps)
    z = gelu_np(np.matmul(norm, w_in.T) + b_in)
    gated = spatial_gating_unit_np(z, w_sgu, b_sgu)
    out = np.matmul(gated, w_out.T) + b_out
    return x + out


def resmlp_block_np(
    x: np.ndarray,
    scale_token: np.ndarray,
    shift_token: np.ndarray,
    w_token: np.ndarray,
    b_token: np.ndarray,
    ls_token: np.ndarray,
    scale_chan: np.ndarray,
    shift_chan: np.ndarray,
    w1_chan: np.ndarray,
    b1_chan: np.ndarray,
    w2_chan: np.ndarray,
    b2_chan: np.ndarray,
    ls_chan: np.ndarray,
) -> np.ndarray:
    """Evaluates 1 ResMLP block with Affine channel-scaling and LayerScale."""
    # Token mixing
    aff1 = x * scale_token + shift_token
    tok = np.swapaxes(np.matmul(np.swapaxes(aff1, 1, 2), w_token.T) + b_token, 1, 2)
    y = x + ls_token * tok

    # Channel mixing
    aff2 = y * scale_chan + shift_chan
    chan = np.matmul(gelu_np(np.matmul(aff2, w1_chan.T) + b1_chan), w2_chan.T) + b2_chan
    z = y + ls_chan * chan
    return z


def verify_against_checkpoint(ckpt_path: Path | str) -> None:
    weights = load_file(str(ckpt_path))
    x = weights["x"]

    # 1. Verify MLP-Mixer
    mixer_out = mlp_mixer_block_np(
        x,
        weights["w1_token"], weights["b1_token"], weights["w2_token"], weights["b2_token"],
        weights["w1_chan"], weights["b1_chan"], weights["w2_chan"], weights["b2_chan"],
        weights["ln1_w"], weights["ln1_b"], weights["ln2_w"], weights["ln2_b"],
    )
    err_mixer = float(np.max(np.abs(mixer_out - weights["mixer_out"])))
    assert err_mixer < 1e-5, f"JAX/NumPy MLP-Mixer mismatch: {err_mixer}"

    # 2. Verify gMLP
    gmlp_out = gmlp_block_np(
        x,
        weights["w_in"], weights["b_in"],
        weights["w_sgu"], weights["b_sgu"],
        weights["w_out"], weights["b_out"],
        weights["gmlp_ln_w"], weights["gmlp_ln_b"],
    )
    err_gmlp = float(np.max(np.abs(gmlp_out - weights["gmlp_out"])))
    assert err_gmlp < 1e-5, f"JAX/NumPy gMLP mismatch: {err_gmlp}"

    # 3. Verify Anchor Parity
    test_in = weights["test_input"]
    anchor_out = np.matmul(test_in, weights["weight"]) + weights["bias"]
    err_anchor = float(np.max(np.abs(anchor_out - weights["test_output"])))
    assert err_anchor < 1e-5, f"JAX/NumPy Anchor mismatch: {err_anchor}"

    print(f"JAX/NumPy verified against checkpoint! Mixer err={err_mixer:.2e}, gMLP err={err_gmlp:.2e}, Anchor err={err_anchor:.2e}")


if __name__ == "__main__":
    ckpt = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors")
    verify_against_checkpoint(ckpt)
