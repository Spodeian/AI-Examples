"""Functional JAX implementation for Category 05 Subfolder 02 Topic 02:
Specialized Convolutions & Channel Attention (CoordConv, Squeeze-and-Excitation, ECA-Net)
in 05_architectures/02_convolutional_and_spatiotemporal.
"""

from __future__ import annotations

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def coord_conv2d(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """CoordConv 2D forward with 1x1 projection.
    
    Args:
        x: Input tensor [B, C, H, W]
        weight: Conv weight [C_out, C+2, 1, 1]
        bias: Conv bias [C_out]
    """
    B, C, H, W = x.shape
    y_coords = jnp.linspace(-1.0, 1.0, H).reshape(1, 1, H, 1)
    y_coords = jnp.tile(y_coords, (B, 1, 1, W))
    x_coords = jnp.linspace(-1.0, 1.0, W).reshape(1, 1, 1, W)
    x_coords = jnp.tile(x_coords, (B, 1, H, 1))

    x_with_coords = jnp.concatenate([x, x_coords, y_coords], axis=1)  # [B, C+2, H, W]
    w_2d = weight[:, :, 0, 0]  # [C_out, C+2]
    out = jnp.einsum("bchw,oc->bohw", x_with_coords, w_2d) + bias[None, :, None, None]
    return out


def squeeze_and_excitation(
    x: jnp.ndarray,
    fc1_w: jnp.ndarray,
    fc1_b: jnp.ndarray,
    fc2_w: jnp.ndarray,
    fc2_b: jnp.ndarray,
) -> jnp.ndarray:
    """Squeeze-and-Excitation (SE) Block forward.
    
    Args:
        x: Input tensor [B, C, H, W]
        fc1_w: First projection weight [r_dim, C]
        fc1_b: First projection bias [r_dim]
        fc2_w: Second projection weight [C, r_dim]
        fc2_b: Second projection bias [C]
    """
    gap = jnp.mean(x, axis=(2, 3))  # [B, C]
    hidden = jax.nn.relu(gap @ fc1_w.T + fc1_b)
    scale = jax.nn.sigmoid(hidden @ fc2_w.T + fc2_b)  # [B, C]
    return x * scale[:, :, None, None]


def efficient_channel_attention(
    x: jnp.ndarray,
    conv_w: jnp.ndarray,
    conv_b: jnp.ndarray,
) -> jnp.ndarray:
    """Efficient Channel Attention (ECA) Block forward.
    
    Args:
        x: Input tensor [B, C, H, W]
        conv_w: 1D Conv kernel [1, 1, 3]
        conv_b: 1D Conv bias [1]
    """
    B, C, H, W = x.shape
    gap = jnp.mean(x, axis=(2, 3))  # [B, C]
    padded = jnp.pad(gap, ((0, 0), (1, 1)), mode="constant")
    w = conv_w[0, 0]  # [3]
    b = conv_b[0]
    conv_out = padded[:, 0:C] * w[0] + padded[:, 1 : C + 1] * w[1] + padded[:, 2 : C + 2] * w[2] + b
    scale = jax.nn.sigmoid(conv_out)  # [B, C]
    return x * scale[:, :, None, None]


def verify_against_golden() -> None:
    """Verifies JAX implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/02_specialized_convolutions_and_channel_attention.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = float(jnp.max(jnp.abs(test_out - weights["test_output"])))
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]

    # 2. CoordConv verification
    out_coord = coord_conv2d(x, weights["coord_conv_w"], weights["coord_conv_b"])
    err_coord = float(jnp.max(jnp.abs(out_coord - weights["coord_output"])))
    assert err_coord < 1e-6, f"CoordConv parity error {err_coord:.2e} exceeds 1e-6"

    # 3. Squeeze-and-Excitation verification
    out_se = squeeze_and_excitation(
        x,
        weights["se_fc1_w"],
        weights["se_fc1_b"],
        weights["se_fc2_w"],
        weights["se_fc2_b"],
    )
    err_se = float(jnp.max(jnp.abs(out_se - weights["se_output"])))
    assert err_se < 1e-6, f"SE parity error {err_se:.2e} exceeds 1e-6"

    # 4. Efficient Channel Attention verification
    out_eca = efficient_channel_attention(x, weights["eca_conv_w"], weights["eca_conv_b"])
    err_eca = float(jnp.max(jnp.abs(out_eca - weights["eca_output"])))
    assert err_eca < 1e-6, f"ECA parity error {err_eca:.2e} exceeds 1e-6"

    print("Functional JAX CoordConv, SE & ECA parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
