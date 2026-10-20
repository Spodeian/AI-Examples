"""Functional JAX implementation for Category 05 Subfolder 02 Topic 03:
U-Net & Multiscale Feature Pyramids (U-Net skip concatenation, BiFPN fast normalized fusion)
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


def conv2d_same(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """2D convolution with padding 1 and stride 1."""
    dn = jax.lax.ConvDimensionNumbers((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))
    out = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dimension_numbers=dn,
    )
    return out + b[None, :, None, None]


def unet_mini_forward(
    x: jnp.ndarray,
    enc_w: jnp.ndarray,
    enc_b: jnp.ndarray,
    bot_w: jnp.ndarray,
    bot_b: jnp.ndarray,
    dec_w: jnp.ndarray,
    dec_b: jnp.ndarray,
) -> jnp.ndarray:
    """U-Net Mini forward pass."""
    B, C, H, W = x.shape

    # 1. Encoder Conv (4 -> 8, pad 1) + ReLU
    e1 = jax.nn.relu(conv2d_same(x, enc_w, enc_b))

    # 2. 2x2 Max Pooling
    p1 = jnp.max(jnp.reshape(e1, (B, e1.shape[1], H // 2, 2, W // 2, 2)), axis=(3, 5))

    # 3. Bottleneck Conv (8 -> 8, pad 1) + ReLU
    b_feat = jax.nn.relu(conv2d_same(p1, bot_w, bot_b))

    # 4. 2x Nearest Upsample
    up = jnp.repeat(jnp.repeat(b_feat, 2, axis=2), 2, axis=3)

    # 5. Skip Concatenation
    cat_feat = jnp.concatenate([up, e1], axis=1)

    # 6. Decoder Conv (16 -> 4, pad 1) + ReLU
    out = jax.nn.relu(conv2d_same(cat_feat, dec_w, dec_b))
    return out


def bifpn_fusion(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    w: jnp.ndarray,
    conv_w: jnp.ndarray,
    conv_b: jnp.ndarray,
    eps: float = 1e-4,
) -> jnp.ndarray:
    """BiFPN fast normalized weighted fusion."""
    w_relu = jax.nn.relu(w)
    norm_w = w_relu / (jnp.sum(w_relu) + eps)
    fused = norm_w[0] * p1 + norm_w[1] * p2
    w_2d = conv_w[:, :, 0, 0]
    return jnp.einsum("bchw,oc->bohw", fused, w_2d) + conv_b[None, :, None, None]


def verify_against_golden() -> None:
    """Verifies JAX implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/03_unet_and_multiscale_feature_pyramids.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = float(jnp.max(jnp.abs(test_out - weights["test_output"])))
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]

    # 2. U-Net verification
    out_unet = unet_mini_forward(
        x,
        weights["enc_w"],
        weights["enc_b"],
        weights["bot_w"],
        weights["bot_b"],
        weights["dec_w"],
        weights["dec_b"],
    )
    err_unet = float(jnp.max(jnp.abs(out_unet - weights["unet_output"])))
    assert err_unet < 1e-5, f"U-Net parity error {err_unet:.2e} exceeds 1e-5"

    # 3. BiFPN verification
    out_bifpn = bifpn_fusion(
        x,
        out_unet,
        weights["bifpn_w"],
        weights["bifpn_conv_w"],
        weights["bifpn_conv_b"],
    )
    err_bifpn = float(jnp.max(jnp.abs(out_bifpn - weights["bifpn_output"])))
    assert err_bifpn < 1e-5, f"BiFPN parity error {err_bifpn:.2e} exceeds 1e-5"

    print("Functional JAX U-Net & BiFPN parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
