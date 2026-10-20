"""Functional JAX implementation for Category 05 Subfolder 02 Topic 04:
Spatiotemporal Video: I3D Weight Inflation & TimeSformer Factorized Space-Time Attention
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


def i3d_inflate_weights(conv2d_w: jnp.ndarray, k_t: int = 3) -> jnp.ndarray:
    """Inflates a 2D convolutional filter into a 3D spatiotemporal filter."""
    expanded = jnp.expand_dims(conv2d_w, axis=2)
    repeated = jnp.repeat(expanded, k_t, axis=2)
    return repeated / float(k_t)


def conv3d_forward(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """3D Convolution forward with padding 1 along all 3 dimensions."""
    dn = jax.lax.ConvDimensionNumbers((0, 1, 2, 3, 4), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4))
    out = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1, 1),
        padding=((1, 1), (1, 1), (1, 1)),
        dimension_numbers=dn,
    )
    return out + b[None, :, None, None, None]


def timesformer_attention_forward(
    z: jnp.ndarray,
    spatial_q_w: jnp.ndarray,
    spatial_k_w: jnp.ndarray,
    spatial_v_w: jnp.ndarray,
    spatial_out_w: jnp.ndarray,
    temporal_q_w: jnp.ndarray,
    temporal_k_w: jnp.ndarray,
    temporal_v_w: jnp.ndarray,
    temporal_out_w: jnp.ndarray,
) -> jnp.ndarray:
    """TimeSformer Factorized Space-Time Attention forward pass."""
    B, T, S, D = z.shape
    scale = jnp.sqrt(D)

    # 1. Spatial Attention across S tokens
    z_s = jnp.reshape(z, (B * T, S, D))
    q_s = z_s @ spatial_q_w
    k_s = z_s @ spatial_k_w
    v_s = z_s @ spatial_v_w
    scores_s = (q_s @ jnp.swapaxes(k_s, -1, -2)) / scale
    attn_s = jax.nn.softmax(scores_s, axis=-1)
    out_s = jnp.reshape((attn_s @ v_s) @ spatial_out_w, (B, T, S, D))
    z_mid = z + out_s

    # 2. Temporal Attention across T frames
    z_t = jnp.reshape(jnp.transpose(z_mid, (0, 2, 1, 3)), (B * S, T, D))
    q_t = z_t @ temporal_q_w
    k_t = z_t @ temporal_k_w
    v_t = z_t @ temporal_v_w
    scores_t = (q_t @ jnp.swapaxes(k_t, -1, -2)) / scale
    attn_t = jax.nn.softmax(scores_t, axis=-1)
    out_t = (attn_t @ v_t) @ temporal_out_w
    out_t_reshaped = jnp.transpose(jnp.reshape(out_t, (B, S, T, D)), (0, 2, 1, 3))

    return z_mid + out_t_reshaped


def verify_against_golden() -> None:
    """Verifies JAX implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/04_spatiotemporal_video_i3d_inflation_and_vivit_timesformer.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = float(jnp.max(jnp.abs(test_out - weights["test_output"])))
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    # 2. I3D Weight Inflation verification
    inflated_w = i3d_inflate_weights(weights["conv2d_w"], k_t=3)
    err_inflate = float(jnp.max(jnp.abs(inflated_w - weights["i3d_w"])))
    assert err_inflate < 1e-6, f"I3D inflation weight error: {err_inflate}"

    # 3. 3D Conv Forward verification
    out_3d = conv3d_forward(weights["x_video"], weights["i3d_w"], weights["i3d_b"])
    err_3d = float(jnp.max(jnp.abs(out_3d - weights["y_3d"])))
    assert err_3d < 1e-5, f"3D conv forward error: {err_3d}"

    # 4. TimeSformer Attention verification
    out_tf = timesformer_attention_forward(
        weights["tokens"],
        weights["spatial_q_w"],
        weights["spatial_k_w"],
        weights["spatial_v_w"],
        weights["spatial_out_w"],
        weights["temporal_q_w"],
        weights["temporal_k_w"],
        weights["temporal_v_w"],
        weights["temporal_out_w"],
    )
    err_tf = float(jnp.max(jnp.abs(out_tf - weights["timesformer_output"])))
    assert err_tf < 1e-5, f"TimeSformer parity error {err_tf:.2e} exceeds 1e-5"

    print("Functional JAX I3D & TimeSformer parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
