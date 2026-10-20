"""Functional JAX implementation for Category 05 Subfolder 02 Topic 05:
Wavelet Scattering, WTConv, Steerable & Morphological Networks
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


def haar_dwt2d(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """2D Orthonormal Haar Discrete Wavelet Transform."""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 + x01 - x10 - x11)
    hl = 0.5 * (x00 - x01 + x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh


def haar_idwt2d(ll: jnp.ndarray, lh: jnp.ndarray, hl: jnp.ndarray, hh: jnp.ndarray) -> jnp.ndarray:
    """2D Orthonormal Haar Inverse Discrete Wavelet Transform."""
    B, C, H2, W2 = ll.shape
    H, W = H2 * 2, W2 * 2
    out = jnp.zeros((B, C, H, W), dtype=ll.dtype)

    out = out.at[:, :, 0::2, 0::2].set(0.5 * (ll + lh + hl + hh))
    out = out.at[:, :, 0::2, 1::2].set(0.5 * (ll + lh - hl - hh))
    out = out.at[:, :, 1::2, 0::2].set(0.5 * (ll - lh + hl - hh))
    out = out.at[:, :, 1::2, 1::2].set(0.5 * (ll - lh - hl + hh))
    return out


def wtconv_forward(
    x: jnp.ndarray,
    wt_ll_w: jnp.ndarray,
    wt_ll_b: jnp.ndarray,
    wt_proj_w: jnp.ndarray,
    wt_proj_b: jnp.ndarray,
) -> jnp.ndarray:
    """Wavelet Convolution Block forward pass."""
    ll, lh, hl, hh = haar_dwt2d(x)
    
    # Depthwise 1x1 conv on LL subband
    ll_proc = ll * wt_ll_w.reshape(1, -1, 1, 1) + wt_ll_b.reshape(1, -1, 1, 1)
    
    # Reconstruct via IDWT
    x_rec = haar_idwt2d(ll_proc, lh, hl, hh)
    
    # Base 1x1 projection
    # wt_proj_w is [C_out, C_in, 1, 1]
    base = jnp.einsum("bchw,oc->bohw", x, wt_proj_w[:, :, 0, 0]) + wt_proj_b.reshape(1, -1, 1, 1)
    
    return base + x_rec


def morphological_dilation_forward(x: jnp.ndarray, se: jnp.ndarray) -> jnp.ndarray:
    """Tropical max-plus morphological dilation layer forward pass."""
    B, C, H, W = x.shape
    padded = jnp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    out = jnp.full((B, C, H, W), -1e9, dtype=x.dtype)

    for kh in range(3):
        for kw in range(3):
            val = padded[:, :, kh : kh + H, kw : kw + W] + se.reshape(1, C, 3, 3)[:, :, kh, kw, None, None]
            out = jnp.maximum(out, val)
    return out


def verify_against_golden() -> None:
    """Verifies JAX implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/02_convolutional_and_spatiotemporal/05_wavelet_scattering_wtconv_steerable_and_morphological_networks.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor verification
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = float(jnp.max(jnp.abs(test_out - weights["test_output"])))
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]

    # 2. WTConv Verification
    out_wtconv = wtconv_forward(
        x,
        weights["wt_ll_w"],
        weights["wt_ll_b"],
        weights["wt_proj_w"],
        weights["wt_proj_b"],
    )
    err_wt = float(jnp.max(jnp.abs(out_wtconv - weights["wtconv_output"])))
    assert err_wt < 1e-6, f"WTConv parity error {err_wt:.2e} exceeds 1e-6"

    # 3. Morphological Dilation Verification
    out_morph = morphological_dilation_forward(x, weights["morph_se"])
    err_morph = float(jnp.max(jnp.abs(out_morph - weights["morph_output"])))
    assert err_morph < 1e-6, f"Morphological parity error {err_morph:.2e} exceeds 1e-6"

    print("Functional JAX WTConv & Morphological dilation successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
