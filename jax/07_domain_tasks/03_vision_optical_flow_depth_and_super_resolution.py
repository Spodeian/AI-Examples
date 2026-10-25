"""Vision Optical Flow Depth and Super Resolution — functional JAX implementation.

Topic: 07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution. Run ``python jax/07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def pixel_shuffle(x: jnp.ndarray, r: int = 2) -> jnp.ndarray:
    b, c_r2, h, w = x.shape
    c = c_r2 // (r * r)
    x_reshaped = x.reshape((b, c, r, r, h, w))
    x_perm = jnp.transpose(x_reshaped, (0, 1, 4, 2, 5, 3))
    return x_perm.reshape((b, c, h * r, w * r))


def scale_invariant_depth_loss(
    pred_depth: jnp.ndarray,
    target_depth: jnp.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-4,
) -> jnp.ndarray:
    d = jnp.log(pred_depth + eps) - jnp.log(target_depth + eps)
    mean_d2 = jnp.mean(d**2, axis=(-2, -1))
    mean_d = jnp.mean(d, axis=(-2, -1))
    return mean_d2 - alpha * (mean_d**2)


def charbonnier_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    eps: float = 1e-3,
) -> jnp.ndarray:
    diff = pred - target
    return jnp.mean(jnp.sqrt(diff**2 + eps**2), axis=(-2, -1))


def raft_correlation(
    fmap1: jnp.ndarray,
    fmap2: jnp.ndarray,
) -> jnp.ndarray:
    b, c, h, w = fmap1.shape
    v1 = fmap1.reshape((b, c, h * w))
    v2 = fmap2.reshape((b, c, h * w))
    scale = 1.0 / jnp.sqrt(c)
    return (jnp.swapaxes(v1, 1, 2) @ v2) * scale


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. PixelShuffle
    ps_out = pixel_shuffle(g["sr_input"], r=2)
    err_ps = float(jnp.max(jnp.abs(ps_out - g["sr_output"])))
    assert err_ps < 1e-5, f"pixel_shuffle error {err_ps:.2e}"

    # 2. Scale-Invariant Depth Loss
    depth_loss = scale_invariant_depth_loss(g["pred_depth"], g["target_depth"], alpha=0.5)
    err_depth = float(jnp.max(jnp.abs(depth_loss - g["si_depth_ref"])))
    assert err_depth < 1e-5, f"depth loss error {err_depth:.2e}"

    # 3. Charbonnier Flow Loss
    flow_loss = charbonnier_loss(g["pred_flow"], g["target_flow"], eps=1e-3)
    err_flow = float(jnp.max(jnp.abs(flow_loss - g["charb_flow_ref"])))
    assert err_flow < 1e-5, f"charbonnier error {err_flow:.2e}"

    # 4. RAFT Correlation
    corr_out = raft_correlation(g["fmap1"], g["fmap2"])
    err_corr = float(jnp.max(jnp.abs(corr_out - g["corr_ref"])))
    assert err_corr < 1e-5, f"raft corr error {err_corr:.2e}"

    print("07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
