"""Diffusion Conditioning Controlnet and Ip Adapter — functional JAX implementation.

Topic: 05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter. Run ``python jax/05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


import jax.nn


def controlnet_residual_block(
    x: jnp.ndarray,
    control_signal: jnp.ndarray,
    backbone_w: jnp.ndarray,
    control_w: jnp.ndarray,
    z_in_w: jnp.ndarray,
    z_in_b: jnp.ndarray,
    z_out_w: jnp.ndarray,
    z_out_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ctrl_in = x + (control_signal @ z_in_w + z_in_b)
    h_backbone = x @ backbone_w
    h_control = ctrl_in @ control_w
    ctrl_out = h_control @ z_out_w + z_out_b
    y = h_backbone + ctrl_out
    return y, ctrl_out


def ip_adapter_decoupled_cross_attention(
    q: jnp.ndarray,
    k_text: jnp.ndarray,
    v_text: jnp.ndarray,
    k_img: jnp.ndarray,
    v_img: jnp.ndarray,
    scale: float,
    lambda_img: float,
) -> jnp.ndarray:
    scores_text = (q @ jnp.swapaxes(k_text, -2, -1)) * scale
    attn_text = jax.nn.softmax(scores_text, axis=-1) @ v_text

    scores_img = (q @ jnp.swapaxes(k_img, -2, -1)) * scale
    attn_img = jax.nn.softmax(scores_img, axis=-1) @ v_img

    return attn_text + lambda_img * attn_img


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. ControlNet zero-conv
    y_ctrl, _ = controlnet_residual_block(
        g["x"], g["control_signal"], g["backbone_w"], g["control_w"],
        g["z_in_w"], g["z_in_b"], g["z_out_w"], g["z_out_b"]
    )
    err_ctrl = float(jnp.max(jnp.abs(y_ctrl - g["y_ctrl"])))
    assert err_ctrl < 1e-5, f"y_ctrl error {err_ctrl:.2e}"

    # 3. ControlNet adapted
    y_adapt, _ = controlnet_residual_block(
        g["x"], g["control_signal"], g["backbone_w"], g["control_w"],
        g["z_in_w_adapt"], g["z_in_b"], g["z_out_w_adapt"], g["z_out_b"]
    )
    err_adapt = float(jnp.max(jnp.abs(y_adapt - g["y_adapted"])))
    assert err_adapt < 1e-5, f"y_adapted error {err_adapt:.2e}"

    # 4. IP-Adapter
    scale = float(g["scale"][0])
    lambda_img = float(g["lambda_img"][0])
    ip_out = ip_adapter_decoupled_cross_attention(
        g["q"], g["k_text"], g["v_text"], g["k_img"], g["v_img"], scale, lambda_img
    )
    err_ip = float(jnp.max(jnp.abs(ip_out - g["ip_out"])))
    assert err_ip < 1e-5, f"ip_out error {err_ip:.2e}"

    print("05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
