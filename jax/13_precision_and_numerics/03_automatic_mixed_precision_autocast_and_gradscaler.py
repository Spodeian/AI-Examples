"""Automatic Mixed Precision Autocast and Gradscaler — functional JAX implementation.

Topic: 13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler. Run ``python jax/13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def dynamic_grad_scaler_step(
    grads: jnp.ndarray,
    scale: float,
    growth_tracker: int,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 4,
) -> tuple[jnp.ndarray, float, int, float]:
    has_inf_nan = bool(jnp.any(jnp.isinf(grads)) or jnp.any(jnp.isnan(grads)))
    unscaled = grads * (1.0 / float(scale))

    if has_inf_nan:
        new_scale = float(scale) * float(backoff_factor)
        new_tracker = 0
        step_skipped = 1.0
    else:
        new_tracker = int(growth_tracker) + 1
        if new_tracker >= int(growth_interval):
            new_scale = float(scale) * float(growth_factor)
            new_tracker = 0
        else:
            new_scale = float(scale)
        step_skipped = 0.0

    return unscaled, new_scale, new_tracker, step_skipped


def amp_autocast_matmul_softmax(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    x_half = x.astype(jnp.float16)
    w_half = w.astype(jnp.float16)
    b_half = b.astype(jnp.float16)
    z_half = x_half @ w_half + b_half

    z_fp32 = z_half.astype(jnp.float32)
    m = jnp.max(z_fp32, axis=-1, keepdims=True)
    e = jnp.exp(z_fp32 - m)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def unscale_and_clip_grads(
    scaled_grads: jnp.ndarray,
    scale: float,
    max_norm: float = 1.0,
    eps: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    unscaled = scaled_grads * (1.0 / float(scale))
    total_norm = jnp.sqrt(jnp.sum(unscaled ** 2))
    coeff = jnp.minimum(1.0, float(max_norm) / (total_norm + float(eps)))
    clipped = unscaled * coeff
    return clipped, total_norm


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 1. AMP Autocast Matmul + Softmax
    amp_probs = amp_autocast_matmul_softmax(g["amp_x"], g["amp_w"], g["amp_b"])
    assert float(jnp.max(jnp.abs(amp_probs - g["amp_probs_ref"]))) < 1e-5

    # 2. Dynamic Scaler: Clean step
    unscaled_c, scale_c, tracker_c, skipped_c = dynamic_grad_scaler_step(
        g["clean_grads"], scale=65536.0, growth_tracker=3, growth_factor=2.0, backoff_factor=0.5, growth_interval=4
    )
    assert float(jnp.max(jnp.abs(unscaled_c - g["unscaled_clean_ref"]))) < 1e-5
    assert abs(scale_c - float(g["scale_clean_ref"])) < 1e-5
    assert tracker_c == int(g["tracker_clean_ref"])
    assert skipped_c == float(g["skipped_clean_ref"])

    # 3. Dynamic Scaler: Overflow step
    unscaled_o, scale_o, tracker_o, skipped_o = dynamic_grad_scaler_step(
        g["unscaled_over_ref"] * 65536.0, scale=65536.0, growth_tracker=2, growth_factor=2.0, backoff_factor=0.5, growth_interval=4
    )
    assert abs(scale_o - float(g["scale_over_ref"])) < 1e-5
    assert tracker_o == int(g["tracker_over_ref"])
    assert skipped_o == float(g["skipped_over_ref"])

    # 4. Unscale and Clip Gradients
    clipped, norm = unscale_and_clip_grads(g["clip_scaled_grads"], scale=1000.0, max_norm=1.0)
    assert float(jnp.max(jnp.abs(clipped - g["clipped_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(norm - g["norm_ref"]))) < 1e-5

    print("13_precision_and_numerics/03_automatic_mixed_precision_autocast_and_gradscaler: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
