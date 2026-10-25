"""PEFT SVD and Adaptive PiSSA Adalora Vera — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera. Run ``python jax/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def pissa_forward(
    x: jnp.ndarray,
    w_res: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
) -> jnp.ndarray:
    return x @ w_res + (x @ b) @ a


def adalora_forward(
    x: jnp.ndarray,
    w0: jnp.ndarray,
    p: jnp.ndarray,
    lambda_diag: jnp.ndarray,
    q: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    effective_lambda = jnp.diag(lambda_diag * mask)
    delta_w = p @ effective_lambda @ q
    h = x @ w0 + x @ delta_w
    return h, delta_w


def vera_forward(
    x: jnp.ndarray,
    w0: jnp.ndarray,
    a_shared: jnp.ndarray,
    b_shared: jnp.ndarray,
    d_vec: jnp.ndarray,
    b_vec: jnp.ndarray,
) -> jnp.ndarray:
    base_out = x @ w0
    x_a = x @ a_shared
    scaled_xa = x_a * d_vec
    xa_b = scaled_xa @ b_shared
    adapter_out = xa_b * b_vec
    return base_out + adapter_out


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. PiSSA
    p_out = pissa_forward(g["x"], g["w_res"], g["b_pissa"], g["a_pissa"])
    err = float(jnp.max(jnp.abs(p_out - g["pissa_out"])))
    assert err < 1e-5, f"pissa_out error {err:.2e}"

    # 3. AdaLoRA
    a_out, d_w = adalora_forward(g["x"], g["w0"], g["p"], g["lambda_diag"], g["q"], g["ada_mask"])
    err = float(jnp.max(jnp.abs(a_out - g["ada_out"])))
    assert err < 1e-5, f"ada_out error {err:.2e}"
    err = float(jnp.max(jnp.abs(d_w - g["delta_w"])))
    assert err < 1e-5, f"delta_w error {err:.2e}"

    # 4. VeRA
    v_out = vera_forward(g["x"], g["w0"], g["a_shared"], g["b_shared"], g["d_vec"], g["b_vec"])
    err = float(jnp.max(jnp.abs(v_out - g["vera_out"])))
    assert err < 1e-5, f"vera_out error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
