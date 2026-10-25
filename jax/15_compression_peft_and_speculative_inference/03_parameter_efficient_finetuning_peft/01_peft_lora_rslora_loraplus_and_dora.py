"""PEFT LoRA Rslora Loraplus and DoRA — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora. Run ``python jax/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def lora_forward(
    x: jnp.ndarray,
    w0: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    alpha: float = 4.0,
    r: int = 2,
) -> jnp.ndarray:
    scale = alpha / float(r)
    return x @ w0 + scale * ((x @ a) @ b)


def rslora_forward(
    x: jnp.ndarray,
    w0: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    alpha: float = 4.0,
    r: int = 2,
) -> jnp.ndarray:
    scale = alpha / math.sqrt(float(r))
    return x @ w0 + scale * ((x @ a) @ b)


def dora_forward(
    x: jnp.ndarray,
    w0: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    m: jnp.ndarray,
    alpha: float = 4.0,
    r: int = 2,
    eps: float = 1e-8,
) -> jnp.ndarray:
    scale = alpha / float(r)
    delta_v = a @ b
    v = w0 + scale * delta_v
    col_norm = jnp.maximum(jnp.linalg.norm(v, ord=2, axis=0, keepdims=True), eps)
    w_dora = (m / col_norm) * v
    return x @ w_dora


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. LoRA
    l_out = lora_forward(g["x"], g["w0"], g["a"], g["b"], alpha=4.0, r=2)
    err = float(jnp.max(jnp.abs(l_out - g["lora_out"])))
    assert err < 1e-5, f"lora_out error {err:.2e}"

    # 3. rsLoRA
    rs_out = rslora_forward(g["x"], g["w0"], g["a"], g["b"], alpha=4.0, r=2)
    err = float(jnp.max(jnp.abs(rs_out - g["rslora_out"])))
    assert err < 1e-5, f"rslora_out error {err:.2e}"

    # 4. DoRA
    d_out = dora_forward(g["x"], g["w0"], g["a"], g["b"], g["m"], alpha=4.0, r=2)
    err = float(jnp.max(jnp.abs(d_out - g["dora_out"])))
    assert err < 1e-5, f"dora_out error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
