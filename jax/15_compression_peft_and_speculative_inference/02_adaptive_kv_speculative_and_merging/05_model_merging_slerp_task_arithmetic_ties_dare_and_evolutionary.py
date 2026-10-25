"""Model Merging Slerp Task Arithmetic TIES DARE and Evolutionary — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary. Run ``python jax/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def slerp_merge(
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    t: float,
    eps: float = 1e-6,
) -> jnp.ndarray:
    orig_shape = v0.shape
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()

    norm0 = float(jnp.linalg.norm(v0_flat))
    norm1 = float(jnp.linalg.norm(v1_flat))

    if norm0 < eps or norm1 < eps:
        return (1.0 - t) * v0 + t * v1

    u0 = v0_flat / norm0
    u1 = v1_flat / norm1

    dot = float(jnp.clip(jnp.sum(u0 * u1), -1.0, 1.0))
    theta = float(jnp.arccos(dot))

    if abs(jnp.sin(theta)) < eps or abs(theta) < eps:
        return (1.0 - t) * v0 + t * v1

    sin_theta = float(jnp.sin(theta))
    w0 = float(jnp.sin((1.0 - t) * theta)) / sin_theta
    w1 = float(jnp.sin(t * theta)) / sin_theta

    res_flat = w0 * v0_flat + w1 * v1_flat
    return res_flat.reshape(orig_shape)


def task_arithmetic_merge(
    base: jnp.ndarray,
    deltas: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    num_tasks = deltas.shape[0]
    merged = base
    for k in range(num_tasks):
        merged = merged + weights[k] * deltas[k]
    return merged


def ties_merge(
    base: jnp.ndarray,
    deltas: jnp.ndarray,
    trim_ratio: float = 0.5,
    scale: float = 0.5,
) -> jnp.ndarray:
    num_tasks, dim1, dim2 = deltas.shape
    total_elements = dim1 * dim2
    keep_k = max(1, int(round(total_elements * (1.0 - trim_ratio))))

    trimmed_deltas = jnp.zeros_like(deltas)
    for m in range(num_tasks):
        d_flat = deltas[m].flatten()
        abs_d = jnp.abs(d_flat)
        top_idx = jnp.argsort(abs_d)[-keep_k:]
        trimmed_flat = jnp.zeros_like(d_flat)
        trimmed_flat = trimmed_flat.at[top_idx].set(d_flat[top_idx])
        trimmed_deltas = trimmed_deltas.at[m].set(trimmed_flat.reshape((dim1, dim2)))

    sum_deltas = jnp.sum(trimmed_deltas, axis=0)
    elected_sign = jnp.sign(sum_deltas)

    merged_delta = jnp.zeros((dim1, dim2), dtype=deltas.dtype)
    counts = jnp.zeros((dim1, dim2), dtype=deltas.dtype)

    for m in range(num_tasks):
        agree_mask = (jnp.sign(trimmed_deltas[m]) == elected_sign) & (trimmed_deltas[m] != 0.0)
        merged_delta = merged_delta + jnp.where(agree_mask, trimmed_deltas[m], 0.0)
        counts = counts + agree_mask.astype(deltas.dtype)

    avg_delta = jnp.where(counts > 0, merged_delta / counts, 0.0)
    return base + scale * avg_delta


def dare_rescale(
    delta: jnp.ndarray,
    mask: jnp.ndarray,
    drop_prob: float = 0.5,
) -> jnp.ndarray:
    rescale_factor = 1.0 / (1.0 - drop_prob)
    return delta * mask * rescale_factor


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. SLERP
    s_out = slerp_merge(g["v0"], g["v1"], t=0.3)
    err = float(jnp.max(jnp.abs(s_out - g["slerp_out"])))
    assert err < 1e-5, f"slerp_out error {err:.2e}"

    # 3. Task Arithmetic
    a_out = task_arithmetic_merge(g["base_w"], g["deltas"], g["task_weights"])
    err = float(jnp.max(jnp.abs(a_out - g["arithmetic_out"])))
    assert err < 1e-5, f"arithmetic_out error {err:.2e}"

    # 4. TIES Merge
    t_out = ties_merge(g["base_w"], g["deltas"], trim_ratio=0.5, scale=0.6)
    err = float(jnp.max(jnp.abs(t_out - g["ties_out"])))
    assert err < 1e-5, f"ties_out error {err:.2e}"

    # 5. DARE
    d_out = dare_rescale(g["deltas"][0], g["dare_mask"], drop_prob=0.5)
    err = float(jnp.max(jnp.abs(d_out - g["dare_out"])))
    assert err < 1e-5, f"dare_out error {err:.2e}"

    print("15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
