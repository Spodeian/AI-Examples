"""Production Serving Dynamic Batcher Pd Disaggregation and Streaming — functional JAX implementation.

Topic: 17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming. Run ``python jax/17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def prefill_step(
    x: jnp.ndarray,
    wq: jnp.ndarray,
    wk: jnp.ndarray,
    wv: jnp.ndarray,
    wo: jnp.ndarray,
    bo: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prefill worker: compute-bound causal self-attention over prompt sequence."""
    b, s, d = x.shape
    q = x @ wq
    k = x @ wk
    v = x @ wv

    scores = (q @ jnp.swapaxes(k, -2, -1)) / math.sqrt(d)
    mask = jnp.triu(jnp.full((s, s), -10000.0), k=1)
    attn = jax.nn.softmax(scores + mask, axis=-1)
    context = attn @ v
    y = context @ wo + bo
    return y, k, v


def decode_step(
    x_new: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    wq: jnp.ndarray,
    wk: jnp.ndarray,
    wv: jnp.ndarray,
    wo: jnp.ndarray,
    bo: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Decode worker: memory-bound incremental step consuming KV-cache from prefill worker."""
    b, _, d = x_new.shape
    q_new = x_new @ wq
    k_new = x_new @ wk
    v_new = x_new @ wv

    new_k = jnp.concatenate([k_cache, k_new], axis=1)
    new_v = jnp.concatenate([v_cache, v_new], axis=1)

    scores = (q_new @ jnp.swapaxes(new_k, -2, -1)) / math.sqrt(d)
    attn = jax.nn.softmax(scores, axis=-1)
    context = attn @ new_v
    y_token = context @ wo + bo
    return y_token, new_k, new_v


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    err_linear = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_linear < 1e-5, f"linear anchor error {err_linear:.2e}"

    y_prefill, k_cache, v_cache = prefill_step(g["x_prefill"], g["wq"], g["wk"], g["wv"], g["wo"], g["bo"])
    err_prefill = float(jnp.max(jnp.abs(y_prefill - g["y_prefill"])))
    assert err_prefill < 1e-5, f"prefill y error {err_prefill:.2e}"
    assert float(jnp.max(jnp.abs(k_cache - g["k_cache"]))) < 1e-5
    assert float(jnp.max(jnp.abs(v_cache - g["v_cache"]))) < 1e-5

    y_decode, new_k, new_v = decode_step(
        g["x_decode"], k_cache, v_cache, g["wq"], g["wk"], g["wv"], g["wo"], g["bo"]
    )
    err_decode = float(jnp.max(jnp.abs(y_decode - g["y_decode"])))
    assert err_decode < 1e-5, f"decode y error {err_decode:.2e}"
    assert float(jnp.max(jnp.abs(new_k - g["new_k"]))) < 1e-5
    assert float(jnp.max(jnp.abs(new_v - g["new_v"]))) < 1e-5

    print("17_deployment_low_level_and_capstones/05_production_serving_dynamic_batcher_pd_disaggregation_and_streaming: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
