"""Quantile and Information Theoretic NF4 Sf4 FP4 Lloyd Max and Dynamic Tree — functional JAX implementation.

Topic: 14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree. Run ``python jax/14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree.safetensors"

NF4_LEVELS = jnp.array([
    -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0,
])

SF4_LEVELS = jnp.array([
    -1.0, -0.7303, -0.5512, -0.4072, -0.2885, -0.1852, -0.0905, 0.0,
    0.0792, 0.1614, 0.2486, 0.3448, 0.4571, 0.5982, 0.7681, 1.0,
])


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def nf4_quantize_dequantize(
    x: jnp.ndarray,
    block_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)

    amax = jnp.max(jnp.abs(x_reshaped), axis=-1, keepdims=True)
    is_zero = (amax == 0.0)
    scale = jnp.where(is_zero, jnp.ones_like(amax), amax)

    norm_x = x_reshaped / scale
    diff = jnp.abs(jnp.expand_dims(norm_x, -1) - NF4_LEVELS)
    idx = jnp.argmin(diff, axis=-1)

    dequant = NF4_LEVELS[idx] * scale
    dequant = jnp.where(is_zero, jnp.zeros_like(dequant), dequant)
    return dequant.reshape(orig_shape), scale.reshape(-1)


def sf4_quantize_dequantize(
    x: jnp.ndarray,
    block_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)

    amax = jnp.max(jnp.abs(x_reshaped), axis=-1, keepdims=True)
    is_zero = (amax == 0.0)
    scale = jnp.where(is_zero, jnp.ones_like(amax), amax)

    norm_x = x_reshaped / scale
    diff = jnp.abs(jnp.expand_dims(norm_x, -1) - SF4_LEVELS)
    idx = jnp.argmin(diff, axis=-1)

    dequant = SF4_LEVELS[idx] * scale
    dequant = jnp.where(is_zero, jnp.zeros_like(dequant), dequant)
    return dequant.reshape(orig_shape), scale.reshape(-1)


def lloyd_max_scalar_quantize(
    data: jnp.ndarray,
    num_levels: int = 8,
    iters: int = 20,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    flat = data.flatten()
    dmin, dmax = float(jnp.min(flat)), float(jnp.max(flat))
    centroids = jnp.linspace(dmin, dmax, num_levels)
    
    for _ in range(iters):
        diff = jnp.abs(jnp.expand_dims(flat, -1) - centroids)
        cluster_idx = jnp.argmin(diff, axis=-1)
        
        # Update centroids
        new_centroids = []
        for k in range(num_levels):
            mask = (cluster_idx == k)
            c_val = jnp.where(jnp.any(mask), jnp.sum(jnp.where(mask, flat, 0.0)) / jnp.maximum(1.0, jnp.sum(mask)), centroids[k])
            new_centroids.append(c_val)
        centroids = jnp.array(new_centroids)

    diff_final = jnp.abs(jnp.expand_dims(data, -1) - centroids)
    idx_final = jnp.argmin(diff_final, axis=-1)
    return centroids[idx_final], centroids


def dynamic_tree_scalar_quantize(
    x: jnp.ndarray,
    depth: int = 3,
) -> jnp.ndarray:
    flat = [x.flatten()]
    for _ in range(depth):
        next_flat = []
        for leaf in flat:
            if len(leaf) == 0:
                next_flat.extend([leaf, leaf])
                continue
            median = jnp.median(leaf)
            left = leaf[leaf <= median]
            right = leaf[leaf > median]
            next_flat.extend([left, right])
        flat = next_flat
        
    centroids = jnp.array([jnp.mean(l) if len(l) > 0 else 0.0 for l in flat])
    diff = jnp.abs(jnp.expand_dims(x, -1) - centroids)
    idx = jnp.argmin(diff, axis=-1)
    return centroids[idx]


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}
    
    # 1. Linear anchor
    err_anchor = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err_anchor < 1e-5, f"linear anchor error {err_anchor:.2e}"
    
    # 2. NF4
    nf4_deq, nf4_scales = nf4_quantize_dequantize(g["raw_gaussian"], block_size=64)
    assert float(jnp.max(jnp.abs(nf4_deq - g["nf4_dequant_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(nf4_scales - g["nf4_scales_ref"]))) < 1e-5
    
    # 3. SF4
    sf4_deq, sf4_scales = sf4_quantize_dequantize(g["raw_heavy"], block_size=64)
    assert float(jnp.max(jnp.abs(sf4_deq - g["sf4_dequant_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(sf4_scales - g["sf4_scales_ref"]))) < 1e-5
    
    # 4. Lloyd Max
    lloyd_q, lloyd_c = lloyd_max_scalar_quantize(g["lloyd_input"], num_levels=8, iters=15)
    assert float(jnp.max(jnp.abs(lloyd_q - g["lloyd_q_ref"]))) < 1e-5
    assert float(jnp.max(jnp.abs(lloyd_c - g["lloyd_centroids_ref"]))) < 1e-5
    
    # 5. Dynamic tree
    tree_q = dynamic_tree_scalar_quantize(g["tree_input"], depth=3)
    assert float(jnp.max(jnp.abs(tree_q - g["tree_q_ref"]))) < 1e-5
    
    print("14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
