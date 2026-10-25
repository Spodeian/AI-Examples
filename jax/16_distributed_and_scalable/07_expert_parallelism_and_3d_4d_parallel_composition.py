"""Expert Parallelism and 3d 4d Parallel Composition — functional JAX implementation.

Topic: 16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition.
Run ``python jax/16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def compute_3d_mesh_coordinates(dp: int, pp: int, tp: int) -> jnp.ndarray:
    world_size = dp * pp * tp
    coords = []
    for r in range(world_size):
        dp_idx = r // (pp * tp)
        rem = r % (pp * tp)
        pp_idx = rem // tp
        tp_idx = rem % tp
        coords.append([dp_idx, pp_idx, tp_idx])
    return jnp.array(coords, dtype=jnp.int64)


def expert_parallel_forward(
    x: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_experts: jnp.ndarray,
    ep_world_size: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_tokens, d_model = x.shape
    num_experts = w_experts.shape[0]
    experts_per_rank = num_experts // ep_world_size

    gate_logits = x @ w_gate
    top2_logits, top2_indices = jax.lax.top_k(gate_logits, k=2)
    top2_probs = jax.nn.softmax(top2_logits, axis=-1)

    dispatch_mask = []
    for r in range(ep_world_size):
        r_start = r * experts_per_rank
        r_end = (r + 1) * experts_per_rank
        mask_r = (top2_indices >= r_start) & (top2_indices < r_end)
        dispatch_mask.append(mask_r)
    dispatch_mask_arr = jnp.stack(dispatch_mask, axis=0).astype(jnp.float32)

    moe_out = jnp.zeros_like(x)
    for t in range(num_tokens):
        for k in range(2):
            exp_idx = int(top2_indices[t, k])
            weight_k = top2_probs[t, k]
            exp_out = x[t] @ w_experts[exp_idx]
            moe_out = moe_out.at[t].add(weight_k * exp_out)

    return moe_out, top2_indices, top2_probs, dispatch_mask_arr


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # Anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # Mesh coords
    coords = compute_3d_mesh_coordinates(2, 2, 2)
    err = float(jnp.max(jnp.abs(coords - g["mesh_coords"])))
    assert err < 1e-5, f"mesh coords error {err:.2e}"

    # Expert Parallel
    moe_out, top2_indices, top2_probs, dispatch_mask = expert_parallel_forward(
        g["x"], g["w_gate"], g["w_experts"], ep_world_size=2
    )
    err = float(jnp.max(jnp.abs(moe_out - g["moe_out"])))
    assert err < 1e-5, f"moe_out error {err:.2e}"
    err = float(jnp.max(jnp.abs(top2_indices - g["top2_indices"])))
    assert err < 1e-5, f"top2_indices error {err:.2e}"
    err = float(jnp.max(jnp.abs(top2_probs - g["top2_probs"])))
    assert err < 1e-5, f"top2_probs error {err:.2e}"
    err = float(jnp.max(jnp.abs(dispatch_mask - g["dispatch_mask"])))
    assert err < 1e-5, f"dispatch_mask error {err:.2e}"

    print("16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
