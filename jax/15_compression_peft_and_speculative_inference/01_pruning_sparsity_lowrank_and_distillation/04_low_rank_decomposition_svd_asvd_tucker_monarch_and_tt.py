"""Low Rank Decomposition SVD Asvd Tucker Monarch and Tt — functional JAX implementation.

Topic: 15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt.
Run ``python jax/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt.py`` to verify against the golden checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return x @ weight + bias


def truncated_svd(w: jnp.ndarray, rank: int = 4) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    U, S, Vh = jnp.linalg.svd(w, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    w_recon = (U_r * S_r[None, :]) @ Vh_r
    frob_err = jnp.linalg.norm(w - w_recon)
    return w_recon, S_r, frob_err


def asvd_decompose(
    w: jnp.ndarray, act_scales: jnp.ndarray, rank: int = 4
) -> tuple[jnp.ndarray, jnp.ndarray]:
    s = jnp.clip(act_scales, a_min=1e-5)
    w_scaled = w * s[None, :]
    U, S, Vh = jnp.linalg.svd(w_scaled, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    w_scaled_r = (U_r * S_r[None, :]) @ Vh_r
    w_asvd = w_scaled_r / s[None, :]
    return w_asvd, S_r


def monarch_forward(x: jnp.ndarray, b1: jnp.ndarray, b2: jnp.ndarray) -> jnp.ndarray:
    batch = x.shape[0]
    x_blk = x.reshape(batch, 4, 4)
    # Stage 1: Block-diagonal 1
    h1 = jnp.einsum("bki,kij->bkj", x_blk, b1)
    # Stage 2: Transpose block axes
    h1_perm = jnp.transpose(h1, (0, 2, 1))
    # Stage 3: Block-diagonal 2
    h2 = jnp.einsum("bki,kij->bkj", h1_perm, b2)
    return h2.reshape(batch, 16)


def verify_against_golden() -> None:
    g = {k: jnp.asarray(v) for k, v in load_file(str(GOLDEN_PATH)).items()}

    # 1. Linear anchor
    err = float(jnp.max(jnp.abs(linear_anchor(g["test_input"], g["weight"], g["bias"]) - g["test_output"])))
    assert err < 1e-5, f"linear anchor error {err:.2e}"

    # 2. Truncated SVD
    w_svd, s_svd, _ = truncated_svd(g["w"], rank=4)
    assert float(jnp.max(jnp.abs(w_svd - g["w_svd"]))) < 1e-5
    assert float(jnp.max(jnp.abs(s_svd - g["s_svd"]))) < 1e-5

    # 3. ASVD
    w_asvd, s_asvd = asvd_decompose(g["w"], g["act_scales"], rank=4)
    assert float(jnp.max(jnp.abs(w_asvd - g["w_asvd"]))) < 1e-5
    assert float(jnp.max(jnp.abs(s_asvd - g["s_asvd"]))) < 1e-5

    # 4. Monarch forward
    y_monarch = monarch_forward(g["x"], g["b1"], g["b2"])
    assert float(jnp.max(jnp.abs(y_monarch - g["y_monarch"]))) < 1e-5

    print("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
