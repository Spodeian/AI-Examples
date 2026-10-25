"""Memory Architectures: NTM, PKM, PEER, Infini-attention, and Titans — JAX reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans.safetensors
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ntm_content_addressing_read(
    memory: jnp.ndarray,
    key: jnp.ndarray,
    beta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """NTM content-based addressing and read head in JAX."""
    key_norm = key / jnp.linalg.norm(key, axis=-1, keepdims=True)
    key_norm = jnp.expand_dims(key_norm, 1)

    mem_norm = memory / jnp.linalg.norm(memory, axis=-1, keepdims=True)
    cos_sim = jnp.sum(key_norm * mem_norm, axis=-1)

    # Softmax over memory slots
    scaled_sim = beta * cos_sim
    exp_sim = jnp.exp(scaled_sim - jnp.max(scaled_sim, axis=-1, keepdims=True))
    weights = exp_sim / jnp.sum(exp_sim, axis=-1, keepdims=True)

    read_vec = jnp.einsum("bn,bnw->bw", weights, memory)
    return read_vec, weights


def infini_memory_step(
    m_prev: jnp.ndarray,
    z_prev: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Infini-attention compressive memory retrieval and update in JAX."""
    def elu(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x > 0, x, jnp.exp(x) - 1.0)

    phi_q = elu(q) + 1.0
    phi_k = elu(k) + 1.0

    num = jnp.einsum("bd,bdd->bd", phi_q, m_prev)
    denom = jnp.sum(phi_q * z_prev, axis=-1, keepdims=True) + 1e-4
    a_mem = num / denom

    m_delta = jnp.einsum("bd,bv->bdv", phi_k, v)
    m_next = m_prev + m_delta
    z_next = z_prev + phi_k

    return a_mem, m_next, z_next


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation matches the committed golden checkpoint."""
    raw = load_file(str(path))
    golden = {k: jnp.array(v) for k, v in raw.items()}

    # 1. Anchor
    anchor_out = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err_a = float(jnp.max(jnp.abs(anchor_out - golden["test_output"])))
    assert err_a < 1e-5, f"linear_anchor max abs error {err_a:.2e}"

    # 2. NTM Read
    read_vec, weights = ntm_content_addressing_read(golden["memory"], golden["key"], golden["beta"])
    err_r = float(jnp.max(jnp.abs(read_vec - golden["read_vec"])))
    err_w = float(jnp.max(jnp.abs(weights - golden["ntm_weights"])))
    assert err_r < 1e-5, f"read_vec max abs error {err_r:.2e}"
    assert err_w < 1e-5, f"weights max abs error {err_w:.2e}"

    # 3. Infini-Memory
    a_mem, m_next, z_next = infini_memory_step(
        golden["m_prev"], golden["z_prev"], golden["q"], golden["k"], golden["v"]
    )
    err_am = float(jnp.max(jnp.abs(a_mem - golden["a_mem"])))
    err_mn = float(jnp.max(jnp.abs(m_next - golden["m_next"])))
    err_zn = float(jnp.max(jnp.abs(z_next - golden["z_next"])))
    assert err_am < 1e-5, f"a_mem max abs error {err_am:.2e}"
    assert err_mn < 1e-5, f"m_next max abs error {err_mn:.2e}"
    assert err_zn < 1e-5, f"z_next max abs error {err_zn:.2e}"

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
