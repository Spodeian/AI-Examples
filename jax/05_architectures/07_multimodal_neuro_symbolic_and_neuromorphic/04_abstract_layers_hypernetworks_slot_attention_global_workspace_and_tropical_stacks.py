"""Abstract Layers: Hypernetworks, Slot Attention, Global Workspace, and Tropical Stacks — JAX reference implementation.

Topic: 05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks
Golden weights: weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks.safetensors

``python jax/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks.py`` verifies against the golden checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file

TOPIC = "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Cross-track parity anchor shared by every topic."""
    return jnp.matmul(x, weight) + bias


def hypernetwork_linear(
    x: jax.Array,
    z: jax.Array,
    w_h: jax.Array,
    b_h: jax.Array,
    bias: jax.Array,
) -> jax.Array:
    """Hypernetwork: generates dynamic weights for x from conditioning code z."""
    b, d_in = x.shape
    d_out = bias.shape[0]
    w_flat = jnp.matmul(z, w_h) + b_h
    w_dyn = jnp.reshape(w_flat, (b, d_in, d_out))
    y = jnp.squeeze(jnp.matmul(jnp.expand_dims(x, 1), w_dyn), axis=1) + bias
    return y


def slot_attention(
    slots: jax.Array,
    inputs: jax.Array,
    w_q: jax.Array,
    w_k: jax.Array,
    w_v: jax.Array,
) -> jax.Array:
    """Slot Attention: competitive slot attention over input tokens."""
    d = slots.shape[-1]
    q = jnp.matmul(slots, w_q)
    k = jnp.matmul(inputs, w_k)
    v = jnp.matmul(inputs, w_v)

    scale = 1.0 / math.sqrt(d)
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
    attn = jax.nn.softmax(scores, axis=1)

    attn_norm = attn / (jnp.sum(attn, axis=-1, keepdims=True) + 1e-8)
    updates = jnp.matmul(attn_norm, v)
    return slots + updates


def tropical_max_plus_matmul(x: jax.Array, w: jax.Array) -> jax.Array:
    """Tropical (max-plus) semiring matrix multiplication."""
    summed = jnp.expand_dims(x, -1) + jnp.expand_dims(w, 0)
    return jnp.max(summed, axis=1)


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this JAX implementation reproduces the golden checkpoint."""
    golden = {k: jnp.asarray(v) for k, v in load_file(str(path)).items()}

    y = linear_anchor(golden["test_input"], golden["weight"], golden["bias"])
    err = float(jnp.max(jnp.abs(y - golden["test_output"])))
    assert err < 1e-5, f"linear_anchor parity failed: max abs error {err:.2e}"

    y_hyp = hypernetwork_linear(
        golden["x_hyper"], golden["z_code"], golden["w_h"], golden["b_h"], golden["bias_hyper"]
    )
    assert float(jnp.max(jnp.abs(y_hyp - golden["y_hyper"]))) < 1e-5

    slots_out = slot_attention(
        golden["slots_in"], golden["slot_inputs"], golden["w_q"], golden["w_k"], golden["w_v"]
    )
    assert float(jnp.max(jnp.abs(slots_out - golden["slots_out"]))) < 1e-5

    y_trop = tropical_max_plus_matmul(golden["x_trop"], golden["w_trop"])
    assert float(jnp.max(jnp.abs(y_trop - golden["y_trop"]))) < 1e-5

    print(f"{TOPIC}: JAX verified against golden checkpoint")


if __name__ == "__main__":
    verify_against_golden()
