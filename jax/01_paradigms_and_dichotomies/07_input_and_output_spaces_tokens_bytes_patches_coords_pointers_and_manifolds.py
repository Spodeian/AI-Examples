"""Functional JAX implementation for Category 01 Topic 07:
07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds
"""

from __future__ import annotations

from typing import Any

try:
    import jax.numpy as jnp

    import jax
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
    _ = jax
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


def rotation_6d_to_matrix(d6: Any) -> Any:
    """Maps unconstrained 6D representation to SO(3) 3x3 rotation matrix via Gram-Schmidt."""
    d6_arr = jnp.asarray(d6, dtype=jnp.float32)
    a1 = d6_arr[..., 0:3]
    a2 = d6_arr[..., 3:6]

    norm_a1 = jnp.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = a1 / jnp.maximum(norm_a1, 1e-8)

    dot_prod = jnp.sum(b1 * a2, axis=-1, keepdims=True)
    u2 = a2 - dot_prod * b1
    norm_u2 = jnp.linalg.norm(u2, axis=-1, keepdims=True)
    b2 = u2 / jnp.maximum(norm_u2, 1e-8)

    b3 = jnp.cross(b1, b2, axis=-1)
    return jnp.stack([b1, b2, b3], axis=-1)


def pointer_network_forward(
    encoder_seq: Any,
    decoder_state: Any,
    w_enc: Any,
    w_dec: Any,
    b_ptr: Any,
    v_ptr: Any,
) -> tuple[Any, Any]:
    """Computes pointer logits and softmax distribution over variable-length inputs."""
    enc = jnp.asarray(encoder_seq, dtype=jnp.float32)
    dec = jnp.asarray(decoder_state, dtype=jnp.float32)
    we = jnp.asarray(w_enc, dtype=jnp.float32)
    wd = jnp.asarray(w_dec, dtype=jnp.float32)
    bp = jnp.asarray(b_ptr, dtype=jnp.float32)
    vp = jnp.asarray(v_ptr, dtype=jnp.float32)

    e_proj = jnp.dot(enc, we)
    d_proj = jnp.expand_dims(jnp.dot(dec, wd), axis=1)
    u_scores = jnp.tanh(e_proj + d_proj + bp)
    logits = jnp.squeeze(jnp.dot(u_scores, vp), axis=-1)

    # Numerically stable softmax
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - max_logits)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)
    return logits, probs


def fourier_coordinate_encode(coords: Any, num_bands: int = 4) -> Any:
    """Maps continuous coordinates into sinusoidal multi-scale Fourier basis."""
    c_arr = jnp.asarray(coords, dtype=jnp.float32)
    scales = (2.0 ** jnp.arange(num_bands, dtype=jnp.float32)) * jnp.pi
    scaled = jnp.expand_dims(c_arr, axis=-1) * jnp.reshape(scales, (1, 1, -1))
    sin_part = jnp.sin(scaled)
    cos_part = jnp.cos(scaled)
    out = jnp.stack([sin_part, cos_part], axis=-1)
    return jnp.reshape(out, (c_arr.shape[0], -1))


def patch_embedding_forward(patches: Any, w_patch: Any, b_patch: Any) -> Any:
    """Linearly projects flattened patches into transformer latent dimension."""
    p_arr = jnp.asarray(patches, dtype=jnp.float32)
    wp = jnp.asarray(w_patch, dtype=jnp.float32)
    bp = jnp.asarray(b_patch, dtype=jnp.float32)
    return jnp.dot(p_arr, wp) + bp
