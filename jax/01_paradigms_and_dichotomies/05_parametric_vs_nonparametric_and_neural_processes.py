"""Functional JAX implementation for Category 01 Topic 05:
05_parametric_vs_nonparametric_and_neural_processes
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


def parametric_forward(
    x_target: Any,
    w_mlp1: Any,
    b_mlp1: Any,
    w_mlp2: Any,
    b_mlp2: Any,
) -> Any:
    """Computes parametric MLP forward pass: ReLU(x @ w1 + b1) @ w2 + b2."""
    xt = jnp.asarray(x_target, dtype=jnp.float32)
    w1 = jnp.asarray(w_mlp1, dtype=jnp.float32)
    b1 = jnp.asarray(b_mlp1, dtype=jnp.float32)
    w2 = jnp.asarray(w_mlp2, dtype=jnp.float32)
    b2 = jnp.asarray(b_mlp2, dtype=jnp.float32)

    h = jnp.maximum(jnp.dot(xt, w1) + b1, 0.0)
    return jnp.dot(h, w2) + b2


def gp_rbf_kernel(x1: Any, x2: Any, lengthscale: Any) -> Any:
    """Computes RBF kernel matrix between two coordinate sets."""
    x1_arr = jnp.asarray(x1, dtype=jnp.float32)
    x2_arr = jnp.asarray(x2, dtype=jnp.float32)
    ls = jnp.asarray(lengthscale, dtype=jnp.float32)

    x1_sq = jnp.sum(jnp.square(x1_arr), axis=-1, keepdims=True)
    x2_sq = jnp.sum(jnp.square(x2_arr), axis=-1, keepdims=True)
    dist_sq = x1_sq + jnp.swapaxes(x2_sq, -2, -1) - 2.0 * jnp.matmul(x1_arr, jnp.swapaxes(x2_arr, -2, -1))
    dist_sq = jnp.maximum(dist_sq, 0.0)
    return jnp.exp(-0.5 * dist_sq / jnp.square(ls))


def gp_forward(
    x_context: Any,
    y_context: Any,
    x_target: Any,
    lengthscale: Any,
    noise_var: Any,
) -> tuple[Any, Any]:
    """Computes Gaussian Process posterior mean via regularized kernel solve."""
    xc = jnp.asarray(x_context, dtype=jnp.float32)
    yc = jnp.asarray(y_context, dtype=jnp.float32)
    xt = jnp.asarray(x_target, dtype=jnp.float32)
    ls = jnp.asarray(lengthscale, dtype=jnp.float32)
    nv = jnp.asarray(noise_var, dtype=jnp.float32)

    num_c = xc.shape[1]
    batch_size = xc.shape[0]

    k_cc = gp_rbf_kernel(xc, xc, ls)
    eye = jnp.broadcast_to(jnp.eye(num_c, dtype=jnp.float32), (batch_size, num_c, num_c))
    k_cc_reg = k_cc + nv * eye

    alpha = jnp.linalg.solve(k_cc_reg, yc)
    k_tc = gp_rbf_kernel(xt, xc, ls)
    y_pred = jnp.matmul(k_tc, alpha)
    return y_pred, alpha


def anp_forward(
    x_context: Any,
    y_context: Any,
    x_target: Any,
    w_enc: Any,
    b_enc: Any,
    w_q: Any,
    w_k: Any,
    w_v: Any,
    w_dec: Any,
    b_dec: Any,
) -> tuple[Any, Any]:
    """Computes Attentive Neural Process prediction via cross-attention."""
    xc = jnp.asarray(x_context, dtype=jnp.float32)
    yc = jnp.asarray(y_context, dtype=jnp.float32)
    xt = jnp.asarray(x_target, dtype=jnp.float32)

    we = jnp.asarray(w_enc, dtype=jnp.float32)
    be = jnp.asarray(b_enc, dtype=jnp.float32)
    wq = jnp.asarray(w_q, dtype=jnp.float32)
    wk = jnp.asarray(w_k, dtype=jnp.float32)
    wv = jnp.asarray(w_v, dtype=jnp.float32)
    wd = jnp.asarray(w_dec, dtype=jnp.float32)
    bd = jnp.asarray(b_dec, dtype=jnp.float32)

    xy_c = jnp.concatenate([xc, yc], axis=-1)
    v_c = jnp.maximum(jnp.dot(xy_c, we) + be, 0.0)

    q = jnp.dot(xt, wq)
    k = jnp.dot(xc, wk)
    v = jnp.dot(v_c, wv)

    scale = 1.0 / (float(q.shape[-1]) ** 0.5)
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale

    # Stable Softmax
    max_scores = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - max_scores)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    r_anp = jnp.matmul(attn_weights, v)
    xt_r = jnp.concatenate([xt, r_anp], axis=-1)
    y_pred = jnp.dot(xt_r, wd) + bd
    return y_pred, attn_weights


def cnp_forward(
    x_context: Any,
    y_context: Any,
    x_target: Any,
    w_enc: Any,
    b_enc: Any,
    w_dec: Any,
    b_dec: Any,
) -> Any:
    """Computes Conditional Neural Process prediction via uniform mean pooling."""
    xc = jnp.asarray(x_context, dtype=jnp.float32)
    yc = jnp.asarray(y_context, dtype=jnp.float32)
    xt = jnp.asarray(x_target, dtype=jnp.float32)

    we = jnp.asarray(w_enc, dtype=jnp.float32)
    be = jnp.asarray(b_enc, dtype=jnp.float32)
    wd = jnp.asarray(w_dec, dtype=jnp.float32)
    bd = jnp.asarray(b_dec, dtype=jnp.float32)

    xy_c = jnp.concatenate([xc, yc], axis=-1)
    v_c = jnp.maximum(jnp.dot(xy_c, we) + be, 0.0)

    r_pooled = jnp.mean(v_c, axis=1, keepdims=True)
    num_target = xt.shape[1]
    r_expanded = jnp.broadcast_to(r_pooled, (xt.shape[0], num_target, r_pooled.shape[-1]))

    xt_r = jnp.concatenate([xt, r_expanded], axis=-1)
    return jnp.dot(xt_r, wd) + bd
