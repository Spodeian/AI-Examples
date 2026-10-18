"""Functional JAX/NumPy Implementation of Topic 12:
Inductive vs. Transductive Learning and Online Regret vs. Batch IID.

Functional implementations of:
1. Transductive Label Spreading: Zhou et al. (2003) harmonic solution on normalized graph Laplacian.
2. Inductive Linear Predictor: Out-of-sample prediction via learned ridge weights.
3. Online FTRL-Proximal: Coordinate descent updates with exact L1 sparsity.
4. Concept Drift Detection: Page-Hinkley / CUSUM test for non-stationary error streams.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

from safetensors.numpy import load_file

try:
    import jax
    import jax.numpy as jnp
    from jax import Array  # type: ignore[attr-defined]

    HAS_JAX = True
    _ = jax
except (ImportError, AttributeError):
    import numpy as jnp  # type: ignore[no-redef]

    Array = Any  # type: ignore[assignment, misc]
    HAS_JAX = False


def transductive_label_spreading(
    laplacian_s: Any,
    y_init: Any,
    alpha: float = 0.8,
) -> Tuple[Any, Any, Any]:
    """Computes harmonic function on graph manifold F* = (I - alpha * S)^{-1} Y."""
    s = jnp.asarray(laplacian_s, dtype=jnp.float32)
    y = jnp.asarray(y_init, dtype=jnp.float32)

    num_nodes = s.shape[0]
    eye_n = jnp.eye(num_nodes, dtype=jnp.float32)
    inv_mat = jnp.linalg.inv(eye_n - alpha * s)
    f_star = jnp.dot(inv_mat, y)

    # Softmax probabilities
    f_max = jnp.max(f_star, axis=-1, keepdims=True)
    exp_f = jnp.exp(f_star - f_max)
    f_probs = exp_f / jnp.sum(exp_f, axis=-1, keepdims=True)

    preds = jnp.argmax(f_probs, axis=-1).astype(jnp.float32)
    return f_star, f_probs, preds


def inductive_linear_predictor(
    x_test: Any,
    w_ind: Any,
) -> Tuple[Any, Any]:
    """Evaluates inductive linear predictor on out-of-sample points."""
    x = jnp.asarray(x_test, dtype=jnp.float32)
    w = jnp.asarray(w_ind, dtype=jnp.float32)

    scores = jnp.dot(x, w)
    preds = (scores >= 0.5).astype(jnp.float32)
    return scores, preds


def ftrl_proximal_optimizer(
    grad_stream: Any,
    dim: int = 4,
    alpha: float = 0.1,
    beta: float = 1.0,
    lambda1: float = 0.5,
    lambda2: float = 0.1,
) -> Tuple[Any, Any, Any]:
    """Executes FTRL-Proximal online coordinate updates producing exact zeros."""
    grads = jnp.asarray(grad_stream, dtype=jnp.float32)
    time_steps = grads.shape[0]

    z = [0.0] * dim
    n = [0.0] * dim

    for t in range(time_steps):
        g = [float(grads[t, i]) for i in range(dim)]
        w_t = [0.0] * dim
        for i in range(dim):
            if abs(z[i]) <= lambda1:
                w_t[i] = 0.0
            else:
                sign_z = 1.0 if z[i] > 0 else -1.0
                denom = (beta + math.sqrt(n[i])) / alpha + lambda2
                w_t[i] = - (z[i] - sign_z * lambda1) / denom

        for i in range(dim):
            sigma_i = (math.sqrt(n[i] + g[i] ** 2) - math.sqrt(n[i])) / alpha
            z[i] += g[i] - sigma_i * w_t[i]
            n[i] += g[i] ** 2

    final_w = [0.0] * dim
    for i in range(dim):
        if abs(z[i]) <= lambda1:
            final_w[i] = 0.0
        else:
            sign_z = 1.0 if z[i] > 0 else -1.0
            denom = (beta + math.sqrt(n[i])) / alpha + lambda2
            final_w[i] = - (z[i] - sign_z * lambda1) / denom

    return (
        jnp.asarray(final_w, dtype=jnp.float32),
        jnp.asarray(z, dtype=jnp.float32),
        jnp.asarray(n, dtype=jnp.float32),
    )


def page_hinkley_cusum(
    stream_errors: Any,
    delta: float = 0.05,
    threshold: float = 0.5,
    mean_est: float = 0.10,
) -> Tuple[Any, Any, Any]:
    """Performs streaming Page-Hinkley CUSUM change detection on error sequence."""
    errors = jnp.asarray(stream_errors, dtype=jnp.float32)
    time_steps = errors.shape[0]

    cusum_s = []
    ph_stat = []
    drift_flags = []

    cum_s = 0.0
    min_s = 0.0
    for t in range(time_steps):
        e_val = float(errors[t])
        cum_s += e_val - mean_est - delta
        if cum_s < min_s:
            min_s = cum_s
        stat = cum_s - min_s
        cusum_s.append(cum_s)
        ph_stat.append(stat)
        drift_flags.append(1.0 if stat > threshold else 0.0)

    return (
        jnp.asarray(cusum_s, dtype=jnp.float32),
        jnp.asarray(ph_stat, dtype=jnp.float32),
        jnp.asarray(drift_flags, dtype=jnp.float32),
    )


def linear_anchor_forward(x: Any, weight: Any, bias: Any) -> Any:
    """Canonical linear forward pass."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_arr = jnp.asarray(weight, dtype=jnp.float32)
    b_arr = jnp.asarray(bias, dtype=jnp.float32)
    if w_arr.shape[0] == x_arr.shape[-1]:
        return jnp.dot(x_arr, w_arr) + b_arr
    return jnp.dot(x_arr, w_arr.T) + b_arr


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    raw = load_file(checkpoint_path)
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in raw.items()}


if __name__ == "__main__":
    ckpt = (
        Path(__file__).parents[2]
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "12_inductive_vs_transductive_and_online_regret_vs_batch_iid.safetensors"
    )
    tensors = load_checkpoint(str(ckpt))

    # 1. Linear Anchor Parity
    out_anchor = linear_anchor_forward(tensors["x_anchor"], tensors["weight"], tensors["bias"])
    assert jnp.allclose(out_anchor, tensors["linear_anchor_out"], atol=1e-5), "Anchor mismatch"

    # 2. Transductive Label Spreading
    f_star, f_probs, t_preds = transductive_label_spreading(tensors["laplacian_s"], tensors["y_init"])
    assert jnp.allclose(f_star, tensors["f_star"], atol=1e-5), "F* mismatch"
    assert jnp.allclose(f_probs, tensors["f_probs"], atol=1e-5), "F probs mismatch"
    assert jnp.allclose(t_preds, tensors["transductive_preds"], atol=1e-5), "Transductive preds mismatch"

    # 3. Inductive Predictor
    x_unlabeled = tensors["node_coords"][4:]
    ind_scores, ind_preds = inductive_linear_predictor(x_unlabeled, tensors["w_ind"])
    assert jnp.allclose(ind_scores, tensors["inductive_scores"], atol=1e-5), "Inductive scores mismatch"
    assert jnp.allclose(ind_preds, tensors["inductive_preds"], atol=1e-5), "Inductive preds mismatch"

    # 4. FTRL-Proximal Online Optimizer
    w_ftrl, z_acc, n_acc = ftrl_proximal_optimizer(tensors["grad_stream"])
    assert jnp.allclose(w_ftrl, tensors["final_w_ftrl"], atol=1e-5), "FTRL w mismatch"
    assert jnp.allclose(z_acc, tensors["z_accum"], atol=1e-5), "FTRL z mismatch"
    assert jnp.allclose(n_acc, tensors["n_accum"], atol=1e-5), "FTRL n mismatch"

    # 5. Page-Hinkley CUSUM Drift Detector
    s_cusum, ph_s, d_flags = page_hinkley_cusum(tensors["stream_errors"])
    assert jnp.allclose(s_cusum, tensors["cusum_s"], atol=1e-5), "CUSUM s mismatch"
    assert jnp.allclose(ph_s, tensors["ph_stat"], atol=1e-5), "PH stat mismatch"
    assert jnp.allclose(d_flags, tensors["drift_flags"], atol=1e-5), "Drift flags mismatch"

    print("JAX Inductive vs. Transductive & Online Regret successfully verified with zero errors!")
