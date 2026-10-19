"""Differentiable Decision Trees, NODE, and Neural Additive Models in JAX.

Features:
- Functional JIT-compiled Neural Oblivious Decision Ensemble (NODE) forward pass
- Neural Additive Model (NAM) forward evaluation with modular feature branches
- Pool-Adjacent-Violators (PAV) monotonic isotonic regression
"""

from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
import safetensors.torch
import numpy as np


@jax.jit
def node_forward(
    X: jnp.ndarray,
    W_split: jnp.ndarray,
    b_split: jnp.ndarray,
    leaf_values: jnp.ndarray
) -> jnp.ndarray:
    """Computes NODE forward evaluation across trees with depth=2."""
    # X: [N, D]
    # W_split: [n_trees, depth, D], b_split: [n_trees, depth]
    # leaf_values: [n_trees, 4]
    n_trees = W_split.shape[0]
    N = X.shape[0]

    def tree_eval(t, ensemble_acc):
        p0 = jax.nn.sigmoid(X @ W_split[t, 0] + b_split[t, 0]) # [N]
        p1 = jax.nn.sigmoid(X @ W_split[t, 1] + b_split[t, 1]) # [N]
        w0 = (1.0 - p0) * (1.0 - p1)
        w1 = (1.0 - p0) * p1
        w2 = p0 * (1.0 - p1)
        w3 = p0 * p1
        leaf_probs = jnp.stack([w0, w1, w2, w3], axis=1) # [N, 4]
        tree_out = leaf_probs @ leaf_values[t] # [N]
        return ensemble_acc + tree_out

    return jax.lax.fori_loop(0, n_trees, tree_eval, jnp.zeros(N))


@jax.jit
def nam_forward(
    X: jnp.ndarray,
    nam_w1: jnp.ndarray,
    nam_b1: jnp.ndarray,
    nam_w2: jnp.ndarray,
    nam_b2: jnp.ndarray,
    nam_bias0: jnp.ndarray
) -> jnp.ndarray:
    """Computes Neural Additive Model forward pass: y_hat = bias0 + sum_d f_d(x_d)."""
    # X: [N, D]
    # nam_w1: [D, H, 1], nam_b1: [D, H]
    # nam_w2: [D, 1, H], nam_b2: [D, 1]
    D = X.shape[1]
    N = X.shape[0]

    def feature_eval(d, acc):
        x_d = jnp.expand_dims(X[:, d], 1) # [N, 1]
        h = jax.nn.relu(x_d @ nam_w1[d].T + nam_b1[d]) # [N, H]
        out_d = jnp.squeeze(h @ nam_w2[d].T + nam_b2[d], axis=1) # [N]
        return acc + out_d

    initial = jnp.zeros(N) + nam_bias0[0]
    return jax.lax.fori_loop(0, D, feature_eval, initial)


def pav_isotonic_regression(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators (PAV) algorithm for monotonic regression."""
    y_list = y.copy().tolist()
    blocks = [[i, i, 1.0, float(y_list[i])] for i in range(len(y_list))]
    
    i = 0
    while i < len(blocks) - 1:
        mean_curr = blocks[i][3] / blocks[i][2]
        mean_next = blocks[i + 1][3] / blocks[i + 1][2]
        if mean_curr > mean_next:
            blocks[i][1] = blocks[i + 1][1]
            blocks[i][2] += blocks[i + 1][2]
            blocks[i][3] += blocks[i + 1][3]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
            
    out = np.empty_like(y)
    for b in blocks:
        val = b[3] / b[2]
        out[b[0]:b[1] + 1] = val
    return out


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    X = jnp.array(chk["X"].numpy())
    W_split = jnp.array(chk["W_split"].numpy())
    b_split = jnp.array(chk["b_split"].numpy())
    leaf_values = jnp.array(chk["leaf_values"].numpy())
    golden_node_out = chk["node_out"].numpy()

    nam_w1 = jnp.array(chk["nam_w1"].numpy())
    nam_b1 = jnp.array(chk["nam_b1"].numpy())
    nam_w2 = jnp.array(chk["nam_w2"].numpy())
    nam_b2 = jnp.array(chk["nam_b2"].numpy())
    nam_bias0 = jnp.array(chk["nam_bias0"].numpy())
    golden_nam_out = chk["nam_out"].numpy()

    y_noisy = chk["y_noisy"].numpy()
    golden_iso_fit = chk["iso_fit"].numpy()

    # 1. NODE Parity
    node_out = node_forward(X, W_split, b_split, leaf_values)
    diff_node = float(np.max(np.abs(np.array(node_out) - golden_node_out)))
    assert diff_node < 1e-5, f"JAX NODE output mismatch: {diff_node:.2e}"

    # 2. NAM Parity
    nam_out = nam_forward(X, nam_w1, nam_b1, nam_w2, nam_b2, nam_bias0)
    diff_nam = float(np.max(np.abs(np.array(nam_out) - golden_nam_out)))
    assert diff_nam < 1e-5, f"JAX NAM output mismatch: {diff_nam:.2e}"

    # 3. PAV Isotonic Parity
    iso_fit = pav_isotonic_regression(y_noisy)
    diff_iso = float(np.max(np.abs(iso_fit - golden_iso_fit)))
    assert diff_iso < 1e-5, f"JAX PAV Isotonic mismatch: {diff_iso:.2e}"

    print("JAX Topic 07 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
