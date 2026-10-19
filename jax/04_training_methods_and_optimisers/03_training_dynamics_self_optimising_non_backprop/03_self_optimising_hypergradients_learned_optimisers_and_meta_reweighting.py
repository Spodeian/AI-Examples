"""JAX / Equinox implementation for Category 04 Topic 17:
Self-Optimising Hypergradients, Learned Optimisers, and Meta-Reweighting.

Provides functional, jit-compiled implementations of:
1. Hypergradient Descent (HD) online learning rate adaptation.
2. Meta-Sample Reweighting for noisy label robustness.
3. Coordinate-wise Learned Optimizer MLP step.
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


@jax.jit
def step_hypergradient_descent_jax(
    w: jnp.ndarray,
    g_curr: jnp.ndarray,
    g_prev: jnp.ndarray,
    lr: jnp.ndarray,
    hyper_lr: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Hypergradient Descent (HD) update in JAX."""
    h_t = jnp.dot(g_curr, g_prev)
    lr_next = lr + hyper_lr * h_t
    w_next = w - lr_next * g_curr
    return w_next, lr_next


@jax.jit
def meta_sample_reweight_jax(
    per_sample_grads: jnp.ndarray,
    val_grad: jnp.ndarray,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Meta-Sample Reweighting in JAX."""
    alignments = per_sample_grads @ val_grad
    weights_relu = jnp.maximum(0.0, alignments)
    sum_weights = jnp.sum(weights_relu) + eps
    norm_weights = weights_relu / sum_weights
    reweighted_grad = norm_weights @ per_sample_grads
    return norm_weights, reweighted_grad


@jax.jit
def step_learned_optimizer_jax(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    momentum: jnp.ndarray,
    w1: jnp.ndarray,
    b1: jnp.ndarray,
    w2: jnp.ndarray,
    b2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Coordinate-wise Learned Optimizer MLP step in JAX."""
    coords_in = jnp.stack([grad, momentum], axis=-1) # [D, 2]
    h = jax.nn.relu(coords_in @ w1 + b1) # [D, 4]
    delta_w = jnp.squeeze(h @ w2 + b2, axis=-1) # [D]
    p_next = param - delta_w
    m_next = 0.9 * momentum + 0.1 * grad
    return p_next, m_next, delta_w


@jax.jit
def anchor_forward_jax(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Linear anchor forward in JAX."""
    return x @ w + b


def verify_invariants(safetensors_path: str | Path, atol: float = 1e-5) -> dict[str, float]:
    """Verifies JAX implementation against golden reference tensors."""
    weights = load_file(str(safetensors_path))
    errors: dict[str, float] = {}

    # 1. Hypergradient Descent
    w_hd = jnp.array(weights["w_hd"])
    g_curr = jnp.array(weights["g_curr"])
    g_prev = jnp.array(weights["g_prev"])
    lr_init = jnp.array(weights["lr_init"])
    hyper_lr = jnp.array(weights["hyper_lr"])
    golden_lr = jnp.array(weights["golden_lr_next"])
    golden_w = jnp.array(weights["golden_w_hd"])

    w_out, lr_out = step_hypergradient_descent_jax(w_hd, g_curr, g_prev, lr_init, hyper_lr)
    err_lr = float(jnp.max(jnp.abs(lr_out - golden_lr)))
    err_w = float(jnp.max(jnp.abs(w_out - golden_w)))
    err_hd = max(err_lr, err_w)
    assert err_hd < atol, f"JAX HD failed: {err_hd:.2e}"
    errors["hd_err"] = err_hd

    # 2. Meta-Sample Reweighting
    per_sample_grads = jnp.array(weights["per_sample_grads"])
    val_grad = jnp.array(weights["val_grad"])
    golden_weights = jnp.array(weights["golden_sample_weights"])
    golden_rew_g = jnp.array(weights["golden_reweighted_grad"])

    weights_out, rew_g_out = meta_sample_reweight_jax(per_sample_grads, val_grad)
    err_weights = float(jnp.max(jnp.abs(weights_out - golden_weights)))
    err_rew_g = float(jnp.max(jnp.abs(rew_g_out - golden_rew_g)))
    err_reweight = max(err_weights, err_rew_g)
    assert err_reweight < atol, f"JAX Reweight failed: {err_reweight:.2e}"
    errors["reweight_err"] = err_reweight

    # 3. Learned Optimizer MLP
    param_l = jnp.array(weights["param_learned"])
    grad_l = jnp.array(weights["grad_learned"])
    mom_l = jnp.array(weights["momentum_learned"])
    w1 = jnp.array(weights["mlp_w1"])
    b1 = jnp.array(weights["mlp_b1"])
    w2 = jnp.array(weights["mlp_w2"])
    b2 = jnp.array(weights["mlp_b2"])
    golden_p_l = jnp.array(weights["golden_param_learned"])
    golden_m_l = jnp.array(weights["golden_momentum_learned"])
    golden_d_w = jnp.array(weights["golden_delta_w"])

    p_out, m_out, d_w_out = step_learned_optimizer_jax(param_l, grad_l, mom_l, w1, b1, w2, b2)
    err_p = float(jnp.max(jnp.abs(p_out - golden_p_l)))
    err_m = float(jnp.max(jnp.abs(m_out - golden_m_l)))
    err_d = float(jnp.max(jnp.abs(d_w_out - golden_d_w)))
    err_learned = max(err_p, err_m, err_d)
    assert err_learned < atol, f"JAX Learned failed: {err_learned:.2e}"
    errors["learned_err"] = err_learned

    # 4. Linear Anchor
    anchor_x = jnp.array(weights["anchor_x"])
    anchor_w = jnp.array(weights["anchor_w"])
    anchor_b = jnp.array(weights["anchor_b"])
    golden_anchor = jnp.array(weights["golden_anchor_out"])

    anchor_out = anchor_forward_jax(anchor_x, anchor_w, anchor_b)
    err_anchor = float(jnp.max(jnp.abs(anchor_out - golden_anchor)))
    assert err_anchor < atol, f"JAX Anchor failed: {err_anchor:.2e}"
    errors["anchor_err"] = err_anchor

    return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.safetensors")
    errs = verify_invariants(p, 1e-5)
    print(f"JAX Invariant Parity Checks Passed: {errs}")
