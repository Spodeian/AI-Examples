"""JAX / Equinox implementation for Category 04 Topic 20:
Zeroth-Order Optimization: MeZO, Synthetic Gradients (DNI), and HSIC Bottleneck.

Provides functional, jit-compiled implementations of:
1. MeZO directional derivative and parameter update (Malladi et al., 2023).
2. Decoupled Neural Interfaces (DNI) synthetic gradient prediction and MSE loss (Jaderberg et al., 2017).
3. Hilbert-Schmidt Independence Criterion (HSIC) with linear kernels (Ma et al., 2020).
4. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file


@jax.jit
def mezo_step_jax(
    theta: jnp.ndarray,
    z: jnp.ndarray,
    theta_star: jnp.ndarray,
    eps: float = 1e-3,
    eta: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MeZO step in JAX."""
    def loss_fn(p: jnp.ndarray) -> jnp.ndarray:
        diff = p - theta_star
        return 0.5 * jnp.sum(diff ** 2)

    loss_pos = loss_fn(theta + eps * z)
    loss_neg = loss_fn(theta - eps * z)
    dir_deriv = (loss_pos - loss_neg) / (2.0 * eps)
    theta_next = theta - eta * dir_deriv * z
    return loss_pos, loss_neg, dir_deriv, theta_next


@jax.jit
def dni_synthetic_gradient_jax(
    h: jnp.ndarray,
    w_synth: jnp.ndarray,
    b_synth: jnp.ndarray,
    target_grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Decoupled Neural Interfaces synthetic gradient in JAX."""
    pred_grad = h @ w_synth + b_synth
    dni_loss = 0.5 * jnp.mean((pred_grad - target_grad) ** 2)
    return pred_grad, dni_loss


@jax.jit
def hsic_bottleneck_jax(
    h: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Hilbert-Schmidt Independence Criterion with linear kernels in JAX."""
    b = h.shape[0]
    k_mat = h @ h.T
    l_mat = y @ y.T

    eye = jnp.eye(b, dtype=h.dtype)
    ones = jnp.ones((b, b), dtype=h.dtype) / float(b)
    h_c = eye - ones

    k_c = h_c @ k_mat @ h_c
    l_c = h_c @ l_mat @ h_c

    score = (1.0 / ((b - 1.0) ** 2)) * jnp.trace(k_c @ l_c)
    return score


@jax.jit
def linear_anchor_forward_jax(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Linear anchor forward pass in JAX."""
    return x @ w + b


class ZerothOrderUnifiedModuleJAX:
    """Unified container for JAX Topic 20 verification."""

    def __init__(self, weights_path: Path):
        weights = load_file(str(weights_path))

        # MeZO
        self.theta = jnp.array(weights["theta"])
        self.z = jnp.array(weights["z"])
        self.theta_star = jnp.array(weights["theta_star"])
        self.golden_loss_pos = jnp.array(weights["golden_loss_pos"])
        self.golden_loss_neg = jnp.array(weights["golden_loss_neg"])
        self.golden_dir_deriv = jnp.array(weights["golden_dir_deriv"])
        self.golden_theta_next = jnp.array(weights["golden_theta_next"])

        # DNI
        self.h = jnp.array(weights["h"])
        self.w_synth = jnp.array(weights["w_synth"])
        self.b_synth = jnp.array(weights["b_synth"])
        self.target_grad = jnp.array(weights["target_grad"])
        self.golden_pred_grad = jnp.array(weights["golden_pred_grad"])
        self.golden_dni_loss = jnp.array(weights["golden_dni_loss"])

        # HSIC
        self.hsic_h = jnp.array(weights["hsic_h"])
        self.hsic_y = jnp.array(weights["hsic_y"])
        self.golden_hsic_score = jnp.array(weights["golden_hsic_score"])

        # Anchor
        self.anchor_x = jnp.array(weights["anchor_x"])
        self.anchor_w = jnp.array(weights["anchor_w"])
        self.anchor_b = jnp.array(weights["anchor_b"])
        self.golden_anchor_out = jnp.array(weights["golden_anchor_out"])

    def verify_invariants(self, atol: float = 1e-5) -> dict[str, float]:
        """Verifies mathematical invariants against golden references."""
        errors: dict[str, float] = {}

        # 1. MeZO
        loss_p, loss_n, dir_d, th_next = mezo_step_jax(
            self.theta, self.z, self.theta_star, eps=1e-3, eta=0.05
        )
        err_lp = float(jnp.abs(loss_p - self.golden_loss_pos[0]))
        err_ln = float(jnp.abs(loss_n - self.golden_loss_neg[0]))
        err_dd = float(jnp.abs(dir_d - self.golden_dir_deriv[0]))
        err_th = float(jnp.max(jnp.abs(th_next - self.golden_theta_next)))
        err_mezo = max(err_lp, err_ln, err_dd, err_th)
        assert err_mezo < atol, f"JAX MeZO invariant failed: {err_mezo:.2e}"
        errors["mezo_err"] = err_mezo

        # 2. DNI
        pred_g, dni_l = dni_synthetic_gradient_jax(
            self.h, self.w_synth, self.b_synth, self.target_grad
        )
        err_pg = float(jnp.max(jnp.abs(pred_g - self.golden_pred_grad)))
        err_dl = float(jnp.abs(dni_l - self.golden_dni_loss[0]))
        err_dni = max(err_pg, err_dl)
        assert err_dni < atol, f"JAX DNI invariant failed: {err_dni:.2e}"
        errors["dni_err"] = err_dni

        # 3. HSIC
        hsic_s = hsic_bottleneck_jax(self.hsic_h, self.hsic_y)
        err_hsic = float(jnp.abs(hsic_s - self.golden_hsic_score[0]))
        assert err_hsic < atol, f"JAX HSIC invariant failed: {err_hsic:.2e}"
        errors["hsic_err"] = err_hsic

        # 4. Anchor
        anchor_out = linear_anchor_forward_jax(self.anchor_x, self.anchor_w, self.anchor_b)
        err_anchor = float(jnp.max(jnp.abs(anchor_out - self.golden_anchor_out)))
        assert err_anchor < atol, f"JAX Anchor invariant failed: {err_anchor:.2e}"
        errors["anchor_err"] = err_anchor

        return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.safetensors")
    module = ZerothOrderUnifiedModuleJAX(p)
    errs = module.verify_invariants(1e-5)
    print(f"JAX Topic 20 Invariant Parity Checks Passed: {errs}")
