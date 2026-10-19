"""JAX / Equinox implementation for Category 02 Topic 01:
Generalized Linear Models and Proximal Lasso.

Provides functional, jit-compiled implementations of:
1. Ordinary Least Squares (OLS) via QR decomposition.
2. Ridge Regression with L2 regularizer.
3. Lasso Regression via FISTA.
4. Poisson GLM gradient and step.
5. Robust Huber loss regression.
6. Linear anchor module for transitive multi-track parity.
"""

from pathlib import Path
import math
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from safetensors.numpy import load_file


@jax.jit
def solve_ols_qr_jax(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solves Ordinary Least Squares via QR decomposition in JAX."""
    q, r = jnp.linalg.qr(x)
    qty = jnp.matmul(q.T, y)
    w = jsp.linalg.solve_triangular(r, qty, lower=False)
    mse = jnp.mean((jnp.matmul(x, w) - y) ** 2)
    return w, mse


@jax.jit
def solve_ridge_jax(x: jnp.ndarray, y: jnp.ndarray, lambda_reg: float = 0.5) -> jnp.ndarray:
    """Solves Ridge regression in JAX: (X^T X + lambda I)^{-1} X^T y."""
    xtx = jnp.matmul(x.T, x)
    reg_mat = xtx + lambda_reg * jnp.eye(x.shape[1])
    return jnp.linalg.solve(reg_mat, jnp.matmul(x.T, y))


def solve_lasso_fista_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    lambda_reg: float = 0.2,
    steps: int = 10,
) -> jnp.ndarray:
    """Solves Lasso regression via FISTA in JAX."""
    xtx = jnp.matmul(x.T, x)
    eig_max = float(jnp.max(jnp.linalg.eigvalsh(xtx)))
    step_size = 1.0 / (eig_max + 1e-4)

    w = jnp.zeros(x.shape[1], dtype=jnp.float32)
    y_k = w
    t_k = 1.0

    for _ in range(steps):
        grad = jnp.matmul(x.T, jnp.matmul(x, y_k) - y)
        z = y_k - step_size * grad
        threshold = step_size * lambda_reg
        w_next = jnp.sign(z) * jnp.clip(jnp.abs(z) - threshold, a_min=0.0)

        t_next = (1.0 + math.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
        y_k = w_next + ((t_k - 1.0) / t_next) * (w_next - w)
        w = w_next
        t_k = t_next

    return w


@jax.jit
def step_poisson_glm_jax(
    x: jnp.ndarray,
    y_counts: jnp.ndarray,
    w_init: jnp.ndarray,
    lr: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes Poisson GLM gradient and performs 1 gradient step in JAX."""
    mu = jnp.exp(jnp.matmul(x, w_init))
    grad = jnp.matmul(x.T, mu - y_counts)
    w_next = w_init - lr * grad
    return grad, w_next


@jax.jit
def compute_huber_mean_loss_jax(residual: jnp.ndarray, delta: float = 0.5) -> jnp.ndarray:
    """Computes mean Huber loss in JAX."""
    abs_r = jnp.abs(residual)
    quad = jnp.clip(abs_r, a_max=delta)
    lin = abs_r - quad
    loss = 0.5 * (quad ** 2) + delta * lin
    return jnp.mean(loss)


@jax.jit
def anchor_forward_jax(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Forward pass for linear anchor in JAX."""
    return jnp.matmul(x, w) + b


class GLMUnifiedModuleJAX:
    """Unified JAX module for Category 02 Topic 01."""

    def __init__(self, weights_path: Path):
        weights = load_file(str(weights_path))

        self.x_data = jnp.array(weights["x_data"])
        self.y_data = jnp.array(weights["y_data"])
        self.golden_w_ols = jnp.array(weights["golden_w_ols"])
        self.golden_mse_ols = jnp.array(weights["golden_mse_ols"])
        self.golden_w_ridge = jnp.array(weights["golden_w_ridge"])
        self.golden_w_lasso = jnp.array(weights["golden_w_lasso"])

        self.w_poisson_init = jnp.array(weights["w_poisson_init"])
        self.y_counts = jnp.array(weights["y_counts"])
        self.golden_poisson_grad = jnp.array(weights["golden_poisson_grad"])
        self.golden_w_poisson_next = jnp.array(weights["golden_w_poisson_next"])
        self.golden_mean_huber = jnp.array(weights["golden_mean_huber"])

        self.anchor_w = jnp.array(weights["anchor_w"])
        self.anchor_b = jnp.array(weights["anchor_b"])
        self.anchor_x = jnp.array(weights["anchor_x"])
        self.golden_anchor_out = jnp.array(weights["golden_anchor_out"])

    def verify_invariants(self, tolerance: float = 1e-5) -> dict[str, float]:
        """Verifies parity against golden reference tensors."""
        w_ols, mse_ols = solve_ols_qr_jax(self.x_data, self.y_data)
        ols_err = max(
            float(jnp.max(jnp.abs(w_ols - self.golden_w_ols))),
            float(jnp.abs(jnp.squeeze(mse_ols) - jnp.squeeze(self.golden_mse_ols))),
        )

        w_ridge = solve_ridge_jax(self.x_data, self.y_data, lambda_reg=0.5)
        ridge_err = float(jnp.max(jnp.abs(w_ridge - self.golden_w_ridge)))

        w_lasso = solve_lasso_fista_jax(self.x_data, self.y_data, lambda_reg=0.2, steps=10)
        lasso_err = float(jnp.max(jnp.abs(w_lasso - self.golden_w_lasso)))

        p_grad, w_p_next = step_poisson_glm_jax(self.x_data, self.y_counts, self.w_poisson_init, lr=0.01)
        poisson_err = max(
            float(jnp.max(jnp.abs(p_grad - self.golden_poisson_grad))),
            float(jnp.max(jnp.abs(w_p_next - self.golden_w_poisson_next))),
        )

        res = jnp.matmul(self.x_data, w_ols) - self.y_data
        huber = compute_huber_mean_loss_jax(res, delta=0.5)
        huber_err = float(jnp.abs(jnp.squeeze(huber) - jnp.squeeze(self.golden_mean_huber)))

        anchor_out = anchor_forward_jax(self.anchor_x, self.anchor_w, self.anchor_b)
        anchor_err = float(jnp.max(jnp.abs(anchor_out - self.golden_anchor_out)))

        assert ols_err < tolerance, f"OLS error {ols_err} exceeds {tolerance}"
        assert ridge_err < tolerance, f"Ridge error {ridge_err} exceeds {tolerance}"
        assert lasso_err < tolerance, f"Lasso error {lasso_err} exceeds {tolerance}"
        assert poisson_err < tolerance, f"Poisson error {poisson_err} exceeds {tolerance}"
        assert huber_err < tolerance, f"Huber error {huber_err} exceeds {tolerance}"
        assert anchor_err < tolerance, f"Anchor error {anchor_err} exceeds {tolerance}"

        return {
            "ols_err": ols_err,
            "ridge_err": ridge_err,
            "lasso_err": lasso_err,
            "poisson_err": poisson_err,
            "huber_err": huber_err,
            "anchor_err": anchor_err,
        }


if __name__ == "__main__":
    ckpt = Path("weights/02_classical_and_probabilistic_ml/01_generalized_linear_models_and_proximal_lasso.safetensors")
    module = GLMUnifiedModuleJAX(ckpt)
    errors = module.verify_invariants()
    print("JAX Category 02 Topic 01 Invariants Verified:", errors)
