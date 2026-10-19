"""JAX / Functional implementation for Category 04 Topic 14:
Minimax Saddle Point, Extragradient, OGDA, SGA, and Trust-Region BOBYQA.

Provides pure functional, memory-efficient implementations of:
1. Korpelevich Extragradient step on bilinear and minimax games.
2. Optimistic Gradient Descent-Ascent (OGDA) with negative momentum.
3. Symplectic Gradient Adjustment (SGA) Hamiltonian vector field alignment.
4. Nelder-Mead derivative-free simplex reflection dynamics.
"""

from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def step_extragradient(
    x: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Korpelevich Extragradient step on bilinear game min_x max_y x^T A y."""
    grad_x = np.matmul(A, y)
    grad_y = np.matmul(A.T, x)
    x_tilde = x - eta_x * grad_x
    y_tilde = y + eta_y * grad_y

    grad_x_tilde = np.matmul(A, y_tilde)
    grad_y_tilde = np.matmul(A.T, x_tilde)
    x_next = x - eta_x * grad_x_tilde
    y_next = y + eta_y * grad_y_tilde
    return x_next, y_next


def step_ogda(
    x: np.ndarray,
    y: np.ndarray,
    x_prev: np.ndarray,
    y_prev: np.ndarray,
    A: np.ndarray,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimistic Gradient Descent-Ascent (OGDA) step."""
    grad_x = np.matmul(A, y)
    grad_y = np.matmul(A.T, x)
    grad_x_prev = np.matmul(A, y_prev)
    grad_y_prev = np.matmul(A.T, x_prev)

    x_next = x - 2.0 * eta_x * grad_x + eta_x * grad_x_prev
    y_next = y + 2.0 * eta_y * grad_y - eta_y * grad_y_prev
    return x_next, y_next


def step_sga(
    x: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    lambda_sga: float = 0.5,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Symplectic Gradient Adjustment (SGA) step."""
    AAT_x = np.matmul(A, np.matmul(A.T, x))
    ATA_y = np.matmul(A.T, np.matmul(A, y))
    grad_x = np.matmul(A, y)
    grad_y = np.matmul(A.T, x)

    update_x = grad_x + lambda_sga * AAT_x
    update_y = grad_y - lambda_sga * ATA_y
    x_next = x - eta_x * update_x
    y_next = y + eta_y * update_y
    return x_next, y_next


def nelder_mead_reflect(
    x_best: np.ndarray,
    x_second: np.ndarray,
    x_worst: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Nelder-Mead reflection point calculation."""
    centroid = 0.5 * (x_best + x_second)
    return centroid + alpha * (centroid - x_worst)


def linear_anchor_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear anchor forward pass: x @ w + b."""
    return np.matmul(x, w) + b


def verify_invariants(weights_path: str | Path, atol: float = 1e-5) -> dict[str, float]:
    """Verify parity against golden reference tensors."""
    weights = load_file(str(weights_path))
    errors: dict[str, float] = {}

    # 1. Extragradient Invariant
    x_eg, y_eg = step_extragradient(weights["state_x"], weights["state_y"], weights["matrix_A"], eta_x=0.1, eta_y=0.1)
    err_x_eg = float(np.max(np.abs(x_eg - weights["golden_x_eg"])))
    err_y_eg = float(np.max(np.abs(y_eg - weights["golden_y_eg"])))
    err_eg = max(err_x_eg, err_y_eg)
    assert err_eg < atol, f"Extragradient invariant failed: err={err_eg:.2e} >= {atol:.2e}"
    errors["extragradient_err"] = err_eg

    # 2. OGDA Invariant
    x_ogda, y_ogda = step_ogda(
        weights["state_x"], weights["state_y"], weights["state_x_prev"], weights["state_y_prev"], weights["matrix_A"], eta_x=0.1, eta_y=0.1
    )
    err_x_ogda = float(np.max(np.abs(x_ogda - weights["golden_x_ogda"])))
    err_y_ogda = float(np.max(np.abs(y_ogda - weights["golden_y_ogda"])))
    err_ogda = max(err_x_ogda, err_y_ogda)
    assert err_ogda < atol, f"OGDA invariant failed: err={err_ogda:.2e} >= {atol:.2e}"
    errors["ogda_err"] = err_ogda

    # 3. SGA Invariant
    x_sga, y_sga = step_sga(weights["state_x"], weights["state_y"], weights["matrix_A"], lambda_sga=0.5, eta_x=0.1, eta_y=0.1)
    err_x_sga = float(np.max(np.abs(x_sga - weights["golden_x_sga"])))
    err_y_sga = float(np.max(np.abs(y_sga - weights["golden_y_sga"])))
    err_sga = max(err_x_sga, err_y_sga)
    assert err_sga < atol, f"SGA invariant failed: err={err_sga:.2e} >= {atol:.2e}"
    errors["sga_err"] = err_sga

    # 4. Nelder-Mead Reflection Invariant
    x_refl = nelder_mead_reflect(weights["x_best"], weights["x_second"], weights["x_worst"], alpha=1.0)
    err_refl = float(np.max(np.abs(x_refl - weights["golden_x_reflect"])))
    assert err_refl < atol, f"Nelder-Mead invariant failed: err={err_refl:.2e} >= {atol:.2e}"
    errors["nelder_mead_err"] = err_refl

    # 5. Linear Anchor Invariant
    anchor_out = linear_anchor_forward(weights["anchor_x"], weights["anchor_w"], weights["anchor_b"])
    err_anchor = float(np.max(np.abs(anchor_out - weights["golden_anchor_out"])))
    assert err_anchor < atol, f"Anchor invariant failed: err={err_anchor:.2e} >= {atol:.2e}"
    errors["anchor_err"] = err_anchor

    return errors


if __name__ == "__main__":
    p = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa.safetensors")
    errs = verify_invariants(p)
    print(f"JAX Invariant Parity Checks Passed: {errs}")
