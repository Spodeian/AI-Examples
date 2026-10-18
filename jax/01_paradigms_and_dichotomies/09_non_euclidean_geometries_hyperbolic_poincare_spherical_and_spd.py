"""Functional JAX implementation for Category 01 Topic 09:
09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors.torch import load_file

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


def poincare_project(x: Any, c: float = 1.0, eps: float = 1e-5) -> Any:
    """Projects points strictly inside the Poincaré ball ||x|| < 1/sqrt(c)."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    max_norm = (1.0 - eps) / (c ** 0.5)
    norm = jnp.linalg.norm(x_arr, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, 1e-15)
    cond = norm > max_norm
    projected = x_arr / norm * max_norm
    return jnp.where(cond, projected, x_arr)


def poincare_exp_zero(v: Any, c: float = 1.0) -> Any:
    """Exponential map from tangent space at origin to the Poincaré ball."""
    v_arr = jnp.asarray(v, dtype=jnp.float32)
    sqrt_c = c ** 0.5
    v_norm = jnp.linalg.norm(v_arr, axis=-1, keepdims=True)
    v_norm = jnp.maximum(v_norm, 1e-8)
    tanh_term = jnp.tanh(sqrt_c * v_norm / 2.0)
    return (tanh_term / (sqrt_c * v_norm)) * v_arr


def poincare_log_zero(x: Any, c: float = 1.0) -> Any:
    """Logarithmic map from the Poincaré ball to tangent space at origin."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    sqrt_c = c ** 0.5
    x_norm = jnp.linalg.norm(x_arr, axis=-1, keepdims=True)
    x_norm = jnp.maximum(x_norm, 1e-8)
    clamped_norm = jnp.clip(sqrt_c * x_norm, a_min=0.0, a_max=0.9999)
    atanh_term = jnp.arctanh(clamped_norm)
    return (2.0 * atanh_term / (sqrt_c * x_norm)) * x_arr


def poincare_mobius_add(x: Any, y: Any, c: float = 1.0) -> Any:
    """Möbius addition x (+) y in the Poincaré ball."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    y_arr = jnp.asarray(y, dtype=jnp.float32)

    xy_inner = jnp.sum(x_arr * y_arr, axis=-1, keepdims=True)
    x_sq = jnp.sum(jnp.square(x_arr), axis=-1, keepdims=True)
    y_sq = jnp.sum(jnp.square(y_arr), axis=-1, keepdims=True)

    num = (1.0 + 2.0 * c * xy_inner + c * y_sq) * x_arr + (1.0 - c * x_sq) * y_arr
    denom = 1.0 + 2.0 * c * xy_inner + (c ** 2) * x_sq * y_sq
    denom = jnp.maximum(denom, 1e-15)
    return num / denom


def poincare_distance(x: Any, y: Any, c: float = 1.0) -> Any:
    """Geodesic distance between x and y in the Poincaré ball."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    y_arr = jnp.asarray(y, dtype=jnp.float32)
    neg_x = -x_arr
    diff = poincare_mobius_add(neg_x, y_arr, c=c)
    diff_norm = jnp.linalg.norm(diff, axis=-1)
    diff_norm = jnp.clip(diff_norm, a_min=0.0, a_max=0.9999)
    return (2.0 / (c ** 0.5)) * jnp.arctanh(diff_norm)


def spherical_project(x: Any) -> Any:
    """Normalizes vectors onto the unit hypersphere S^(d-1)."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    norm = jnp.linalg.norm(x_arr, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, 1e-15)
    return x_arr / norm


def spherical_distance(p: Any, q: Any) -> Any:
    """Geodesic angular distance on the hypersphere: arccos(<p, q>)."""
    p_arr = jnp.asarray(p, dtype=jnp.float32)
    q_arr = jnp.asarray(q, dtype=jnp.float32)
    dot = jnp.sum(p_arr * q_arr, axis=-1)
    dot = jnp.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    return jnp.arccos(dot)


def spherical_alignment_loss(p: Any, q: Any) -> Any:
    """Contrastive alignment loss on the sphere: ||p - q||_2^2."""
    p_arr = jnp.asarray(p, dtype=jnp.float32)
    q_arr = jnp.asarray(q, dtype=jnp.float32)
    return jnp.sum(jnp.square(p_arr - q_arr), axis=-1)


def spd_matrix_log(P: Any) -> Any:
    """Computes matrix logarithm for symmetric positive-definite matrices."""
    P_arr = jnp.asarray(P, dtype=jnp.float32)
    L, Q_mat = jnp.linalg.eigh(P_arr)
    L_clamped = jnp.maximum(L, 1e-6)
    log_L = jnp.vectorize(jnp.diag, signature="(m)->(m,m)")(jnp.log(L_clamped))
    return jnp.matmul(Q_mat, jnp.matmul(log_L, jnp.swapaxes(Q_mat, -1, -2)))


def spd_log_euclidean_distance(P: Any, Q: Any) -> Any:
    """Log-Euclidean Riemannian metric: ||log(P) - log(Q)||_F."""
    log_P = spd_matrix_log(P)
    log_Q = spd_matrix_log(Q)
    diff = log_P - log_Q
    return jnp.linalg.norm(diff, axis=(-2, -1))


def linear_forward(x: Any, w: Any, b: Any) -> Any:
    """Canonical linear projection matching test_output."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    w_arr = jnp.asarray(w, dtype=jnp.float32)
    b_arr = jnp.asarray(b, dtype=jnp.float32)
    return jnp.dot(x_arr, w_arr) + b_arr


class NonEuclideanJAXModel:
    """Encapsulates weights and functional evaluation in JAX/NumPy."""

    def __init__(self, weight: Any, bias: Any, c: float = 1.0) -> None:
        self.weight = jnp.asarray(weight, dtype=jnp.float32)
        self.bias = jnp.asarray(bias, dtype=jnp.float32)
        self.c = float(c)

    def forward(self, x: Any) -> Any:
        return linear_forward(x, self.weight, self.bias)


def load_from_safetensors(path: str | Path) -> NonEuclideanJAXModel:
    """Loads weights and returns an initialized NonEuclideanJAXModel."""
    tensors = load_file(str(path))
    return NonEuclideanJAXModel(
        weight=tensors["layers.weight"].numpy(),
        bias=tensors["layers.bias"].numpy(),
        c=float(tensors["layers.c"].item()),
    )


if __name__ == "__main__":
    import numpy as np

    ckpt_path = Path("weights/01_paradigms_and_dichotomies/09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd.safetensors")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    tensors = load_file(str(ckpt_path))
    model = load_from_safetensors(ckpt_path)

    x = tensors["test_input"].numpy()
    y_gold = tensors["test_output"].numpy()
    y_pred = np.asarray(model.forward(x))
    diff_linear = float(np.max(np.abs(y_pred - y_gold)))
    assert diff_linear < 1e-5, f"Linear anchor mismatch: {diff_linear}"

    c_val = model.c
    v = tensors["layers.euclidean_v"].numpy()
    exp_calc = np.asarray(poincare_exp_zero(v, c=c_val))
    assert float(np.max(np.abs(exp_calc - tensors["layers.exp_map_0"].numpy()))) < 1e-5, "Exp map mismatch"

    x_hyp = tensors["layers.poincare_x"].numpy()
    log_calc = np.asarray(poincare_log_zero(x_hyp, c=c_val))
    assert float(np.max(np.abs(log_calc - tensors["layers.log_map_0"].numpy()))) < 1e-5, "Log map mismatch"

    y_hyp = tensors["layers.poincare_y"].numpy()
    mobius_calc = np.asarray(poincare_mobius_add(x_hyp, y_hyp, c=c_val))
    assert float(np.max(np.abs(mobius_calc - tensors["layers.mobius_add"].numpy()))) < 1e-5, "Mobius add mismatch"

    dist_calc = np.asarray(poincare_distance(x_hyp, y_hyp, c=c_val))
    assert float(np.max(np.abs(dist_calc - tensors["layers.poincare_dist"].numpy()))) < 1e-5, "Poincare dist mismatch"

    p = tensors["layers.sphere_p"].numpy()
    q = tensors["layers.sphere_q"].numpy()
    s_dist_calc = np.asarray(spherical_distance(p, q))
    assert float(np.max(np.abs(s_dist_calc - tensors["layers.sphere_dist"].numpy()))) < 1e-5, "Sphere dist mismatch"

    align_calc = np.asarray(spherical_alignment_loss(p, q))
    assert float(np.max(np.abs(align_calc - tensors["layers.alignment_loss"].numpy()))) < 1e-5, "Sphere align mismatch"

    P = tensors["layers.spd_P"].numpy()
    Q = tensors["layers.spd_Q"].numpy()
    spd_dist_calc = np.asarray(spd_log_euclidean_distance(P, Q))
    assert float(np.max(np.abs(spd_dist_calc - tensors["layers.log_euclidean_dist"].numpy()))) < 1e-5, "SPD dist mismatch"

    print("JAX/NumPy Module 09 passed all smoke tests and golden parity assertions!")
