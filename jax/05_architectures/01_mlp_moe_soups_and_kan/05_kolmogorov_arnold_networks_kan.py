"""JAX / NumPy implementation for Category 05 Topic 05:
Kolmogorov-Arnold Networks (KAN): B-Spline KAN, FastKAN (RBF), and ChebyKAN
in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def silu_np(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


def chebyshev_polynomials_np(x: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate Chebyshev polynomials T_0(u) to T_degree(u) for u = tanh(x).
    Returns tensor with trailing dimension (degree + 1).
    """
    u = np.tanh(x)
    T = [np.ones_like(u), u]
    for _ in range(2, degree + 1):
        T.append(2.0 * u * T[-1] - T[-2])
    return np.stack(T, axis=-1)


def chebykan_forward_np(x: np.ndarray, cheby_c: np.ndarray, degree: int = 4) -> np.ndarray:
    """ChebyKAN forward pass.
    
    Args:
        x: Input array of shape (..., in_features)
        cheby_c: Coefficient matrix of shape (in_features, out_features, degree + 1)
        degree: Degree of Chebyshev polynomials
        
    Returns:
        Output array of shape (..., out_features)
    """
    T_x = chebyshev_polynomials_np(x, degree)  # (..., in_features, degree + 1)
    return np.einsum("...id,ijd->...j", T_x, cheby_c)


def fastkan_forward_np(
    x: np.ndarray,
    base_w: np.ndarray,
    rbf_w: np.ndarray,
    grid: np.ndarray,
    denominator: float,
) -> np.ndarray:
    """FastKAN forward pass with SiLU base and Gaussian Radial Basis Functions.
    
    Args:
        x: Input array of shape (..., in_features)
        base_w: Linear base weight of shape (in_features, out_features)
        rbf_w: RBF weight of shape (in_features, num_grids, out_features)
        grid: Grid centers of shape (num_grids,)
        denominator: RBF width scaling factor h
        
    Returns:
        Output array of shape (..., out_features)
    """
    # Base branch: silu(x) @ base_w
    base_out = np.matmul(silu_np(x), base_w)

    # RBF branch: exp(-((x - grid) / h)^2)
    diff = np.expand_dims(x, axis=-1) - grid  # (..., in_features, num_grids)
    phi = np.exp(-np.square(diff / denominator))

    # Contract across in_features and num_grids
    rbf_out = np.einsum("...ig,igj->...j", phi, rbf_w)

    return base_out + rbf_out


def linear_anchor(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Standard linear anchor for test_cross_track_parity.py."""
    return np.matmul(x, weight) + bias


def verify_against_golden() -> None:
    """Verifies JAX / NumPy implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = float(np.max(np.abs(test_out - weights["test_output"])))
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]

    # 2. ChebyKAN verification
    out_cheby = chebykan_forward_np(x, weights["cheby_c"], degree=4)
    err_cheby = float(np.max(np.abs(out_cheby - weights["chebykan_output"])))
    assert err_cheby < 1e-6, f"ChebyKAN parity error {err_cheby:.2e} exceeds 1e-6"

    # 3. FastKAN verification
    out_fastkan = fastkan_forward_np(
        x,
        weights["fastkan_base_w"],
        weights["fastkan_rbf_w"],
        weights["fastkan_grid"],
        float(weights["fastkan_denominator"]),
    )
    err_fastkan = float(np.max(np.abs(out_fastkan - weights["fastkan_output"])))
    assert err_fastkan < 1e-6, f"FastKAN parity error {err_fastkan:.2e} exceeds 1e-6"

    print("JAX / NumPy KAN parity successfully verified against golden SafeTensors!")


if __name__ == "__main__":
    verify_against_golden()
