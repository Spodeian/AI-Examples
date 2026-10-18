"""Functional JAX/Equinox implementation for Normalization Layers Complete (C03 Topic 02).

Implements pure functional normalization operators:
1. LayerNorm: Mean and variance normalization over feature dimension (Ba et al. 2016).
2. RMSNorm: Root Mean Square normalization eliminating mean computation (Zhang & Sennrich 2019).
3. Dynamic Tanh (DyT): Elementwise reduction-free scaling (Zhu et al. 2025).
4. adaLN-Zero: Adaptive LayerNorm with conditioning vector modulation (Peebles & Xie 2023).
5. QK-Norm: Head-wise RMS normalization for Query/Key matrices (Dehghani et al. 2023).
6. Scaled Weight Standardization: Standardizing linear weight matrices (Brock et al. 2021).
"""

from pathlib import Path
from typing import Dict, Any

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp

import numpy as np
from safetensors.numpy import load_file


def layernorm_forward(
    x: Any, gamma: Any, beta: Any, eps: float = 1e-5
) -> Any:
    """Computes Layer Normalization along innermost dimension."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return x_norm * gamma + beta


def rmsnorm_forward(x: Any, gamma: Any, eps: float = 1e-5) -> Any:
    """Computes RMS Normalization along innermost dimension."""
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def dyt_forward(
    x: Any, alpha: Any, gamma: Any, beta: Any
) -> Any:
    """Computes Dynamic Tanh (DyT) reduction-free normalization."""
    return gamma * jnp.tanh(alpha * x) + beta


def adaln_zero_forward(
    x: Any,
    cond: Any,
    w_adaln: Any,
    b_adaln: Any,
    eps: float = 1e-5,
) -> Any:
    """Computes adaLN-Zero adaptive normalization with conditioning vector."""
    proj = jnp.matmul(cond, w_adaln) + b_adaln
    dim = x.shape[-1]
    gamma_mod = proj[:, :dim]
    beta_mod = proj[:, dim:]

    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    base_ln = (x - mean) / jnp.sqrt(var + eps)
    return (1.0 + gamma_mod) * base_ln + beta_mod


def qknorm_forward(x: Any, eps: float = 1e-5) -> Any:
    """Computes Query-Key RMS normalization."""
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x / rms


def weight_standardization_forward(w: Any, eps: float = 1e-5) -> Any:
    """Computes Scaled Weight Standardization along fan-in dimension (axis 1)."""
    fan_in = w.shape[1]
    mean = jnp.mean(w, axis=1, keepdims=True)
    var = jnp.var(w, axis=1, keepdims=True)
    return (w - mean) / (jnp.sqrt(var + eps) * (fan_in ** 0.5))


def linear_forward(x: Any, w: Any, b: Any) -> Any:
    """Computes linear anchor map."""
    return jnp.matmul(x, w) + b


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/02_normalization_layers_complete.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    weights: Dict[str, np.ndarray] = load_file(str(weights_path))

    test_in = jnp.array(weights["test_input"])
    weight = jnp.array(weights["weight"])
    bias = jnp.array(weights["bias"])
    gamma = jnp.array(weights["gamma"])
    beta = jnp.array(weights["beta"])
    dyt_alpha = jnp.array(weights["dyt_alpha"])
    dyt_gamma = jnp.array(weights["dyt_gamma"])
    dyt_beta = jnp.array(weights["dyt_beta"])
    cond = jnp.array(weights["cond_vector"])
    w_adaln = jnp.array(weights["w_adaln"])
    b_adaln = jnp.array(weights["b_adaln"])
    w_matrix = jnp.array(weights["w_matrix"])
    eps = float(weights["eps"][0])

    # 1. Linear Anchor Parity
    y_anchor = linear_forward(test_in, weight, bias)
    err_anchor = float(jnp.max(jnp.abs(y_anchor - jnp.array(weights["test_output"]))))
    assert err_anchor < 1e-5, f"Anchor parity failed: {err_anchor:.2e}"

    # 2. LayerNorm Parity
    y_ln = layernorm_forward(test_in, gamma, beta, eps)
    err_ln = float(jnp.max(jnp.abs(y_ln - jnp.array(weights["y_layernorm"]))))
    assert err_ln < 1e-5, f"LayerNorm parity failed: {err_ln:.2e}"

    # 3. RMSNorm Parity
    y_rms = rmsnorm_forward(test_in, gamma, eps)
    err_rms = float(jnp.max(jnp.abs(y_rms - jnp.array(weights["y_rmsnorm"]))))
    assert err_rms < 1e-5, f"RMSNorm parity failed: {err_rms:.2e}"

    # 4. Dynamic Tanh (DyT) Parity
    y_dyt = dyt_forward(test_in, dyt_alpha, dyt_gamma, dyt_beta)
    err_dyt = float(jnp.max(jnp.abs(y_dyt - jnp.array(weights["y_dyt"]))))
    assert err_dyt < 1e-5, f"DyT parity failed: {err_dyt:.2e}"

    # 5. adaLN-Zero Parity
    y_adaln = adaln_zero_forward(test_in, cond, w_adaln, b_adaln, eps)
    err_adaln = float(jnp.max(jnp.abs(y_adaln - jnp.array(weights["y_adaln_zero"]))))
    assert err_adaln < 1e-5, f"adaLN parity failed: {err_adaln:.2e}"

    # 6. QK-Norm Parity
    y_qk = qknorm_forward(test_in, eps)
    err_qk = float(jnp.max(jnp.abs(y_qk - jnp.array(weights["y_qknorm"]))))
    assert err_qk < 1e-5, f"QKNorm parity failed: {err_qk:.2e}"

    # 7. Scaled Weight Standardization Parity
    w_std = weight_standardization_forward(w_matrix, eps)
    err_ws = float(jnp.max(jnp.abs(w_std - jnp.array(weights["w_standardized"]))))
    assert err_ws < 1e-5, f"Weight standardization parity failed: {err_ws:.2e}"

    print(
        f"All JAX assertions passed successfully (Anchor: {err_anchor:.2e}, "
        f"LN: {err_ln:.2e}, RMS: {err_rms:.2e}, DyT: {err_dyt:.2e}, adaLN: {err_adaln:.2e}, "
        f"QK: {err_qk:.2e}, WS: {err_ws:.2e})"
    )


if __name__ == "__main__":
    main()
