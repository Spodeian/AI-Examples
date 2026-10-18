"""Functional JAX/Equinox implementation for Weight Initialization and muP Scaling (C03 Topic 03).

Implements pure functional operators:
1. LayerScale: Residual branch elementwise modulation (Touvron et al. 2021).
2. ReZero: Residual branch scalar gate initialized to zero (Zhang et al. 2019; Bachlechner et al. 2020).
3. Residual Scale: 1/sqrt(2L) deep residual network stabilization.
4. muP Readout: Output projection with 1/width_scale multiplier for zero-shot width scaling (Yang et al. 2022).
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


def layerscale_forward(x: Any, residual: Any, gamma: Any) -> Any:
    """Computes LayerScale: x + gamma * residual."""
    return x + gamma * residual


def rezero_forward(x: Any, residual: Any, alpha: Any) -> Any:
    """Computes ReZero: x + alpha * residual."""
    return x + alpha * residual


def residual_scale_forward(x: Any, residual: Any, scale: float) -> Any:
    """Computes Residual 1/sqrt(2L) scaling: x + scale * residual."""
    return x + scale * residual


def mup_readout_forward(x: Any, weight: Any, width_scale: float) -> Any:
    """Computes muP Readout forward: (x @ weight) / width_scale."""
    return jnp.matmul(x, weight) / width_scale


def linear_forward(x: Any, weight: Any, bias: Any) -> Any:
    """Computes linear anchor map."""
    return jnp.matmul(x, weight) + bias


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/03_weight_initialization_and_mup_scaling.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    weights: Dict[str, np.ndarray] = load_file(str(weights_path))

    test_in = jnp.array(weights["test_input"])
    weight = jnp.array(weights["weight"])
    bias = jnp.array(weights["bias"])
    res_branch = jnp.array(weights["residual_branch"])
    ls_gamma = jnp.array(weights["layerscale_gamma"])
    rz_alpha = jnp.array(weights["rezero_alpha"])
    res_scale = float(weights["residual_scale"][0])
    x_hid = jnp.array(weights["x_hidden"])
    w_mup_hid = jnp.array(weights["w_mup_hidden"])
    w_mup_ro = jnp.array(weights["w_mup_readout"])
    width_scale = float(weights["width_scale"][0])

    # 1. Linear Anchor Parity
    y_anchor = linear_forward(test_in, weight, bias)
    err_anchor = float(jnp.max(jnp.abs(y_anchor - jnp.array(weights["test_output"]))))
    assert err_anchor < 1e-5, f"Anchor parity failed: {err_anchor:.2e}"

    # 2. LayerScale Parity
    y_ls = layerscale_forward(test_in, res_branch, ls_gamma)
    err_ls = float(jnp.max(jnp.abs(y_ls - jnp.array(weights["y_layerscale"]))))
    assert err_ls < 1e-5, f"LayerScale parity failed: {err_ls:.2e}"

    # 3. ReZero Parity
    y_rz = rezero_forward(test_in, res_branch, rz_alpha)
    err_rz = float(jnp.max(jnp.abs(y_rz - jnp.array(weights["y_rezero"]))))
    assert err_rz < 1e-5, f"ReZero parity failed: {err_rz:.2e}"

    # 4. Residual Scaling Parity
    y_scaled = residual_scale_forward(test_in, res_branch, res_scale)
    err_scaled = float(jnp.max(jnp.abs(y_scaled - jnp.array(weights["y_residual_scaled"]))))
    assert err_scaled < 1e-5, f"Residual scaling parity failed: {err_scaled:.2e}"

    # 5. muP Hidden Parity
    y_hid = jnp.matmul(x_hid, w_mup_hid)
    err_hid = float(jnp.max(jnp.abs(y_hid - jnp.array(weights["y_mup_hidden"]))))
    assert err_hid < 1e-5, f"muP hidden parity failed: {err_hid:.2e}"

    # 6. muP Readout Parity
    y_ro = mup_readout_forward(x_hid, w_mup_ro, width_scale)
    err_ro = float(jnp.max(jnp.abs(y_ro - jnp.array(weights["y_mup_readout"]))))
    assert err_ro < 1e-5, f"muP readout parity failed: {err_ro:.2e}"

    print(
        f"All JAX assertions passed successfully (Anchor: {err_anchor:.2e}, "
        f"LS: {err_ls:.2e}, RZ: {err_rz:.2e}, Scale: {err_scaled:.2e}, "
        f"muP_hid: {err_hid:.2e}, muP_ro: {err_ro:.2e})"
    )


if __name__ == "__main__":
    main()
