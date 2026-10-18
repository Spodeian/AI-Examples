"""Functional JAX/Equinox implementation for Positional and Structural Encodings (C03 Topic 04).

Implements pure functional operators:
1. Sinusoidal PE: Absolute harmonic coordinate addition (Vaswani et al. 2017).
2. RoPE: Rotary Position Embedding on 2D coordinate pairs (Su et al. 2021).
3. ALiBi: Attention with Linear Biases penalty matrix (Press et al. 2021).
4. RWPE: Random Walk Positional Encoding landing probabilities (Dwivedi et al. 2022).
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


def sinusoidal_pe_forward(x: Any, pe: Any) -> Any:
    """Computes x + pe[:seq_len]."""
    seq_len = x.shape[0]
    return x + pe[:seq_len]


def rope_forward(x: Any, cos: Any, sin: Any) -> Any:
    """Computes RoPE 2D pair rotation."""
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    out_even = x1 * cos - x2 * sin
    out_odd = x1 * sin + x2 * cos
    out = jnp.zeros_like(x)
    if hasattr(out, "at"):
        out = out.at[:, 0::2].set(out_even)
        out = out.at[:, 1::2].set(out_odd)
    else:
        out[:, 0::2] = out_even
        out[:, 1::2] = out_odd
    return out


def alibi_forward(seq_len: int, slopes: Any) -> Any:
    """Computes ALiBi attention bias matrix -slopes * |i - j|."""
    idx = jnp.arange(seq_len, dtype=jnp.float32)
    dist = jnp.abs(jnp.expand_dims(idx, 0) - jnp.expand_dims(idx, 1))
    return -jnp.reshape(slopes, (-1, 1, 1)) * jnp.expand_dims(dist, 0)


def rwpe_forward(adj: Any, k_steps: int = 4) -> Any:
    """Computes Random Walk landing probabilities P^k_ii."""
    deg = jnp.sum(adj, axis=1)
    inv_deg = jnp.diag(1.0 / deg)
    p_walk = jnp.matmul(inv_deg, adj)

    num_nodes = adj.shape[0]
    cols = []
    p_curr = jnp.eye(num_nodes, dtype=jnp.float32)
    for _ in range(k_steps):
        p_curr = jnp.matmul(p_curr, p_walk)
        cols.append(jnp.diag(p_curr))
    return jnp.stack(cols, axis=1)


def linear_forward(x: Any, weight: Any, bias: Any) -> Any:
    """Computes linear anchor map."""
    return jnp.matmul(x, weight) + bias


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/04_positional_and_structural_encodings.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    weights: Dict[str, np.ndarray] = load_file(str(weights_path))

    test_in = jnp.array(weights["test_input"])
    weight = jnp.array(weights["weight"])
    bias = jnp.array(weights["bias"])
    pe = jnp.array(weights["sinusoidal_pe"])
    cos = jnp.array(weights["cos_rope"])
    sin = jnp.array(weights["sin_rope"])
    slopes = jnp.array(weights["alibi_slopes"])
    adj = jnp.array(weights["graph_adj"])

    # 1. Linear Anchor Parity
    y_anchor = linear_forward(test_in, weight, bias)
    err_anchor = float(jnp.max(jnp.abs(y_anchor - jnp.array(weights["test_output"]))))
    assert err_anchor < 1e-5, f"Anchor parity failed: {err_anchor:.2e}"

    # 2. Sinusoidal PE Parity
    y_sin = sinusoidal_pe_forward(test_in, pe)
    err_sin = float(jnp.max(jnp.abs(y_sin - jnp.array(weights["y_sinusoidal"]))))
    assert err_sin < 1e-5, f"Sinusoidal PE parity failed: {err_sin:.2e}"

    # 3. RoPE Parity
    y_rp = rope_forward(test_in, cos, sin)
    err_rp = float(jnp.max(jnp.abs(y_rp - jnp.array(weights["y_rope"]))))
    assert err_rp < 1e-5, f"RoPE parity failed: {err_rp:.2e}"

    # 4. ALiBi Parity
    y_alibi = alibi_forward(test_in.shape[0], slopes)
    err_alibi = float(jnp.max(jnp.abs(y_alibi - jnp.array(weights["alibi_bias"]))))
    assert err_alibi < 1e-5, f"ALiBi parity failed: {err_alibi:.2e}"

    # 5. RWPE Parity
    y_rw = rwpe_forward(adj, k_steps=4)
    err_rw = float(jnp.max(jnp.abs(y_rw - jnp.array(weights["y_rwpe"]))))
    assert err_rw < 1e-5, f"RWPE parity failed: {err_rw:.2e}"

    print(
        f"All JAX assertions passed successfully (Anchor: {err_anchor:.2e}, "
        f"Sin: {err_sin:.2e}, RoPE: {err_rp:.2e}, ALiBi: {err_alibi:.2e}, RWPE: {err_rw:.2e})"
    )


if __name__ == "__main__":
    main()
