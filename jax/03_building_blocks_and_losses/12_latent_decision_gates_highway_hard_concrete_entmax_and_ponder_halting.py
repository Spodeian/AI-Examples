"""JAX / Functional NumPy implementation of Category 03 Topic 12:
Latent Decision Gates: Highway Networks, Hard Concrete L0, Sparsemax, PonderNet Halting, and FiLM.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import numpy as np
from safetensors.numpy import load_file

try:
    from jaxtyping import Array, Float
except ImportError:
    class _TypeShim:
        def __getitem__(self, item: Any) -> Any:
            return Any

    Array = Any
    Float = _TypeShim()


def linear_anchor_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Linear anchor forward: Y = X @ W + b."""
    return np.matmul(x, weight) + bias


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def relu_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def highway_forward(
    x: np.ndarray,
    w_h: np.ndarray,
    b_h: np.ndarray,
    w_t: np.ndarray,
    b_t: np.ndarray,
) -> np.ndarray:
    """Highway network layer: y = t * relu(x @ W_h + b_h) + (1 - t) * x."""
    h = relu_np(np.matmul(x, w_h) + b_h)
    t = sigmoid_np(np.matmul(x, w_t) + b_t)
    return t * h + (1.0 - t) * x


def glu_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gated Linear Unit: z = x @ W + b, y = z[..., :d] * sigmoid(z[..., d:])."""
    z = np.matmul(x, w) + b
    d = z.shape[-1] // 2
    a = z[..., :d]
    gate = sigmoid_np(z[..., d:])
    return a * gate


def hard_concrete_forward(
    log_alpha: np.ndarray,
    beta: float = 0.67,
    gamma: float = -0.1,
    zeta: float = 1.1,
    u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard Concrete stretched-and-rectified gate with closed-form expected L0 penalty."""
    logit_u = np.log(u) - np.log(1.0 - u)
    s = sigmoid_np((logit_u + log_alpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    g = np.clip(s_bar, 0.0, 1.0)
    offset = beta * math.log(-gamma / zeta)
    l0_prob = sigmoid_np(log_alpha - offset)
    return g, l0_prob


def sparsemax_forward(z: np.ndarray) -> np.ndarray:
    """Exact-zero projection of logits onto the probability simplex."""
    # Sort descending
    z_sorted = np.sort(z, axis=-1)[:, ::-1]
    k = np.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    z_cumsum = np.cumsum(z_sorted, axis=-1)
    support = (1.0 + k * z_sorted) > z_cumsum
    k_star = np.sum(support.astype(k.dtype), axis=-1, keepdims=True)
    # Gather cumsum at k_star - 1
    idx = (k_star - 1).astype(int)
    row_indices = np.arange(z.shape[0])[:, None]
    gathered_cumsum = z_cumsum[row_indices, idx]
    tau = (gathered_cumsum - 1.0) / k_star
    return np.maximum(0.0, z - tau)


def pondernet_halting_forward(
    lambdas: np.ndarray,
    lambda_p: float = 0.3,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PonderNet geometric halting hazard formulation with expected steps and KL prior."""
    B, N = lambdas.shape
    p = np.zeros_like(lambdas)
    cum_unhalted = np.ones((B, 1), dtype=lambdas.dtype)
    for n in range(N):
        p[:, n : n + 1] = lambdas[:, n : n + 1] * cum_unhalted
        cum_unhalted = cum_unhalted * (1.0 - lambdas[:, n : n + 1])

    steps = np.arange(1, N + 1, dtype=lambdas.dtype)[None, :]
    expected_steps = np.sum(p * steps, axis=-1)

    geom_factors = (1.0 - lambda_p) ** (steps - 1.0)
    p_star = lambda_p * geom_factors
    kl = np.sum(p * (np.log(np.maximum(eps, p)) - np.log(np.maximum(eps, p_star))), axis=-1)
    return p, expected_steps, kl


def film_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Feature-wise Linear Modulation: y = gamma * x + beta."""
    return gamma * x + beta


def verify_topic_12_jax() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting.safetensors")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights checkpoint: {weights_path}")

    tensors = load_file(str(weights_path))
    print(f"Loaded {len(tensors)} tensors from {weights_path.name}")

    # 1. Linear Anchor Parity
    pred_linear = linear_anchor_forward(tensors["test_input"], tensors["weight"], tensors["bias"])
    err_linear = float(np.max(np.abs(pred_linear - tensors["test_output"])))
    assert err_linear < 1e-5, f"Linear anchor error {err_linear:.2e} exceeds 1e-5"

    # 2. Highway Parity
    pred_highway = highway_forward(
        tensors["highway_x"], tensors["w_h"], tensors["b_h"], tensors["w_t"], tensors["b_t"]
    )
    err_highway = float(np.max(np.abs(pred_highway - tensors["expected_highway"])))
    assert err_highway < 1e-5, f"Highway error {err_highway:.2e} exceeds 1e-5"

    # 3. GLU Parity
    pred_glu = glu_forward(tensors["glu_x"], tensors["w_glu"], tensors["b_glu"])
    err_glu = float(np.max(np.abs(pred_glu - tensors["expected_glu"])))
    assert err_glu < 1e-5, f"GLU error {err_glu:.2e} exceeds 1e-5"

    # 4. Hard Concrete Parity
    pred_hc_g, pred_hc_l0 = hard_concrete_forward(
        tensors["log_alpha"], beta=0.67, gamma=-0.1, zeta=1.1, u=tensors["hard_concrete_u"]
    )
    err_hc_g = float(np.max(np.abs(pred_hc_g - tensors["expected_hard_concrete_g"])))
    err_hc_l0 = float(np.max(np.abs(pred_hc_l0 - tensors["expected_hard_concrete_l0"])))
    assert err_hc_g < 1e-5, f"Hard concrete g error {err_hc_g:.2e} exceeds 1e-5"
    assert err_hc_l0 < 1e-5, f"Hard concrete L0 error {err_hc_l0:.2e} exceeds 1e-5"

    # 5. Sparsemax Parity
    pred_sparsemax = sparsemax_forward(tensors["sparsemax_logits"])
    err_sparsemax = float(np.max(np.abs(pred_sparsemax - tensors["expected_sparsemax"])))
    assert err_sparsemax < 1e-5, f"Sparsemax error {err_sparsemax:.2e} exceeds 1e-5"

    # 6. PonderNet Halting Parity
    pred_ponder_p, pred_ponder_steps, pred_ponder_kl = pondernet_halting_forward(
        tensors["ponder_lambdas"], lambda_p=0.3
    )
    err_p = float(np.max(np.abs(pred_ponder_p - tensors["expected_ponder_p"])))
    err_steps = float(np.max(np.abs(pred_ponder_steps - tensors["expected_ponder_steps"])))
    err_kl = float(np.max(np.abs(pred_ponder_kl - tensors["expected_ponder_kl"])))
    assert err_p < 1e-5, f"PonderNet p error {err_p:.2e} exceeds 1e-5"
    assert err_steps < 1e-5, f"PonderNet steps error {err_steps:.2e} exceeds 1e-5"
    assert err_kl < 1e-5, f"PonderNet KL error {err_kl:.2e} exceeds 1e-5"

    # 7. FiLM Parity
    pred_film = film_forward(tensors["film_x"], tensors["film_gamma"], tensors["film_beta"])
    err_film = float(np.max(np.abs(pred_film - tensors["expected_film"])))
    assert err_film < 1e-5, f"FiLM error {err_film:.2e} exceeds 1e-5"

    print(
        f"All 7 JAX/NumPy operations verified! Anchor: {err_linear:.2e}, Highway: {err_highway:.2e}, "
        f"GLU: {err_glu:.2e}, HardConcrete: {err_hc_g:.2e}/{err_hc_l0:.2e}, Sparsemax: {err_sparsemax:.2e}, "
        f"PonderNet: {err_p:.2e}/{err_steps:.2e}/{err_kl:.2e}, FiLM: {err_film:.2e}"
    )


if __name__ == "__main__":
    verify_topic_12_jax()
