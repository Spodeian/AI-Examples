"""JAX / Functional NumPy implementation of Category 03 Topic 13:
Differentiable Sorting, Ranking, Top-k, Permutahedron, and I-MLE.
"""

from __future__ import annotations

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


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def soft_sort_forward(scores: np.ndarray, tau: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """SoftSort continuous permutation matrix and sorted values approximation."""
    # s_sorted: sort descending
    s_sorted = np.sort(scores, axis=-1)[:, ::-1]
    # dist: [B, K, K] where row i is sorted position i, col j is original score j
    dist = np.abs(s_sorted[:, :, None] - scores[:, None, :])
    p_soft = softmax_np(-dist / tau, axis=-1)
    sorted_approx = np.squeeze(np.matmul(p_soft, scores[:, :, None]), axis=-1)
    return p_soft, sorted_approx


def neural_sort_forward(scores: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """NeuralSort unimodal row-stochastic permutation matrix."""
    B, K = scores.shape
    diff = scores[:, :, None] - scores[:, None, :]
    l1_sum = np.sum(np.abs(diff), axis=-1)

    j_vec = np.arange(1, K + 1, dtype=scores.dtype)
    weights = (K + 1.0 - 2.0 * j_vec)[None, None, :]

    s_expanded = scores[:, :, None]
    l1_expanded = l1_sum[:, :, None]

    cost = weights * s_expanded - l1_expanded
    cost = np.transpose(cost, (0, 2, 1))
    return softmax_np(cost / tau, axis=-1)


def diff_topk_forward(
    scores: np.ndarray,
    k: int = 2,
    tau: float = 0.2,
    num_bisection: int = 15,
) -> np.ndarray:
    """Differentiable Top-k selection weights via bisection root-finding."""
    lo = np.min(scores, axis=-1, keepdims=True) - 5.0 * tau
    hi = np.max(scores, axis=-1, keepdims=True) + 5.0 * tau
    for _ in range(num_bisection):
        mid = 0.5 * (lo + hi)
        w = sigmoid_np((scores - mid) / tau)
        sum_w = np.sum(w, axis=-1, keepdims=True)
        lo = np.where(sum_w > float(k), mid, lo)
        hi = np.where(sum_w > float(k), hi, mid)
    mid = 0.5 * (lo + hi)
    return sigmoid_np((scores - mid) / tau)


def soft_histogram_forward(
    x: np.ndarray,
    bin_centers: np.ndarray,
    bin_width: float = 1.0,
    tau: float = 0.1,
) -> np.ndarray:
    """Differentiable soft histogram via two-sided sigmoid differences."""
    B, N = x.shape
    M = bin_centers.shape[0]
    x_exp = x[:, :, None]
    c_exp = bin_centers.reshape(1, 1, M)

    lo = c_exp - 0.5 * bin_width
    hi = c_exp + 0.5 * bin_width

    in_bin = sigmoid_np((x_exp - lo) / tau) - sigmoid_np((x_exp - hi) / tau)
    hist = np.sum(in_bin, axis=1)
    return hist / (np.sum(hist, axis=-1, keepdims=True) + 1e-12)


def soft_quantile_forward(
    scores: np.ndarray,
    quantile: float = 0.5,
    tau: float = 0.2,
) -> np.ndarray:
    """Differentiable soft quantile estimation via smooth ranking."""
    B, N = scores.shape
    target_rank = quantile * float(N - 1) + 1.0
    diff = (scores[:, :, None] - scores[:, None, :]) / tau
    ranks = 0.5 + np.sum(sigmoid_np(diff), axis=-1)
    dist_sq = (ranks - target_rank) ** 2
    weights = softmax_np(-dist_sq / (2.0 * tau), axis=-1)
    return np.sum(weights * scores, axis=-1)


def verify_topic_13_jax() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/13_differentiable_sorting_ranking_topk_permutahedron_and_imle.safetensors")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights checkpoint: {weights_path}")

    tensors = load_file(str(weights_path))
    print(f"Loaded {len(tensors)} tensors from {weights_path.name}")

    # 1. Linear Anchor Parity
    pred_linear = linear_anchor_forward(tensors["test_input"], tensors["weight"], tensors["bias"])
    err_linear = float(np.max(np.abs(pred_linear - tensors["test_output"])))
    assert err_linear < 1e-5, f"Linear anchor error {err_linear:.2e} exceeds 1e-5"

    # 2. SoftSort Parity
    pred_p_soft, pred_sorted_approx = soft_sort_forward(tensors["softsort_scores"], tau=0.2)
    err_p_soft = float(np.max(np.abs(pred_p_soft - tensors["expected_p_soft"])))
    err_sorted = float(np.max(np.abs(pred_sorted_approx - tensors["expected_sorted_approx"])))
    assert err_p_soft < 1e-5, f"SoftSort P error {err_p_soft:.2e} exceeds 1e-5"
    assert err_sorted < 1e-5, f"SoftSort sorted approx error {err_sorted:.2e} exceeds 1e-5"

    # 3. NeuralSort Parity
    pred_p_neural = neural_sort_forward(tensors["neuralsort_scores"], tau=0.5)
    err_p_neural = float(np.max(np.abs(pred_p_neural - tensors["expected_p_neural"])))
    assert err_p_neural < 1e-5, f"NeuralSort P error {err_p_neural:.2e} exceeds 1e-5"

    # 4. Diff Top-k Parity
    pred_topk = diff_topk_forward(tensors["topk_scores"], k=2, tau=0.2, num_bisection=15)
    err_topk = float(np.max(np.abs(pred_topk - tensors["expected_topk_weights"])))
    assert err_topk < 1e-5, f"Diff Top-k error {err_topk:.2e} exceeds 1e-5"

    # 5. Soft Histogram Parity
    pred_hist = soft_histogram_forward(tensors["hist_x"], tensors["bin_centers"], bin_width=1.0, tau=0.1)
    err_hist = float(np.max(np.abs(pred_hist - tensors["expected_histogram"])))
    assert err_hist < 1e-5, f"Soft histogram error {err_hist:.2e} exceeds 1e-5"

    # 6. Soft Quantile Parity
    pred_median = soft_quantile_forward(tensors["quantile_scores"], quantile=0.5, tau=0.2)
    err_median = float(np.max(np.abs(pred_median - tensors["expected_median"])))
    assert err_median < 1e-5, f"Soft quantile error {err_median:.2e} exceeds 1e-5"

    print(
        f"All 6 JAX/NumPy operations verified! Anchor: {err_linear:.2e}, SoftSort: {err_p_soft:.2e}/{err_sorted:.2e}, "
        f"NeuralSort: {err_p_neural:.2e}, TopK: {err_topk:.2e}, Histogram: {err_hist:.2e}, Median: {err_median:.2e}"
    )


if __name__ == "__main__":
    verify_topic_13_jax()
