"""PyTorch 2.x implementation of Category 03 Topic 13:
Differentiable Sorting, Ranking, Top-k, Permutahedron, and I-MLE.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


class LinearAnchor(nn.Module):
    """Linear anchor module: Y = X @ W + b."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


def soft_sort_forward(scores: torch.Tensor, tau: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """SoftSort continuous permutation matrix and sorted values approximation."""
    s_sorted, _ = torch.sort(scores, descending=True, dim=-1)
    dist = torch.abs(s_sorted.unsqueeze(-1) - scores.unsqueeze(-2))
    p_soft = torch.softmax(-dist / tau, dim=-1)
    sorted_approx = torch.matmul(p_soft, scores.unsqueeze(-1)).squeeze(-1)
    return p_soft, sorted_approx


def neural_sort_forward(scores: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """NeuralSort unimodal row-stochastic permutation matrix."""
    B, K = scores.shape
    diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    l1_sum = torch.sum(torch.abs(diff), dim=-1)

    j_vec = torch.arange(1, K + 1, device=scores.device, dtype=scores.dtype)
    weights = (K + 1.0 - 2.0 * j_vec).unsqueeze(0).unsqueeze(1)

    s_expanded = scores.unsqueeze(-1)
    l1_expanded = l1_sum.unsqueeze(-1)

    cost = weights * s_expanded - l1_expanded
    cost = cost.permute(0, 2, 1)
    return torch.softmax(cost / tau, dim=-1)


def diff_topk_forward(
    scores: torch.Tensor,
    k: int = 2,
    tau: float = 0.2,
    num_bisection: int = 15,
) -> torch.Tensor:
    """Differentiable Top-k selection weights via bisection root-finding."""
    lo = torch.min(scores, dim=-1, keepdim=True)[0] - 5.0 * tau
    hi = torch.max(scores, dim=-1, keepdim=True)[0] + 5.0 * tau
    for _ in range(num_bisection):
        mid = 0.5 * (lo + hi)
        w = torch.sigmoid((scores - mid) / tau)
        sum_w = torch.sum(w, dim=-1, keepdim=True)
        lo = torch.where(sum_w > float(k), mid, lo)
        hi = torch.where(sum_w > float(k), hi, mid)
    mid = 0.5 * (lo + hi)
    return torch.sigmoid((scores - mid) / tau)


def soft_histogram_forward(
    x: torch.Tensor,
    bin_centers: torch.Tensor,
    bin_width: float = 1.0,
    tau: float = 0.1,
) -> torch.Tensor:
    """Differentiable soft histogram via two-sided sigmoid differences."""
    B, N = x.shape
    M = bin_centers.shape[0]
    x_exp = x.unsqueeze(-1)
    c_exp = bin_centers.view(1, 1, M)

    lo = c_exp - 0.5 * bin_width
    hi = c_exp + 0.5 * bin_width

    in_bin = torch.sigmoid((x_exp - lo) / tau) - torch.sigmoid((x_exp - hi) / tau)
    hist = torch.sum(in_bin, dim=1)
    return hist / (torch.sum(hist, dim=-1, keepdim=True) + 1e-12)


def soft_quantile_forward(
    scores: torch.Tensor,
    quantile: float = 0.5,
    tau: float = 0.2,
) -> torch.Tensor:
    """Differentiable soft quantile estimation via smooth ranking."""
    B, N = scores.shape
    target_rank = quantile * float(N - 1) + 1.0
    diff = (scores.unsqueeze(2) - scores.unsqueeze(1)) / tau
    ranks = 0.5 + torch.sum(torch.sigmoid(diff), dim=-1)
    dist_sq = (ranks - target_rank).pow(2)
    weights = torch.softmax(-dist_sq / (2.0 * tau), dim=-1)
    return torch.sum(weights * scores, dim=-1)


def verify_topic_13_pytorch() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/13_differentiable_sorting_ranking_topk_permutahedron_and_imle.safetensors")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights checkpoint: {weights_path}")

    tensors = load_file(str(weights_path))
    print(f"Loaded {len(tensors)} tensors from {weights_path.name}")

    # 1. Linear Anchor Parity
    anchor = LinearAnchor(4, 4)
    with torch.no_grad():
        anchor.weight.copy_(tensors["weight"])
        anchor.bias.copy_(tensors["bias"])
    pred_linear = anchor(tensors["test_input"])
    err_linear = (pred_linear - tensors["test_output"]).abs().max().item()
    assert err_linear < 1e-5, f"Linear anchor error {err_linear:.2e} exceeds 1e-5"

    # 2. SoftSort Parity
    pred_p_soft, pred_sorted_approx = soft_sort_forward(tensors["softsort_scores"], tau=0.2)
    err_p_soft = (pred_p_soft - tensors["expected_p_soft"]).abs().max().item()
    err_sorted = (pred_sorted_approx - tensors["expected_sorted_approx"]).abs().max().item()
    assert err_p_soft < 1e-5, f"SoftSort P error {err_p_soft:.2e} exceeds 1e-5"
    assert err_sorted < 1e-5, f"SoftSort sorted approx error {err_sorted:.2e} exceeds 1e-5"

    # 3. NeuralSort Parity
    pred_p_neural = neural_sort_forward(tensors["neuralsort_scores"], tau=0.5)
    err_p_neural = (pred_p_neural - tensors["expected_p_neural"]).abs().max().item()
    assert err_p_neural < 1e-5, f"NeuralSort P error {err_p_neural:.2e} exceeds 1e-5"

    # 4. Diff Top-k Parity
    pred_topk = diff_topk_forward(tensors["topk_scores"], k=2, tau=0.2, num_bisection=15)
    err_topk = (pred_topk - tensors["expected_topk_weights"]).abs().max().item()
    assert err_topk < 1e-5, f"Diff Top-k error {err_topk:.2e} exceeds 1e-5"

    # 5. Soft Histogram Parity
    pred_hist = soft_histogram_forward(tensors["hist_x"], tensors["bin_centers"], bin_width=1.0, tau=0.1)
    err_hist = (pred_hist - tensors["expected_histogram"]).abs().max().item()
    assert err_hist < 1e-5, f"Soft histogram error {err_hist:.2e} exceeds 1e-5"

    # 6. Soft Quantile Parity
    pred_median = soft_quantile_forward(tensors["quantile_scores"], quantile=0.5, tau=0.2)
    err_median = (pred_median - tensors["expected_median"]).abs().max().item()
    assert err_median < 1e-5, f"Soft quantile error {err_median:.2e} exceeds 1e-5"

    print(
        f"All 6 operations verified! Anchor: {err_linear:.2e}, SoftSort: {err_p_soft:.2e}/{err_sorted:.2e}, "
        f"NeuralSort: {err_p_neural:.2e}, TopK: {err_topk:.2e}, Histogram: {err_hist:.2e}, Median: {err_median:.2e}"
    )


if __name__ == "__main__":
    verify_topic_13_pytorch()
