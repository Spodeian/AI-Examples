"""Generate deterministic reference safetensors checkpoint for Category 03 Topic 13:
13_differentiable_sorting_ranking_topk_permutahedron_and_imle
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
from safetensors.torch import save_file

import core


def compute_soft_sort(scores: torch.Tensor, tau: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    # scores: [B, K]
    # SoftSort (Prillo & Eisenschlos 2020):
    # s_sorted: sort descending
    s_sorted, _ = torch.sort(scores, descending=True, dim=-1) # [B, K]
    # Pairwise L1 distance: |s_sorted[:, :, None] - scores[:, None, :]| -> [B, K, K]
    dist = torch.abs(s_sorted.unsqueeze(-1) - scores.unsqueeze(-2))
    # Soft permutation matrix: softmax(-dist / tau, dim=-1)
    p_soft = torch.softmax(-dist / tau, dim=-1) # [B, K, K]
    # Reconstructed sorted values: P_soft @ scores
    sorted_approx = torch.matmul(p_soft, scores.unsqueeze(-1)).squeeze(-1) # [B, K]
    return p_soft, sorted_approx


def compute_neural_sort(scores: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    # NeuralSort (Grover et al. 2019):
    # scores: [B, K]
    B, K = scores.shape
    # diff: scores[:, :, None] - scores[:, None, :] -> [B, K, K]
    diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    l1_sum = torch.sum(torch.abs(diff), dim=-1) # [B, K], sum_k |s_i - s_k|
    
    # Weights vector: (K + 1 - 2j) for j = 1..K
    j_vec = torch.arange(1, K + 1, device=scores.device, dtype=scores.dtype)
    weights = (K + 1.0 - 2.0 * j_vec).unsqueeze(0).unsqueeze(1) # [1, 1, K]
    
    # A_ij = (K + 1 - 2j) s_i - sum_k |s_i - s_k|
    # A has shape [B, K, K] where row i is item i, col j is rank position j
    # But usually NeuralSort defines row j as rank position j, col i as item i:
    # C_j = (K + 1 - 2j) s - sum_k |s - s_k|
    s_expanded = scores.unsqueeze(-1) # [B, K, 1]
    l1_expanded = l1_sum.unsqueeze(-1) # [B, K, 1]
    
    # Score for item i at position j: (K + 1 - 2j) * s_i - l1_sum_i
    cost = weights * s_expanded - l1_expanded # [B, K, K], item i, rank j
    # Transpose so row j is rank position j, cols are items:
    cost = cost.permute(0, 2, 1) # [B, K, K]
    p_neural = torch.softmax(cost / tau, dim=-1)
    return p_neural


def compute_diff_topk(scores: torch.Tensor, k: int = 2, tau: float = 0.2, num_bisection: int = 15) -> torch.Tensor:
    # Entropic Top-k via sigmoid root-finding:
    # Find threshold lambda per batch such that sum_i sigmoid((s_i - lambda) / tau) == k
    B, N = scores.shape
    # Binary search bounds for lambda
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


def compute_soft_histogram(
    x: torch.Tensor,
    bin_centers: torch.Tensor,
    bin_width: float = 1.0,
    tau: float = 0.1,
) -> torch.Tensor:
    # x: [B, N], bin_centers: [M]
    # Soft bin indicator: sigmoid((x - (c - w/2))/tau) - sigmoid((x - (c + w/2))/tau)
    # x: [B, N, 1], centers: [1, 1, M]
    B, N = x.shape
    M = bin_centers.shape[0]
    x_exp = x.unsqueeze(-1) # [B, N, 1]
    c_exp = bin_centers.view(1, 1, M) # [1, 1, M]
    
    lo = c_exp - 0.5 * bin_width
    hi = c_exp + 0.5 * bin_width
    
    in_bin = torch.sigmoid((x_exp - lo) / tau) - torch.sigmoid((x_exp - hi) / tau)
    # Sum over N samples in batch: [B, M]
    hist = torch.sum(in_bin, dim=1)
    # Normalize per batch to sum to 1.0
    return hist / (torch.sum(hist, dim=-1, keepdim=True) + 1e-12)


def compute_soft_quantile(
    scores: torch.Tensor,
    quantile: float = 0.5,
    tau: float = 0.2,
) -> torch.Tensor:
    # Continuous quantile estimation:
    # Target rank: q * (N - 1) + 1
    # Weight per item i is based on proximity of soft rank R_i to target rank
    B, N = scores.shape
    target_rank = quantile * float(N - 1) + 1.0
    # Soft ranks (ascending): R_i = 0.5 + sum_j sigmoid((s_i - s_j) / tau)
    diff = (scores.unsqueeze(2) - scores.unsqueeze(1)) / tau
    ranks = 0.5 + torch.sum(torch.sigmoid(diff), dim=-1) # [B, N]
    # Soft weights centered at target rank: softmax(- (R_i - target_rank)^2 / tau)
    dist_sq = (ranks - target_rank).pow(2)
    weights = torch.softmax(-dist_sq / (2.0 * tau), dim=-1)
    return torch.sum(weights * scores, dim=-1)


def generate_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    dim = 4

    # 1. Linear Anchor
    weight = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(dim, generator=gen, dtype=torch.float32)
    test_input = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    test_output = test_input @ weight + bias

    # 2. SoftSort
    softsort_scores = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    expected_p_soft, expected_sorted_approx = compute_soft_sort(softsort_scores, tau=0.2)

    # 3. NeuralSort
    neuralsort_scores = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    expected_p_neural = compute_neural_sort(neuralsort_scores, tau=0.5)

    # 4. Differentiable Top-k
    topk_scores = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    expected_topk_weights = compute_diff_topk(topk_scores, k=2, tau=0.2, num_bisection=15)

    # 5. Soft Histogram
    hist_x = torch.randn(batch_size, 16, generator=gen, dtype=torch.float32) # [8, 16]
    bin_centers = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32) # 4 bins
    expected_histogram = compute_soft_histogram(hist_x, bin_centers, bin_width=1.0, tau=0.1)

    # 6. Soft Quantile (Median q=0.5)
    quantile_scores = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    expected_median = compute_soft_quantile(quantile_scores, quantile=0.5, tau=0.2)

    tensors = {
        "weight": weight,
        "bias": bias,
        "test_input": test_input,
        "test_output": test_output,
        # SoftSort
        "softsort_scores": softsort_scores,
        "expected_p_soft": expected_p_soft,
        "expected_sorted_approx": expected_sorted_approx,
        # NeuralSort
        "neuralsort_scores": neuralsort_scores,
        "expected_p_neural": expected_p_neural,
        # Diff Top-k
        "topk_scores": topk_scores,
        "expected_topk_weights": expected_topk_weights,
        # Soft Histogram
        "hist_x": hist_x,
        "bin_centers": bin_centers,
        "expected_histogram": expected_histogram,
        # Soft Quantile
        "quantile_scores": quantile_scores,
        "expected_median": expected_median,
    }

    out_dir = Path("weights/03_building_blocks_and_losses")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "13_differentiable_sorting_ranking_topk_permutahedron_and_imle.safetensors"
    save_file(tensors, out_path)
    print(f"Saved {len(tensors)} tensors to {out_path}")


if __name__ == "__main__":
    generate_weights()
