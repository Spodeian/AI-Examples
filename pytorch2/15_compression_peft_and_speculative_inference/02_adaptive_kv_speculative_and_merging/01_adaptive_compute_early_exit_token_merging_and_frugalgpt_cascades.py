"""Adaptive Compute Early Exit Token Merging and Frugalgpt Cascades — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades
Golden weights: weights/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def early_exit_predict(
    hidden: torch.Tensor, head_weight: torch.Tensor, head_bias: torch.Tensor, threshold: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Early exit classification head evaluating confidence and exit decision.

    Args:
        hidden: Layer hidden representations of shape [batch, dim].
        head_weight: Exit classifier weight of shape [dim, classes].
        head_bias: Exit classifier bias of shape [classes].
        threshold: Confidence threshold in (0, 1].

    Returns:
        probs: Softmax class probabilities of shape [batch, classes].
        confidence: Maximum class probability per sample of shape [batch].
        exit_mask: Binary indicator of shape [batch] (1 = exit now, 0 = continue to next layer).
    """
    logits = hidden @ head_weight + head_bias
    probs = F.softmax(logits, dim=-1)
    confidence, _ = torch.max(probs, dim=-1)
    exit_mask = (confidence >= threshold).to(torch.float32)
    return probs, confidence, exit_mask


def tome_bipartite_merge(
    x: torch.Tensor, r_merge: int = 2, token_sizes: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token Merging (ToMe) bipartite soft matching.

    Partitions sequence into source set A (even tokens) and target set B (odd tokens),
    computes cosine similarity, and merges the top r_merge most similar pairs with weighted averaging.

    Args:
        x: Input token sequence of shape [batch, num_tokens, dim].
        r_merge: Number of tokens to merge per batch item.
        token_sizes: Size count vector tracking merged tokens, shape [batch, num_tokens].

    Returns:
        x_out: Reduced token sequence of shape [batch, num_tokens - r_merge, dim].
        token_sizes_out: Updated size counts of shape [batch, num_tokens - r_merge].
    """
    batch, n, dim = x.shape
    if token_sizes is None:
        token_sizes = torch.ones(batch, n, dtype=torch.float32, device=x.device)

    # Partition into A (even) and B (odd)
    idx_a = torch.arange(0, n, 2, device=x.device)
    idx_b = torch.arange(1, n, 2, device=x.device)
    a = x[:, idx_a, :]
    b = x[:, idx_b, :]
    sizes_a = token_sizes[:, idx_a]
    sizes_b = token_sizes[:, idx_b]

    # Normalized cosine similarity: [batch, len_a, len_b]
    a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-8)
    sim = a_norm @ b_norm.transpose(-1, -2)

    # For each a in A, find best match in B: best_b = argmax(sim)
    best_sim, best_b = torch.max(sim, dim=-1)  # [batch, len_a]

    # Select top r_merge nodes in A with highest similarity to their best B match
    _, top_a_sub = torch.topk(best_sim, r_merge, dim=-1, largest=True)  # [batch, r_merge]

    # Execute merge for each batch
    x_out_list = []
    sizes_out_list = []
    for b_idx in range(batch):
        cur_b = b[b_idx].clone()
        cur_sizes_b = sizes_b[b_idx].clone()
        a_unmerged = []
        sizes_a_unmerged = []

        merged_a_set = set(top_a_sub[b_idx].tolist())
        for a_idx in range(len(idx_a)):
            if a_idx in merged_a_set:
                target_b = best_b[b_idx, a_idx].item()
                # Weighted average merge into target_b
                s_a = sizes_a[b_idx, a_idx].item()
                s_b = cur_sizes_b[target_b].item()
                cur_b[target_b] = (s_b * cur_b[target_b] + s_a * a[b_idx, a_idx]) / (s_a + s_b)
                cur_sizes_b[target_b] = s_a + s_b
            else:
                a_unmerged.append(a[b_idx, a_idx])
                sizes_a_unmerged.append(sizes_a[b_idx, a_idx])

        if a_unmerged:
            combined_x = torch.cat([cur_b, torch.stack(a_unmerged)], dim=0)
            combined_sizes = torch.cat([cur_sizes_b, torch.tensor(sizes_a_unmerged, device=x.device)], dim=0)
        else:
            combined_x = cur_b
            combined_sizes = cur_sizes_b

        x_out_list.append(combined_x)
        sizes_out_list.append(combined_sizes)

    return torch.stack(x_out_list), torch.stack(sizes_out_list)


def frugal_cascade_route(
    confidences: torch.Tensor,
    costs: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FrugalGPT multi-tier model cascade router.

    Evaluates tiered predictions sequentially; if confidence >= threshold, accept and stop.

    Args:
        confidences: Model confidence scores of shape [batch, num_models].
        costs: Execution cost per model of shape [num_models].
        thresholds: Early-acceptance threshold per tier of shape [num_models].

    Returns:
        routed_model_idx: Selected model index per query of shape [batch].
        total_costs: Cost incurred per query of shape [batch].
    """
    batch, num_models = confidences.shape
    routed_idx = torch.full((batch,), num_models - 1, dtype=torch.long, device=confidences.device)
    query_cost = torch.zeros(batch, dtype=torch.float32, device=confidences.device)
    resolved = torch.zeros(batch, dtype=torch.bool, device=confidences.device)

    for m in range(num_models):
        active = ~resolved
        query_cost[active] += costs[m]
        # Check acceptance threshold
        if m < num_models - 1:
            accepted = active & (confidences[:, m] >= thresholds[m])
            routed_idx[accepted] = m
            resolved[accepted] = True
        else:
            routed_idx[active] = m
            resolved[active] = True

    return routed_idx, query_cost


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Early-Exit Classifier
    hidden = torch.randn(8, 16)
    head_w = torch.randn(16, 4) * 0.2
    head_b = torch.randn(4) * 0.1
    probs, conf, exit_mask = early_exit_predict(hidden, head_w, head_b, threshold=0.6)

    # 2. Token Merging (ToMe)
    # 4 batch items, 8 tokens per sequence, dim 16 -> merge 2 tokens to get 6 tokens
    x_tokens = torch.randn(4, 8, 16)
    init_sizes = torch.ones(4, 8)
    x_merged, merged_sizes = tome_bipartite_merge(x_tokens, r_merge=2, token_sizes=init_sizes)

    # 3. FrugalGPT Cascades (8 queries, 3 tiers: Small, Medium, Large)
    confidences = torch.tensor([
        [0.85, 0.90, 0.95],  # Easy -> exits tier 0
        [0.40, 0.88, 0.92],  # Medium -> exits tier 1
        [0.30, 0.50, 0.91],  # Hard -> exits tier 2
        [0.75, 0.85, 0.90],  # Easy -> exits tier 0
        [0.65, 0.75, 0.85],  # Medium -> exits tier 1
        [0.10, 0.20, 0.80],  # Hard -> exits tier 2
        [0.90, 0.95, 0.99],  # Easy -> exits tier 0
        [0.55, 0.60, 0.70],  # Hard -> exits tier 2
    ], dtype=torch.float32)
    costs = torch.tensor([1.0, 5.0, 20.0], dtype=torch.float32)
    thresholds = torch.tensor([0.70, 0.80, 0.0], dtype=torch.float32)
    routed_idx, total_cost = frugal_cascade_route(confidences, costs, thresholds)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "hidden": hidden,
        "head_w": head_w,
        "head_b": head_b,
        "probs": probs,
        "conf": conf,
        "exit_mask": exit_mask,
        "x_tokens": x_tokens,
        "init_sizes": init_sizes,
        "x_merged": x_merged,
        "merged_sizes": merged_sizes,
        "confidences": confidences,
        "costs": costs,
        "thresholds": thresholds,
        "routed_idx": routed_idx.to(torch.float32),
        "total_cost": total_cost,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-5, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
