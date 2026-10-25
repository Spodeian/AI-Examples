"""Active Learning Bald Coreset and Badge — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge
Golden weights: weights/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/07_active_learning_bald_coreset_and_badge"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def bald_score(mc_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Bayesian Active Learning by Disagreement: H(mean(p)) - mean(H(p))."""
    mean_probs = mc_probs.mean(dim=0)
    entropy_mean = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=-1)
    mean_entropy = -(mc_probs * torch.log(mc_probs + eps)).sum(dim=-1).mean(dim=0)
    return entropy_mean - mean_entropy


def coreset_min_distances(unlabeled_feats: torch.Tensor, labeled_feats: torch.Tensor) -> torch.Tensor:
    """Core-Set k-Center Greedy min distance from each unlabeled point to labeled points."""
    dist_matrix = torch.cdist(unlabeled_feats, labeled_feats, p=2)
    min_dists, _ = dist_matrix.min(dim=1)
    return min_dists


def badge_gradient_embeddings(feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """BADGE gradient embedding: g_i = (p_i - e_yhat) (x) x_i in R^(C x D), flattened to R^(C*D)."""
    pred_class = probs.argmax(dim=-1)
    c = probs.shape[-1]
    one_hot = F.one_hot(pred_class, num_classes=c).float()
    prob_diff = probs - one_hot
    return (prob_diff.unsqueeze(2) * feats.unsqueeze(1)).flatten(start_dim=1)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Linear anchor
    test_input = torch.randn(4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    test_output = linear_anchor(test_input, weight, bias)

    # 2. BALD
    mc_logits = torch.randn(4, 4, 3)
    mc_probs = F.softmax(mc_logits, dim=-1)
    bald_score_ref = bald_score(mc_probs)

    # 3. Core-Set
    labeled_feats = torch.randn(2, 4)
    unlabeled_feats = torch.randn(4, 4)
    coreset_min_dists_ref = coreset_min_distances(unlabeled_feats, labeled_feats)
    best_unlabeled_idx_ref = coreset_min_dists_ref.argmax().unsqueeze(0)

    # 4. BADGE
    feats = torch.randn(4, 4)
    probs = F.softmax(torch.randn(4, 3), dim=-1)
    badge_embeds_ref = badge_gradient_embeddings(feats, probs)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "mc_probs": mc_probs,
        "bald_score_ref": bald_score_ref,
        "labeled_feats": labeled_feats,
        "unlabeled_feats": unlabeled_feats,
        "coreset_min_dists_ref": coreset_min_dists_ref,
        "best_unlabeled_idx_ref": best_unlabeled_idx_ref,
        "feats": feats,
        "probs": probs,
        "badge_embeds_ref": badge_embeds_ref,
    }


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
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
