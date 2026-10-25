"""Statistical Model Comparison Bayesian RoPE Friedman and Bootstrap — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap
Golden weights: weights/10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/11_statistical_model_comparison_bayesian_rope_friedman_and_bootstrap"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def mcnemar_test_statistic(b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """McNemar's test with continuity correction (Dietterich, 1998).
    
    chi^2 = (|b - c| - 1)^2 / (b + c + eps)
    """
    diff = torch.abs(b - c) - 1.0
    return (diff * diff) / (b + c + 1e-8)


def nadeau_bengio_corrected_t(
    diffs: torch.Tensor,
    n_train: float,
    n_test: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nadeau-Bengio corrected resampled t-test for cross-validation (Nadeau & Bengio, 2003).
    
    c = 1/K + n_test / n_train
    sigma = sqrt(c * S^2)
    t = d_bar / sigma
    """
    k = float(diffs.shape[0])
    d_bar = diffs.mean()
    s_sq = torch.sum((diffs - d_bar) ** 2) / (k - 1.0)
    c_factor = 1.0 / k + float(n_test) / float(n_train)
    sigma = torch.sqrt(c_factor * s_sq + 1e-8)
    t_stat = d_bar / sigma
    return t_stat, sigma


def friedman_rank_statistic(ranks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Friedman non-parametric multi-classifier comparison test (Demšar, 2006).
    
    ranks: [N, K] matrix of algorithm ranks across N datasets.
    Returns: (chi_F^2, avg_ranks)
    """
    n = float(ranks.shape[0])
    k = float(ranks.shape[1])
    avg_ranks = ranks.mean(dim=0)
    sum_sq = torch.sum(avg_ranks * avg_ranks)
    chi_f = (12.0 * n / (k * (k + 1.0))) * (sum_sq - (k * (k + 1.0) ** 2) / 4.0)
    return chi_f, avg_ranks


def bayesian_rope_probabilities(
    diffs: torch.Tensor,
    n_train: float,
    n_test: float,
    rope_radius: float = 0.01,
) -> torch.Tensor:
    """Bayesian correlated t-test with Region of Practical Equivalence (ROPE) (Benavoli et al., 2017).
    
    Returns: [P(M1 > M2), P(ROPE), P(M2 > M1)]
    """
    k = float(diffs.shape[0])
    d_bar = diffs.mean()
    s_sq = torch.sum((diffs - d_bar) ** 2) / (k - 1.0)
    c_factor = 1.0 / k + float(n_test) / float(n_train)
    sigma = torch.sqrt(c_factor * s_sq + 1e-8)

    r = float(rope_radius)
    z1 = (r - d_bar) / sigma
    z2 = (-r - d_bar) / sigma

    # Standard normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    sqrt2 = 1.4142135623730951
    phi1 = 0.5 * (1.0 + torch.erf(z1 / sqrt2))
    phi2 = 0.5 * (1.0 + torch.erf(z2 / sqrt2))

    p_rope = phi1 - phi2
    p_better = 1.0 - phi1
    p_worse = phi2

    return torch.stack([p_better, p_rope, p_worse])


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. McNemar setup: b=24 discordants (M1 ok, M2 err), c=9 discordants (M1 err, M2 ok)
    mcnemar_b = torch.tensor(24.0, dtype=torch.float32)
    mcnemar_c = torch.tensor(9.0, dtype=torch.float32)
    mcnemar_stat_ref = mcnemar_test_statistic(mcnemar_b, mcnemar_c)

    # 2. Nadeau-Bengio corrected t-test: K=5 CV folds
    cv_diffs = torch.tensor([0.025, 0.018, 0.032, 0.015, 0.020], dtype=torch.float32)
    nb_t_ref, nb_sigma_ref = nadeau_bengio_corrected_t(cv_diffs, n_train=800.0, n_test=200.0)

    # 3. Friedman test: N=4 benchmarks, K=3 algorithms
    ranks = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 2.0],
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 3.0],
    ], dtype=torch.float32)
    friedman_stat_ref, avg_ranks_ref = friedman_rank_statistic(ranks)

    # 4. Bayesian ROPE probabilities
    rope_probs_ref = bayesian_rope_probabilities(cv_diffs, n_train=800.0, n_test=200.0, rope_radius=0.01)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "mcnemar_b": mcnemar_b,
        "mcnemar_c": mcnemar_c,
        "mcnemar_stat_ref": mcnemar_stat_ref,
        "cv_diffs": cv_diffs,
        "nb_t_ref": nb_t_ref,
        "nb_sigma_ref": nb_sigma_ref,
        "ranks": ranks,
        "friedman_stat_ref": friedman_stat_ref,
        "avg_ranks_ref": avg_ranks_ref,
        "rope_probs_ref": rope_probs_ref,
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
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
