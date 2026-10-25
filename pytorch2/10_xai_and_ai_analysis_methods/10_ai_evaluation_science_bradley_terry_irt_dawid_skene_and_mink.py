"""AI Evaluation Science Bradley Terry IRT Dawid Skene and Min-K% — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink
Golden weights: weights/10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/10_ai_evaluation_science_bradley_terry_irt_dawid_skene_and_mink"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def btl_win_probability(skills: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry-Luce pairwise win probability matrix P_{ij} = sigma(s_i - s_j) (Bradley & Terry, 1952)."""
    diff = skills.unsqueeze(1) - skills.unsqueeze(0)
    return torch.sigmoid(diff)


def irt_2pl_probability(
    abilities: torch.Tensor,
    difficulties: torch.Tensor,
    discriminations: torch.Tensor,
) -> torch.Tensor:
    """Two-Parameter Logistic Item Response Theory (2PL IRT) P(Y_{mi}=1) = sigma(a_i * (theta_m - b_i)) (Lord, 1980)."""
    diff = abilities.unsqueeze(1) - difficulties.unsqueeze(0)
    return torch.sigmoid(discriminations.unsqueeze(0) * diff)


def dawid_skene_e_step(
    annotations: torch.Tensor,
    class_priors: torch.Tensor,
    confusion_matrices: torch.Tensor,
) -> torch.Tensor:
    """Dawid-Skene annotator consensus expectation step (Dawid & Skene, 1979).
    
    annotations: [N, J] float containing class indices in {0, ..., K-1}.
    class_priors: [K].
    confusion_matrices: [J, K, K] where [j, k, l] is P(annotator j output l | true class k).
    Returns posterior: [N, K].
    """
    n, j_count = annotations.shape
    k_count = class_priors.shape[0]

    log_post = torch.zeros(n, k_count, dtype=annotations.dtype, device=annotations.device)
    for i in range(n):
        for k in range(k_count):
            lp = torch.log(class_priors[k] + 1e-8)
            for j in range(j_count):
                obs = int(annotations[i, j].item())
                lp = lp + torch.log(confusion_matrices[j, k, obs] + 1e-8)
            log_post[i, k] = lp

    return torch.softmax(log_post, dim=-1)


def min_k_percent_prob(token_log_probs: torch.Tensor, k_fraction: float = 0.25) -> torch.Tensor:
    """Min-K% Prob pretraining data / memorization detector (Shi et al., 2023)."""
    sorted_lp, _ = torch.sort(token_log_probs)
    k_count = max(1, int(round(k_fraction * token_log_probs.shape[0])))
    return sorted_lp[:k_count].mean()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. BTL setup: M=4 models
    btl_skills = torch.tensor([1.2, 0.5, -0.2, -1.5], dtype=torch.float32)
    btl_win_prob_ref = btl_win_probability(btl_skills)

    # 2. 2PL IRT setup: M=3 models, I=4 evaluation items
    irt_abilities = torch.tensor([1.5, 0.0, -1.5], dtype=torch.float32)
    irt_difficulties = torch.tensor([-1.0, 0.0, 0.5, 2.0], dtype=torch.float32)
    irt_discriminations = torch.tensor([1.2, 0.8, 1.5, 2.0], dtype=torch.float32)
    irt_prob_ref = irt_2pl_probability(irt_abilities, irt_difficulties, irt_discriminations)

    # 3. Dawid-Skene setup: N=4 instances, J=3 judges/annotators, K=2 binary classes
    ds_annotations = torch.tensor([
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ], dtype=torch.float32)
    ds_priors = torch.tensor([0.5, 0.5], dtype=torch.float32)
    ds_confusion = torch.tensor([
        [[0.85, 0.15], [0.10, 0.90]],
        [[0.75, 0.25], [0.20, 0.80]],
        [[0.90, 0.10], [0.15, 0.85]],
    ], dtype=torch.float32)
    ds_posterior_ref = dawid_skene_e_step(ds_annotations, ds_priors, ds_confusion)

    # 4. Min-K% Prob setup: T=8 tokens, k=0.25 (2 lowest)
    mink_log_probs = torch.tensor([-0.12, -0.45, -2.85, -0.22, -1.95, -0.05, -3.40, -0.35], dtype=torch.float32)
    mink_score_ref = min_k_percent_prob(mink_log_probs, k_fraction=0.25)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "btl_skills": btl_skills,
        "btl_win_prob_ref": btl_win_prob_ref,
        "irt_abilities": irt_abilities,
        "irt_difficulties": irt_difficulties,
        "irt_discriminations": irt_discriminations,
        "irt_prob_ref": irt_prob_ref,
        "ds_annotations": ds_annotations,
        "ds_priors": ds_priors,
        "ds_confusion": ds_confusion,
        "ds_posterior_ref": ds_posterior_ref,
        "mink_log_probs": mink_log_probs,
        "mink_score_ref": mink_score_ref,
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
