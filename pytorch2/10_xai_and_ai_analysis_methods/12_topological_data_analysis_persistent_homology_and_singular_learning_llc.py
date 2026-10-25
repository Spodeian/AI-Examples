"""Topological Data Analysis Persistent Homology and Singular Learning LLC — PyTorch 2.x reference implementation.

Topic: 10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc
Golden weights: weights/10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc.safetensors

``python pytorch2/10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc.py`` verifies against the golden checkpoint;
``python pytorch2/10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "10_xai_and_ai_analysis_methods/12_topological_data_analysis_persistent_homology_and_singular_learning_llc"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def persistence_diagram_lifespans(birth_death_pairs: torch.Tensor) -> torch.Tensor:
    """Computes topological feature lifespans pers_i = death_i - birth_i (Edelsbrunner & Harer, 2008)."""
    return birth_death_pairs[:, 1] - birth_death_pairs[:, 0]


def persistence_landscape_level1(
    birth_death_pairs: torch.Tensor,
    eval_points: torch.Tensor,
) -> torch.Tensor:
    """Evaluates level-1 persistence landscape lambda_1(t) = max_i relu(min(t - b_i, d_i - t)) (Bubenik, 2015)."""
    t = eval_points.unsqueeze(0)  # [1, T]
    b = birth_death_pairs[:, 0:1]  # [M, 1]
    d = birth_death_pairs[:, 1:2]  # [M, 1]
    tri = torch.clamp(torch.minimum(t - b, d - t), min=0.0)  # [M, T]
    return torch.max(tri, dim=0).values


def singular_learning_llc_energy(
    sample_losses: torch.Tensor,
    loss_star: torch.Tensor,
    n_samples: float,
    beta_inv_temp: float = 1.0,
) -> torch.Tensor:
    """Singular Learning Theory Local Learning Coefficient (LLC) hat{lambda} via SGLD (Watanabe, 2009; Lau et al., 2023)."""
    mean_diff = torch.mean(sample_losses - loss_star)
    return (float(n_samples) * float(beta_inv_temp) / 2.0) * mean_diff


def betti_0_filtration_counts(
    pairwise_distances: torch.Tensor,
    epsilon_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Counts connected components (Betti-0 number beta_0) along a Vietoris-Rips filtration."""
    n = pairwise_distances.shape[0]
    e_count = epsilon_thresholds.shape[0]
    counts = torch.zeros(e_count, dtype=torch.float32, device=pairwise_distances.device)

    for idx in range(e_count):
        eps = float(epsilon_thresholds[idx].item())
        adj = pairwise_distances <= eps
        visited = [False] * n
        comp = 0
        for i in range(n):
            if not visited[i]:
                comp += 1
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    for nbr in range(n):
                        if bool(adj[curr, nbr].item()) and not visited[nbr]:
                            visited[nbr] = True
                            queue.append(nbr)
        counts[idx] = float(comp)

    return counts


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. Persistence diagram & lifespans: M=4 topological features
    birth_death_pairs = torch.tensor([
        [0.0, 1.5],
        [0.2, 0.8],
        [0.5, 2.0],
        [1.0, 1.2],
    ], dtype=torch.float32)
    lifespans_ref = persistence_diagram_lifespans(birth_death_pairs)

    # 2. Persistence landscape level 1: T=5 grid evaluation points
    eval_points = torch.tensor([0.2, 0.6, 1.0, 1.4, 1.8], dtype=torch.float32)
    landscape_ref = persistence_landscape_level1(birth_death_pairs, eval_points)

    # 3. Singular Learning Theory LLC: S=6 SGLD loss samples
    sample_losses = torch.tensor([0.155, 0.162, 0.148, 0.159, 0.165, 0.151], dtype=torch.float32)
    loss_star = torch.tensor(0.140, dtype=torch.float32)
    llc_ref = singular_learning_llc_energy(sample_losses, loss_star, n_samples=500.0, beta_inv_temp=1.0)

    # 4. Vietoris-Rips filtration Betti-0: N=4 point cloud distances, E=4 epsilon thresholds
    pairwise_distances = torch.tensor([
        [0.0, 0.5, 1.2, 2.0],
        [0.5, 0.0, 0.8, 1.5],
        [1.2, 0.8, 0.0, 0.6],
        [2.0, 1.5, 0.6, 0.0],
    ], dtype=torch.float32)
    epsilon_thresholds = torch.tensor([0.2, 0.55, 0.9, 1.6], dtype=torch.float32)
    betti0_ref = betti_0_filtration_counts(pairwise_distances, epsilon_thresholds)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "birth_death_pairs": birth_death_pairs,
        "lifespans_ref": lifespans_ref,
        "eval_points": eval_points,
        "landscape_ref": landscape_ref,
        "sample_losses": sample_losses,
        "loss_star": loss_star,
        "llc_ref": llc_ref,
        "pairwise_distances": pairwise_distances,
        "epsilon_thresholds": epsilon_thresholds,
        "betti0_ref": betti0_ref,
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
