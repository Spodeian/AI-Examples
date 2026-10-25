"""Quality Diversity Map Elites and Differentiable Qd — PyTorch 2.x reference implementation.

Topic: 09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd
Golden weights: weights/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.safetensors

``python pytorch2/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.py`` verifies against the golden checkpoint;
``python pytorch2/09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "09_causal_ai_and_evolutionary_search/05_quality_diversity_map_elites_and_differentiable_qd"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def map_elites_grid_indices(
    descriptors: torch.Tensor,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    grid_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map continuous behavioral descriptors to discrete multi-index and flattened 1D cell index.
    
    idx_k = clamp(floor((b_k - min_k) / (max_k - min_k) * B_k), 0, B_k - 1)
    cell_idx = idx_0 * B_1 + idx_1
    """
    scale = (descriptors - bounds_min.unsqueeze(0)) / (bounds_max.unsqueeze(0) - bounds_min.unsqueeze(0) + 1e-8)
    indices = torch.floor(scale * grid_size.unsqueeze(0)).long()
    indices = torch.clamp(indices, torch.zeros_like(grid_size.long()), (grid_size - 1).long())

    # For 2D grid: idx_0 * B_1 + idx_1
    cell_idx = indices[:, 0] * grid_size[1].long() + indices[:, 1]
    return indices.float(), cell_idx.float()


def map_elites_archive_update(
    archive_fitness: torch.Tensor,
    candidate_fitness: torch.Tensor,
    cell_indices: torch.Tensor,
) -> torch.Tensor:
    """Update 1D flattened archive with candidate fitness if greater than current occupant.
    
    archive[c] = max(archive[c], f_i)
    """
    updated = archive_fitness.clone()
    for i in range(candidate_fitness.shape[0]):
        c = int(cell_indices[i].item())
        if candidate_fitness[i] > updated[c]:
            updated[c] = candidate_fitness[i]
    return updated


def dqd_gradient_arborescence_step(
    grad_f: torch.Tensor,
    jac_b: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Quality Diversity (DQD / OMG-MEGA) multi-objective directional step.
    
    Delta theta = c_0 * grad_f + sum_{k=1}^K c_k * grad_b_k
    """
    c0 = coeffs[0]
    c_b = coeffs[1:].unsqueeze(-1)  # [K, 1]
    step = c0 * grad_f + torch.sum(c_b * jac_b, dim=0)
    return step


def dqd_orthogonalize_behavior_gradients(
    grad_f: torch.Tensor,
    jac_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Gram-Schmidt project behavioral gradients orthogonal to objective gradient.
    
    g_k^perp = g_k - (g_k . g_f) / (||g_f||^2 + eps) * g_f
    """
    norm_sq = torch.sum(grad_f * grad_f) + eps
    proj_coeffs = (jac_b @ grad_f) / norm_sq  # [K]
    ortho_jac = jac_b - proj_coeffs.unsqueeze(-1) * grad_f.unsqueeze(0)
    return ortho_jac


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 1. MAP-Elites grid setup (2D descriptor space, bounds [0, 1] x [0, 1], 4x4 grid)
    b_min = torch.tensor([0.0, 0.0], dtype=torch.float32)
    b_max = torch.tensor([1.0, 1.0], dtype=torch.float32)
    grid_size = torch.tensor([4.0, 4.0], dtype=torch.float32)

    descriptors = torch.tensor([
        [0.15, 0.22],
        [0.85, 0.12],
        [0.45, 0.78],
        [0.92, 0.95],
        [0.18, 0.28],  # Same cell (0, 0) as first individual
        [0.62, 0.35],
        [0.05, 0.88],
        [0.72, 0.65],
    ], dtype=torch.float32)

    grid_indices_ref, cell_indices_ref = map_elites_grid_indices(descriptors, b_min, b_max, grid_size)

    # Archive fitness: 16 cells (4x4), initialized to -1e4 (empty)
    archive_fit = torch.full((16,), -10000.0, dtype=torch.float32)
    archive_fit[0] = 5.0  # Cell 0 already has fitness 5.0
    archive_fit[5] = 12.0 # Cell 5 has 12.0

    cand_fitness = torch.tensor([4.0, 15.0, 8.5, 20.0, 7.5, 11.0, 3.2, 9.8], dtype=torch.float32)
    # Notice candidate 0 has f=4.0 for cell 0 (won't overwrite 5.0), but candidate 4 has f=7.5 for cell 0 (will overwrite to 7.5)
    archive_fit_updated_ref = map_elites_archive_update(archive_fit, cand_fitness, cell_indices_ref)

    # 2. Differentiable Quality Diversity (DQD) tensors
    # Parameter dimension D=4, K=2 behavioral descriptors
    grad_f = torch.tensor([0.8, -1.2, 0.5, 0.1], dtype=torch.float32)
    jac_b = torch.tensor([
        [0.2, 0.9, -0.4, 0.6],
        [-0.5, 0.3, 0.8, -0.2],
    ], dtype=torch.float32)
    dqd_coeffs = torch.tensor([1.0, 0.5, -0.3], dtype=torch.float32)

    dqd_step_ref = dqd_gradient_arborescence_step(grad_f, jac_b, dqd_coeffs)
    dqd_ortho_jac_ref = dqd_orthogonalize_behavior_gradients(grad_f, jac_b)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "b_min": b_min,
        "b_max": b_max,
        "grid_size": grid_size,
        "descriptors": descriptors,
        "grid_indices_ref": grid_indices_ref,
        "cell_indices_ref": cell_indices_ref,
        "archive_fit": archive_fit,
        "cand_fitness": cand_fitness,
        "archive_fit_updated_ref": archive_fit_updated_ref,
        "grad_f": grad_f,
        "jac_b": jac_b,
        "dqd_coeffs": dqd_coeffs,
        "dqd_step_ref": dqd_step_ref,
        "dqd_ortho_jac_ref": dqd_ortho_jac_ref,
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
