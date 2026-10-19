"""Generate reference weights and golden parity outputs for Category 04 Topic 14:
Minimax Saddle Point, Extragradient, OGDA, SGA, and Trust-Region BOBYQA.
"""

from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import save_file


def step_extragradient(
    x: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Korpelevich Extragradient step on bilinear game min_x max_y x^T A y."""
    grad_x = torch.matmul(A, y)
    grad_y = torch.matmul(A.T, x)
    x_tilde = x - eta_x * grad_x
    y_tilde = y + eta_y * grad_y

    grad_x_tilde = torch.matmul(A, y_tilde)
    grad_y_tilde = torch.matmul(A.T, x_tilde)
    x_next = x - eta_x * grad_x_tilde
    y_next = y + eta_y * grad_y_tilde
    return x_next, y_next


def step_ogda(
    x: torch.Tensor,
    y: torch.Tensor,
    x_prev: torch.Tensor,
    y_prev: torch.Tensor,
    A: torch.Tensor,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimistic Gradient Descent-Ascent (OGDA) step."""
    grad_x = torch.matmul(A, y)
    grad_y = torch.matmul(A.T, x)
    grad_x_prev = torch.matmul(A, y_prev)
    grad_y_prev = torch.matmul(A.T, x_prev)

    x_next = x - 2.0 * eta_x * grad_x + eta_x * grad_x_prev
    y_next = y + 2.0 * eta_y * grad_y - eta_y * grad_y_prev
    return x_next, y_next


def step_sga(
    x: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
    lambda_sga: float = 0.5,
    eta_x: float = 0.1,
    eta_y: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symplectic Gradient Adjustment (SGA) step."""
    AAT_x = torch.matmul(A, torch.matmul(A.T, x))
    ATA_y = torch.matmul(A.T, torch.matmul(A, y))
    grad_x = torch.matmul(A, y)
    grad_y = torch.matmul(A.T, x)

    update_x = grad_x + lambda_sga * AAT_x
    update_y = grad_y - lambda_sga * ATA_y
    x_next = x - eta_x * update_x
    y_next = y + eta_y * update_y
    return x_next, y_next


def nelder_mead_reflect(
    x_best: torch.Tensor,
    x_second: torch.Tensor,
    x_worst: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Nelder-Mead reflection point calculation."""
    centroid = 0.5 * (x_best + x_second)
    return centroid + alpha * (centroid - x_worst)


def main():
    torch.manual_seed(42)

    # 1. Bilinear game matrix and state
    A = torch.tensor([
        [0.8, -0.4],
        [0.3, 0.9],
    ], dtype=torch.float32)
    x = torch.tensor([1.5, -1.0], dtype=torch.float32)
    y = torch.tensor([-0.8, 1.2], dtype=torch.float32)
    x_prev = torch.tensor([1.4, -0.9], dtype=torch.float32)
    y_prev = torch.tensor([-0.7, 1.1], dtype=torch.float32)

    # Invariant 1: Extragradient
    gold_x_eg, gold_y_eg = step_extragradient(x, y, A, eta_x=0.1, eta_y=0.1)

    # Invariant 2: OGDA
    gold_x_ogda, gold_y_ogda = step_ogda(x, y, x_prev, y_prev, A, eta_x=0.1, eta_y=0.1)

    # Invariant 3: SGA
    gold_x_sga, gold_y_sga = step_sga(x, y, A, lambda_sga=0.5, eta_x=0.1, eta_y=0.1)

    # Invariant 4: Nelder-Mead Reflection
    x_best = torch.tensor([0.2, 0.4], dtype=torch.float32)
    x_second = torch.tensor([0.5, 0.6], dtype=torch.float32)
    x_worst = torch.tensor([1.2, 1.4], dtype=torch.float32)
    gold_x_reflect = nelder_mead_reflect(x_best, x_second, x_worst, alpha=1.0)

    # Invariant 5: Linear Anchor for Cross-Track Parity
    anchor_x = torch.randn(2, 8, dtype=torch.float32)
    anchor_w = torch.randn(8, 4, dtype=torch.float32)
    anchor_b = torch.randn(4, dtype=torch.float32)
    golden_anchor_out = torch.matmul(anchor_x, anchor_w) + anchor_b

    tensors = {
        # Bilinear game tensors
        "matrix_A": A,
        "state_x": x,
        "state_y": y,
        "state_x_prev": x_prev,
        "state_y_prev": y_prev,
        "golden_x_eg": gold_x_eg,
        "golden_y_eg": gold_y_eg,
        "golden_x_ogda": gold_x_ogda,
        "golden_y_ogda": gold_y_ogda,
        "golden_x_sga": gold_x_sga,
        "golden_y_sga": gold_y_sga,
        # Nelder-Mead
        "x_best": x_best,
        "x_second": x_second,
        "x_worst": x_worst,
        "golden_x_reflect": gold_x_reflect,
        # Linear Anchor
        "anchor_x": anchor_x,
        "anchor_w": anchor_w,
        "anchor_b": anchor_b,
        "golden_anchor_out": golden_anchor_out,
        # Standard Parity Keys for test_cross_track_parity.py
        "weight": anchor_w.clone(),
        "bias": anchor_b.clone(),
        "test_input": anchor_x.clone(),
        "test_output": golden_anchor_out.clone(),
    }

    out_dir = Path("weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {len(tensors)} tensors to {out_path}")


if __name__ == "__main__":
    main()
