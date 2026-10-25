"""Diloco Local SGD Powersgd and Distributed Checkpointing — PyTorch 2.x reference implementation.

Topic: 16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing
Golden weights: weights/16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing.safetensors

``python pytorch2/16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing.py`` verifies against the golden checkpoint;
``python pytorch2/16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def diloco_outer_step(
    theta_0: torch.Tensor,
    theta_workers: torch.Tensor,
    v_outer: torch.Tensor,
    lr_outer: float = 0.7,
    beta_outer: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DiLoCo outer optimization step across worker clusters.

    Args:
        theta_0: Initial global model parameters before H local steps [M, N].
        theta_workers: Final parameters of each worker after H local steps [W, M, N].
        v_outer: Running outer momentum tensor [M, N].
        lr_outer: Outer learning rate (typically 0.7).
        beta_outer: Outer momentum coefficient (typically 0.9).

    Returns:
        theta_new: Updated global parameters [M, N].
        v_new: Updated outer momentum buffer [M, N].
        delta_avg: Averaged pseudo-gradient across workers [M, N].
    """
    world_size = theta_workers.shape[0]
    # Worker pseudo-gradients: Delta^(w) = theta_0 - theta^(w)
    deltas = theta_0.unsqueeze(0) - theta_workers  # [W, M, N]
    delta_avg = torch.mean(deltas, dim=0)          # [M, N]

    v_new = beta_outer * v_outer + delta_avg
    theta_new = theta_0 - lr_outer * v_new
    return theta_new, v_new, delta_avg


def powersgd_compress_decompress(
    grad_matrix: torch.Tensor,
    probe_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PowerSGD rank-r gradient matrix compression and reconstruction.

    Args:
        grad_matrix: Dense gradient tensor [M, N].
        probe_q: Initial right-side probing matrix [N, r].

    Returns:
        recon_grad: Reconstructed low-rank gradient approximation [M, N].
        p_norm: Left-side factor normalized [M, r].
        q_new: Right-side factor updated [N, r].
    """
    # 1. Project onto left subspace: P = G @ Q
    p = grad_matrix @ probe_q  # [M, r]
    norm_p = torch.norm(p, p=2) + 1e-8
    p_norm = p / norm_p

    # 2. Project onto right subspace: Q_new = G^T @ P_norm
    q_new = grad_matrix.t() @ p_norm  # [N, r]

    # 3. Decompress: G_hat = P_norm @ Q_new^T
    recon_grad = p_norm @ q_new.t()
    return recon_grad, p_norm, q_new


def distributed_checkpoint_reshard(
    shards_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed checkpoint resharding from 2 shards to 4 shards.

    Args:
        shards_2: Checkpoint saved on 2 workers [2, 2, 4].

    Returns:
        full_tensor: Reconstructed full parameter tensor [4, 4].
        shards_4: Resharded checkpoint for 4 workers [4, 1, 4].
    """
    full_tensor = torch.cat([shards_2[0], shards_2[1]], dim=0)  # [4, 4]
    chunks = torch.chunk(full_tensor, 4, dim=0)
    shards_4 = torch.stack(chunks, dim=0)  # [4, 1, 4]
    return full_tensor, shards_4


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. DiLoCo setup
    theta_0 = torch.randn(4, 4)
    # Simulate 2 workers drifting during local training
    w0 = theta_0 - torch.randn(4, 4) * 0.05
    w1 = theta_0 - torch.randn(4, 4) * 0.05
    theta_workers = torch.stack([w0, w1], dim=0)
    v_outer_init = torch.zeros(4, 4)

    theta_new, v_outer_new, delta_avg = diloco_outer_step(
        theta_0, theta_workers, v_outer_init, lr_outer=0.7, beta_outer=0.9
    )

    # 2. PowerSGD setup
    grad_matrix = torch.randn(4, 4)
    probe_q = torch.randn(4, 1)  # rank 1
    recon_grad, p_norm, q_new = powersgd_compress_decompress(grad_matrix, probe_q)

    # 3. DCP Resharding setup
    shards_2 = torch.randn(2, 2, 4)
    full_tensor, shards_4 = distributed_checkpoint_reshard(shards_2)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "theta_0": theta_0,
        "theta_workers": theta_workers,
        "v_outer_init": v_outer_init,
        "theta_new": theta_new,
        "v_outer_new": v_outer_new,
        "delta_avg": delta_avg,
        "grad_matrix": grad_matrix,
        "probe_q": probe_q,
        "powersgd_recon": recon_grad,
        "powersgd_p": p_norm,
        "powersgd_q": q_new,
        "shards_2": shards_2,
        "shards_4": shards_4,
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
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
