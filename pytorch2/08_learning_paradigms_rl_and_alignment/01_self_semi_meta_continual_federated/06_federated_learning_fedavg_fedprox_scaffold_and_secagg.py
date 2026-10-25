"""Federated Learning Fedavg Fedprox Scaffold and Secagg — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg
Golden weights: weights/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/06_federated_learning_fedavg_fedprox_scaffold_and_secagg"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def fedavg_aggregate(client_weights: torch.Tensor, client_counts: torch.Tensor) -> torch.Tensor:
    """FedAvg weighted parameter aggregation: sum_k (n_k / N) * w_k."""
    weights_ratio = client_counts / client_counts.sum()
    return (weights_ratio.unsqueeze(1) * client_weights).sum(dim=0)


def fedprox_gradient_step(
    w_local: torch.Tensor,
    w_global: torch.Tensor,
    grad_local: torch.Tensor,
    eta: float = 0.05,
    mu: float = 0.1,
) -> torch.Tensor:
    """FedProx local update with proximal constraint: w_local - eta * (grad + mu * (w_local - w_global))."""
    prox_grad = grad_local + mu * (w_local - w_global)
    return w_local - eta * prox_grad


def scaffold_correct_gradient(
    grad_i: torch.Tensor,
    c_i: torch.Tensor,
    c_global: torch.Tensor,
) -> torch.Tensor:
    """SCAFFOLD client drift correction: grad_i - c_i + c_global."""
    return grad_i - c_i + c_global


def secagg_aggregate_sum(masked_models: torch.Tensor) -> torch.Tensor:
    """Secure Aggregation cancellation: pairwise masks cancel out upon summing over all clients."""
    return masked_models.sum(dim=0)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Linear anchor
    test_input = torch.randn(4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    test_output = linear_anchor(test_input, weight, bias)

    # 2. FedAvg
    client_weights = torch.randn(3, 8)
    client_counts = torch.tensor([100.0, 200.0, 300.0])
    fedavg_aggregated_ref = fedavg_aggregate(client_weights, client_counts)

    # 3. FedProx
    w_local = torch.randn(8)
    w_global = torch.randn(8)
    grad_local = torch.randn(8)
    w_prox_ref = fedprox_gradient_step(w_local, w_global, grad_local, eta=0.05, mu=0.1)

    # 4. SCAFFOLD
    grad_i = torch.randn(8)
    c_i = torch.randn(8)
    c_global = torch.randn(8)
    scaffold_corrected_ref = scaffold_correct_gradient(grad_i, c_i, c_global)

    # 5. SecAgg
    raw_models = torch.randn(3, 8)
    s12 = torch.randn(8)
    s13 = torch.randn(8)
    s23 = torch.randn(8)
    y1 = raw_models[0] + s12 + s13
    y2 = raw_models[1] - s12 + s23
    y3 = raw_models[2] - s13 - s23
    masked_models = torch.stack([y1, y2, y3], dim=0)
    secagg_unmasked_sum_ref = secagg_aggregate_sum(masked_models)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "client_weights": client_weights,
        "client_counts": client_counts,
        "fedavg_aggregated_ref": fedavg_aggregated_ref,
        "w_local": w_local,
        "w_global": w_global,
        "grad_local": grad_local,
        "w_prox_ref": w_prox_ref,
        "grad_i": grad_i,
        "c_i": c_i,
        "c_global": c_global,
        "scaffold_corrected_ref": scaffold_corrected_ref,
        "masked_models": masked_models,
        "secagg_unmasked_sum_ref": secagg_unmasked_sum_ref,
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
