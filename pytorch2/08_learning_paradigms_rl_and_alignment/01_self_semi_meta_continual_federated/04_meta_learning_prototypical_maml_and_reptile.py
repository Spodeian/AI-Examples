"""Meta Learning Prototypical MAML and Reptile — PyTorch 2.x reference implementation.

Topic: 08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile
Golden weights: weights/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile.safetensors

``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile.py`` verifies against the golden checkpoint;
``python pytorch2/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/04_meta_learning_prototypical_maml_and_reptile"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def protonet_prototypes(support_embeds: torch.Tensor) -> torch.Tensor:
    """Computes class prototypes as mean support embeddings: [N, K, D] -> [N, D]."""
    return support_embeds.mean(dim=1)


def protonet_logits(query_embeds: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Computes negative squared Euclidean distances between queries and prototypes: [Q, N]."""
    return -(torch.cdist(query_embeds, prototypes, p=2) ** 2)


def maml_inner_linear_step(
    x_s: torch.Tensor,
    y_s: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step inner gradient adaptation for linear regression under MSE loss."""
    pred_s = x_s @ w + b
    err = pred_s - y_s
    n_elements = err.numel()
    grad_w = (2.0 / n_elements) * (x_s.T @ err)
    grad_b = (2.0 / n_elements) * err.sum(dim=0)
    w_adapted = w - alpha * grad_w
    b_adapted = b - alpha * grad_b
    return w_adapted, b_adapted


def reptile_outer_step(
    theta: torch.Tensor,
    theta_adapted: torch.Tensor,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Reptile outer meta-update: theta + epsilon * (theta_adapted - theta)."""
    return theta + epsilon * (theta_adapted - theta)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Linear anchor
    test_input = torch.randn(4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    test_output = linear_anchor(test_input, weight, bias)

    # 2. ProtoNet
    support_embeds = torch.randn(3, 4, 8)
    query_embeds = torch.randn(6, 8)
    prototypes = protonet_prototypes(support_embeds)
    protonet_logits_ref = protonet_logits(query_embeds, prototypes)

    # 3. MAML
    x_s = torch.randn(4, 8)
    y_s = torch.randn(4, 2)
    w_maml = torch.randn(8, 2)
    b_maml = torch.randn(2)
    w_adapted_ref, b_adapted_ref = maml_inner_linear_step(x_s, y_s, w_maml, b_maml, alpha=0.05)

    # 4. Reptile
    theta_base = torch.randn(8)
    theta_adapted = torch.randn(8)
    theta_new_ref = reptile_outer_step(theta_base, theta_adapted, epsilon=0.1)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "support_embeds": support_embeds,
        "query_embeds": query_embeds,
        "prototypes_ref": prototypes,
        "protonet_logits_ref": protonet_logits_ref,
        "x_s": x_s,
        "y_s": y_s,
        "w_maml": w_maml,
        "b_maml": b_maml,
        "w_adapted_ref": w_adapted_ref,
        "b_adapted_ref": b_adapted_ref,
        "theta_base": theta_base,
        "theta_adapted": theta_adapted,
        "theta_new_ref": theta_new_ref,
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
