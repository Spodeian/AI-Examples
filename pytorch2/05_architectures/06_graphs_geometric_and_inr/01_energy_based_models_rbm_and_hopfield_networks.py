"""Energy-Based Models, RBM, and Hopfield Networks — PyTorch 2.x reference implementation.

Topic: 05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks
Golden weights: weights/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.safetensors

``python pytorch2/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/06_graphs_geometric_and_inr/01_energy_based_models_rbm_and_hopfield_networks"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def rbm_free_energy(v: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes RBM free energy marginalizing hidden states: F(v) = -v @ a - sum_j softplus(v @ w + b).
    
    v has shape [B, V], w has shape [V, H], a has shape [V], b has shape [H].
    Returns free energy of shape [B, 1].
    """
    v_bias = (v @ a).unsqueeze(1)
    h_act = F.softplus(v @ w + b)
    h_term = torch.sum(h_act, dim=-1, keepdim=True)
    return -v_bias - h_term


def langevin_dynamics_step(
    x: torch.Tensor,
    grad_energy: torch.Tensor,
    step_size: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Simulates one step of overdamped Langevin dynamics: x_next = x - 0.5*eps*grad_E + sqrt(eps)*noise."""
    return x - 0.5 * step_size * grad_energy + math.sqrt(step_size) * noise


def modern_hopfield_energy(x: torch.Tensor, memories: torch.Tensor, beta: float) -> torch.Tensor:
    """Computes continuous modern Hopfield energy: E(x) = -1/beta * log(sum exp(beta * memories @ x)) + 0.5 * ||x||^2.
    
    x has shape [B, D], memories has shape [M, D].
    Returns energy of shape [B, 1].
    """
    sim = x @ memories.T  # [B, M]
    lse = torch.logsumexp(beta * sim, dim=-1, keepdim=True) / beta
    norm_term = 0.5 * torch.sum(x ** 2, dim=-1, keepdim=True)
    return -lse + norm_term


def modern_hopfield_retrieve(query: torch.Tensor, memories: torch.Tensor, beta: float) -> torch.Tensor:
    """Retrieves associative memory pattern via attention mechanism: x_new = softmax(beta * query @ M^T) @ M.
    
    query has shape [B, D], memories has shape [M, D].
    Returns retrieved state of shape [B, D].
    """
    scores = query @ memories.T
    weights = F.softmax(beta * scores, dim=-1)
    return weights @ memories


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # RBM parameters: B=4, V=6, H=8
    B, V, H = 4, 6, 8
    v = torch.randn(B, V)
    w_rbm = torch.randn(V, H) * 0.2
    a_rbm = torch.randn(V) * 0.1
    b_rbm = torch.randn(H) * 0.1
    free_energy = rbm_free_energy(v, w_rbm, a_rbm, b_rbm)

    # Langevin dynamics setup: state x [4, 8]
    D = 8
    x_ebm = torch.randn(B, D)
    grad_energy = torch.randn(B, D) * 0.5
    noise = torch.randn(B, D)
    x_langevin = langevin_dynamics_step(x_ebm, grad_energy, step_size=0.01, noise=noise)

    # Modern Hopfield setup: M=5 memories of dimension D=8
    M = 5
    memories = torch.randn(M, D)
    beta = 2.0
    hopfield_energy = modern_hopfield_energy(x_ebm, memories, beta=beta)
    retrieved = modern_hopfield_retrieve(x_ebm, memories, beta=beta)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "v": v,
        "w_rbm": w_rbm,
        "a_rbm": a_rbm,
        "b_rbm": b_rbm,
        "free_energy": free_energy,
        "x_ebm": x_ebm,
        "grad_energy": grad_energy,
        "noise": noise,
        "x_langevin": x_langevin,
        "memories": memories,
        "hopfield_energy": hopfield_energy,
        "retrieved": retrieved,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous().cpu() for k, v in build_golden_tensors().items()}
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
