"""Spiking Neural Networks, LIF, Surrogate Gradients, and Capsules — PyTorch 2.x reference implementation.

Topic: 05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules
Golden weights: weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.safetensors

``python pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_spiking_neural_networks_lif_surrogate_gradients_and_capsules"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def lif_simulation(
    currents: torch.Tensor,
    beta: float = 0.8,
    v_th: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leaky Integrate-and-Fire (LIF) multi-step discrete simulation with hard reset.
    
    currents has shape [T, B, D].
    Returns (spikes, voltages) each of shape [T, B, D].
    """
    t_steps, b, d = currents.shape
    v = torch.zeros(b, d, dtype=currents.dtype, device=currents.device)
    s = torch.zeros(b, d, dtype=currents.dtype, device=currents.device)

    all_v = []
    all_s = []

    for t in range(t_steps):
        # Subthreshold membrane potential with hard reset
        v = beta * v * (1.0 - s) + currents[t]
        # Spike generation via threshold comparison
        s = (v >= v_th).to(currents.dtype)
        all_v.append(v)
        all_s.append(s)

    return torch.stack(all_s, dim=0), torch.stack(all_v, dim=0)


def capsule_squash(s: torch.Tensor) -> torch.Tensor:
    """Vector squashing non-linearity: scales vectors to length in [0, 1)."""
    norm_sq = torch.sum(s ** 2, dim=-1, keepdim=True)
    norm = torch.sqrt(norm_sq + 1e-8)
    scale = norm_sq / (1.0 + norm_sq)
    return scale * (s / norm)


def capsule_dynamic_routing(u_hat: torch.Tensor, num_iters: int = 3) -> torch.Tensor:
    """Dynamic routing agreement over capsule prediction vectors.
    
    u_hat has shape [B, N_in, N_out, D_out].
    Returns output capsule vectors v of shape [B, N_out, D_out].
    """
    b, n_in, n_out, d_out = u_hat.shape
    # Routing logits b initialized to 0
    b_logits = torch.zeros(b, n_in, n_out, dtype=u_hat.dtype, device=u_hat.device)

    v = torch.zeros(b, n_out, d_out, dtype=u_hat.dtype, device=u_hat.device)
    for i in range(num_iters):
        c = F.softmax(b_logits, dim=-1)  # [B, N_in, N_out]
        # s_j = sum_i c_ij * u_hat_j|i
        s = torch.sum(c.unsqueeze(-1) * u_hat, dim=1)  # [B, N_out, D_out]
        v = capsule_squash(s)
        if i < num_iters - 1:
            # Agreement update: b_ij += u_hat_j|i . v_j
            agreement = torch.sum(u_hat * v.unsqueeze(1), dim=-1)  # [B, N_in, N_out]
            b_logits = b_logits + agreement

    return v


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # LIF simulation: T=4, B=2, D=6
    T, B, D = 4, 2, 6
    currents = torch.randn(T, B, D) * 0.6 + 0.4
    spikes, voltages = lif_simulation(currents, beta=0.8, v_th=1.0)

    # Capsule Dynamic Routing: B=2, N_in=4, N_out=3, D_out=4
    u_hat = torch.randn(2, 4, 3, 4) * 0.5
    v_capsules = capsule_dynamic_routing(u_hat, num_iters=3)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "currents": currents,
        "lif_spikes": spikes,
        "lif_voltages": voltages,
        "u_hat": u_hat,
        "v_capsules": v_capsules,
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
