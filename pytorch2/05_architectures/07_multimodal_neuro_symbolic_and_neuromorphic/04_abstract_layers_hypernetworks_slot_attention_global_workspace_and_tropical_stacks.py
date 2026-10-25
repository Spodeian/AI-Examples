"""Abstract Layers: Hypernetworks, Slot Attention, Global Workspace, and Tropical Stacks — PyTorch 2.x reference implementation.

Topic: 05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks
Golden weights: weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks.safetensors

``python pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_layers_hypernetworks_slot_attention_global_workspace_and_tropical_stacks"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def hypernetwork_linear(
    x: torch.Tensor,
    z: torch.Tensor,
    w_h: torch.Tensor,
    b_h: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Hypernetwork: generates dynamic weights for x from conditioning code z.
    
    x: [B, D_in]
    z: [B, D_z]
    w_h: [D_z, D_in * D_out]
    b_h: [D_in * D_out]
    bias: [D_out]
    Returns y of shape [B, D_out].
    """
    b, d_in = x.shape
    d_out = bias.shape[0]
    w_flat = z @ w_h + b_h  # [B, D_in * D_out]
    w_dyn = w_flat.reshape(b, d_in, d_out)  # [B, D_in, D_out]
    y = (x.unsqueeze(1) @ w_dyn).squeeze(1) + bias  # [B, D_out]
    return y


def slot_attention(
    slots: torch.Tensor,
    inputs: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
) -> torch.Tensor:
    """Slot Attention: competitive slot attention over input tokens.
    
    slots: [B, K, D]
    inputs: [B, N, D]
    Returns updated slots S_new of shape [B, K, D].
    """
    d = slots.shape[-1]
    q = slots @ w_q    # [B, K, D]
    k = inputs @ w_k   # [B, N, D]
    v = inputs @ w_v   # [B, N, D]

    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(-2, -1)) * scale  # [B, K, N]
    # Softmax over slots (dim 1) — slots compete for each input key
    attn = F.softmax(scores, dim=1)  # [B, K, N]

    # Weighted mean normalization
    attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K, N]
    updates = attn_norm @ v  # [B, K, D]
    return slots + updates


def tropical_max_plus_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Tropical (max-plus) semiring matrix multiplication: Y_ij = max_k (X_ik + W_kj).
    
    x: [B, D_in]
    w: [D_in, D_out]
    Returns Y of shape [B, D_out].
    """
    # x.unsqueeze(-1) is [B, D_in, 1], w.unsqueeze(0) is [1, D_in, D_out]
    summed = x.unsqueeze(-1) + w.unsqueeze(0)  # [B, D_in, D_out]
    return torch.max(summed, dim=1).values  # [B, D_out]


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Hypernetwork setup: B=2, D_in=4, D_out=4, D_z=3
    B, D_in, D_out, D_z = 2, 4, 4, 3
    x_hyper = torch.randn(B, D_in)
    z = torch.randn(B, D_z)
    w_h = torch.randn(D_z, D_in * D_out) * 0.2
    b_h = torch.randn(D_in * D_out) * 0.1
    bias_hyper = torch.randn(D_out) * 0.1
    y_hyper = hypernetwork_linear(x_hyper, z, w_h, b_h, bias_hyper)

    # Slot Attention: B=2, K=3, N=5, D=4
    K, N, D = 3, 5, 4
    slots = torch.randn(B, K, D)
    inputs = torch.randn(B, N, D)
    w_q = torch.randn(D, D) * 0.2
    w_k = torch.randn(D, D) * 0.2
    w_v = torch.randn(D, D) * 0.2
    slots_out = slot_attention(slots, inputs, w_q, w_k, w_v)

    # Tropical Max-Plus Matmul: B=2, D_in=4, D_out=3
    x_trop = torch.randn(B, 4)
    w_trop = torch.randn(4, 3)
    y_trop = tropical_max_plus_matmul(x_trop, w_trop)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_hyper": x_hyper,
        "z_code": z,
        "w_h": w_h,
        "b_h": b_h,
        "bias_hyper": bias_hyper,
        "y_hyper": y_hyper,
        "slots_in": slots,
        "slot_inputs": inputs,
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "slots_out": slots_out,
        "x_trop": x_trop,
        "w_trop": w_trop,
        "y_trop": y_trop,
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
