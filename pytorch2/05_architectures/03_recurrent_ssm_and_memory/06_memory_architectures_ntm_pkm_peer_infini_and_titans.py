"""Memory Architectures: NTM, PKM, PEER, Infini-attention, and Titans — PyTorch 2.x reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans.safetensors

``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ntm_content_addressing_read(
    memory: torch.Tensor,
    key: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NTM content-based addressing and read head.
    
    memory: [B, N, W] (N memory slots, width W)
    key: [B, W]
    beta: [B, 1] key strength
    
    sim = cosine_similarity(key.unsqueeze(1), memory, dim=-1)  # [B, N]
    weights = softmax(beta * sim, dim=-1)                      # [B, N]
    read_vec = weights.unsqueeze(1) @ memory                  # [B, W]
    """
    key_norm = F.normalize(key, p=2, dim=-1).unsqueeze(1)      # [B, 1, W]
    mem_norm = F.normalize(memory, p=2, dim=-1)               # [B, N, W]
    cos_sim = (key_norm * mem_norm).sum(dim=-1)                # [B, N]

    weights = F.softmax(beta * cos_sim, dim=-1)                # [B, N]
    read_vec = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)  # [B, W]
    return read_vec, weights


def infini_memory_step(
    m_prev: torch.Tensor,
    z_prev: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infini-attention compressive memory retrieval and update step.
    
    phi(x) = elu(x) + 1.0
    A_mem = (phi(q) @ M_prev) / (phi(q) @ z_prev + 1e-4)  [B, D]
    M_next = M_prev + phi(k)^T @ v                         [B, D, D]
    z_next = z_prev + phi(k)                               [B, D]
    """
    # Non-negative feature map phi
    phi_q = F.elu(q) + 1.0  # [B, D]
    phi_k = F.elu(k) + 1.0  # [B, D]

    # Retrieval
    # phi_q: [B, 1, D] @ m_prev: [B, D, D] -> [B, 1, D]
    num = torch.bmm(phi_q.unsqueeze(1), m_prev).squeeze(1)  # [B, D]
    denom = (phi_q * z_prev).sum(dim=-1, keepdim=True) + 1e-4  # [B, 1]
    a_mem = num / denom  # [B, D]

    # Update
    # phi_k: [B, D, 1] @ v: [B, 1, D] -> [B, D, D]
    m_delta = torch.bmm(phi_k.unsqueeze(2), v.unsqueeze(1))
    m_next = m_prev + m_delta
    z_next = z_prev + phi_k

    return a_mem, m_next, z_next


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    batch = 8
    in_dim = 4
    out_dim = 4
    test_in = torch.randn(batch, in_dim)
    w_linear = torch.randn(in_dim, out_dim) * 0.1
    b_linear = torch.randn(out_dim) * 0.1
    test_out = linear_anchor(test_in, w_linear, b_linear)

    # 2. NTM Content Addressing
    B = 2
    N = 8  # 8 slots
    W = 4  # slot width
    memory = torch.randn(B, N, W) * 0.5
    key = torch.randn(B, W) * 0.5
    beta = torch.abs(torch.randn(B, 1)) + 1.0

    read_vec, ntm_weights = ntm_content_addressing_read(memory, key, beta)

    # 3. Infini-Attention Compressive Memory
    D = 4
    m_prev = torch.randn(B, D, D) * 0.2
    z_prev = torch.abs(torch.randn(B, D)) + 0.5
    q = torch.randn(B, D) * 0.2
    k = torch.randn(B, D) * 0.2
    v = torch.randn(B, D) * 0.2

    a_mem, m_next, z_next = infini_memory_step(m_prev, z_prev, q, k, v)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "memory": memory,
        "key": key,
        "beta": beta,
        "read_vec": read_vec,
        "ntm_weights": ntm_weights,
        "m_prev": m_prev,
        "z_prev": z_prev,
        "q": q,
        "k": k,
        "v": v,
        "a_mem": a_mem,
        "m_next": m_next,
        "z_next": z_next,
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
