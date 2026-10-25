"""Structured State Space Models S4 and LRU — PyTorch 2.x reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru.safetensors

``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def lru_recurrent_forward(
    x_seq: torch.Tensor,
    nu: torch.Tensor,
    theta: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
    c_re: torch.Tensor,
    c_im: torch.Tensor,
    d_skip: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear Recurrent Unit (LRU) recurrent forward pass in complex domain."""
    B, T, D_in = x_seq.shape
    D_state = nu.shape[0]

    # Diagonal transition eigenvalue: Lambda = exp(-nu + i * theta)
    diag_mag = torch.exp(-torch.clamp(nu, min=1e-4))
    diag_re = diag_mag * torch.cos(theta)
    diag_im = diag_mag * torch.sin(theta)

    h_re = torch.zeros(B, D_state, device=x_seq.device, dtype=x_seq.dtype)
    h_im = torch.zeros(B, D_state, device=x_seq.device, dtype=x_seq.dtype)

    outs = []
    for t in range(T):
        xt = x_seq[:, t, :]  # [B, D_in]

        # B x_t: [B, D_state]
        bu_re = xt @ b_re.t()
        bu_im = xt @ b_im.t()

        # Complex multiply: h_next = Lambda * h_prev + B u_t
        # (diag_re + i*diag_im) * (h_re + i*h_im) = (diag_re*h_re - diag_im*h_im) + i*(diag_re*h_im + diag_im*h_re)
        h_re_next = diag_re * h_re - diag_im * h_im + bu_re
        h_im_next = diag_re * h_im + diag_im * h_re + bu_im

        h_re = h_re_next
        h_im = h_im_next

        # Readout: y_t = Re(C h_t) + D x_t
        # C h = (c_re + i*c_im) * (h_re + i*h_im) -> Re = c_re @ h_re - c_im @ h_im
        y_t = h_re @ c_re.t() - h_im @ c_im.t() + xt * d_skip
        outs.append(y_t.unsqueeze(1))

    seq_out = torch.cat(outs, dim=1)
    return seq_out, h_re, h_im


def s4_kernel(
    nu: torch.Tensor,
    theta: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
    c_re: torch.Tensor,
    c_im: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """Computes convolutional filter K_t = Re(C Lambda^t B) for t = 0..L-1."""
    diag_mag = torch.exp(-torch.clamp(nu, min=1e-4))
    diag_re = diag_mag * torch.cos(theta)
    diag_im = diag_mag * torch.sin(theta)

    # Powers of Lambda
    D_state = nu.shape[0]
    p_re = torch.ones(D_state, device=nu.device)
    p_im = torch.zeros(D_state, device=nu.device)

    kernels = []
    for t in range(length):
        # C * (Lambda^t * B)
        # For simplicity, elementwise outer response:
        # Re(C * (p * B))
        # shape [D_in]
        pb_re = p_re.unsqueeze(1) * b_re - p_im.unsqueeze(1) * b_im  # [D_state, D_in]
        pb_im = p_re.unsqueeze(1) * b_im + p_im.unsqueeze(1) * b_re  # [D_state, D_in]

        # Readout: c_re @ pb_re - c_im @ pb_im -> [D_in, D_in]
        k_t = c_re @ pb_re - c_im @ pb_im
        kernels.append(k_t.unsqueeze(0))

        # Advance power: p = p * Lambda
        next_re = p_re * diag_re - p_im * diag_im
        next_im = p_re * diag_im + p_im * diag_re
        p_re, p_im = next_re, next_im

    return torch.cat(kernels, dim=0)  # [L, D_in, D_in]


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Linear Anchor
    batch = 8
    in_dim = 4
    out_dim = 4
    test_in = torch.randn(batch, in_dim)
    w_linear = torch.randn(in_dim, out_dim) * 0.1
    b_linear = torch.randn(out_dim) * 0.1
    test_out = linear_anchor(test_in, w_linear, b_linear)

    # 2. LRU / S4 Sequence Data
    B = 2
    T = 4
    D_in = 4
    D_state = 4
    x_seq = torch.randn(B, T, D_in)

    # Diagonal damping nu > 0 and frequency theta in [0, pi]
    nu = torch.tensor([0.2, 0.4, 0.6, 0.8])
    theta = torch.tensor([0.1, 0.5, 1.2, 2.0])

    b_re = torch.randn(D_state, D_in) * 0.2
    b_im = torch.randn(D_state, D_in) * 0.2
    c_re = torch.randn(D_in, D_state) * 0.2
    c_im = torch.randn(D_in, D_state) * 0.2
    d_skip = torch.randn(D_in) * 0.1

    seq_out, h_re, h_im = lru_recurrent_forward(
        x_seq, nu, theta, b_re, b_im, c_re, c_im, d_skip
    )

    conv_kernel = s4_kernel(nu, theta, b_re, b_im, c_re, c_im, length=T)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "x_seq": x_seq,
        "nu": nu,
        "theta": theta,
        "b_re": b_re,
        "b_im": b_im,
        "c_re": c_re,
        "c_im": c_im,
        "d_skip": d_skip,
        "seq_out": seq_out,
        "final_h_re": h_re,
        "final_h_im": h_im,
        "conv_kernel": conv_kernel,
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
