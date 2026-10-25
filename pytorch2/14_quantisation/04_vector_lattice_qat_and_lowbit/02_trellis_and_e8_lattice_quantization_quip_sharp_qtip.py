"""Trellis and E8 Lattice Quantization QuIP# QTIP — PyTorch 2.x reference implementation.

Topic: 14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip
Golden weights: weights/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.safetensors

``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/04_vector_lattice_qat_and_lowbit/02_trellis_and_e8_lattice_quantization_quip_sharp_qtip"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def project_d8(x: torch.Tensor) -> torch.Tensor:
    """Conway & Sloane fast projection onto the D8 checkerboard lattice.
    
    D8 = { x in Z^8 : sum(x_i) = 0 mod 2 }
    x: [..., 8]
    returns: [..., 8]
    """
    f = torch.round(x)
    s = torch.sum(f, dim=-1, keepdim=True).long()
    is_odd = (s % 2 != 0)

    # Errors |x - f|
    diff = x - f
    abs_diff = torch.abs(diff)
    max_idx = torch.argmax(abs_diff, dim=-1, keepdim=True)

    # Where sum is odd, flip the coordinate with largest rounding error
    delta = torch.where(diff >= 0.0, 1.0, -1.0)
    flip_mask = torch.zeros_like(x).scatter_(-1, max_idx, 1.0)

    f_adjusted = torch.where(is_odd & (flip_mask > 0.0), f + delta, f)
    return f_adjusted


def project_e8(x: torch.Tensor) -> torch.Tensor:
    """Fast nearest-point decoding for the 8D Gosset lattice E8:
    
    E8 = D8 U (D8 + 1/2)
    x: [..., 8]
    returns: [..., 8] nearest lattice point in E8
    """
    # Candidate 0: projection onto D8
    p0 = project_d8(x)
    dist0 = torch.sum((x - p0) ** 2, dim=-1, keepdim=True)

    # Candidate 1: projection onto coset D8 + 1/2
    p1 = project_d8(x - 0.5) + 0.5
    dist1 = torch.sum((x - p1) ** 2, dim=-1, keepdim=True)

    return torch.where(dist0 <= dist1, p0, p1)


def quantize_e8_lattice(vectors: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize vectors in R^8 to E8 Gosset lattice scaled by step size `scale`.
    
    vectors: [N, 8]
    scale: [1, 1] or [1, 8]
    returns: (w_deq [N, 8], lattice_points [N, 8])
    """
    x_scaled = vectors / scale
    lattice_points = project_e8(x_scaled)
    w_deq = lattice_points * scale
    return w_deq, lattice_points


def trellis_coded_quantize_1d(
    x: torch.Tensor,
    codebook_subsets: torch.Tensor,
    transitions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trellis-Coded Quantization (TCQ) forward Viterbi decoding over 1D sequence.
    
    x: [L] sequence of scalar weights
    codebook_subsets: [4, 4] 4 trellis subsets each having 4 scalar levels
    transitions: [4, 2] state transition table: next_state = transitions[curr_state, bit]
    returns: (recon [L], state_path [L])
    """
    l = x.shape[0]
    n_states = transitions.shape[0]
    inf = 1e9

    # Viterbi DP table: [L, n_states]
    dp = torch.full((l, n_states), inf)
    back_state = torch.zeros((l, n_states), dtype=torch.long)
    back_level = torch.zeros((l, n_states))

    # Initialize step 0
    for s in range(n_states):
        # find closest level in subset s
        subset = codebook_subsets[s]
        errs = (x[0] - subset) ** 2
        min_idx = torch.argmin(errs)
        dp[0, s] = errs[min_idx]
        back_level[0, s] = subset[min_idx]

    # Viterbi forward pass
    for t in range(1, l):
        for s in range(n_states):
            for bit in (0, 1):
                next_s = int(transitions[s, bit])
                subset = codebook_subsets[next_s]
                errs = (x[t] - subset) ** 2
                best_k = torch.argmin(errs)
                cost = dp[t - 1, s] + errs[best_k]
                if cost < dp[t, next_s]:
                    dp[t, next_s] = cost
                    back_state[t, next_s] = s
                    back_level[t, next_s] = subset[best_k]

    # Backtrack optimal state path
    recon = torch.zeros(l)
    best_last = int(torch.argmin(dp[-1]))
    curr_s = best_last
    for t in range(l - 1, -1, -1):
        recon[t] = back_level[t, curr_s]
        curr_s = int(back_state[t, curr_s])

    return recon, dp


def e8_forward(
    x: torch.Tensor,
    w_orig: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference forward pass using E8 lattice quantized weight matrix."""
    d_in, d_out = w_orig.shape
    w_vecs = w_orig.reshape(-1, 8)
    w_deq_vecs, _ = quantize_e8_lattice(w_vecs, scale)
    w_deq = w_deq_vecs.reshape(d_in, d_out)
    y = x @ w_deq
    if bias is not None:
        y = y + bias
    return y, w_deq


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    b, d_in, d_out = 16, 16, 8
    x_calib = torch.randn(b, d_in)
    w_orig = torch.randn(d_in, d_out) * 0.3
    b_orig = torch.randn(d_out) * 0.05

    # Group into N = 16 vectors in R^8
    w_vecs = w_orig.reshape(16, 8).clone()

    # 1. E8 Lattice Quantization (scale = 0.1)
    e8_scale = torch.tensor([[0.10]])
    w_e8_deq, lattice_points = quantize_e8_lattice(w_vecs, e8_scale)

    # 2. Trellis-Coded Quantization (TCQ) on a 1D slice of 16 weights
    tcq_x = w_orig.flatten()[:16].clone()
    # 4 state subsets, 4 levels each
    tcq_subsets = torch.tensor([
        [-0.6, -0.2, 0.2, 0.6],
        [-0.5, -0.1, 0.3, 0.7],
        [-0.7, -0.3, 0.1, 0.5],
        [-0.4, 0.0, 0.4, 0.8],
    ])
    # 4-state rate-1/2 convolutional trellis transitions: [4, 2]
    tcq_transitions = torch.tensor([
        [0, 1],
        [2, 3],
        [0, 1],
        [2, 3],
    ], dtype=torch.long)

    tcq_recon, tcq_dp = trellis_coded_quantize_1d(tcq_x, tcq_subsets, tcq_transitions)

    # 3. E8 Forward pass
    out_e8, w_deq_full = e8_forward(x_calib, w_orig, e8_scale, b_orig)

    # 4. Pure FP32 ground truth
    out_fp32 = x_calib @ w_orig + b_orig

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_calib": x_calib,
        "w_orig": w_orig,
        "b_orig": b_orig,
        "w_vecs": w_vecs,
        "e8_scale": e8_scale,
        "w_e8_deq": w_e8_deq,
        "lattice_points": lattice_points,
        "tcq_x": tcq_x,
        "tcq_subsets": tcq_subsets,
        "tcq_transitions": tcq_transitions.float(),
        "tcq_recon": tcq_recon,
        "w_deq_full": w_deq_full,
        "out_e8": out_e8,
        "out_fp32": out_fp32,
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
