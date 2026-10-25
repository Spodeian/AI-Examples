"""Reservoir Computing, Echo State Networks, LSM, and NG-RC — PyTorch 2.x reference implementation.

Topic: 05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc
Golden weights: weights/05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc.safetensors

``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def esn_forward(
    x_seq: torch.Tensor,
    w_in: torch.Tensor,
    w_res: torch.Tensor,
    b_res: torch.Tensor,
    alpha: float,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leaky Echo State Network recurrent forward pass.
    
    tilde_h_t = tanh(x_t @ w_in.T + h_{t-1} @ w_res.T + b_res)
    h_t = (1 - alpha) * h_{t-1} + alpha * tilde_h_t
    y_t = h_t @ w_out.T + b_out
    """
    B, T, D_in = x_seq.shape
    D_res = w_in.shape[0]

    h_t = torch.zeros(B, D_res, device=x_seq.device, dtype=x_seq.dtype)
    outs = []

    for t in range(T):
        xt = x_seq[:, t, :]  # [B, D_in]
        pre_act = xt @ w_in.t() + h_t @ w_res.t() + b_res
        tilde_h = torch.tanh(pre_act)
        h_t = (1.0 - alpha) * h_t + alpha * tilde_h
        yt = h_t @ w_out.t() + b_out
        outs.append(yt.unsqueeze(1))

    seq_out = torch.cat(outs, dim=1)
    return seq_out, h_t


def ngrc_features(x_curr: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    """Next-Generation Reservoir Computing (NG-RC) nonlinear feature expansion.
    
    Linear features: [x_curr, x_prev]
    Nonlinear features: quadratic self and cross-terms
    """
    lin = torch.cat([x_curr, x_prev], dim=-1)  # [B, 2 * D]
    quad = (x_curr.unsqueeze(-1) * x_prev.unsqueeze(-2)).flatten(start_dim=1)  # [B, D * D]
    return torch.cat([lin, quad], dim=-1)


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

    # 2. Echo State Network
    B = 2
    T = 4
    D_in = 4
    D_res = 8
    D_out = 4
    alpha = 0.6

    x_seq = torch.randn(B, T, D_in) * 0.2
    w_in = torch.randn(D_res, D_in) * 0.2
    # Ensure spectral radius < 1 for contractive echo state property
    w_res_raw = torch.randn(D_res, D_res)
    eigs = torch.linalg.eigvals(w_res_raw)
    spec_rad = float(torch.max(torch.abs(eigs)))
    w_res = (w_res_raw / (spec_rad * 1.25)) * 0.9  # rho = 0.72 < 1

    b_res = torch.zeros(D_res)
    w_out = torch.randn(D_out, D_res) * 0.1
    b_out = torch.randn(D_out) * 0.1
    alpha_tensor = torch.tensor([alpha])

    seq_out, final_h = esn_forward(x_seq, w_in, w_res, b_res, alpha, w_out, b_out)

    # 3. NG-RC expansion
    x_curr = torch.randn(B, 3) * 0.5
    x_prev = torch.randn(B, 3) * 0.5
    ngrc_feat = ngrc_features(x_curr, x_prev)

    return {
        "test_input": test_in,
        "weight": w_linear,
        "bias": b_linear,
        "test_output": test_out,
        "x_seq": x_seq,
        "w_in": w_in,
        "w_res": w_res,
        "b_res": b_res,
        "alpha": alpha_tensor,
        "w_out": w_out,
        "b_out": b_out,
        "seq_out": seq_out,
        "final_h": final_h,
        "x_curr": x_curr,
        "x_prev": x_prev,
        "ngrc_feat": ngrc_feat,
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
