"""PEFT SVD and Adaptive PiSSA Adalora Vera — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera
Golden weights: weights/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def pissa_decompose(
    w0: torch.Tensor,
    r: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PiSSA (Principal Singular values and Singular vectors Adaptation) decomposition.

    Decomposes base weight W0 via SVD into principal rank-r components and residual:
    W0 = U S V^T
    B = U_r diag(sqrt(S_r))
    A = diag(sqrt(S_r)) V_r^T
    W_res = W0 - B A

    Args:
        w0: Base weight tensor of shape [d_in, d_out].
        r: Adaptation rank.

    Returns:
        w_res: Residual base weight of shape [d_in, d_out].
        b: Down-projection principal adapter of shape [d_in, r].
        a: Up-projection principal adapter of shape [r, d_out].
    """
    u, s, vh = torch.linalg.svd(w0, full_matrices=False)
    u_r = u[:, :r]
    s_r = s[:r]
    vh_r = vh[:r, :]

    sqrt_s = torch.diag(torch.sqrt(s_r))
    b = u_r @ sqrt_s
    a = sqrt_s @ vh_r
    w_res = w0 - b @ a
    return w_res, b, a


def pissa_forward(
    x: torch.Tensor,
    w_res: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """PiSSA forward pass: h = x W_res + (x B) A."""
    return x @ w_res + (x @ b) @ a


def adalora_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    p: torch.Tensor,
    lambda_diag: torch.Tensor,
    q: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AdaLoRA forward pass with dynamic rank mask.

    Delta W = P diag(lambda * mask) Q
    h = x W0 + x Delta W

    Args:
        x: Input activations of shape [batch, d_in].
        w0: Frozen base weights of shape [d_in, d_out].
        p: Left singular matrix adapter of shape [d_in, r].
        lambda_diag: Singular value parameter vector of shape [r].
        q: Right singular matrix adapter of shape [r, d_out].
        mask: Binary pruning mask of shape [r] (1 = keep, 0 = prune).

    Returns:
        h: Output activations of shape [batch, d_out].
        delta_w: Effective low-rank update of shape [d_in, d_out].
    """
    effective_lambda = torch.diag(lambda_diag * mask)
    delta_w = p @ effective_lambda @ q
    h = x @ w0 + x @ delta_w
    return h, delta_w


def vera_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    a_shared: torch.Tensor,
    b_shared: torch.Tensor,
    d_vec: torch.Tensor,
    b_vec: torch.Tensor,
) -> torch.Tensor:
    """VeRA (Vector-based Random Matrix Adaptation) forward pass.

    Uses shared frozen random matrices A_shared and B_shared, with layer-specific
    trainable scaling vectors d_vec (rank dimension) and b_vec (output dimension):
    h = x W0 + ((x A_shared * d_vec) B_shared) * b_vec

    Args:
        x: Input activations of shape [batch, d_in].
        w0: Frozen base weights of shape [d_in, d_out].
        a_shared: Shared frozen random down-matrix of shape [d_in, r].
        b_shared: Shared frozen random up-matrix of shape [r, d_out].
        d_vec: Trainable rank scaling vector of shape [r].
        b_vec: Trainable output scaling vector of shape [d_out].

    Returns:
        h: Output activations of shape [batch, d_out].
    """
    base_out = x @ w0
    x_a = x @ a_shared  # [batch, r]
    scaled_xa = x_a * d_vec  # elementwise multiply along rank dim
    xa_b = scaled_xa @ b_shared  # [batch, d_out]
    adapter_out = xa_b * b_vec  # elementwise multiply along output dim
    return base_out + adapter_out


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    batch = 4
    d_in = 8
    d_out = 8
    r_pissa = 2
    r_ada = 4

    x = torch.randn(batch, d_in)
    w0 = torch.randn(d_in, d_out) * 0.3

    # 1. PiSSA
    w_res, b_pissa, a_pissa = pissa_decompose(w0, r=r_pissa)
    pissa_out = pissa_forward(x, w_res, b_pissa, a_pissa)

    # 2. AdaLoRA
    p = torch.randn(d_in, r_ada) * 0.15
    q = torch.randn(r_ada, d_out) * 0.15
    lambda_diag = torch.tensor([0.5, 0.4, 0.2, 0.1], dtype=torch.float32)
    ada_mask = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32)  # pruned index 2
    ada_out, delta_w = adalora_forward(x, w0, p, lambda_diag, q, ada_mask)

    # 3. VeRA
    a_shared = torch.randn(d_in, r_ada) * 0.2
    b_shared = torch.randn(r_ada, d_out) * 0.2
    d_vec = torch.randn(r_ada) * 0.1
    b_vec = torch.randn(d_out) * 0.1
    vera_out = vera_forward(x, w0, a_shared, b_shared, d_vec, b_vec)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "w0": w0,
        "w_res": w_res,
        "b_pissa": b_pissa,
        "a_pissa": a_pissa,
        "pissa_out": pissa_out,
        "p": p,
        "q": q,
        "lambda_diag": lambda_diag,
        "ada_mask": ada_mask,
        "ada_out": ada_out,
        "delta_w": delta_w,
        "a_shared": a_shared,
        "b_shared": b_shared,
        "d_vec": d_vec,
        "b_vec": b_vec,
        "vera_out": vera_out,
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
        assert err < 1e-5, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
