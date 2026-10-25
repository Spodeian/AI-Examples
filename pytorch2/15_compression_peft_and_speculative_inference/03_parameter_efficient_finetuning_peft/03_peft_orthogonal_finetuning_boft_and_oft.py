"""PEFT Orthogonal Finetuning BOFT and Oft — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft
Golden weights: weights/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def cayley_orthogonal(q_skew: torch.Tensor) -> torch.Tensor:
    """Construct strictly orthogonal matrix R from skew-symmetric Q via Cayley transform:
    R = (I - Q) (I + Q)^(-1)
    Guarantees R^T R = I.

    Args:
        q_skew: Skew-symmetric matrix of shape [b, b] satisfying Q^T = -Q.

    Returns:
        r: Orthogonal rotation matrix of shape [b, b].
    """
    b = q_skew.shape[0]
    eye = torch.eye(b, dtype=q_skew.dtype, device=q_skew.device)
    r = torch.linalg.solve(eye + q_skew, eye - q_skew)
    return r


def block_oft_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    q_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-Diagonal Orthogonal Finetuning (OFT) forward pass.

    Constructs block-diagonal orthogonal matrix R = blkdiag(R_1, ..., R_m) where each R_k
    is derived via Cayley transform from skew-symmetric parameter block Q_k.
    Adapted weights: W_oft = R @ W0
    Forward pass: h = x @ W_oft

    Args:
        x: Input activations of shape [batch, d_in].
        w0: Base frozen weights of shape [d_in, d_out].
        q_blocks: Skew-symmetric parameter blocks of shape [num_blocks, block_size, block_size].

    Returns:
        h: Output activations of shape [batch, d_out].
        r_full: Block-diagonal orthogonal matrix of shape [d_in, d_in].
    """
    num_blocks, block_size, _ = q_blocks.shape
    d_in = num_blocks * block_size

    r_blocks = []
    for k in range(num_blocks):
        # Enforce skew-symmetry: Q = 0.5 * (Q - Q^T)
        skew_k = 0.5 * (q_blocks[k] - q_blocks[k].T)
        r_k = cayley_orthogonal(skew_k)
        r_blocks.append(r_k)

    r_full = torch.block_diag(*r_blocks)  # [d_in, d_in]
    w_oft = r_full @ w0
    h = x @ w_oft
    return h, r_full


def butterfly_boft_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    q_blocks_layer1: torch.Tensor,
    q_blocks_layer2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Butterfly Orthogonal Finetuning (BOFT) forward pass.

    Composes two layers of block-diagonal orthogonal transformations with permuted butterfly interleaving:
    R_total = R_layer2 @ R_layer1
    W_boft = R_total @ W0
    h = x @ W_boft

    Args:
        x: Input activations of shape [batch, d_in].
        w0: Base frozen weights of shape [d_in, d_out].
        q_blocks_layer1: First stage block parameters [num_blocks, block_size, block_size].
        q_blocks_layer2: Second stage block parameters [num_blocks, block_size, block_size].

    Returns:
        h: Output activations of shape [batch, d_out].
        r_total: Composite butterfly orthogonal transformation of shape [d_in, d_in].
    """
    _, r1 = block_oft_forward(x, w0, q_blocks_layer1)
    _, r2 = block_oft_forward(x, w0, q_blocks_layer2)
    r_total = r2 @ r1
    w_boft = r_total @ w0
    h = x @ w_boft
    return h, r_total


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    batch = 4
    d_in = 8
    d_out = 8
    block_size = 4
    num_blocks = 2  # 2 * 4 = 8

    x = torch.randn(batch, d_in)
    w0 = torch.randn(d_in, d_out) * 0.25

    # 1. Single Block Cayley Orthogonal Matrix
    raw_q = torch.randn(block_size, block_size) * 0.2
    q_skew = 0.5 * (raw_q - raw_q.T)
    r_cayley = cayley_orthogonal(q_skew)

    # 2. Block-Diagonal OFT
    raw_blocks1 = torch.randn(num_blocks, block_size, block_size) * 0.2
    # Ensure skew-symmetry
    q_blocks1 = 0.5 * (raw_blocks1 - raw_blocks1.transpose(-1, -2))
    oft_out, r_full = block_oft_forward(x, w0, q_blocks1)

    # 3. Butterfly BOFT
    raw_blocks2 = torch.randn(num_blocks, block_size, block_size) * 0.2
    q_blocks2 = 0.5 * (raw_blocks2 - raw_blocks2.transpose(-1, -2))
    boft_out, r_total = butterfly_boft_forward(x, w0, q_blocks1, q_blocks2)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "w0": w0,
        "q_skew": q_skew,
        "r_cayley": r_cayley,
        "q_blocks1": q_blocks1,
        "oft_out": oft_out,
        "r_full": r_full,
        "q_blocks2": q_blocks2,
        "boft_out": boft_out,
        "r_total": r_total,
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
