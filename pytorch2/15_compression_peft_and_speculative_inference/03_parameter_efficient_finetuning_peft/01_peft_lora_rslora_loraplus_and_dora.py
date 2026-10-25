"""PEFT LoRA Rslora Loraplus and DoRA — PyTorch 2.x reference implementation.

Topic: 15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora
Golden weights: weights/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora.safetensors

``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora.py`` verifies against the golden checkpoint;
``python pytorch2/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def lora_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 4.0,
    r: int = 2,
) -> torch.Tensor:
    """Standard LoRA forward pass: h = x W0 + (alpha / r) * (x A) B.

    Args:
        x: Input activations of shape [batch, d_in].
        w0: Frozen base weights of shape [d_in, d_out].
        a: Down-projection adapter of shape [d_in, r].
        b: Up-projection adapter of shape [r, d_out].
        alpha: LoRA scaling hyperparameter.
        r: Adapter rank.

    Returns:
        h: Adapted output representations of shape [batch, d_out].
    """
    scale = alpha / float(r)
    base_out = x @ w0
    adapter_out = (x @ a) @ b
    return base_out + scale * adapter_out


def rslora_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 4.0,
    r: int = 2,
) -> torch.Tensor:
    """Rank-Stabilized LoRA (rsLoRA) forward pass: h = x W0 + (alpha / sqrt(r)) * (x A) B.

    Stabilizes gradient norms when scaling adapter rank r.

    Args:
        x: Input activations of shape [batch, d_in].
        w0: Frozen base weights of shape [d_in, d_out].
        a: Down-projection adapter of shape [d_in, r].
        b: Up-projection adapter of shape [r, d_out].
        alpha: LoRA scaling hyperparameter.
        r: Adapter rank.

    Returns:
        h: Adapted output representations of shape [batch, d_out].
    """
    scale = alpha / math.sqrt(float(r))
    base_out = x @ w0
    adapter_out = (x @ a) @ b
    return base_out + scale * adapter_out


def dora_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    m: torch.Tensor,
    alpha: float = 4.0,
    r: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weight-Decomposed Low-Rank Adaptation (DoRA) forward pass.

    Decomposes weight into learnable magnitude m and normalized direction matrix V:
    V = W0 + (alpha / r) * (A B)
    W_dora = m * (V / ||V||_c)
    h = x @ W_dora

    Args:
        x: Input activations of shape [batch, d_in].
        w0: Frozen base weights of shape [d_in, d_out].
        a: Down-projection adapter of shape [d_in, r].
        b: Up-projection adapter of shape [r, d_out].
        m: Learnable magnitude vector of shape [d_out].
        alpha: LoRA scaling hyperparameter.
        r: Adapter rank.
        eps: Epsilon for numerical stability.

    Returns:
        h: Adapted output representations of shape [batch, d_out].
    """
    scale = alpha / float(r)
    delta_v = a @ b
    v = w0 + scale * delta_v  # [d_in, d_out]

    # Column-wise norm: norm over dim 0 (d_in) -> [1, d_out]
    col_norm = torch.norm(v, p=2, dim=0, keepdim=True).clamp_min(eps)
    w_dora = (m / col_norm) * v  # broadcasting m [d_out] across rows
    return x @ w_dora


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # Dimensions
    batch = 4
    d_in = 8
    d_out = 8
    r = 2
    alpha = 4.0

    x = torch.randn(batch, d_in)
    w0 = torch.randn(d_in, d_out) * 0.2
    a = torch.randn(d_in, r) * 0.1
    b = torch.randn(r, d_out) * 0.1

    # Compute initial magnitude vector m as column norms of W0
    m = torch.norm(w0, p=2, dim=0)

    # Reference outputs
    lora_out = lora_forward(x, w0, a, b, alpha=alpha, r=r)
    rslora_out = rslora_forward(x, w0, a, b, alpha=alpha, r=r)
    dora_out = dora_forward(x, w0, a, b, m, alpha=alpha, r=r)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x": x,
        "w0": w0,
        "a": a,
        "b": b,
        "m": m,
        "lora_out": lora_out,
        "rslora_out": rslora_out,
        "dora_out": dora_out,
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
