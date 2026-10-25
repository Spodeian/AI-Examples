"""Triton and Tiled Kernel Algorithms Softmax RMSNorm Liger Cross-Entropy — PyTorch 2.x reference implementation.

Topic: 17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce
Golden weights: weights/17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce.safetensors

``python pytorch2/17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce.py`` verifies against the golden checkpoint;
``python pytorch2/17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def fused_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Fused RMSNorm: x / sqrt(mean(x^2) + eps) * gamma."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * gamma


def fused_swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation: silu(x @ w_gate) * (x @ w_up)."""
    gate = F.silu(x @ w_gate)
    up = x @ w_up
    return gate * up


def tiled_linear_cross_entropy(
    h: torch.Tensor,
    w_vocab: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Liger-style tiled linear + cross-entropy computing loss and input gradients chunk-by-chunk.
    
    Avoids materializing full [B, Vocab] logit tensor in global memory.
    """
    b, d = h.shape
    v = w_vocab.shape[1]

    total_loss = torch.tensor(0.0, device=h.device)
    dh = torch.zeros_like(h)

    num_chunks = (b + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, b)
        h_chunk = h[start:end]
        labels_chunk = labels[start:end]

        # Compute logits only for this small chunk
        logits_chunk = h_chunk @ w_vocab
        loss_chunk = F.cross_entropy(logits_chunk, labels_chunk, reduction="sum")
        total_loss = total_loss + loss_chunk

        # Backward gradient on chunk inputs: dH = (probs - 1_y) @ W_vocab^T
        probs = F.softmax(logits_chunk, dim=-1)
        one_hot = F.one_hot(labels_chunk, num_classes=v).to(probs.dtype)
        d_logits = probs - one_hot
        dh[start:end] = (d_logits @ w_vocab.T) / b

    mean_loss = total_loss / b
    return mean_loss, dh


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    b, d, v = 4, 4, 16

    h = torch.randn(b, d)
    gamma = torch.randn(d) * 0.2 + 1.0
    w_gate = torch.randn(d, d) * 0.2
    w_up = torch.randn(d, d) * 0.2
    w_vocab = torch.randn(d, v) * 0.2
    labels = torch.randint(0, v, (b,))

    normed_h = fused_rmsnorm(h, gamma)
    swiglu_out = fused_swiglu(h, w_gate, w_up)
    loss, dh = tiled_linear_cross_entropy(h, w_vocab, labels, chunk_size=2)

    # Verification against standard eager cross entropy
    eager_logits = h @ w_vocab
    eager_loss = F.cross_entropy(eager_logits, labels)
    assert torch.abs(loss - eager_loss) < 1e-5

    linear_w = torch.randn(d, d) * 0.1
    linear_b = torch.randn(d) * 0.1
    test_out = linear_anchor(normed_h, linear_w, linear_b)

    tensors = {
        "h": h,
        "gamma": gamma,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_vocab": w_vocab,
        "labels": labels.to(torch.float32),
        "normed_h": normed_h,
        "swiglu_out": swiglu_out,
        "loss": loss.unsqueeze(0),
        "dh": dh,
        "linear_w": linear_w,
        "linear_b": linear_b,
        "test_input": normed_h.clone(),
        "weight": linear_w.clone(),
        "bias": linear_b.clone(),
        "test_output": test_out,
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
