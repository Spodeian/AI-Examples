"""Neural Audio Codecs Encodec RVQ and Quantizer Dropout — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout
Golden weights: weights/05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout.safetensors

``python pytorch2/05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/02_neural_audio_codecs_encodec_rvq_and_quantizer_dropout"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def snake_activation(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake periodic activation function: Snake(x) = x + (1 / alpha) * sin^2(alpha * x)."""
    return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2)


def residual_vector_quantization(
    z: torch.Tensor,
    codebooks: torch.Tensor,
    n_active: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual Vector Quantization (RVQ) with optional quantizer dropout truncation.
    
    Args:
        z: Input latent tensor [B, D].
        codebooks: Codebook tensor [N_q, K, D].
        n_active: Number of active quantizer stages (1 <= n_active <= N_q).
        
    Returns:
        z_q: Quantized latent representation [B, D].
        indices: Codebook indices [N_q, B].
        residual: Final remaining residual [B, D].
    """
    N_q, K, D = codebooks.shape
    B = z.shape[0]
    active = N_q if n_active is None else n_active

    residual = z.clone()
    z_q = torch.zeros_like(z)
    all_indices = []

    for k in range(N_q):
        cb = codebooks[k]  # [K, D]
        # Distances: ||r - c||^2 = ||r||^2 - 2 r @ c^T + ||c||^2
        dists = (
            residual.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * (residual @ cb.T)
            + cb.pow(2).sum(dim=-1).unsqueeze(0)
        )  # [B, K]
        idx = torch.argmin(dists, dim=-1)  # [B]
        all_indices.append(idx)
        quantized_k = cb[idx]  # [B, D]

        if k < active:
            z_q = z_q + quantized_k
            residual = residual - quantized_k
        else:
            residual = residual - quantized_k

    indices_tensor = torch.stack(all_indices, dim=0)  # [N_q, B]
    return z_q, indices_tensor, residual


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 2. Snake periodic activation
    B = 4
    D = 4
    x_audio = torch.randn(B, D)
    alpha = torch.tensor([1.5])
    snake_out = snake_activation(x_audio, alpha)

    # 3. Multi-scale RVQ with Quantizer Dropout
    N_q = 3
    K = 8
    codebooks = torch.randn(N_q, K, D) * 0.5
    z_in = torch.randn(B, D)
    n_active = 2

    z_q, indices, residual = residual_vector_quantization(z_in, codebooks, n_active=n_active)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "x_audio": x_audio,
        "alpha": alpha,
        "snake_out": snake_out,
        "z_in": z_in,
        "codebooks": codebooks,
        "n_active": torch.tensor([n_active], dtype=torch.int64),
        "z_q": z_q,
        "indices": indices.float(),
        "residual": residual,
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
