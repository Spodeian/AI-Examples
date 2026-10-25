"""Multimodal Fusion FiLM, Perceiver Resampler, and VLM Connectors — PyTorch 2.x reference implementation.

Topic: 05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors
Golden weights: weights/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.safetensors

``python pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def film_modulation(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Feature-wise Linear Modulation (FiLM): gamma * x + beta.
    
    x has shape [B, L, D], gamma and beta have shape [B, 1, D].
    """
    return gamma * x + beta


def perceiver_resampler(
    latent_queries: torch.Tensor,
    visual_tokens: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
) -> torch.Tensor:
    """Perceiver Resampler cross-attention: maps variable visual tokens down to fixed K latents.
    
    latent_queries has shape [B, K, D], visual_tokens has shape [B, L, D].
    Returns resampled tokens of shape [B, K, D].
    """
    d = latent_queries.shape[-1]
    q = latent_queries @ w_q
    k = visual_tokens @ w_k
    v = visual_tokens @ w_v

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(scores, dim=-1)
    return attn @ v


def vlm_spatial_pool_connector(
    patches: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """VLM pixel unshuffle connector: compresses 2x2 adjacent patches into 1 LLM token via 2-layer MLP.
    
    patches has shape [B, 4, D].
    Returns projected LLM tokens of shape [B, 1, D_llm].
    """
    b, num_patches, d = patches.shape
    # Flatten 2x2 patches into a single 4D feature vector
    unshuffled = patches.reshape(b, 1, num_patches * d)
    h = F.gelu(unshuffled @ w1 + b1)
    return h @ w2 + b2


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # FiLM setup: B=2, L=6, D=8
    B, L, K, D = 2, 6, 3, 8
    x_feat = torch.randn(B, L, D)
    gamma = torch.randn(B, 1, D) * 0.5 + 1.0
    beta = torch.randn(B, 1, D) * 0.2
    h_film = film_modulation(x_feat, gamma, beta)

    # Perceiver Resampler: K=3 latents, L=6 visual tokens
    latent_queries = torch.randn(B, K, D)
    w_q = torch.randn(D, D) * 0.2
    w_k = torch.randn(D, D) * 0.2
    w_v = torch.randn(D, D) * 0.2
    h_resampled = perceiver_resampler(latent_queries, x_feat, w_q, w_k, w_v)

    # VLM Connector: 4 patches -> 1 token
    patches = torch.randn(B, 4, D)
    D_llm = 8
    w1 = torch.randn(4 * D, 16) * 0.2
    b1 = torch.randn(16) * 0.1
    w2 = torch.randn(16, D_llm) * 0.2
    b2 = torch.randn(D_llm) * 0.1
    h_vlm = vlm_spatial_pool_connector(patches, w1, b1, w2, b2)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_feat": x_feat,
        "gamma": gamma,
        "beta": beta,
        "h_film": h_film,
        "latent_queries": latent_queries,
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "h_resampled": h_resampled,
        "patches": patches,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "h_vlm": h_vlm,
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
