"""GANs Dcgan Wgan GP and Stylegan Concepts — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts
Golden weights: weights/05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts.safetensors

``python pytorch2/05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/03_gans_dcgan_wgan_gp_and_stylegan_concepts"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def adain(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Adaptive Instance Normalization (AdaIN): y = scale * (x - mean) / sqrt(var + eps) + bias."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    norm = (x - mean) / torch.sqrt(var + eps)
    return norm * scale + bias


def stylegan2_weight_demodulation(
    x: torch.Tensor,
    weights: torch.Tensor,
    styles: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """StyleGAN2 Weight Modulation and Demodulation.
    
    Args:
        x: Input activations [B, C_in].
        weights: Base layer weight matrix [C_out, C_in].
        styles: Style scale modulation factors [B, C_in].
        eps: Small epsilon for numerical stability.
        
    Returns:
        Demodulated forward output [B, C_out].
    """
    # 1. Modulate: w'_b,i,j = w_i,j * s_b,j -> shape [B, C_out, C_in]
    w_mod = weights.unsqueeze(0) * styles.unsqueeze(1)
    # 2. Demodulate: normalize over incoming channel dimension C_in
    # d_b,i = sqrt(sum_j (w'_b,i,j)^2 + eps) -> shape [B, C_out, 1]
    demod_factors = torch.rsqrt(w_mod.pow(2).sum(dim=-1, keepdim=True) + eps)
    w_demod = w_mod * demod_factors
    # 3. Apply to input: y_b,i = sum_j x_b,j * w''_b,i,j
    # x: [B, 1, C_in], w_demod: [B, C_out, C_in] -> y: [B, C_out]
    y = (w_demod @ x.unsqueeze(-1)).squeeze(-1)
    return y


def wgan_critic_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """WGAN Earth Mover's critic loss: E[D(x_fake)] - E[D(x_real)]."""
    return fake_scores.mean() - real_scores.mean()


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 2. AdaIN
    B = 4
    C = 6
    x_adain = torch.randn(B, C)
    scale = torch.randn(B, C) * 0.5 + 1.0
    bias_adain = torch.randn(B, C) * 0.2
    adain_out = adain(x_adain, scale, bias_adain)

    # 3. StyleGAN2 Weight Demodulation
    C_in = 4
    C_out = 4
    x_feat = torch.randn(B, C_in)
    w_conv = torch.randn(C_out, C_in) * 0.5
    s_style = torch.randn(B, C_in).abs() + 0.5
    demod_out = stylegan2_weight_demodulation(x_feat, w_conv, s_style)

    # 4. WGAN critic loss
    real_scores = torch.randn(B, 1) * 2.0 + 1.0
    fake_scores = torch.randn(B, 1) * 2.0 - 1.0
    wgan_loss = wgan_critic_loss(real_scores, fake_scores)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "x_adain": x_adain,
        "scale": scale,
        "bias_adain": bias_adain,
        "adain_out": adain_out,
        "x_feat": x_feat,
        "w_conv": w_conv,
        "s_style": s_style,
        "demod_out": demod_out,
        "real_scores": real_scores,
        "fake_scores": fake_scores,
        "wgan_loss": wgan_loss.unsqueeze(0),
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
