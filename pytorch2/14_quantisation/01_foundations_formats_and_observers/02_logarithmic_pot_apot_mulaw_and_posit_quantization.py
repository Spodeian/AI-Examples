"""Logarithmic PoT APoT μ-Law and Posit Quantization — PyTorch 2.x reference implementation.

Topic: 14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization
Golden weights: weights/14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization.safetensors

``python pytorch2/14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/01_foundations_formats_and_observers/02_logarithmic_pot_apot_mulaw_and_posit_quantization"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def pot_quantize(x: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Single-term Power-of-Two (PoT) multiplier-free quantization.
    Weights are mapped to ±2^(-p), p in {0, 1, ..., 2^(b-1) - 2} or 0.
    """
    x_abs = torch.abs(x)
    max_p = float((1 << (num_bits - 1)) - 2)
    threshold = 2.0 ** (-max_p - 1.0)
    
    is_zero = (x_abs < threshold)
    x_safe = torch.where(is_zero, torch.ones_like(x_abs), x_abs)
    
    p = torch.clamp(torch.round(-torch.log2(x_safe)), 0.0, max_p)
    q = torch.sign(x) * (2.0 ** (-p))
    return torch.where(is_zero, torch.zeros_like(x), q)


def apot_quantize(x: torch.Tensor) -> torch.Tensor:
    """Additive Power-of-Two (APoT) 4-bit quantization (Li et al., 2020).
    Uses 2 terms from {0, 0.125, 0.25, 0.5} producing 8 levels in [0, 1]:
    [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0].
    """
    levels = torch.tensor([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0], device=x.device)
    x_abs = torch.clamp(torch.abs(x), 0.0, 1.0)
    
    # Broadcast distance across levels: (..., N, 8)
    diff = torch.abs(x_abs.unsqueeze(-1) - levels)
    idx = torch.argmin(diff, dim=-1)
    q_mag = levels[idx]
    
    return torch.sign(x) * q_mag


def mu_law_compand_expand(
    x: torch.Tensor,
    mu: float = 255.0,
    num_bits: int = 8,
) -> torch.Tensor:
    """ITU-T G.711 / WaveNet μ-law non-linear companding and expansion.
    F(x) = sign(x) * ln(1 + μ|x|) / ln(1 + μ) for x in [-1, 1].
    """
    x_clamped = torch.clamp(x, -1.0, 1.0)
    ln_1_mu = math.log(1.0 + float(mu))
    
    # 1. Compand
    y = torch.sign(x_clamped) * torch.log(1.0 + float(mu) * torch.abs(x_clamped)) / ln_1_mu
    
    # 2. Uniform Quantize
    qmax = float((1 << num_bits) - 1)
    y_norm = (y + 1.0) * 0.5  # in [0, 1]
    y_q = torch.round(y_norm * qmax) / qmax
    y_quant = y_q * 2.0 - 1.0  # back in [-1, 1]
    
    # 3. Expand (Inverse compand)
    x_hat = torch.sign(y_quant) * ((1.0 + float(mu)) ** torch.abs(y_quant) - 1.0) / float(mu)
    return x_hat


def base_sqrt2_log_quantize(x: torch.Tensor) -> torch.Tensor:
    """Base-sqrt(2) logarithmic quantization providing 2 bins per octave."""
    x_abs = torch.abs(x)
    threshold = 2.0 ** (-7.5)
    is_zero = (x_abs < threshold)
    x_safe = torch.where(is_zero, torch.ones_like(x_abs), x_abs)
    
    # k in {-14, ..., 0}
    k = torch.clamp(torch.round(2.0 * torch.log2(x_safe)), -14.0, 0.0)
    q = torch.sign(x) * (2.0 ** (k * 0.5))
    return torch.where(is_zero, torch.zeros_like(x), q)


def posit4_quantize(x: torch.Tensor) -> torch.Tensor:
    """4-bit Posit tapered quantization codebook (8 non-negative levels)."""
    levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0], device=x.device)
    x_abs = torch.clamp(torch.abs(x), 0.0, 4.0)
    
    diff = torch.abs(x_abs.unsqueeze(-1) - levels)
    idx = torch.argmin(diff, dim=-1)
    q_mag = levels[idx]
    return torch.sign(x) * q_mag


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. PoT Quantization
    raw_pot = torch.tensor([0.9, -0.4, 0.22, -0.06, 0.005, 0.0])
    pot_ref = pot_quantize(raw_pot, num_bits=4)

    # 2. APoT Quantization
    raw_apot = torch.tensor([0.95, -0.68, 0.32, -0.19, 0.05, 0.0])
    apot_ref = apot_quantize(raw_apot)

    # 3. μ-Law Compand-Expand
    raw_audio = torch.linspace(-0.95, 0.95, 16)
    mulaw_ref = mu_law_compand_expand(raw_audio, mu=255.0, num_bits=8)

    # 4. Base-sqrt(2) Log Quantization
    raw_sqrt2 = torch.tensor([0.9, -0.7, 0.45, -0.32, 0.16, 0.0])
    sqrt2_ref = base_sqrt2_log_quantize(raw_sqrt2)

    # 5. Posit4 Quantization
    raw_posit = torch.tensor([0.8, -1.4, 0.65, -0.2, 3.2, 0.0])
    posit4_ref = posit4_quantize(raw_posit)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "raw_pot": raw_pot,
        "pot_ref": pot_ref,
        "raw_apot": raw_apot,
        "apot_ref": apot_ref,
        "raw_audio": raw_audio,
        "mulaw_ref": mulaw_ref,
        "raw_sqrt2": raw_sqrt2,
        "sqrt2_ref": sqrt2_ref,
        "raw_posit": raw_posit,
        "posit4_ref": posit4_ref,
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
