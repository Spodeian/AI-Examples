"""AI Watermarking AI Detection and Model Obfuscation — PyTorch 2.x reference implementation.

Topic: 11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation
Golden weights: weights/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.safetensors

``python pytorch2/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.py`` verifies against the golden checkpoint;
``python pytorch2/11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def watermark_logit_bias(
    logits: torch.Tensor,
    green_mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Kirchenbauer et al. (2023) green-list logit biasing: l' = l + delta * m_green."""
    return logits + float(delta) * green_mask.unsqueeze(0)


def watermark_z_score(
    green_counts: torch.Tensor,
    seq_lens: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """One-tailed z-score hypothesis test for green token frequency:
    
    z = (|S_G| - gamma * T) / sqrt(T * gamma * (1 - gamma) + eps)
    """
    expected = float(gamma) * seq_lens
    std = torch.sqrt(seq_lens * float(gamma) * (1.0 - float(gamma)) + 1e-8)
    return (green_counts - expected) / std


def fast_detect_gpt_score(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Fast-DetectGPT (Bao et al. 2023) 1-pass conditional likelihood curvature score:
    
    d = (1 / T) * sum_t (log p_model(x_t) - log p_ref(x_t))
    """
    return torch.mean(log_probs - ref_log_probs, dim=-1)


def passport_layer_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    passport_key: torch.Tensor,
    passport_weight: torch.Tensor,
) -> torch.Tensor:
    """Passport Layer (Fan et al. 2019) secret key gating:
    
    s = sigmoid(K @ W_pass)
    y = (x @ W + b) * s
    """
    scale = torch.sigmoid(passport_key @ passport_weight)
    base = x @ weight + bias
    return base * scale.unsqueeze(0)


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Kirchenbauer logit bias: B=4, V=8
    logits = torch.randn(4, 8)
    green_mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    watermarked_logits_ref = watermark_logit_bias(logits, green_mask, delta=2.5)

    # 2. Z-score detection: B=4
    green_counts = torch.tensor([85.0, 52.0, 92.0, 48.0])
    seq_lens = torch.tensor([100.0, 100.0, 100.0, 100.0])
    z_scores_ref = watermark_z_score(green_counts, seq_lens, gamma=0.5)

    # 3. Fast-DetectGPT curvature: B=4, T=16
    log_probs = torch.randn(4, 16) - 1.5
    ref_log_probs = torch.randn(4, 16) - 2.5
    detect_scores_ref = fast_detect_gpt_score(log_probs, ref_log_probs)

    # 4. Passport Layer: B=4, D=6, D_k=4
    pass_x = torch.randn(4, 6)
    pass_w = torch.randn(6, 6) * 0.2
    pass_b = torch.randn(6) * 0.1
    passport_key = torch.randn(4)
    passport_weight = torch.randn(4, 6) * 0.5
    passport_out_ref = passport_layer_forward(pass_x, pass_w, pass_b, passport_key, passport_weight)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "logits": logits,
        "green_mask": green_mask,
        "watermarked_logits_ref": watermarked_logits_ref,
        "green_counts": green_counts,
        "seq_lens": seq_lens,
        "z_scores_ref": z_scores_ref,
        "log_probs": log_probs,
        "ref_log_probs": ref_log_probs,
        "detect_scores_ref": detect_scores_ref,
        "pass_x": pass_x,
        "pass_w": pass_w,
        "pass_b": pass_b,
        "passport_key": passport_key,
        "passport_weight": passport_weight,
        "passport_out_ref": passport_out_ref,
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
