"""Numerical fuzzing and extreme precision boundary stress tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.metrics import relative_l2_error, sqnr_db


def test_subnormal_float_stability() -> None:
    """Verifies that subnormal/denormal inputs flush safely to zero without NaN propagation."""
    subnormal_tensor = torch.tensor([1e-39, 1e-42, -1e-40], dtype=torch.float32)
    linear = nn.Linear(3, 2, bias=True)

    with torch.no_grad():
        out = linear(subnormal_tensor)

    assert not torch.isnan(out).any(), "Subnormals generated NaN output"
    assert not torch.isinf(out).any(), "Subnormals generated Inf output"


def test_extreme_batch_dimensions() -> None:
    """Verifies operators execute stably on singleton (B=1) and large (B=1024) batch sizes."""
    layer = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.SiLU(), nn.Linear(16, 4))

    # Singleton batch
    out_1 = layer(torch.randn(1, 8))
    assert out_1.shape == (1, 4)
    assert not torch.isnan(out_1).any()

    # Large batch
    out_1024 = layer(torch.randn(1024, 8))
    assert out_1024.shape == (1024, 4)
    assert not torch.isnan(out_1024).any()


def test_nan_inf_isolation_in_attention_masking() -> None:
    """Verifies that large negative values (-1e9) in softmax attention masking cleanly isolate padded tokens."""
    scores = torch.randn(2, 4, 16, 16)
    # Mask out last 4 tokens
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[:, 12:] = True

    masked_scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
    attn_weights = torch.softmax(masked_scores, dim=-1)

    assert not torch.isnan(attn_weights).any(), "Attention masking generated NaNs in softmax"
    # Masked positions must be identically 0.0
    masked_out = attn_weights[:, :, :, 12:]
    assert torch.all(masked_out == 0.0), "Masked positions are non-zero after softmax"


def test_reduced_precision_simulation() -> None:
    """Verifies that simulating FP16 downcasting preserves numerical fidelity within acceptable SQNR."""
    weights = torch.randn(64, 64, dtype=torch.float32)
    weights_fp16 = weights.to(dtype=torch.float16).to(dtype=torch.float32)

    sqnr = sqnr_db(weights, weights_fp16)
    rel_err = relative_l2_error(weights, weights_fp16)

    # FP16 (10-bit mantissa) has ~60 dB SQNR and < 1e-3 relative error
    assert sqnr > 50.0, f"FP16 SQNR too low: {sqnr:.2f} dB"
    assert rel_err < 5e-3, f"FP16 relative error too high: {rel_err:.3e}"
