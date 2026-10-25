"""Floating Point Anatomy Subnormal Traps and Stochastic Logic — PyTorch 2.x reference implementation.

Topic: 13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic
Golden weights: weights/13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic.safetensors

``python pytorch2/13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic.py`` verifies against the golden checkpoint;
``python pytorch2/13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "13_precision_and_numerics/01_floating_point_anatomy_subnormal_traps_and_stochastic_logic"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def ftz_daz_simulate(x: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    """Simulates hardware Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ)."""
    return torch.where(torch.abs(x) < float(threshold), torch.zeros_like(x), x)


def fp32_to_bf16_rtne(x: torch.Tensor) -> torch.Tensor:
    """Simulates round-to-nearest-even conversion from FP32 to BF16."""
    return x.bfloat16().float()


def stochastic_unipolar_mul(
    p1: torch.Tensor,
    p2: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
) -> torch.Tensor:
    """Stochastic computing unipolar probability multiplication via bitwise AND:
    p1, p2: (B,) probabilities in [0, 1]
    u1, u2: (B, L) quasi-random sequences in [0, 1]
    b1 = (u1 < p1), b2 = (u2 < p2)
    b_out = b1 & b2
    p_hat = mean(b_out, dim=-1)
    """
    b1 = (u1 < p1.unsqueeze(1)).float()
    b2 = (u2 < p2.unsqueeze(1)).float()
    b_mul = b1 * b2
    return torch.mean(b_mul, dim=-1)


def stochastic_bipolar_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
) -> torch.Tensor:
    """Stochastic computing bipolar value multiplication via bitwise XNOR:
    x1, x2: (B,) values in [-1, 1]
    p = (x + 1) / 2 in [0, 1]
    b_xnor = b1 * b2 + (1 - b1) * (1 - b2)
    x_hat = 2 * mean(b_xnor) - 1
    """
    p1 = (x1 + 1.0) * 0.5
    p2 = (x2 + 1.0) * 0.5
    b1 = (u1 < p1.unsqueeze(1)).float()
    b2 = (u2 < p2.unsqueeze(1)).float()
    b_xnor = b1 * b2 + (1.0 - b1) * (1.0 - b2)
    p_out = torch.mean(b_xnor, dim=-1)
    return 2.0 * p_out - 1.0


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. FTZ / DAZ Simulation
    raw_vals = torch.tensor([-2.5, 0.00005, -0.00002, 1.2, 0.00008, -3.1, 0.000001, 0.5])
    ftz_ref = ftz_daz_simulate(raw_vals, threshold=1e-4)

    # 2. BF16 RTNE Truncation
    bf16_in = torch.randn(6, 4) * 5.0
    bf16_ref = fp32_to_bf16_rtne(bf16_in)

    # 3. Stochastic Computing Bitstreams: Batch=4, Length=256
    p1 = torch.tensor([0.2, 0.5, 0.8, 0.35])
    p2 = torch.tensor([0.6, 0.4, 0.5, 0.7])
    u1 = torch.rand(4, 256)
    u2 = torch.rand(4, 256)
    unipolar_ref = stochastic_unipolar_mul(p1, p2, u1, u2)

    # 4. Bipolar Stochastic Computing: Batch=4, Length=256
    x1 = torch.tensor([-0.6, 0.2, 0.8, -0.4])
    x2 = torch.tensor([0.5, -0.5, -0.2, 0.75])
    bipolar_ref = stochastic_bipolar_mul(x1, x2, u1, u2)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "raw_vals": raw_vals,
        "ftz_ref": ftz_ref,
        "bf16_in": bf16_in,
        "bf16_ref": bf16_ref,
        "p1": p1,
        "p2": p2,
        "u1": u1,
        "u2": u2,
        "unipolar_ref": unipolar_ref,
        "x1": x1,
        "x2": x2,
        "bipolar_ref": bipolar_ref,
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
    if "--export-golden" in sys.argv:
        export_golden()
    else:
        verify_against_golden()
