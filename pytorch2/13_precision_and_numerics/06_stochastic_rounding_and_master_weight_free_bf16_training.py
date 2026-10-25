"""Stochastic Rounding and Master Weight Free BF16 Training — PyTorch 2.x reference implementation.

Topic: 13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training
Golden weights: weights/13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training.safetensors

``python pytorch2/13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training.py`` verifies against the golden checkpoint;
``python pytorch2/13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "13_precision_and_numerics/06_stochastic_rounding_and_master_weight_free_bf16_training"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def stochastic_rounding_bf16(x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Simulates unbiased Stochastic Rounding (SR) to BF16 (7 mantissa bits).
    
    P(round up) = (x - floor(x)) / ULP(x).
    Expectation E[SR(x)] == x.
    """
    x_abs = torch.abs(x)
    is_zero = (x_abs == 0.0)
    x_safe = torch.where(is_zero, torch.ones_like(x_abs), x_abs)
    
    # BF16 exponent and ULP
    e = torch.floor(torch.log2(x_safe))
    ulp = 2.0 ** (e - 7.0)
    
    step_ratio = x_safe / ulp
    floor_ratio = torch.floor(step_ratio)
    frac = step_ratio - floor_ratio
    
    round_up = (noise < frac)
    q_mag = torch.where(round_up, (floor_ratio + 1.0) * ulp, floor_ratio * ulp)
    return torch.where(is_zero, torch.zeros_like(x), torch.sign(x) * q_mag)


def deterministic_rtne_bf16(x: torch.Tensor) -> torch.Tensor:
    """Deterministic Round-to-Nearest-Even (RTNE) to BF16."""
    x_abs = torch.abs(x)
    is_zero = (x_abs == 0.0)
    x_safe = torch.where(is_zero, torch.ones_like(x_abs), x_abs)
    
    e = torch.floor(torch.log2(x_safe))
    ulp = 2.0 ** (e - 7.0)
    
    step_ratio = x_safe / ulp
    q_mag = torch.round(step_ratio) * ulp
    return torch.where(is_zero, torch.zeros_like(x), torch.sign(x) * q_mag)


def master_weight_free_sgd_step(
    w_bf16: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    noise: torch.Tensor,
    use_stochastic_rounding: bool = True,
) -> torch.Tensor:
    """Executes single optimizer step directly on BF16 weights without FP32 master weights."""
    target_w = w_bf16 - float(lr) * grad
    if use_stochastic_rounding:
        return stochastic_rounding_bf16(target_w, noise)
    return deterministic_rtne_bf16(target_w)


def kahan_accumulate_step(
    accum: torch.Tensor,
    val: torch.Tensor,
    comp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kahan compensated summation step: recovers discarded low-order bits."""
    y = val - comp
    t = accum + y
    comp_new = (t - accum) - y
    accum_new = t
    return accum_new, comp_new


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. Stochastic Rounding BF16
    x_float = torch.tensor([1.002, -2.501, 0.00781, 10.125, 0.0, 3.1415926])
    noise = torch.tensor([0.2, 0.7, 0.1, 0.9, 0.4, 0.3])
    sr_out_ref = stochastic_rounding_bf16(x_float, noise)
    rtne_out_ref = deterministic_rtne_bf16(x_float)

    # 2. Master-weight-free SGD step
    w_init = torch.ones(6)
    grad_tiny = torch.tensor([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005])
    lr = 0.01
    sgd_sr_ref = master_weight_free_sgd_step(w_init, grad_tiny, lr=lr, noise=noise, use_stochastic_rounding=True)
    sgd_rtne_ref = master_weight_free_sgd_step(w_init, grad_tiny, lr=lr, noise=noise, use_stochastic_rounding=False)

    # 3. Kahan Accumulation
    k_accum = torch.tensor([1000.0, 500.0, 200.0])
    k_val = torch.tensor([0.00012, 0.00034, 0.00056])
    k_comp = torch.zeros(3)
    k_accum_ref, k_comp_ref = kahan_accumulate_step(k_accum, k_val, k_comp)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "x_float": x_float,
        "noise": noise,
        "sr_out_ref": sr_out_ref,
        "rtne_out_ref": rtne_out_ref,
        "w_init": w_init,
        "grad_tiny": grad_tiny,
        "sgd_sr_ref": sgd_sr_ref,
        "sgd_rtne_ref": sgd_rtne_ref,
        "k_accum": k_accum,
        "k_val": k_val,
        "k_comp": k_comp,
        "k_accum_ref": k_accum_ref,
        "k_comp_ref": k_comp_ref,
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
