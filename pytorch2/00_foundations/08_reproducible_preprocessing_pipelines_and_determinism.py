"""PyTorch 2.x Reproducible Pipelines, Deterministic Algorithms & Hash Partitioning.

Topic 08: 08_reproducible_preprocessing_pipelines_and_determinism (Category 00: Foundations)

Theoretical Overview:
Non-determinism in deep learning pipelines stems from three distinct sources:
1. Pseudorandom Number Generation (PRNG) Drift:
   Sharing global PRNG state across multithreaded DataLoader workers causes non-reproducible
   data orders. Deterministic pipelines encapsulate isolated torch.Generator instances.
2. Non-Associative Floating-Point Reductions:
   Because (a + b) + c != a + (b + c) in IEEE-754 floating point, nondeterministic thread scheduling
   in GPU atomicAdd operations produces divergent gradients across identical runs.
   PyTorch provides torch.use_deterministic_algorithms(True) to enforce deterministic kernels.
   For numerical reductions, Kahan Compensated Summation tracks running lost low-order bits:
       y = x_i - c
       t = sum + y
       c = (t - sum) - y
       sum = t
   reducing error accumulation from O(N * eps_mach) to O(eps_mach + N * eps_mach^2).
3. Dataset Partition Drift:
   Random permutation splitting (train_test_split) invalidates existing test splits when new data
   is ingested. Salted hash partitioning (hash(id) mod M) assigns samples deterministically based
   on immutable identifiers without global shuffle dependencies.
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


@contextmanager
def deterministic_context(seed: int = 42) -> Generator[None, None, None]:
    """Context manager enforcing strict bitwise reproducibility and deterministic algorithms."""
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark

    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.use_deterministic_algorithms(prev_deterministic)
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cudnn.benchmark = prev_cudnn_bench


def hash_deterministic_split(
    sample_id: str,
    salt: str = "ai-examples-v1",
    train_ratio: float = 0.8,
) -> str:
    """Deterministically assigns sample to 'train' or 'test' based on cryptographic row hash."""
    hasher = hashlib.sha256()
    hasher.update(f"{salt}:{sample_id}".encode())
    digest = hasher.hexdigest()
    # Normalize integer digest to [0.0, 1.0)
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if fraction < train_ratio else "test"


def kahan_sum(x: Float[Tensor, "batch dim"], dim: int = 0) -> Float[Tensor, "*dim"]:
    """Performs Kahan compensated summation along the specified dimension."""
    x_t = x.transpose(0, dim)
    s = torch.zeros_like(x_t[0])
    c = torch.zeros_like(s)

    for i in range(x_t.shape[0]):
        y = x_t[i] - c
        t = s + y
        c = (t - s) - y
        s = t

    return s


class ReproducibleLinearPipeline(nn.Module):
    """Deterministic linear pipeline layer with isolated generator state."""

    def __init__(self, in_features: int, out_features: int, seed: int = 42) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed

        gen = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features, generator=gen, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.randn(out_features, generator=gen, dtype=torch.float32)
        )

    def forward(self, x: Float[Tensor, "batch in_features"]) -> Float[Tensor, "batch out_features"]:
        return x @ self.weight + self.bias


if __name__ == "__main__":
    import core

    env = core.setup_env(seed=42)
    print(f"Testing PyTorch 2.x Determinism & Reproducible Pipelines on {env.device}...")

    ckpt_path = Path("weights/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.safetensors")
    weights, test_in, golden_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w_gold = weights["weight"]
    b_gold = weights["bias"]
    kahan_mean_gold = weights["kahan_mean"]
    target_gold = weights["target"]
    golden_gw = weights["grad_weight"]
    golden_gb = weights["grad_bias"]
    golden_gx = weights["grad_input"]

    dim = int(meta["dim"])
    out_dim = w_gold.shape[1]

    # 1. Deterministic Context & Kahan Summation Verification
    with deterministic_context(seed=42):
        k_sum = kahan_sum(test_in, dim=0)
        k_mean = k_sum / float(test_in.shape[0])
        torch.testing.assert_close(k_mean, kahan_mean_gold, atol=1e-6, rtol=1e-6)
    print("[PASS] Kahan compensated summation matches golden reference across float reductions.")

    # 2. Hash-Based Deterministic Partitioning Invariant
    splits_run1 = [hash_deterministic_split(f"row_{i}") for i in range(100)]
    splits_run2 = [hash_deterministic_split(f"row_{i}") for i in range(100)]
    assert splits_run1 == splits_run2, "Hash partition must be completely deterministic across executions"
    print("[PASS] Hash-based dataset partitioning verified invariant to execution environment.")

    # 3. Model Forward Parity
    model = ReproducibleLinearPipeline(in_features=dim, out_features=out_dim, seed=42)
    with torch.no_grad():
        model.weight.copy_(w_gold)
        model.bias.copy_(b_gold)

    y_pred = model(test_in)
    torch.testing.assert_close(y_pred, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] Reproducible pipeline forward output strictly matches golden reference (<1e-5).")

    # 4. Autograd Backward Gradient Parity
    x_param = test_in.clone().detach().requires_grad_(True)
    model.weight.grad = None
    model.bias.grad = None

    pred = model(x_param)
    loss = 0.5 * torch.sum((pred - target_gold) ** 2)
    loss.backward()

    torch.testing.assert_close(model.weight.grad, golden_gw, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.bias.grad, golden_gb, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_param.grad, golden_gx, atol=1e-5, rtol=1e-5)
    print("[PASS] Autograd backward gradient parity strictly verified (<1e-5).")

    # 5. Compilation & Export Verification
    ok_exp, msg_exp = core.export.verify_torch_export(model, test_in)
    print(f"torch.export: {msg_exp}")
    assert ok_exp

    ok_comp, msg_comp = core.export.verify_torch_compile_fullgraph(model, test_in)
    print(f"torch.compile: {msg_comp}")
    assert ok_comp

    print("[PASS] All Topic 08 PyTorch 2.x tests passed cleanly!")
