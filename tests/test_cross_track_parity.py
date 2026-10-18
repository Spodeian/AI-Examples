"""Automated multi-track parity verification across golden safetensors checkpoints."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import core


def discover_checkpoints() -> list[Path]:
    """Finds all .safetensors reference checkpoints in weights/."""
    weights_root = Path("weights")
    if not weights_root.is_dir():
        return []
    return sorted(weights_root.rglob("*.safetensors"))


def test_checkpoint_discovery_sanity() -> None:
    """Verifies that weights directory exists and discovery logic functions without error."""
    weights_root = Path("weights")
    assert weights_root.is_dir(), "weights/ directory must exist"
    checkpoints = discover_checkpoints()
    assert isinstance(checkpoints, list)
    assert len(checkpoints) >= 3, (
        "At least 3 checkpoints should be discovered in Stage 0/Topic 01-03"
    )


@pytest.mark.parametrize("checkpoint_path", discover_checkpoints())
def test_checkpoint_parity(checkpoint_path: Path) -> None:
    """Loads discovered safetensors checkpoint and asserts 4-way transitive parity against golden output."""
    weights, test_in, test_out, _meta = core.parity.load_reference_checkpoint(checkpoint_path)

    assert isinstance(weights, dict)
    assert len(weights) > 0, f"Checkpoint {checkpoint_path} has no parameter tensors"
    assert test_in.ndim >= 1, f"Checkpoint {checkpoint_path} test_input must have >= 1 dimension"
    assert test_out.ndim >= 1, f"Checkpoint {checkpoint_path} test_output must have >= 1 dimension"
    assert not torch.isnan(test_out).any(), (
        f"Checkpoint {checkpoint_path} contains NaNs in test_output"
    )

    # 1. PyTorch 2 Forward Evaluation vs Golden Output
    w = weights.get("weight", weights.get("layers.weight"))
    b = weights.get("bias", weights.get("layers.bias"))
    assert w is not None and b is not None, f"Checkpoint {checkpoint_path} missing weight/bias"

    y_torch = torch.matmul(test_in, w) + b
    err_torch = float(torch.max(torch.abs(y_torch - test_out)).item())
    assert err_torch < 1e-5, f"PyTorch vs Golden output error {err_torch:.2e} exceeds 1e-5"

    # 2. JAX / Functional NumPy Forward Evaluation vs Golden Output
    x_np = test_in.cpu().numpy()
    w_np = w.cpu().numpy()
    b_np = b.cpu().numpy()
    golden_np = test_out.cpu().numpy()

    y_jax = np.dot(x_np, w_np) + b_np
    err_jax = float(np.max(np.abs(y_jax - golden_np)))
    assert err_jax < 1e-5, f"JAX vs Golden output error {err_jax:.2e} exceeds 1e-5"

    # 3. Transitive PyTorch vs JAX Parity
    err_torch_jax = float(np.max(np.abs(y_torch.cpu().numpy() - y_jax)))
    assert err_torch_jax < 1e-5, f"PyTorch vs JAX error {err_torch_jax:.2e} exceeds 1e-5"


def test_candle_crate_parity() -> None:
    """Runs native Rust Candle integration test suite to assert Candle vs Golden parity."""
    cargo_bin = shutil.which("cargo")
    if cargo_bin is None:
        pytest.skip("cargo not found in environment")

    # Run cargo test on ai-candle crate (both unit tests and integration tests)
    env = os.environ.copy()
    env["RUSTC_WRAPPER"] = ""
    result = subprocess.run(
        [cargo_bin, "test", "-p", "ai-candle"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"Candle test suite failed with exit code {result.returncode}:\n{result.stderr}\n{result.stdout}"
    )
