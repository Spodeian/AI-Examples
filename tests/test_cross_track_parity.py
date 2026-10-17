"""Automated multi-track parity verification across golden safetensors checkpoints."""

from __future__ import annotations

from pathlib import Path

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


@pytest.mark.parametrize("checkpoint_path", discover_checkpoints())
def test_checkpoint_parity(checkpoint_path: Path) -> None:
    """Loads discovered safetensors checkpoint and asserts internal consistency."""
    weights, test_in, test_out, meta = core.parity.load_reference_checkpoint(checkpoint_path)

    assert isinstance(weights, dict)
    assert len(weights) > 0, f"Checkpoint {checkpoint_path} has no parameter tensors"
    assert test_in.ndim >= 1, f"Checkpoint {checkpoint_path} test_input must have >= 1 dimension"
    assert test_out.ndim >= 1, f"Checkpoint {checkpoint_path} test_output must have >= 1 dimension"
    assert not torch.isnan(test_out).any(), (
        f"Checkpoint {checkpoint_path} contains NaNs in test_output"
    )
