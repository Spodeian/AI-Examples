"""Unit tests verifying graph exportability and compile invariants."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.export import verify_onnx_export, verify_torch_compile_fullgraph, verify_torch_export


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc2(self.act(self.fc1(x)))
        return out


def test_torch_export_clean_graph() -> None:
    model = SimpleMLP()
    x = torch.randn(4, 16)
    ok, msg = verify_torch_export(model, x)
    assert ok, msg


def test_torch_compile_fullgraph_capture() -> None:
    model = SimpleMLP()
    x = torch.randn(4, 16)
    ok, msg = verify_torch_compile_fullgraph(model, x)
    assert ok, msg


def test_onnx_export_pipeline() -> None:
    model = SimpleMLP()
    x = torch.randn(2, 16)
    ok, msg = verify_onnx_export(model, x)
    assert ok, msg
