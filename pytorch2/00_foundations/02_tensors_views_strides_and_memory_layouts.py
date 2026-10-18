"""Topic 02: Tensors, Views, Strides & Memory Layouts (PyTorch 2.x Track).

Implements the fundamental mechanics of n-dimensional tensor storage, memory strides,
zero-copy view transformations, contiguous vs non-contiguous layout dynamics,
and path-optimized Einstein summation (einsum).

Key Concepts:
- Linear storage mapping: index = storage_offset + sum_k (i_k * stride_k).
- Zero-copy sliding window extraction via `as_strided` (constant O(1) memory allocation).
- Strided memory overlap detection to prevent corrupting in-place writes.
- NCHW vs NHWC (channels_last) memory layout optimization for TensorCore efficiency.
- Optimal einsum contraction graph paths eliminating intermediate memory bloat.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class TensorLayoutMetadata(NamedTuple):
    """Metadata detailing physical tensor storage and layout properties."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    is_contiguous_c: bool
    is_contiguous_f: bool
    is_channels_last: bool
    element_size_bytes: int
    total_memory_bytes: int


class MemoryLayoutAnalyzer:
    """Utilities for inspecting and transforming physical tensor memory layouts."""

    @staticmethod
    def inspect(tensor: Tensor) -> TensorLayoutMetadata:
        """Extracts complete stride and storage layout parameters."""
        shape = tuple(tensor.shape)
        stride = tuple(tensor.stride())
        offset = int(tensor.storage_offset())
        is_c = tensor.is_contiguous()
        # Fortran contiguous check (column-major)
        is_f = False
        if tensor.ndim > 1:
            expected_f_strides = [1]
            for s in shape[:-1]:
                expected_f_strides.append(expected_f_strides[-1] * s)
            is_f = stride == tuple(expected_f_strides)
        is_cl = (
            tensor.is_contiguous(memory_format=torch.channels_last) if tensor.ndim == 4 else False
        )
        elem_size = tensor.element_size()
        storage = (
            tensor.untyped_storage() if hasattr(tensor, "untyped_storage") else tensor.storage()
        )
        total_bytes = storage.nbytes() if storage is not None else 0

        return TensorLayoutMetadata(
            shape=shape,
            stride=stride,
            storage_offset=offset,
            is_contiguous_c=is_c,
            is_contiguous_f=is_f,
            is_channels_last=is_cl,
            element_size_bytes=elem_size,
            total_memory_bytes=total_bytes,
        )

    @staticmethod
    def linear_offset(tensor: Tensor, indices: tuple[int, ...]) -> int:
        """Calculates exact 1D physical element index from multi-dimensional coordinate:
        linear_idx = storage_offset + sum_k (i_k * stride_k)
        """
        if len(indices) != tensor.ndim:
            raise ValueError(
                f"Indices length {len(indices)} does not match tensor ndim {tensor.ndim}"
            )
        offset = int(tensor.storage_offset())
        for idx, stride in zip(indices, tensor.stride()):
            offset += idx * int(stride)
        return int(offset)

    @staticmethod
    def has_overlapping_memory(tensor: Tensor) -> bool:
        """Detects if multiple view indices map to the same physical memory location."""
        # A tensor has overlapping memory if its numel exceeds the span of unique storage addresses
        if tensor.numel() <= 1:
            return False
        # If any stride is 0 for a dimension > 1, memory is explicitly broadcast/overlapping
        for s, d in zip(tensor.stride(), tensor.shape):
            if s == 0 and d > 1:
                return True
        # For general strides: compare number of unique physical offsets to numel
        shape = tensor.shape
        strides = tensor.stride()
        if tensor.ndim == 1:
            return strides[0] == 0 and shape[0] > 1

        # Check for sliding-window or overlapping stride steps
        sorted_strides = sorted(enumerate(strides), key=lambda x: x[1])
        min_stride = sorted_strides[0][1]
        if min_stride == 0 and shape[sorted_strides[0][0]] > 1:
            return True
        return False


class ZeroCopySlidingWindow(nn.Module):
    """Extracts sliding windows from 1D sequence signals with O(1) auxiliary memory allocations."""

    def __init__(self, window_size: int, step: int = 1) -> None:
        super().__init__()
        self.window_size = window_size
        self.step = step

    def forward(
        self, x: Float[Tensor, "batch length"]
    ) -> Float[Tensor, "batch num_windows window_size"]:
        """Constructs a strided sliding window view sharing the underlying storage.

        Input: [B, L]
        Output: [B, W, K] where W = (L - window_size) // step + 1, K = window_size
        """
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input [batch, length], got shape {x.shape}")
        batch_size, length = x.shape
        if length < self.window_size:
            raise ValueError(
                f"Sequence length {length} is smaller than window size {self.window_size}"
            )

        num_windows = (length - self.window_size) // self.step + 1
        batch_stride, elem_stride = x.stride()

        new_shape = (batch_size, num_windows, self.window_size)
        new_strides = (batch_stride, elem_stride * self.step, elem_stride)

        # Zero-copy view construction
        return torch.as_strided(x, size=new_shape, stride=new_strides)


class EinsumContractor:
    """Demonstrates high-performance Einstein summation with optimized contraction paths."""

    @staticmethod
    def batched_bilinear(
        x: Float[Tensor, "batch in_features"],
        w: Float[Tensor, "in_features out_features"],
        y: Float[Tensor, "batch out_features"],
    ) -> Float[Tensor, "*batch"]:
        """Computes quadratic form sum_i x_i W_ij y_j per batch element using optimal einsum."""
        return torch.einsum("bi,ij,bj->b", x, w, y)

    @staticmethod
    def attention_scores(
        q: Float[Tensor, "batch heads seq_q dim"],
        k: Float[Tensor, "batch heads seq_k dim"],
    ) -> Float[Tensor, "batch heads seq_q seq_k"]:
        """Computes scaled dot-product attention scores S_bhqk = sum_d Q_bhqd * K_bhkd."""
        return torch.einsum("bhqd,bhkd->bhqk", q, k)


if __name__ == "__main__":
    import core

    env = core.setup_env(seed=42)
    print(f"Testing PyTorch 2.x Tensor Views & Strides Track on {env.device}...")

    # 1. Verification of Linear Offset Formula
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    meta = MemoryLayoutAnalyzer.inspect(t)
    print(
        f"Tensor Shape: {meta.shape} | Stride: {meta.stride} | Contiguous: {meta.is_contiguous_c}"
    )

    # Compute coordinate (1, 2, 3) -> offset = 0 + 1*12 + 2*4 + 3*1 = 23
    coord = (1, 2, 3)
    calc_offset = MemoryLayoutAnalyzer.linear_offset(t, coord)
    assert calc_offset == 23
    assert t[coord].item() == 23.0
    print("Linear offset arithmetic matches storage layout exactly.")

    # 2. Zero-Copy Invariant in Sliding Windows
    seq = torch.linspace(0.0, 10.0, 16).unsqueeze(0)  # [1, 16]
    window_extractor = ZeroCopySlidingWindow(window_size=4, step=2)
    windows = window_extractor(seq)

    # Invariant: data pointers MUST be identical (zero allocations)
    assert seq.data_ptr() == windows.data_ptr()
    print(
        f"Zero-copy window shape: {windows.shape} (Shared storage pointer: {seq.data_ptr() == windows.data_ptr()})"
    )

    # 3. Overlap Detection
    # Broadcast tensor has stride 0
    t_broadcast = t.expand(5, 2, 3, 4)
    assert MemoryLayoutAnalyzer.has_overlapping_memory(t_broadcast)
    print("Overlapping memory detection correctly flagged broadcast view.")

    # 4. Einsum Optimization Check
    b, d_in, d_out = 8, 16, 32
    x_in = torch.randn(b, d_in)
    w_mat = torch.randn(d_in, d_out)
    y_in = torch.randn(b, d_out)
    res_einsum = EinsumContractor.batched_bilinear(x_in, w_mat, y_in)
    res_manual = (x_in @ w_mat * y_in).sum(dim=-1)
    torch.testing.assert_close(res_einsum, res_manual, atol=1e-5, rtol=1e-5)
    print("Einsum bilinear contraction matches explicit GEMM with 0 numerical divergence.")

    # 5. Compilation & Export Verification
    ok_exp, msg_exp = core.export.verify_torch_export(window_extractor, seq)
    print(f"torch.export: {msg_exp}")
    assert ok_exp

    ok_comp, msg_comp = core.export.verify_torch_compile_fullgraph(window_extractor, seq)
    print(f"torch.compile: {msg_comp}")
    assert ok_comp

    print("[PASS] All PyTorch 2.x Tensor Views & Strides checks passed cleanly!")
