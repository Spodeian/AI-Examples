"""Production PyTorch 2.x implementation for Topic 09:
09_nested_ragged_and_specialized_sparse_tensors
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float


class RaggedBatch(NamedTuple):
    """Encapsulates a variable-length ragged batch of token features."""

    values: Float[torch.Tensor, "num_tokens dim"]
    offsets: torch.Tensor  # 1D int64 tensor of length [batch_size + 1]


def ragged_to_padded(
    values: Float[torch.Tensor, "num_tokens dim"],
    offsets: torch.Tensor,
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts a flat ragged token tensor into a padded dense batch [B, max_len, D] and boolean mask."""
    num_seqs = offsets.shape[0] - 1
    seq_lens = [int((offsets[i + 1] - offsets[i]).item()) for i in range(num_seqs)]
    max_len = max(seq_lens)
    dim = values.shape[-1]

    padded = torch.full(
        (num_seqs, max_len, dim),
        fill_value=padding_value,
        dtype=values.dtype,
        device=values.device,
    )
    mask = torch.zeros((num_seqs, max_len), dtype=torch.bool, device=values.device)

    for i in range(num_seqs):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        length = end - start
        padded[i, :length] = values[start:end]
        mask[i, :length] = True

    return padded, mask


def padded_to_ragged(
    padded: Float[torch.Tensor, "batch max_len dim"],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs a padded dense batch back into contiguous ragged format with offsets."""
    num_seqs = padded.shape[0]
    lengths = mask.sum(dim=1).to(torch.int64)
    offsets = torch.zeros(num_seqs + 1, dtype=torch.int64, device=padded.device)
    offsets[1:] = torch.cumsum(lengths, dim=0)

    total_tokens = int(offsets[-1].item())
    dim = padded.shape[-1]
    values = torch.zeros((total_tokens, dim), dtype=padded.dtype, device=padded.device)

    for i in range(num_seqs):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        length = int(lengths[i].item())
        values[start:end] = padded[i, :length]

    return values, offsets


def ragged_mean_pool(
    values: Float[torch.Tensor, "num_tokens dim"],
    offsets: torch.Tensor,
) -> Float[torch.Tensor, "batch dim"]:
    """Computes per-sequence mean reduction across ragged token slices."""
    num_seqs = offsets.shape[0] - 1
    dim = values.shape[-1]
    pooled = torch.zeros((num_seqs, dim), dtype=values.dtype, device=values.device)

    for i in range(num_seqs):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        length = float(end - start)
        if length > 0:
            pooled[i] = torch.sum(values[start:end], dim=0) / length

    return pooled


def csr_spmv(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    x: torch.Tensor,
    num_rows: int,
    num_cols: int,
) -> torch.Tensor:
    """Evaluates Compressed Sparse Row matrix-vector/matrix product: Y = A @ X."""
    dim_out = x.shape[-1]
    y = torch.zeros((num_rows, dim_out), dtype=x.dtype, device=x.device)

    for r in range(num_rows):
        start = int(crow_indices[r].item())
        end = int(crow_indices[r + 1].item())
        row_sum = torch.zeros(dim_out, dtype=x.dtype, device=x.device)
        for idx in range(start, end):
            c = int(col_indices[idx].item())
            val = values[idx]
            row_sum += val * x[c]
        y[r] = row_sum

    return y


class RaggedLinear(nn.Module):
    """Linear projection layer tailored for variable-length ragged sequences."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / (self.in_features**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        values: Float[torch.Tensor, "num_tokens in_features"],
    ) -> Float[torch.Tensor, "num_tokens out_features"]:
        return values @ self.weight + self.bias

    def forward_pooled(
        self,
        values: Float[torch.Tensor, "num_tokens in_features"],
        offsets: torch.Tensor,
    ) -> Float[torch.Tensor, "batch out_features"]:
        tokens = self.forward(values)
        return ragged_mean_pool(tokens, offsets)


def create_ragged_linear(
    in_features: int,
    out_features: int,
) -> RaggedLinear:
    """Factory creating an initialized RaggedLinear module."""
    return RaggedLinear(in_features, out_features)
