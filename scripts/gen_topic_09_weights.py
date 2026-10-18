"""Generate deterministic reference safetensors checkpoint for Topic 09:
09_nested_ragged_and_specialized_sparse_tensors
"""

from __future__ import annotations

from pathlib import Path

import torch

import core


def generate_topic_09_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    # Ragged batch specification:
    # 3 sequences with lengths [2, 4, 3], total N = 9 tokens, feature dim = 4, out dim = 2
    seq_lens = [2, 4, 3]
    num_seqs = len(seq_lens)
    total_tokens = sum(seq_lens)
    dim = 4
    out_dim = 2

    # 1. Contiguous ragged token features (N_total, D)
    values = torch.randn(total_tokens, dim, generator=gen, dtype=torch.float32)

    # Offsets vector: [0, 2, 6, 9]
    offsets = torch.tensor([0, 2, 6, 9], dtype=torch.int64)

    # 2. Linear projection parameters
    weight = torch.randn(dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)

    # 3. Dense padded representation [B, max_len, D]
    max_len = max(seq_lens)
    dense_padded = torch.zeros(num_seqs, max_len, dim, dtype=torch.float32)
    mask = torch.zeros(num_seqs, max_len, dtype=torch.float32)
    for i, length in enumerate(seq_lens):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        dense_padded[i, :length] = values[start:end]
        mask[i, :length] = 1.0

    # 4. Token-level projection
    token_out = values @ weight + bias

    # 5. Sequence-level mean pooling via offsets
    pooled_out = torch.zeros(num_seqs, out_dim, dtype=torch.float32)
    for i in range(num_seqs):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        pooled_out[i] = torch.mean(token_out[start:end], dim=0)

    # 6. CSR Representation of the pooling matrix (M: [num_seqs, total_tokens])
    # Each row i has 1/L_i at column indices start..end
    csr_crow = torch.tensor([0, 2, 6, 9], dtype=torch.int64)
    csr_col = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
    csr_val = torch.tensor(
        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        dtype=torch.float32,
    )

    # Verify CSR SpMV matches pooled_out
    csr_mat = torch.sparse_csr_tensor(csr_crow, csr_col, csr_val, (num_seqs, total_tokens))
    csr_pooled = torch.matmul(csr_mat, token_out)
    torch.testing.assert_close(csr_pooled, pooled_out, atol=1e-6, rtol=1e-6)

    # 7. Differentiable backward pass for autograd parity
    v_param = values.clone().detach().requires_grad_(True)
    w_param = weight.clone().detach().requires_grad_(True)
    b_param = bias.clone().detach().requires_grad_(True)
    target = torch.randn(total_tokens, out_dim, generator=gen, dtype=torch.float32)

    pred = v_param @ w_param + b_param
    loss = 0.5 * torch.sum((pred - target) ** 2)
    loss.backward()

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "offsets": offsets,
        "dense_padded": dense_padded,
        "mask": mask,
        "pooled_out": pooled_out.detach(),
        "csr_crow": csr_crow,
        "csr_col": csr_col,
        "csr_val": csr_val,
        "target": target.detach(),
        "grad_weight": w_param.grad.clone().detach(),
        "grad_bias": b_param.grad.clone().detach(),
        "grad_input": v_param.grad.clone().detach(),
    }

    metadata = {
        "topic": "00_foundations/09_nested_ragged_and_specialized_sparse_tensors",
        "description": "Ragged jagged nested tensors, CSR SpMV pooling, and sparse tensor parity",
        "num_seqs": str(num_seqs),
        "total_tokens": str(total_tokens),
        "dim": str(dim),
        "out_dim": str(out_dim),
    }

    out_path = Path("weights/00_foundations/09_nested_ragged_and_specialized_sparse_tensors.safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core.parity.save_reference_checkpoint(
        path=out_path,
        weights_dict=weights_dict,
        input_tensor=values,
        output_tensor=token_out.detach(),
        metadata=metadata,
    )
    print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_topic_09_weights()
