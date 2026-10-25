"""Positional and Structural Encodings: Sinusoidal, RoPE, ALiBi, and RWPE (C03 Topic 04).

Implements fundamental sequence and graph positional encoding schemes:
1. Sinusoidal Absolute Positional Encoding: Harmonic frequency tables (Vaswani et al. 2017).
2. Rotary Position Embedding (RoPE): Orthogonal 2D pair rotations preserving relative distances in attention dot-products (Su et al. 2021).
3. Attention with Linear Biases (ALiBi): Static geometric penalties on query-key attention logits (Press et al. 2021).
4. Random Walk Positional Encoding (RWPE): Landing probabilities on graph random walks invariant to Laplacian sign ambiguity (Dwivedi et al. 2022).
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from safetensors.torch import load_file


class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani et al. (2017) Sinusoidal Positional Encoding."""

    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, x: Float[torch.Tensor, "seq_len dim"]
    ) -> Float[torch.Tensor, "seq_len dim"]:
        seq_len = x.shape[0]
        return x + self.pe[:seq_len]


class RotaryPositionEmbedding(nn.Module):
    """Su et al. (2021) Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        x: Float[torch.Tensor, "seq_len dim"],
        cos: Float[torch.Tensor, "seq_len half_dim"],
        sin: Float[torch.Tensor, "seq_len half_dim"],
    ) -> Float[torch.Tensor, "seq_len dim"]:
        x1 = x[:, 0::2]
        x2 = x[:, 1::2]
        out = torch.zeros_like(x)
        out[:, 0::2] = x1 * cos - x2 * sin
        out[:, 1::2] = x1 * sin + x2 * cos
        return out


class ALiBi(nn.Module):
    """Press et al. (2021) Attention with Linear Biases."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor(
            [2.0 ** (-8.0 * h / num_heads) for h in range(1, num_heads + 1)],
            dtype=torch.float32,
        )
        self.register_buffer("slopes", slopes)

    def forward(self, seq_len: int) -> Float[torch.Tensor, "num_heads seq_len seq_len"]:
        idx = torch.arange(seq_len, dtype=torch.float32, device=self.slopes.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        return -self.slopes.view(self.num_heads, 1, 1) * dist.unsqueeze(0)


def compute_rwpe(
    adj: Float[torch.Tensor, "num_nodes num_nodes"],
    k_steps: int = 4,
) -> Float[torch.Tensor, "num_nodes k_steps"]:
    """Dwivedi et al. (2022) Random Walk Positional Encoding (RWPE)."""
    deg = adj.sum(dim=1)
    inv_deg = torch.diag(1.0 / deg)
    p_walk = torch.matmul(inv_deg, adj)

    num_nodes = adj.shape[0]
    rwpe = torch.zeros(num_nodes, k_steps, dtype=torch.float32, device=adj.device)
    p_curr = torch.eye(num_nodes, dtype=torch.float32, device=adj.device)
    for k in range(k_steps):
        p_curr = torch.matmul(p_curr, p_walk)
        rwpe[:, k] = torch.diagonal(p_curr)
    return rwpe


class PositionalEncodingsUnifiedModule(nn.Module):
    """Unified module combining positional encodings with a linear anchor."""

    def __init__(
        self,
        dim: int = 8,
        out_dim: int = 4,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalPositionalEncoding(dim, max_len=64)
        self.rope = RotaryPositionEmbedding(dim)
        self.alibi = ALiBi(num_heads)
        self.linear = nn.Linear(dim, out_dim)

    def forward(
        self, x: Float[torch.Tensor, "seq_len dim"]
    ) -> Float[torch.Tensor, "seq_len out_dim"]:
        return self.linear(x)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[PositionalEncodingsUnifiedModule, Dict[str, torch.Tensor]]:
    weights = load_file(checkpoint_path)
    module = PositionalEncodingsUnifiedModule(dim=8, out_dim=4, num_heads=2)

    with torch.no_grad():
        module.linear.weight.copy_(weights["weight"].t())
        module.linear.bias.copy_(weights["bias"])
        module.alibi.slopes.copy_(weights["alibi_slopes"])

    return module, weights


def main() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/04_positional_and_structural_encodings.safetensors")
    if not weights_path.exists():
        print(f"Skipping: {weights_path} not found.")
        return

    module, weights = load_checkpoint(str(weights_path))
    module.eval()

    test_in = weights["test_input"]
    cos_rope = weights["cos_rope"]
    sin_rope = weights["sin_rope"]
    graph_adj = weights["graph_adj"]

    with torch.no_grad():
        # 1. Linear Anchor Parity
        y_anchor = module(test_in)
        diff_anchor = (y_anchor - weights["test_output"]).abs().max().item()
        assert diff_anchor < 1e-5, f"Anchor mismatch: {diff_anchor:.2e}"

        # 2. Sinusoidal PE Parity
        y_sin = module.sinusoidal(test_in)
        diff_sin = (y_sin - weights["y_sinusoidal"]).abs().max().item()
        assert diff_sin < 1e-5, f"Sinusoidal PE mismatch: {diff_sin:.2e}"

        # 3. RoPE Parity
        y_rp = module.rope(test_in, cos_rope, sin_rope)
        diff_rp = (y_rp - weights["y_rope"]).abs().max().item()
        assert diff_rp < 1e-5, f"RoPE mismatch: {diff_rp:.2e}"

        # 4. ALiBi Parity
        y_alibi = module.alibi(seq_len=test_in.shape[0])
        diff_alibi = (y_alibi - weights["alibi_bias"]).abs().max().item()
        assert diff_alibi < 1e-5, f"ALiBi mismatch: {diff_alibi:.2e}"

        # 5. RWPE Parity
        y_rw = compute_rwpe(graph_adj, k_steps=4)
        diff_rw = (y_rw - weights["y_rwpe"]).abs().max().item()
        assert diff_rw < 1e-5, f"RWPE mismatch: {diff_rw:.2e}"

    print(
        f"All PyTorch assertions passed successfully (Anchor: {diff_anchor:.2e}, "
        f"Sin: {diff_sin:.2e}, RoPE: {diff_rp:.2e}, ALiBi: {diff_alibi:.2e}, RWPE: {diff_rw:.2e})"
    )


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    from safetensors.torch import load_file, save_file
    torch.manual_seed(42)
    ckpt = Path("weights/03_building_blocks_and_losses/04_positional_and_structural_encodings.safetensors")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(ckpt)) if ckpt.is_file() else {}
    if "test_input" not in tensors:
        tensors["test_input"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["weight"] = torch.randn(4, 4, dtype=torch.float32)
        tensors["bias"] = torch.randn(4, dtype=torch.float32)
        tensors["test_output"] = tensors["test_input"] @ tensors["weight"] + tensors["bias"]
    tensors = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(tensors, str(ckpt), metadata={"format": "ai-examples-v1", "seed": "42", "topic": "03_building_blocks_and_losses/04_positional_and_structural_encodings"})
    print(f"Exported golden checkpoint to {ckpt}")

if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    main()
