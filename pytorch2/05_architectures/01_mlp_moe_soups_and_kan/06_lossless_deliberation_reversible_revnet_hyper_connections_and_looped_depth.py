"""PyTorch 2.x implementation for Category 05 Topic 06:
Lossless Deliberation: Reversible RevNet, Hyper-Connections, and Recurrent Looped Depth
in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class ReversibleBlock(nn.Module):
    """Algebraically Reversible Deliberation Layer (RevNet / Reformer).
    
    Splits input into [x1, x2] and applies measure-preserving coupling:
    y1 = x1 + F(x2)
    y2 = x2 + G(y1)
    
    Exact analytical backward inversion without caching activations:
    x2 = y2 - G(y1)
    x1 = y1 - F(x2)
    
    References:
        - Gomez et al. (2017): The Reversible Residual Network: Backpropagation Without Storing Activations
        - Kitaev et al. (2020): Reformer: The Efficient Transformer
    """

    def __init__(self, d_sub: int, d_ff: int) -> None:
        super().__init__()
        self.d_sub = d_sub
        self.d_ff = d_ff

        # F block: d_sub -> d_ff -> d_sub
        self.f_fc1 = nn.Linear(d_sub, d_ff)
        self.f_fc2 = nn.Linear(d_ff, d_sub)

        # G block: d_sub -> d_ff -> d_sub
        self.g_fc1 = nn.Linear(d_sub, d_ff)
        self.g_fc2 = nn.Linear(d_ff, d_sub)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward coupling pass."""
        x1 = x[..., :self.d_sub]
        x2 = x[..., self.d_sub:]

        f_x2 = self.f_fc2(F.gelu(self.f_fc1(x2)))
        y1 = x1 + f_x2

        g_y1 = self.g_fc2(F.gelu(self.g_fc1(y1)))
        y2 = x2 + g_y1

        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Exact algebraic inverse reconstruction."""
        y1 = y[..., :self.d_sub]
        y2 = y[..., self.d_sub:]

        g_y1 = self.g_fc2(F.gelu(self.g_fc1(y1)))
        x2 = y2 - g_y1

        f_x2 = self.f_fc2(F.gelu(self.f_fc1(x2)))
        x1 = y1 - f_x2

        return torch.cat([x1, x2], dim=-1)


class HyperConnectionsBlock(nn.Module):
    """Multi-Lane Hyper-Connections Layer (HC).
    
    Expands residual stream into n parallel lanes H with learned depth-mixing:
    u = sum_i alpha_i * h_i
    v = FFN(u)
    h_i' = h_i + beta_i * v
    
    References:
        - Zhu et al. (2024): Hyper-Connections: Expanding Residual Streams
    """

    def __init__(self, d_model: int, d_ff: int, num_lanes: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_lanes = num_lanes

        self.alpha = nn.Parameter(torch.ones(num_lanes) / num_lanes)
        self.beta = nn.Parameter(torch.ones(num_lanes) / num_lanes)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass over multi-lane residual stream."""
        # Depth-mixing read
        u = self.alpha[0] * h1 + self.alpha[1] * h2
        v = self.fc2(F.gelu(self.fc1(u)))

        # Depth-mixing write
        h1_next = h1 + self.beta[0] * v
        h2_next = h2 + self.beta[1] * v
        combined = 0.5 * (h1_next + h2_next)

        return h1_next, h2_next, combined


class LoopedDepthBlock(nn.Module):
    """Input-Anchored Recurrent-Depth Deliberation (Looped Transformers).
    
    Iterates a shared block across recurrent depth with persistent input anchoring:
    z^{(k+1)} = LayerNorm(z^{(k)} + FFN(z^{(k)} + gamma * x))
    
    References:
        - Giannou et al. (2023): Looped Transformers as Programmable Computers
        - Geiping et al. (2025): Scaling Test-Time Compute with Recurrent Depth
    """

    def __init__(self, d_model: int, d_ff: int, steps: int = 4, gamma: float = 0.3) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.steps = steps
        self.gamma = gamma

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recurrent deliberation forward pass."""
        z = x.clone()
        for _ in range(self.steps):
            z_in = z + self.gamma * x
            delta = self.fc2(F.gelu(self.fc1(z_in)))
            z = F.layer_norm(z + delta, normalized_shape=(self.d_model,))
        return z


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Standard linear anchor for test_cross_track_parity.py."""
    return x @ weight + bias


def verify_against_golden() -> None:
    """Verifies PyTorch implementation against seed-42 SafeTensors checkpoint."""
    ckpt_path = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors")
    if not ckpt_path.exists():
        ckpt_path = Path("../../../" + str(ckpt_path))
    weights = load_file(str(ckpt_path))

    # 1. Linear anchor
    test_out = linear_anchor(weights["test_input"], weights["weight"], weights["bias"])
    err_linear = torch.max(torch.abs(test_out - weights["test_output"])).item()
    assert err_linear < 1e-6, f"Linear anchor parity failed: {err_linear}"

    x = weights["input"]
    d_model = 8
    d_sub = 4
    d_ff = 16

    # 2. Reversible RevNet verification
    rev = ReversibleBlock(d_sub=d_sub, d_ff=d_ff)
    with torch.no_grad():
        rev.f_fc1.weight.copy_(weights["f_w1"].t())
        rev.f_fc1.bias.copy_(weights["f_b1"])
        rev.f_fc2.weight.copy_(weights["f_w2"].t())
        rev.f_fc2.bias.copy_(weights["f_b2"])
        rev.g_fc1.weight.copy_(weights["g_w1"].t())
        rev.g_fc1.bias.copy_(weights["g_b1"])
        rev.g_fc2.weight.copy_(weights["g_w2"].t())
        rev.g_fc2.bias.copy_(weights["g_b2"])

    out_rev = rev(x)
    err_rev = torch.max(torch.abs(out_rev - weights["revnet_output"])).item()
    assert err_rev < 1e-6, f"RevNet parity error {err_rev:.2e} exceeds 1e-6"

    # Exact Reconstruction Check
    x_rec = rev.inverse(out_rev)
    err_rec = torch.max(torch.abs(x_rec - x)).item()
    assert err_rec < 1e-6, f"RevNet inverse reconstruction error {err_rec:.2e} exceeds 1e-6"

    # 3. Hyper-Connections verification
    hc = HyperConnectionsBlock(d_model=d_model, d_ff=d_ff, num_lanes=2)
    with torch.no_grad():
        hc.alpha.copy_(weights["alpha"])
        hc.beta.copy_(weights["beta"])
        hc.fc1.weight.copy_(weights["hc_w1"].t())
        hc.fc1.bias.copy_(weights["hc_b1"])
        hc.fc2.weight.copy_(weights["hc_w2"].t())
        hc.fc2.bias.copy_(weights["hc_b2"])

    h1 = x.clone()
    h2 = x * 0.5
    _, _, out_hc = hc(h1, h2)
    err_hc = torch.max(torch.abs(out_hc - weights["hc_output"])).item()
    assert err_hc < 1e-6, f"Hyper-Connections parity error {err_hc:.2e} exceeds 1e-6"

    # 4. Looped Recurrent Depth verification
    loop = LoopedDepthBlock(d_model=d_model, d_ff=d_ff, steps=4, gamma=float(weights["gamma"]))
    with torch.no_grad():
        loop.fc1.weight.copy_(weights["loop_w1"].t())
        loop.fc1.bias.copy_(weights["loop_b1"])
        loop.fc2.weight.copy_(weights["loop_w2"].t())
        loop.fc2.bias.copy_(weights["loop_b2"])

    out_loop = loop(x)
    err_loop = torch.max(torch.abs(out_loop - weights["looped_output"])).item()
    assert err_loop < 1e-6, f"Looped Depth parity error {err_loop:.2e} exceeds 1e-6"

    print("PyTorch 2.x Lossless Deliberation parity successfully verified against golden SafeTensors!")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    #!/usr/bin/env python3
    """Generate golden weights and inputs/outputs for Category 05 Topic 06:
    Lossless Deliberation: Reversible RevNet, Hyper-Connections, and Recurrent Looped Depth.
    """

    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import save_file


    def main():
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 4
        d_model = 8       # Total hidden dimension (split into d_sub = 4 for revnet)
        d_sub = d_model // 2
        d_ff = 16

        # Input tensor: (B, S, D)
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

        # ==============================================================
        # 1. Reversible RevNet Parameters (F and G blocks)
        # ==============================================================
        f_w1 = torch.randn(d_sub, d_ff, dtype=torch.float32) * 0.2
        f_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
        f_w2 = torch.randn(d_ff, d_sub, dtype=torch.float32) * 0.2
        f_b2 = torch.randn(d_sub, dtype=torch.float32) * 0.1

        g_w1 = torch.randn(d_sub, d_ff, dtype=torch.float32) * 0.2
        g_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
        g_w2 = torch.randn(d_ff, d_sub, dtype=torch.float32) * 0.2
        g_b2 = torch.randn(d_sub, dtype=torch.float32) * 0.1

        def eval_f(v):
            return F.gelu(v @ f_w1 + f_b1) @ f_w2 + f_b2

        def eval_g(v):
            return F.gelu(v @ g_w1 + g_b1) @ g_w2 + g_b2

        # RevNet Forward
        x1 = x[:, :, :d_sub]
        x2 = x[:, :, d_sub:]
        y1 = x1 + eval_f(x2)
        y2 = x2 + eval_g(y1)
        revnet_output = torch.cat([y1, y2], dim=-1)

        # Exact Reconstruction Check (Invariance)
        x2_rec = y2 - eval_g(y1)
        x1_rec = y1 - eval_f(x2_rec)
        rec_err = torch.max(torch.abs(torch.cat([x1_rec, x2_rec], dim=-1) - x)).item()
        assert rec_err < 1e-6, f"RevNet reconstruction failed: {rec_err}"

        # ==============================================================
        # 2. Hyper-Connections (HC) Parameters (2 residual lanes)
        # ==============================================================
        # Lane 1 and Lane 2 initial states
        h1 = x.clone()
        h2 = x * 0.5

        alpha = torch.tensor([0.6, 0.4], dtype=torch.float32)  # depth-mixing read
        beta = torch.tensor([0.5, 0.5], dtype=torch.float32)   # depth-mixing write

        hc_w1 = torch.randn(d_model, d_ff, dtype=torch.float32) * 0.2
        hc_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
        hc_w2 = torch.randn(d_ff, d_model, dtype=torch.float32) * 0.2
        hc_b2 = torch.randn(d_model, dtype=torch.float32) * 0.1

        # Read from lanes
        u_hc = alpha[0] * h1 + alpha[1] * h2
        v_hc = F.gelu(u_hc @ hc_w1 + hc_b1) @ hc_w2 + hc_b2

        # Write to lanes
        h1_out = h1 + beta[0] * v_hc
        h2_out = h2 + beta[1] * v_hc
        # Combined output: average of updated lanes
        hc_output = 0.5 * (h1_out + h2_out)

        # ==============================================================
        # 3. Looped Recurrent Depth with Input Anchoring
        # ==============================================================
        loop_steps = 4
        gamma = 0.3  # Persistent input injection strength

        loop_w1 = torch.randn(d_model, d_ff, dtype=torch.float32) * 0.2
        loop_b1 = torch.randn(d_ff, dtype=torch.float32) * 0.1
        loop_w2 = torch.randn(d_ff, d_model, dtype=torch.float32) * 0.2
        loop_b2 = torch.randn(d_model, dtype=torch.float32) * 0.1

        z = x.clone()
        for k in range(loop_steps):
            # Anchor with initial representation x
            z_in = z + gamma * x
            delta = F.gelu(z_in @ loop_w1 + loop_b1) @ loop_w2 + loop_b2
            z = F.layer_norm(z + delta, normalized_shape=(d_model,))
        looped_output = z

        # Standard linear anchor for cross-track parity verification
        D_core = 4
        test_input = torch.randn(8, D_core)
        weight = torch.randn(D_core, D_core)
        bias = torch.randn(D_core)
        test_output = test_input @ weight + bias

        # Output directory
        out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors"

        tensors = {
            "weight": weight.clone().contiguous(),
            "bias": bias.clone().contiguous(),
            "test_input": test_input.clone().contiguous(),
            "test_output": test_output.clone().contiguous(),
            "input": x,
            "f_w1": f_w1,
            "f_b1": f_b1,
            "f_w2": f_w2,
            "f_b2": f_b2,
            "g_w1": g_w1,
            "g_b1": g_b1,
            "g_w2": g_w2,
            "g_b2": g_b2,
            "revnet_output": revnet_output,
            "alpha": alpha,
            "beta": beta,
            "hc_w1": hc_w1,
            "hc_b1": hc_b1,
            "hc_w2": hc_w2,
            "hc_b2": hc_b2,
            "hc_output": hc_output,
            "loop_w1": loop_w1,
            "loop_b1": loop_b1,
            "loop_w2": loop_w2,
            "loop_b2": loop_b2,
            "gamma": torch.tensor(gamma, dtype=torch.float32),
            "looped_output": looped_output,
        }

        save_file(tensors, str(out_file))
        print(f"Generated {out_file} successfully.")
        print(f"revnet_output shape: {revnet_output.shape}, norm: {revnet_output.norm():.6f}")
        print(f"hc_output shape: {hc_output.shape}, norm: {hc_output.norm():.6f}")
        print(f"looped_output shape: {looped_output.shape}, norm: {looped_output.norm():.6f}")


    main()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_against_golden()
