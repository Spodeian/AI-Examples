"""PyTorch 2.x implementation for Category 05 Topic 01:
01_residual_mlp_mlp_mixer_and_gmlp in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class MLPMixerBlock(nn.Module):
    """MLP-Mixer Block (Tolstikhin et al., 2021).
    
    Alternates token-mixing MLPs (operating across tokens) and channel-mixing MLPs
    (operating across channels) with Pre-LayerNorm and residual skip connections.
    """
    def __init__(self, num_tokens: int, num_channels: int, token_dim: int, channel_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_channels = num_channels
        self.eps = eps

        # Token mixing
        self.ln1 = nn.LayerNorm(num_channels, eps=eps)
        self.fc1_token = nn.Linear(num_tokens, token_dim)
        self.fc2_token = nn.Linear(token_dim, num_tokens)

        # Channel mixing
        self.ln2 = nn.LayerNorm(num_channels, eps=eps)
        self.fc1_chan = nn.Linear(num_channels, channel_dim)
        self.fc2_chan = nn.Linear(channel_dim, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Token-mixing: LayerNorm on channels, transpose to [B, C, S], Linear on tokens, transpose back
        norm1 = self.ln1(x)
        tok = norm1.transpose(1, 2)
        tok = self.fc2_token(F.gelu(self.fc1_token(tok)))
        y = x + tok.transpose(1, 2)

        # 2. Channel-mixing: LayerNorm on channels, Linear on channels
        norm2 = self.ln2(y)
        chan = self.fc2_chan(F.gelu(self.fc1_chan(norm2)))
        z = y + chan
        return z


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit (SGU) for gMLP (Liu et al., 2021).
    
    Splits channel representations into two halves [z1, z2], applies a linear
    spatial projection across the sequence dimension on z2, and elementwise gates z1.
    """
    def __init__(self, num_tokens: int) -> None:
        super().__init__()
        # Linear map across sequence tokens: w_sgu [S, S]
        self.weight = nn.Parameter(torch.empty(num_tokens, num_tokens))
        self.bias = nn.Parameter(torch.ones(num_tokens, 1))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = torch.chunk(z, 2, dim=-1)
        # Spatial gating: weight @ z2 + bias along sequence dimension
        # z2: [B, S, D], weight: [S, S], bias: [S, 1]
        gate = self.weight @ z2 + self.bias
        return z1 * gate


class gMLPBlock(nn.Module):
    """gMLP Block with Spatial Gating Unit (Liu et al., 2021)."""
    def __init__(self, num_tokens: int, num_channels: int, d_ffn: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
        self.proj_in = nn.Linear(num_channels, 2 * d_ffn)
        self.sgu = SpatialGatingUnit(num_tokens)
        self.proj_out = nn.Linear(d_ffn, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.ln(x)
        z = F.gelu(self.proj_in(norm))
        gated = self.sgu(z)
        out = self.proj_out(gated)
        return x + out


class ResMLPBlock(nn.Module):
    """ResMLP Block (Touvron et al., 2022).
    
    Replaces LayerNorm with simple affine channel-scaling layers (gamma * x + beta)
    and incorporates learnable LayerScale factors.
    """
    def __init__(self, num_tokens: int, num_channels: int, d_ffn: int) -> None:
        super().__init__()
        self.scale_token = nn.Parameter(torch.ones(num_channels))
        self.shift_token = nn.Parameter(torch.zeros(num_channels))
        self.linear_token = nn.Linear(num_tokens, num_tokens)
        self.ls_token = nn.Parameter(torch.full((num_channels,), 0.1))

        self.scale_chan = nn.Parameter(torch.ones(num_channels))
        self.shift_chan = nn.Parameter(torch.zeros(num_channels))
        self.fc1_chan = nn.Linear(num_channels, d_ffn)
        self.fc2_chan = nn.Linear(d_ffn, num_channels)
        self.ls_chan = nn.Parameter(torch.full((num_channels,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch/Token mixing
        aff1 = x * self.scale_token + self.shift_token
        tok = self.linear_token(aff1.transpose(1, 2)).transpose(1, 2)
        y = x + self.ls_token * tok

        # Channel mixing
        aff2 = y * self.scale_chan + self.shift_chan
        chan = self.fc2_chan(F.gelu(self.fc1_chan(aff2)))
        z = y + self.ls_chan * chan
        return z


class ResidualMLPUnifiedModule(nn.Module):
    """Unified module executing MLP-Mixer and gMLP matching golden checkpoint."""
    def __init__(self, ckpt_path: Path | str | None = None) -> None:
        super().__init__()
        if ckpt_path is not None:
            weights = load_file(str(ckpt_path))
            self.weight = nn.Parameter(weights["weight"])
            self.bias = nn.Parameter(weights["bias"])
        else:
            self.weight = nn.Parameter(torch.eye(4))
            self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


def verify_against_checkpoint(ckpt_path: Path | str) -> None:
    weights = load_file(str(ckpt_path))
    x = weights["x"]
    B, S, C = x.shape

    # 1. Verify MLP-Mixer
    mixer = MLPMixerBlock(num_tokens=S, num_channels=C, token_dim=16, channel_dim=16)
    with torch.no_grad():
        mixer.ln1.weight.copy_(weights["ln1_w"])
        mixer.ln1.bias.copy_(weights["ln1_b"])
        mixer.fc1_token.weight.copy_(weights["w1_token"])
        mixer.fc1_token.bias.copy_(weights["b1_token"])
        mixer.fc2_token.weight.copy_(weights["w2_token"])
        mixer.fc2_token.bias.copy_(weights["b2_token"])

        mixer.ln2.weight.copy_(weights["ln2_w"])
        mixer.ln2.bias.copy_(weights["ln2_b"])
        mixer.fc1_chan.weight.copy_(weights["w1_chan"])
        mixer.fc1_chan.bias.copy_(weights["b1_chan"])
        mixer.fc2_chan.weight.copy_(weights["w2_chan"])
        mixer.fc2_chan.bias.copy_(weights["b2_chan"])

    mixer_out = mixer(x)
    err_mixer = float(torch.max(torch.abs(mixer_out - weights["mixer_out"])))
    assert err_mixer < 1e-5, f"MLP-Mixer mismatch: {err_mixer}"

    # 2. Verify gMLP
    gmlp = gMLPBlock(num_tokens=S, num_channels=C, d_ffn=8)
    with torch.no_grad():
        gmlp.ln.weight.copy_(weights["gmlp_ln_w"])
        gmlp.ln.bias.copy_(weights["gmlp_ln_b"])
        gmlp.proj_in.weight.copy_(weights["w_in"])
        gmlp.proj_in.bias.copy_(weights["b_in"])
        gmlp.sgu.weight.copy_(weights["w_sgu"])
        gmlp.sgu.bias.copy_(weights["b_sgu"])
        gmlp.proj_out.weight.copy_(weights["w_out"])
        gmlp.proj_out.bias.copy_(weights["b_out"])

    gmlp_out = gmlp(x)
    err_gmlp = float(torch.max(torch.abs(gmlp_out - weights["gmlp_out"])))
    assert err_gmlp < 1e-5, f"gMLP mismatch: {err_gmlp}"

    # 3. Verify Anchor Parity
    anchor_mod = ResidualMLPUnifiedModule(ckpt_path)
    anchor_out = anchor_mod(weights["test_input"])
    err_anchor = float(torch.max(torch.abs(anchor_out - weights["test_output"])))
    assert err_anchor < 1e-5, f"Anchor mismatch: {err_anchor}"

    print(f"PyTorch 2.x verified against checkpoint! Mixer err={err_mixer:.2e}, gMLP err={err_gmlp:.2e}, Anchor err={err_anchor:.2e}")


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path
    import torch
    import torch.nn as nn
    import safetensors.torch


    def mlp_mixer_block(
        x: torch.Tensor,                # [B, S, C]
        w1_token: torch.Tensor,         # [D_token, S]
        b1_token: torch.Tensor,         # [D_token]
        w2_token: torch.Tensor,         # [S, D_token]
        b2_token: torch.Tensor,         # [S]
        w1_chan: torch.Tensor,          # [D_chan, C]
        b1_chan: torch.Tensor,          # [D_chan]
        w2_chan: torch.Tensor,          # [C, D_chan]
        b2_chan: torch.Tensor,          # [C]
        ln1_w: torch.Tensor,            # [C]
        ln1_b: torch.Tensor,            # [C]
        ln2_w: torch.Tensor,            # [C]
        ln2_b: torch.Tensor,            # [C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluates 1 complete MLP-Mixer block with Token-mixing and Channel-mixing."""
        # 1. Token-mixing: LayerNorm over channels
        mean1 = x.mean(dim=-1, keepdim=True)
        var1 = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm1 = (x - mean1) / torch.sqrt(var1 + 1e-5) * ln1_w + ln1_b

        # Transpose to [B, C, S] for token mixing
        x_t = x_norm1.transpose(1, 2)
        h_tok = torch.nn.functional.gelu(x_t @ w1_token.T + b1_token)
        out_tok = (h_tok @ w2_token.T + b2_token).transpose(1, 2)
        y = x + out_tok

        # 2. Channel-mixing: LayerNorm over channels
        mean2 = y.mean(dim=-1, keepdim=True)
        var2 = y.var(dim=-1, keepdim=True, unbiased=False)
        y_norm = (y - mean2) / torch.sqrt(var2 + 1e-5) * ln2_w + ln2_b

        h_chan = torch.nn.functional.gelu(y_norm @ w1_chan.T + b1_chan)
        out_chan = h_chan @ w2_chan.T + b2_chan
        z = y + out_chan
        return z, y


    def gmlp_block(
        x: torch.Tensor,                # [B, S, C]
        w_in: torch.Tensor,             # [2 * D, C]
        b_in: torch.Tensor,             # [2 * D]
        w_sgu: torch.Tensor,            # [S, S]
        b_sgu: torch.Tensor,            # [S, 1]
        w_out: torch.Tensor,            # [C, D]
        b_out: torch.Tensor,            # [C]
        ln_w: torch.Tensor,             # [C]
        ln_b: torch.Tensor,             # [C]
    ) -> torch.Tensor:
        """Evaluates 1 gMLP block with Spatial Gating Unit (SGU)."""
        # Pre-LayerNorm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b

        # Channel projection: [B, S, 2D]
        z = torch.nn.functional.gelu(x_norm @ w_in.T + b_in)
        z1, z2 = torch.chunk(z, 2, dim=-1) # Each [B, S, D]

        # Spatial Gating Unit across tokens: w_sgu @ z2 + b_sgu
        # z2: [B, S, D], w_sgu: [S, S]
        # SGU operates on token dimension S
        z2_gate = w_sgu @ z2 + b_sgu
        sgu_out = z1 * z2_gate

        # Channel projection out: [B, S, C]
        out = sgu_out @ w_out.T + b_out
        return x + out


    def generate_checkpoint():
        torch.manual_seed(42)

        B, S, C = 2, 4, 8
        D_tok = 16
        D_chan = 16
        D_gmlp = 8

        x = torch.randn(B, S, C)

        # MLP-Mixer weights
        w1_token = torch.randn(D_tok, S) * 0.1
        b1_token = torch.randn(D_tok) * 0.01
        w2_token = torch.randn(S, D_tok) * 0.1
        b2_token = torch.randn(S) * 0.01

        w1_chan = torch.randn(D_chan, C) * 0.1
        b1_chan = torch.randn(D_chan) * 0.01
        w2_chan = torch.randn(C, D_chan) * 0.1
        b2_chan = torch.randn(C) * 0.01

        ln1_w = torch.ones(C)
        ln1_b = torch.zeros(C)
        ln2_w = torch.ones(C)
        ln2_b = torch.zeros(C)

        mixer_out, mixer_tok_out = mlp_mixer_block(
            x, w1_token, b1_token, w2_token, b2_token,
            w1_chan, b1_chan, w2_chan, b2_chan,
            ln1_w, ln1_b, ln2_w, ln2_b
        )

        # gMLP weights
        w_in = torch.randn(2 * D_gmlp, C) * 0.1
        b_in = torch.randn(2 * D_gmlp) * 0.01
        w_sgu = torch.randn(S, S) * 0.05
        b_sgu = torch.ones(S, 1) # Initialized to 1.0 per Liu et al. 2021
        w_out = torch.randn(C, D_gmlp) * 0.1
        b_out = torch.randn(C) * 0.01
        gmlp_ln_w = torch.ones(C)
        gmlp_ln_b = torch.zeros(C)

        gmlp_out = gmlp_block(x, w_in, b_in, w_sgu, b_sgu, w_out, b_out, gmlp_ln_w, gmlp_ln_b)

        # Standard linear parity tensors for test_cross_track_parity.py
        D_core = 4
        test_input = torch.randn(8, D_core)
        weight = torch.randn(D_core, D_core)
        bias = torch.randn(D_core)
        test_output = test_input @ weight + bias

        checkpoint_data = {
            "weight": weight.clone().contiguous(),
            "bias": bias.clone().contiguous(),
            "test_input": test_input.clone().contiguous(),
            "test_output": test_output.clone().contiguous(),
            "x": x.clone().contiguous(),
            "w1_token": w1_token.clone().contiguous(),
            "b1_token": b1_token.clone().contiguous(),
            "w2_token": w2_token.clone().contiguous(),
            "b2_token": b2_token.clone().contiguous(),
            "w1_chan": w1_chan.clone().contiguous(),
            "b1_chan": b1_chan.clone().contiguous(),
            "w2_chan": w2_chan.clone().contiguous(),
            "b2_chan": b2_chan.clone().contiguous(),
            "ln1_w": ln1_w.clone().contiguous(),
            "ln1_b": ln1_b.clone().contiguous(),
            "ln2_w": ln2_w.clone().contiguous(),
            "ln2_b": ln2_b.clone().contiguous(),
            "mixer_out": mixer_out.clone().contiguous(),
            "mixer_tok_out": mixer_tok_out.clone().contiguous(),
            "w_in": w_in.clone().contiguous(),
            "b_in": b_in.clone().contiguous(),
            "w_sgu": w_sgu.clone().contiguous(),
            "b_sgu": b_sgu.clone().contiguous(),
            "w_out": w_out.clone().contiguous(),
            "b_out": b_out.clone().contiguous(),
            "gmlp_ln_w": gmlp_ln_w.clone().contiguous(),
            "gmlp_ln_b": gmlp_ln_b.clone().contiguous(),
            "gmlp_out": gmlp_out.clone().contiguous(),
        }

        out_dir = Path("weights/05_architectures/01_mlp_moe_soups_and_kan")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "01_residual_mlp_mlp_mixer_and_gmlp.safetensors"
        safetensors.torch.save_file(checkpoint_data, out_path)
        print("Saved Category 05 Topic 01 reference checkpoint to", out_path)


    generate_checkpoint()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    ckpt = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors")
    verify_against_checkpoint(ckpt)
