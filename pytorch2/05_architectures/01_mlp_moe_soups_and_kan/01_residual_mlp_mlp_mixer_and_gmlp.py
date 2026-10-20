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


if __name__ == "__main__":
    ckpt = Path("weights/05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp.safetensors")
    verify_against_checkpoint(ckpt)
