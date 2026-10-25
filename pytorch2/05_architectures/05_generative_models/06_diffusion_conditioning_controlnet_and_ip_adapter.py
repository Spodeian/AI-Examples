"""Diffusion Conditioning Controlnet and Ip Adapter — PyTorch 2.x reference implementation.

Topic: 05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter
Golden weights: weights/05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter.safetensors

``python pytorch2/05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter.py`` verifies against the golden checkpoint;
``python pytorch2/05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOPIC = "05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def controlnet_residual_block(
    x: torch.Tensor,
    control_signal: torch.Tensor,
    backbone_w: torch.Tensor,
    control_w: torch.Tensor,
    z_in_w: torch.Tensor,
    z_in_b: torch.Tensor,
    z_out_w: torch.Tensor,
    z_out_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ControlNet architecture block with zero-convolution connections.
    
    y = x @ W_backbone + ZeroConv_out(ControlNet_branch(x + ZeroConv_in(control_signal)))
    """
    ctrl_in = x + (control_signal @ z_in_w + z_in_b)
    h_backbone = x @ backbone_w
    h_control = ctrl_in @ control_w
    ctrl_out = h_control @ z_out_w + z_out_b
    y = h_backbone + ctrl_out
    return y, ctrl_out


def ip_adapter_decoupled_cross_attention(
    q: torch.Tensor,
    k_text: torch.Tensor,
    v_text: torch.Tensor,
    k_img: torch.Tensor,
    v_img: torch.Tensor,
    scale: float,
    lambda_img: float,
) -> torch.Tensor:
    """IP-Adapter decoupled cross-attention:
    Z = Attn(Q, K_text, V_text) + lambda_img * Attn(Q, K_img, V_img)
    """
    scores_text = (q @ k_text.transpose(-2, -1)) * scale
    attn_text = F.softmax(scores_text, dim=-1) @ v_text

    scores_img = (q @ k_img.transpose(-2, -1)) * scale
    attn_img = F.softmax(scores_img, dim=-1) @ v_img

    return attn_text + lambda_img * attn_img


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)

    # 1. Standard Linear Anchor
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1
    test_output = linear_anchor(test_input, weight, bias)

    # 2. ControlNet Zero-Convolution Block
    B = 2
    T = 4
    D = 4
    x = torch.randn(B, T, D)
    control_signal = torch.randn(B, T, D)
    backbone_w = torch.randn(D, D) * 0.2
    control_w = torch.randn(D, D) * 0.2

    # Zero-conv parameters initialized to zero
    z_in_w = torch.zeros(D, D)
    z_in_b = torch.zeros(D)
    z_out_w = torch.zeros(D, D)
    z_out_b = torch.zeros(D)

    y_ctrl, ctrl_out = controlnet_residual_block(
        x, control_signal, backbone_w, control_w, z_in_w, z_in_b, z_out_w, z_out_b
    )

    # Also test with non-zero adapted weights
    z_in_w_adapt = torch.randn(D, D) * 0.1
    z_out_w_adapt = torch.randn(D, D) * 0.1
    y_adapted, _ = controlnet_residual_block(
        x, control_signal, backbone_w, control_w, z_in_w_adapt, z_in_b, z_out_w_adapt, z_out_b
    )

    # 3. IP-Adapter Decoupled Cross-Attention
    M_text = 6
    M_img = 4
    scale = 1.0 / math.sqrt(D)
    lambda_img = 0.8

    q = torch.randn(B, T, D)
    k_text = torch.randn(B, M_text, D)
    v_text = torch.randn(B, M_text, D)
    k_img = torch.randn(B, M_img, D)
    v_img = torch.randn(B, M_img, D)

    ip_out = ip_adapter_decoupled_cross_attention(q, k_text, v_text, k_img, v_img, scale, lambda_img)

    return {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": test_output,
        "x": x,
        "control_signal": control_signal,
        "backbone_w": backbone_w,
        "control_w": control_w,
        "z_in_w": z_in_w,
        "z_in_b": z_in_b,
        "z_out_w": z_out_w,
        "z_out_b": z_out_b,
        "y_ctrl": y_ctrl,
        "z_in_w_adapt": z_in_w_adapt,
        "z_out_w_adapt": z_out_w_adapt,
        "y_adapted": y_adapted,
        "q": q,
        "k_text": k_text,
        "v_text": v_text,
        "k_img": k_img,
        "v_img": v_img,
        "scale": torch.tensor([scale]),
        "lambda_img": torch.tensor([lambda_img]),
        "ip_out": ip_out,
    }


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().clone().contiguous().cpu() for k, v in build_golden_tensors().items()}
    save_file(tensors, str(path), metadata={"format": "ai-examples-v1", "seed": "42", "topic": TOPIC})
    print(f"Saved {len(tensors)} tensors to {path.relative_to(REPO_ROOT)}")


def verify_against_golden(path: Path = GOLDEN_PATH) -> None:
    """Asserts that this implementation reproduces the committed golden checkpoint."""
    golden = load_file(str(path))
    for name, expected in build_golden_tensors().items():
        err = float(torch.max(torch.abs(expected - golden[name])))
        assert err < 1e-6, f"{name}: max abs error {err:.2e}"
    print(f"{TOPIC}: verified against golden checkpoint")


if __name__ == "__main__":
    if "--export-golden" in sys.argv[1:]:
        export_golden()
    else:
        verify_against_golden()
