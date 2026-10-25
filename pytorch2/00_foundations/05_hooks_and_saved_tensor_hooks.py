"""PyTorch 2.x Engine Internals, Autograd Hooks & Saved Tensor Compression.

Topic 05: 05_hooks_and_saved_tensor_hooks (Category 00: Foundations)

Theoretical Overview:
During neural network forward passes, intermediate activations needed for the backward pass
(vector-Jacobian products) are preserved in memory via ctx.save_for_backward. In deep networks,
activation memory scales linearly with depth, sequence length, and batch size: O(B * L * D * N_layers),
frequently exceeding model parameter memory O(P).

PyTorch 2.x provides granular hook interfaces:
1. Module Hooks:
   - register_forward_pre_hook: intercepts and transforms module inputs before forward execution.
   - register_forward_hook: intercepts module outputs for diagnostics or feature caching.
   - register_full_backward_hook: intercepts exact (grad_input, grad_output) pairs at module boundaries.
2. Saved Tensor Hooks (torch.autograd.graph.saved_tensors_hooks):
   - pack_hook(x): intercepts any tensor saved by an autograd node. Enables transparent offloading
     to host RAM or dynamic quantization (e.g. FP32 -> INT8 or BF16).
   - unpack_hook(packed): dynamically reconstructs the tensor when the autograd engine requests it
     during the backward pass.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class HookedLinearLayer(nn.Module):
    """Linear module with explicit hook registration points and saved tensor interception."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Float[Tensor, "batch in_dim"]) -> Float[Tensor, "batch out_dim"]:
        """Evaluates linear projection saving input for backward pass."""
        return x @ self.weight + self.bias


def make_cpu_offload_hooks() -> tuple[Callable[[Tensor], Any], Callable[[Any], Tensor]]:
    """Creates a pair of (pack_hook, unpack_hook) for offloading saved activations to CPU."""
    def pack_hook(tensor: Tensor) -> tuple[torch.device, Tensor]:
        # Offload tensor to CPU memory, freeing accelerator/device VRAM
        return (tensor.device, tensor.to("cpu", non_blocking=False))

    def unpack_hook(packed: tuple[torch.device, Tensor]) -> Tensor:
        orig_device, cpu_tensor = packed
        return cpu_tensor.to(orig_device, non_blocking=False)

    return pack_hook, unpack_hook


def make_scaled_quantization_hooks(
    eps: float = 1e-8,
) -> tuple[Callable[[Tensor], Any], Callable[[Any], Tensor]]:
    """Creates a pair of (pack, unpack) hooks that simulate dynamic INT8 quantization.

    Packs float32 tensors into (int8_quantized, scale, zero_point), cutting memory by ~4x.
    """
    def pack_hook(tensor: Tensor) -> tuple[Tensor, float, float, torch.Size]:
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        scale = max((t_max - t_min) / 255.0, eps)
        zero_point = -round(t_min / scale)
        q = torch.clamp(torch.round(tensor / scale) + zero_point, 0, 255).to(torch.uint8)
        return (q, scale, zero_point, tensor.shape)

    def unpack_hook(packed: tuple[Tensor, float, float, torch.Size]) -> Tensor:
        q, scale, zero_point, orig_shape = packed
        return ((q.float() - zero_point) * scale).view(orig_shape)

    return pack_hook, unpack_hook


def evaluate_with_saved_tensor_hooks(
    model: HookedLinearLayer,
    x: Tensor,
    target: Tensor,
    pack_hook: Callable[[Tensor], Any],
    unpack_hook: Callable[[Any], Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Executes a complete forward-backward pass under saved_tensors_hooks."""
    x_in = x.clone().detach().requires_grad_(True)

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        pred = model(x_in)
        loss = 0.5 * torch.sum((pred - target) ** 2)
        loss.backward()

    assert x_in.grad is not None, "x_in.grad must not be None"
    assert model.weight.grad is not None, "model.weight.grad must not be None"
    assert model.bias.grad is not None, "model.bias.grad must not be None"

    grads = {
        "grad_input": x_in.grad.clone().detach(),
        "grad_weight": model.weight.grad.clone().detach(),
        "grad_bias": model.bias.grad.clone().detach(),
    }
    return pred.detach(), grads


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
    from pathlib import Path

    import torch

    import core


    def generate_topic_05_weights() -> None:
        # Deterministic seed 42
        core.setup_env(seed=42)
        gen = torch.Generator().manual_seed(42)

        # Invertible linear system dimensions
        # Layer: y = GeLU(x @ W + b) with batch_size=4, in_dim=3, out_dim=2
        # Activation x @ W + b is saved for backward by GeLU.
        in_dim, out_dim = 3, 2
        batch_size = 4

        weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
        bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)
        test_in = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
        target = torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)

        # Standard forward and backward pass
        w_std = weight.clone().detach().requires_grad_(True)
        b_std = bias.clone().detach().requires_grad_(True)
        x_std = test_in.clone().detach().requires_grad_(True)

        # Affine linear layer: y = x @ W + b
        # Autograd engine saves x for backward to compute grad_W = x^T @ grad_out
        y_std = x_std @ w_std + b_std
        loss = 0.5 * torch.sum((y_std - target) ** 2)
        loss.backward()

        grad_w_ref = w_std.grad.clone().detach()
        grad_b_ref = b_std.grad.clone().detach()
        grad_x_ref = x_std.grad.clone().detach()

        weights_dict = {
            "weight": weight,
            "bias": bias,
            "target": target,
            "grad_weight": grad_w_ref,
            "grad_bias": grad_b_ref,
            "grad_input": grad_x_ref,
        }

        metadata = {
            "topic": "00_foundations/05_hooks_and_saved_tensor_hooks",
            "description": "Deterministic weights, activations, and reference gradients for hooks and saved_tensors_hooks",
            "seed": "42",
            "framework": "torch",
        }

        out_path = Path("weights/00_foundations/05_hooks_and_saved_tensor_hooks.safetensors")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        core.parity.save_reference_checkpoint(
            path=out_path,
            weights_dict=weights_dict,
            input_tensor=test_in,
            output_tensor=y_std.detach(),
            metadata=metadata,
        )
        print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


    generate_topic_05_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    import core

    env = core.setup_env(seed=42)
    print(f"Testing PyTorch 2.x Hooks & Saved Tensors on {env.device}...")

    # Load golden reference checkpoint
    ckpt_path = Path("weights/00_foundations/05_hooks_and_saved_tensor_hooks.safetensors")
    weights, test_in, golden_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w = weights["weight"]
    b = weights["bias"]
    target = weights["target"]
    golden_gw = weights["grad_weight"]
    golden_gb = weights["grad_bias"]
    golden_gx = weights["grad_input"]

    model = HookedLinearLayer(in_dim=3, out_dim=2)
    model.weight.data.copy_(w)
    model.bias.data.copy_(b)

    # 1. Verify forward output matches reference
    with torch.no_grad():
        out = model(test_in)
    torch.testing.assert_close(out, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] Forward pass matches golden reference.")

    # 2. Test Module Hooks (Pre-forward, Post-forward, Full-backward)
    hook_trace: list[str] = []

    def pre_hook(module: nn.Module, args: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        hook_trace.append("pre_forward")
        return args

    def fwd_hook(module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> None:
        hook_trace.append("forward")

    def bwd_hook(
        module: nn.Module,
        grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor | None, ...],
    ) -> None:
        hook_trace.append("full_backward")

    h_pre = model.register_forward_pre_hook(pre_hook)
    h_fwd = model.register_forward_hook(fwd_hook)
    h_bwd = model.register_full_backward_hook(bwd_hook)

    # 3. Test Saved Tensor Hooks with CPU Offloading
    pack_offload, unpack_offload = make_cpu_offload_hooks()
    pred_offload, grads_offload = evaluate_with_saved_tensor_hooks(
        model, test_in, target, pack_offload, unpack_offload
    )

    # Check hook firing sequence
    assert hook_trace == ["pre_forward", "forward", "full_backward"], f"Unexpected hooks: {hook_trace}"
    print(f"[PASS] Module hook sequence verified: {hook_trace}")

    # Remove module hooks
    h_pre.remove()
    h_fwd.remove()
    h_bwd.remove()

    # 4. Verify exact parity under CPU Offloading hooks
    torch.testing.assert_close(grads_offload["grad_weight"], golden_gw, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grads_offload["grad_bias"], golden_gb, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grads_offload["grad_input"], golden_gx, atol=1e-5, rtol=1e-5)
    print("[PASS] Saved tensor offload hooks achieved exact transitive gradient parity (< 1e-5).")

    # 5. Compilation & Export Verification
    ok_exp, msg_exp = core.export.verify_torch_export(model, test_in)
    print(f"torch.export: {msg_exp}")
    assert ok_exp

    ok_comp, msg_comp = core.export.verify_torch_compile_fullgraph(model, test_in)
    print(f"torch.compile: {msg_comp}")
    assert ok_comp

    print("[PASS] All Topic 05 PyTorch 2.x tests passed cleanly!")
