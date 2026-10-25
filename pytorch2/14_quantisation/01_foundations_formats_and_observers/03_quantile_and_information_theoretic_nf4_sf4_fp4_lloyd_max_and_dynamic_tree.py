"""Quantile and Information Theoretic NF4 Sf4 FP4 Lloyd Max and Dynamic Tree — PyTorch 2.x reference implementation.

Topic: 14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree
Golden weights: weights/14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree.safetensors

``python pytorch2/14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree.py`` verifies against the golden checkpoint;
``python pytorch2/14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree.py --export-golden`` regenerates it deterministically (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPIC = "14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree"
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
GOLDEN_PATH = REPO_ROOT / "weights" / f"{TOPIC}.safetensors"

# Canonical QLoRA NF4 16-level standard codebook (Dettmers et al. 2023)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0,
])

# Student-t SF4 16-level codebook with degrees of freedom nu=4.0 (Dotzel et al. 2024)
SF4_LEVELS = torch.tensor([
    -1.0, -0.7303, -0.5512, -0.4072, -0.2885, -0.1852, -0.0905, 0.0,
    0.0792, 0.1614, 0.2486, 0.3448, 0.4571, 0.5982, 0.7681, 1.0,
])


def linear_anchor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Cross-track parity anchor shared by every topic."""
    return x @ weight + bias


def nf4_quantize_dequantize(
    x: torch.Tensor,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Information-theoretically optimal NormalFloat 4 (NF4) block quantizer."""
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    levels = NF4_LEVELS.to(x.device)

    amax = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
    is_zero = (amax == 0.0)
    scale = torch.where(is_zero, torch.ones_like(amax), amax)

    norm_x = x_reshaped / scale
    diff = torch.abs(norm_x.unsqueeze(-1) - levels)
    idx = torch.argmin(diff, dim=-1)

    dequant = levels[idx] * scale
    dequant = torch.where(is_zero, torch.zeros_like(dequant), dequant)
    return dequant.reshape(orig_shape), scale.reshape(-1)


def sf4_quantize_dequantize(
    x: torch.Tensor,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student-t Float 4 (SF4) block quantizer for heavy-tailed activations/weights."""
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    levels = SF4_LEVELS.to(x.device)

    amax = torch.max(torch.abs(x_reshaped), dim=-1, keepdim=True).values
    is_zero = (amax == 0.0)
    scale = torch.where(is_zero, torch.ones_like(amax), amax)

    norm_x = x_reshaped / scale
    diff = torch.abs(norm_x.unsqueeze(-1) - levels)
    idx = torch.argmin(diff, dim=-1)

    dequant = levels[idx] * scale
    dequant = torch.where(is_zero, torch.zeros_like(dequant), dequant)
    return dequant.reshape(orig_shape), scale.reshape(-1)


def lloyd_max_scalar_quantize(
    data: torch.Tensor,
    num_levels: int = 8,
    iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterative 1D Lloyd-Max optimal scalar quantization algorithm."""
    flat = data.flatten()
    dmin, dmax = float(torch.min(flat)), float(torch.max(flat))
    
    # Initialize centroids linearly between min and max
    centroids = torch.linspace(dmin, dmax, num_levels, device=data.device)
    
    for _ in range(iters):
        # Assign to nearest centroid
        diff = torch.abs(flat.unsqueeze(-1) - centroids)
        cluster_idx = torch.argmin(diff, dim=-1)
        
        # Update centroids to cluster means
        new_centroids = centroids.clone()
        for k in range(num_levels):
            mask = (cluster_idx == k)
            if mask.any():
                new_centroids[k] = torch.mean(flat[mask])
        centroids = new_centroids

    # Final quantization
    diff_final = torch.abs(data.unsqueeze(-1) - centroids)
    idx_final = torch.argmin(diff_final, dim=-1)
    q_out = centroids[idx_final]
    return q_out, centroids


def dynamic_tree_scalar_quantize(
    x: torch.Tensor,
    depth: int = 3,
) -> torch.Tensor:
    """Dynamic tree binary recursive scalar quantization."""
    # Depth 3 produces 2^3 = 8 leaf intervals
    # Bounds: min to max
    leaves = [x.flatten()]
    
    for _ in range(depth):
        next_leaves = []
        for leaf in leaves:
            if len(leaf) == 0:
                next_leaves.extend([leaf, leaf])
                continue
            median = torch.median(leaf)
            left = leaf[leaf <= median]
            right = leaf[leaf > median]
            next_leaves.extend([left, right])
        leaves = next_leaves
        
    centroids = torch.tensor([torch.mean(l) if len(l) > 0 else 0.0 for l in leaves], device=x.device)
    diff = torch.abs(x.unsqueeze(-1) - centroids)
    idx = torch.argmin(diff, dim=-1)
    return centroids[idx]


def build_golden_tensors() -> dict[str, torch.Tensor]:
    """Deterministically construct inputs, parameters and reference outputs (seed 42)."""
    torch.manual_seed(42)
    test_input = torch.randn(8, 4)
    weight = torch.randn(4, 4) * 0.1
    bias = torch.randn(4) * 0.1

    # 1. NF4 Quantization
    raw_gaussian = torch.randn(2, 64) * 1.5
    nf4_dequant_ref, nf4_scales_ref = nf4_quantize_dequantize(raw_gaussian, block_size=64)

    # 2. SF4 Quantization
    raw_heavy = (torch.randn(2, 64) ** 3) * 0.2
    sf4_dequant_ref, sf4_scales_ref = sf4_quantize_dequantize(raw_heavy, block_size=64)

    # 3. Lloyd-Max Quantization
    lloyd_input = torch.randn(64) * 2.0
    lloyd_q_ref, lloyd_centroids_ref = lloyd_max_scalar_quantize(lloyd_input, num_levels=8, iters=15)

    # 4. Dynamic Tree Quantization
    tree_input = torch.randn(32) * 1.8
    tree_q_ref = dynamic_tree_scalar_quantize(tree_input, depth=3)

    tensors = {
        "test_input": test_input,
        "weight": weight,
        "bias": bias,
        "test_output": linear_anchor(test_input, weight, bias),
        "raw_gaussian": raw_gaussian,
        "nf4_dequant_ref": nf4_dequant_ref,
        "nf4_scales_ref": nf4_scales_ref,
        "raw_heavy": raw_heavy,
        "sf4_dequant_ref": sf4_dequant_ref,
        "sf4_scales_ref": sf4_scales_ref,
        "lloyd_input": lloyd_input,
        "lloyd_q_ref": lloyd_q_ref,
        "lloyd_centroids_ref": lloyd_centroids_ref,
        "tree_input": tree_input,
        "tree_q_ref": tree_q_ref,
    }
    return tensors


def export_golden(path: Path = GOLDEN_PATH) -> None:
    """Writes the golden checkpoint consumed by every track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().contiguous().cpu() for k, v in build_golden_tensors().items()}
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
