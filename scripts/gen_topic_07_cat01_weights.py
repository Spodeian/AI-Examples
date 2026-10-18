"""Generate golden weights and reference I/O tensors for Topic 07:
07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.

Pillars:
1. Continuous 6D Rotation Manifold Representation (Zhou et al. CVPR 2019):
   Gram-Schmidt orthogonalization mapping R^{3x2} -> SO(3) without topological singularities.
2. Pointer Mechanism (Vinyals et al. NeurIPS 2015 Pointer Networks):
   Additive attention pointer distribution over variable-length inputs in Delta_N.
3. Multi-Frequency Coordinate Positional Encoding (Mildenhall et al. ECCV 2020 NeRF):
   Sinusoidal Fourier feature mapping for continuous spatial coordinates.
4. Linear Patch Projection (Dosovitskiy et al. ICLR 2021 ViT):
   Flattened patch embedding with learnable affine projection.
5. Canonical Linear Anchor for cross-track parity harness.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

import core


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation to 3x3 SO(3) rotation matrix.

    Maps d6 [B, 6] -> R [B, 3, 3] via Gram-Schmidt orthogonalization.
    Zhou et al. (2019) 'On the Continuity of Rotation Representations in Neural Networks'.
    """
    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]

    # 1. Normalize first vector
    b1 = F.normalize(a1, p=2, dim=-1, eps=1e-8)

    # 2. Gram-Schmidt orthogonal projection for second vector
    dot_prod = torch.sum(b1 * a2, dim=-1, keepdim=True)
    u2 = a2 - dot_prod * b1
    b2 = F.normalize(u2, p=2, dim=-1, eps=1e-8)

    # 3. Cross product for third orthonormal vector
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack column-wise to form rotation matrix [B, 3, 3]
    # R[:, :, 0] = b1, R[:, :, 1] = b2, R[:, :, 2] = b3
    return torch.stack([b1, b2, b3], dim=-1)


def fourier_encode_coords(coords: torch.Tensor, num_bands: int = 4) -> torch.Tensor:
    """Encodes continuous coordinates into multi-scale Fourier features.

    coords: [B, D] -> [B, D * 2 * num_bands]
    gamma(x) = [sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^{L-1} pi x), cos(2^{L-1} pi x)]
    """
    scales = (2.0 ** torch.arange(num_bands, dtype=coords.dtype, device=coords.device)) * torch.pi
    # coords: [B, D, 1], scales: [1, 1, L] -> [B, D, L]
    scaled = coords.unsqueeze(-1) * scales.view(1, 1, -1)
    sin_part = torch.sin(scaled)
    cos_part = torch.cos(scaled)
    # Stack along last dim and flatten
    out = torch.stack([sin_part, cos_part], dim=-1)  # [B, D, L, 2]
    return out.view(coords.shape[0], -1)


def generate_topic_07_cat01_weights(output_path: Path) -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 4
    in_dim = 6
    num_elements = 5
    d_model = 4
    h_attn = 4

    # 1. Input batches
    # Canonical linear test input
    test_input = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)

    # 2. Canonical linear reference anchor
    weight = torch.randn(in_dim, 1, generator=gen, dtype=torch.float32)
    bias = torch.randn(1, generator=gen, dtype=torch.float32)
    test_output = torch.matmul(test_input, weight) + bias

    # 3. 6D Continuous Rotation Manifold
    # test_input is [B, 6] representing two unconstrained 3D vectors
    rot_matrices = rotation_6d_to_matrix(test_input)  # [B, 3, 3]

    # Verify SO(3) invariants: R^T R = I and det(R) = +1
    identity = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
    r_rt = torch.matmul(rot_matrices.transpose(-1, -2), rot_matrices)
    assert torch.allclose(r_rt, identity, atol=1e-5), "R^T R must be identity"
    dets = torch.linalg.det(rot_matrices)
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-5), "det(R) must be 1.0"

    # 4. Pointer Networks
    # Encoder representation of input candidates: E [B, N, D]
    encoder_seq = torch.randn(batch_size, num_elements, d_model, generator=gen, dtype=torch.float32)
    # Decoder query state: d [B, D]
    decoder_state = torch.randn(batch_size, d_model, generator=gen, dtype=torch.float32)

    w_enc = torch.randn(d_model, h_attn, generator=gen, dtype=torch.float32)
    w_dec = torch.randn(d_model, h_attn, generator=gen, dtype=torch.float32)
    b_ptr = torch.randn(h_attn, generator=gen, dtype=torch.float32)
    v_ptr = torch.randn(h_attn, 1, generator=gen, dtype=torch.float32)

    # u_j = v^T tanh(E_j W_enc + d W_dec + b_ptr)
    e_proj = torch.matmul(encoder_seq, w_enc)  # [B, N, H]
    d_proj = torch.matmul(decoder_state, w_dec).unsqueeze(1)  # [B, 1, H]
    u_scores = torch.tanh(e_proj + d_proj + b_ptr)  # [B, N, H]
    ptr_logits = torch.matmul(u_scores, v_ptr).squeeze(-1)  # [B, N]
    ptr_probs = F.softmax(ptr_logits, dim=-1)  # [B, N]

    assert torch.allclose(ptr_probs.sum(-1), torch.ones(batch_size), atol=1e-5), "Pointer probs must sum to 1.0"

    # 5. Multi-Scale Coordinate Positional Encoding (Fourier INR)
    # Continuous 2D coordinates in [-1, 1]: coords [B, 2]
    coords_2d = torch.randn(batch_size, 2, generator=gen, dtype=torch.float32).tanh()
    coords_fourier = fourier_encode_coords(coords_2d, num_bands=4)  # [B, 2 * 2 * 4 = 16]

    # 6. ViT Patch Linear Projection
    # Flattened patches: 4 patches of size 4 (e.g. 2x2 patch with 1 channel)
    num_patches = 4
    patch_dim = 4
    patches = torch.randn(batch_size, num_patches, patch_dim, generator=gen, dtype=torch.float32)
    w_patch = torch.randn(patch_dim, d_model, generator=gen, dtype=torch.float32)
    b_patch = torch.randn(d_model, generator=gen, dtype=torch.float32)
    patch_embeddings = torch.matmul(patches, w_patch) + b_patch  # [B, num_patches, d_model]

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "rot_matrices": rot_matrices.detach(),
        "encoder_seq": encoder_seq.detach(),
        "decoder_state": decoder_state.detach(),
        "w_enc": w_enc.detach(),
        "w_dec": w_dec.detach(),
        "b_ptr": b_ptr.detach(),
        "v_ptr": v_ptr.detach(),
        "ptr_logits": ptr_logits.detach(),
        "ptr_probs": ptr_probs.detach(),
        "coords_2d": coords_2d.detach(),
        "coords_fourier": coords_fourier.detach(),
        "patches": patches.detach(),
        "w_patch": w_patch.detach(),
        "b_patch": b_patch.detach(),
        "patch_embeddings": patch_embeddings.detach(),
    }

    core.parity.save_reference_checkpoint(
        path=output_path,
        weights_dict=weights_dict,
        input_tensor=test_input,
        output_tensor=test_output,
        metadata={
            "topic": "07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds",
            "category": "01_paradigms_and_dichotomies",
            "batch_size": str(batch_size),
            "in_dim": str(in_dim),
            "num_elements": str(num_elements),
            "d_model": str(d_model),
            "h_attn": str(h_attn),
        },
    )

    print(f"Generated Topic 07 Category 01 weights at: {output_path}")
    print(f"rot_matrices shape: {rot_matrices.shape}")
    print(f"ptr_probs shape: {ptr_probs.shape}")
    print(f"coords_fourier shape: {coords_fourier.shape}")
    print(f"patch_embeddings shape: {patch_embeddings.shape}")


if __name__ == "__main__":
    out_file = (
        Path(__file__).resolve().parent.parent
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.safetensors"
    )
    generate_topic_07_cat01_weights(out_file)
