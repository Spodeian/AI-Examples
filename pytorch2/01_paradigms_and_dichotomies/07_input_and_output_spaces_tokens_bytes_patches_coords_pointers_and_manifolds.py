"""Production PyTorch 2.x implementation for Category 01 Topic 07:
07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class ContinuousRotation6D:
    """Continuous 6D Rotation Manifold Representation (Zhou et al. CVPR 2019).

    Inductive Bias:
        Euler angles and Quaternions suffer from topological discontinuities in R^3 and S^3.
        A continuous 6D representation projects two unconstrained 3D vectors onto SO(3)
        via Gram-Schmidt orthogonalization without gimbal lock or sign ambiguity.
    """

    @staticmethod
    def rotation_6d_to_matrix(
        d6: Float[Tensor, "batch 6"],
    ) -> Float[Tensor, "batch 3 3"]:
        """Maps unconstrained 6D vector [a1, a2] to orthogonal 3x3 rotation matrix R in SO(3)."""
        a1 = d6[:, 0:3]
        a2 = d6[:, 3:6]

        b1 = F.normalize(a1, p=2, dim=-1, eps=1e-8)
        dot_prod = torch.sum(b1 * a2, dim=-1, keepdim=True)
        u2 = a2 - dot_prod * b1
        b2 = F.normalize(u2, p=2, dim=-1, eps=1e-8)
        b3 = torch.cross(b1, b2, dim=-1)

        return torch.stack([b1, b2, b3], dim=-1)


class PointerNetworkLayer(nn.Module):
    """Pointer Mechanism for Variable-Length Input Spaces (Vinyals et al. NeurIPS 2015).

    Inductive Bias:
        Instead of predicting over a fixed discrete dictionary, pointer networks score
        variable-length input elements via content-based attention, producing an output
        distribution in Delta_N.
    """

    def __init__(
        self,
        w_enc: Float[Tensor, "d_model h_attn"],
        w_dec: Float[Tensor, "d_model h_attn"],
        b_ptr: Float[Tensor, "*h_attn"],
        v_ptr: Float[Tensor, "h_attn 1"],
    ) -> None:
        super().__init__()
        self.w_enc = nn.Parameter(w_enc.clone())
        self.w_dec = nn.Parameter(w_dec.clone())
        self.b_ptr = nn.Parameter(b_ptr.clone())
        self.v_ptr = nn.Parameter(v_ptr.clone())

    def forward(
        self,
        encoder_seq: Float[Tensor, "batch num_elem d_model"],
        decoder_state: Float[Tensor, "batch d_model"],
    ) -> tuple[Float[Tensor, "batch num_elem"], Float[Tensor, "batch num_elem"]]:
        """Computes pointer logits and probability distribution over input candidates."""
        e_proj = torch.matmul(encoder_seq, self.w_enc)
        d_proj = torch.matmul(decoder_state, self.w_dec).unsqueeze(1)
        u_scores = torch.tanh(e_proj + d_proj + self.b_ptr)
        logits = torch.matmul(u_scores, self.v_ptr).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class FourierCoordinateEncoder:
    """Multi-Scale Sinusoidal Positional Encoding for Continuous Coordinates (Mildenhall et al. 2020).

    Inductive Bias:
        Standard coordinate MLPs suffer from spectral bias, failing to learn high-frequency
        functions. Projecting continuous coordinates onto sinusoidal Fourier basis enables
        representation of fine spatial details.
    """

    @staticmethod
    def encode(
        coords: Float[Tensor, "batch d_coord"],
        num_bands: int = 4,
    ) -> Float[Tensor, "batch out_dim"]:
        """Maps continuous coords in R^D to multi-frequency Fourier embeddings in R^{D * 2 * L}."""
        device = coords.device
        dtype = coords.dtype
        scales = (2.0 ** torch.arange(num_bands, dtype=dtype, device=device)) * torch.pi
        scaled = coords.unsqueeze(-1) * scales.view(1, 1, -1)
        sin_part = torch.sin(scaled)
        cos_part = torch.cos(scaled)
        out = torch.stack([sin_part, cos_part], dim=-1)
        return out.view(coords.shape[0], -1)


class PatchEmbedding(nn.Module):
    """Linear Patch Projection for Vision Transformers (Dosovitskiy et al. ICLR 2021).

    Inductive Bias:
        Tokenizes continuous 2D/3D grids by partitioning into flattened local patches
        and linearly embedding into model latent dimension.
    """

    def __init__(
        self,
        w_patch: Float[Tensor, "patch_dim d_model"],
        b_patch: Float[Tensor, "*d_model"],
    ) -> None:
        super().__init__()
        self.w_patch = nn.Parameter(w_patch.clone())
        self.b_patch = nn.Parameter(b_patch.clone())

    def forward(
        self,
        patches: Float[Tensor, "batch num_patches patch_dim"],
    ) -> Float[Tensor, "batch num_patches d_model"]:
        return torch.matmul(patches, self.w_patch) + self.b_patch
