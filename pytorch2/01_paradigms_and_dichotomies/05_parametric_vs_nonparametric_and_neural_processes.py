"""Production PyTorch 2.x implementation for Category 01 Topic 05:
05_parametric_vs_nonparametric_and_neural_processes
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class ParametricMLP(nn.Module):
    """Parametric Multi-Layer Perceptron (Fixed Memory Capacity).

    Inductive Bias:
        Fixed number of learned parameters theta. Once trained, the model has O(1)
        evaluation cost w.r.t dataset size, but cannot dynamically absorb new context
        observations at test time without weight updates.
    """

    def __init__(
        self,
        w_mlp1: Float[Tensor, "dim_x dim_h"],
        b_mlp1: Float[Tensor, "*dim_h"],
        w_mlp2: Float[Tensor, "dim_h dim_y"],
        b_mlp2: Float[Tensor, "*dim_y"],
    ) -> None:
        super().__init__()
        self.w_mlp1 = nn.Parameter(w_mlp1.clone())
        self.b_mlp1 = nn.Parameter(b_mlp1.clone())
        self.w_mlp2 = nn.Parameter(w_mlp2.clone())
        self.b_mlp2 = nn.Parameter(b_mlp2.clone())

    def forward(
        self, x_target: Float[Tensor, "batch num_target dim_x"]
    ) -> Float[Tensor, "batch num_target dim_y"]:
        h = F.relu(torch.matmul(x_target, self.w_mlp1) + self.b_mlp1)
        return torch.matmul(h, self.w_mlp2) + self.b_mlp2


class ExactGaussianProcess(nn.Module):
    """Non-Parametric Gaussian Process with RBF Kernel (Rasmussen & Williams 2006).

    Inductive Bias:
        Memorizes all training observations. Adapts to arbitrary data complexity as N -> inf,
        but incurs O(N^3) Cholesky computation and O(N^2) memory scaling.
    """

    def __init__(
        self,
        lengthscale: Float[Tensor, "*scalar"] | float = 1.0,
        noise_var: Float[Tensor, "*scalar"] | float = 0.1,
    ) -> None:
        super().__init__()
        if isinstance(lengthscale, (float, int)):
            self.lengthscale = nn.Parameter(
                torch.tensor([float(lengthscale)], dtype=torch.float32)
            )
        else:
            self.lengthscale = nn.Parameter(lengthscale.clone())

        if isinstance(noise_var, (float, int)):
            self.noise_var = nn.Parameter(
                torch.tensor([float(noise_var)], dtype=torch.float32)
            )
        else:
            self.noise_var = nn.Parameter(noise_var.clone())

    @staticmethod
    def rbf_kernel(
        x1: Float[Tensor, "batch n1 dim_x"],
        x2: Float[Tensor, "batch n2 dim_x"],
        lengthscale: Float[Tensor, "*scalar"],
    ) -> Float[Tensor, "batch n1 n2"]:
        x1_norm = torch.sum(x1**2, dim=-1, keepdim=True)  # [B, n1, 1]
        x2_norm = torch.sum(x2**2, dim=-1, keepdim=True)  # [B, n2, 1]
        dist_sq = x1_norm + x2_norm.transpose(-2, -1) - 2.0 * torch.matmul(x1, x2.transpose(-2, -1))
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return torch.exp(-0.5 * dist_sq / (lengthscale**2))

    def forward(
        self,
        x_context: Float[Tensor, "batch num_context dim_x"],
        y_context: Float[Tensor, "batch num_context dim_y"],
        x_target: Float[Tensor, "batch num_target dim_x"],
    ) -> tuple[Float[Tensor, "batch num_target dim_y"], Float[Tensor, "batch num_context dim_y"]]:
        num_c = x_context.shape[1]
        batch_size = x_context.shape[0]

        k_cc = self.rbf_kernel(x_context, x_context, self.lengthscale)
        eye = torch.eye(num_c, device=x_context.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
        k_cc_reg = k_cc + self.noise_var * eye

        alpha = torch.linalg.solve(k_cc_reg, y_context)  # [B, num_c, dim_y]
        k_tc = self.rbf_kernel(x_target, x_context, self.lengthscale)  # [B, num_t, num_c]
        y_pred = torch.matmul(k_tc, alpha)  # [B, num_t, dim_y]
        return y_pred, alpha


class AttentiveNeuralProcess(nn.Module):
    """Semi-Parametric Attentive Neural Process (Kim et al. 2019).

    Inductive Bias:
        Combines deep parametric representation learning with non-parametric cross-attention
        retrieval over dynamic context sets in O(C * T) complexity.
    """

    def __init__(
        self,
        w_enc: Float[Tensor, "in_enc dim_h"],
        b_enc: Float[Tensor, "*dim_h"],
        w_q: Float[Tensor, "dim_x dim_h"],
        w_k: Float[Tensor, "dim_x dim_h"],
        w_v: Float[Tensor, "dim_h dim_h"],
        w_dec: Float[Tensor, "in_dec dim_y"],
        b_dec: Float[Tensor, "*dim_y"],
    ) -> None:
        super().__init__()
        self.w_enc = nn.Parameter(w_enc.clone())
        self.b_enc = nn.Parameter(b_enc.clone())
        self.w_q = nn.Parameter(w_q.clone())
        self.w_k = nn.Parameter(w_k.clone())
        self.w_v = nn.Parameter(w_v.clone())
        self.w_dec = nn.Parameter(w_dec.clone())
        self.b_dec = nn.Parameter(b_dec.clone())

    def encode(
        self,
        x_context: Float[Tensor, "batch num_context dim_x"],
        y_context: Float[Tensor, "batch num_context dim_y"],
    ) -> Float[Tensor, "batch num_context dim_h"]:
        xy_c = torch.cat([x_context, y_context], dim=-1)
        return F.relu(torch.matmul(xy_c, self.w_enc) + self.b_enc)

    def forward(
        self,
        x_context: Float[Tensor, "batch num_context dim_x"],
        y_context: Float[Tensor, "batch num_context dim_y"],
        x_target: Float[Tensor, "batch num_target dim_x"],
    ) -> tuple[
        Float[Tensor, "batch num_target dim_y"],
        Float[Tensor, "batch num_target num_context"],
    ]:
        v_c = self.encode(x_context, y_context)  # [B, N_c, dim_h]

        q = torch.matmul(x_target, self.w_q)  # [B, N_t, dim_h]
        k = torch.matmul(x_context, self.w_k)  # [B, N_c, dim_h]
        v = torch.matmul(v_c, self.w_v)  # [B, N_c, dim_h]

        scale = 1.0 / (float(q.shape[-1]) ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, N_t, N_c]
        attn_weights = F.softmax(scores, dim=-1)

        r_anp = torch.matmul(attn_weights, v)  # [B, N_t, dim_h]
        xt_r = torch.cat([x_target, r_anp], dim=-1)  # [B, N_t, dim_x + dim_h]
        y_pred = torch.matmul(xt_r, self.w_dec) + self.b_dec
        return y_pred, attn_weights


class ConditionalNeuralProcess(nn.Module):
    """Semi-Parametric Conditional Neural Process with Mean Pooling (Garnelo et al. 2018).

    Inductive Bias:
        Permutation-invariant aggregation through uniform mean-pooling across context points.
        Suffers from information bottlenecks on complex multi-frequency functions.
    """

    def __init__(
        self,
        w_enc: Float[Tensor, "in_enc dim_h"],
        b_enc: Float[Tensor, "*dim_h"],
        w_dec: Float[Tensor, "in_dec dim_y"],
        b_dec: Float[Tensor, "*dim_y"],
    ) -> None:
        super().__init__()
        self.w_enc = nn.Parameter(w_enc.clone())
        self.b_enc = nn.Parameter(b_enc.clone())
        self.w_dec = nn.Parameter(w_dec.clone())
        self.b_dec = nn.Parameter(b_dec.clone())

    def forward(
        self,
        x_context: Float[Tensor, "batch num_context dim_x"],
        y_context: Float[Tensor, "batch num_context dim_y"],
        x_target: Float[Tensor, "batch num_target dim_x"],
    ) -> Float[Tensor, "batch num_target dim_y"]:
        xy_c = torch.cat([x_context, y_context], dim=-1)
        v_c = F.relu(torch.matmul(xy_c, self.w_enc) + self.b_enc)  # [B, N_c, dim_h]

        r_pooled = torch.mean(v_c, dim=1, keepdim=True)  # [B, 1, dim_h]
        num_target = x_target.shape[1]
        r_expanded = r_pooled.expand(-1, num_target, -1)  # [B, N_t, dim_h]

        xt_r = torch.cat([x_target, r_expanded], dim=-1)  # [B, N_t, dim_x + dim_h]
        return torch.matmul(xt_r, self.w_dec) + self.b_dec
