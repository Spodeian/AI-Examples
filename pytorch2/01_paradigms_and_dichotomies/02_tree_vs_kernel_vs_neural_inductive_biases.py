"""Production PyTorch 2.x implementation for Category 01 Topic 02:
02_tree_vs_kernel_vs_neural_inductive_biases
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class SoftDecisionTree(nn.Module):
    """Differentiable Soft Decision Tree with axis-aligned routing projections.

    Inductive Bias:
        Partitions input space with smooth hyperplanes orthogonal to coordinate axes:
        p(go right at node i) = sigmoid(beta * (x @ routing_weights[i] - threshold[i])).
        Invariant to monotonic marginal scaling; sensitive to orthogonal coordinate rotations.
    """

    def __init__(
        self,
        routing_weights: Float[Tensor, "nodes in_dim"],
        thresholds: Float[Tensor, "*nodes"],
        temperature: Float[Tensor, "*scalar"] | float,
        leaf_weights: Float[Tensor, "leaves out_dim"],
    ) -> None:
        super().__init__()
        self.routing_weights = nn.Parameter(routing_weights.clone())
        self.thresholds = nn.Parameter(thresholds.clone())
        if isinstance(temperature, (float, int)):
            self.temperature = nn.Parameter(torch.tensor([float(temperature)], dtype=torch.float32))
        else:
            self.temperature = nn.Parameter(temperature.clone())
        self.leaf_weights = nn.Parameter(leaf_weights.clone())

    def forward(self, x: Float[Tensor, "batch in_dim"]) -> Float[Tensor, "batch out_dim"]:
        # Routing logits: beta * (x @ W_route.T - thresholds)
        logits = self.temperature * (torch.matmul(x, self.routing_weights.T) - self.thresholds)
        p_right = torch.sigmoid(logits)  # [batch, 3]
        p_left = 1.0 - p_right

        # Depth-2 leaf probabilities
        mu0 = p_left[:, 0:1] * p_left[:, 1:2]
        mu1 = p_left[:, 0:1] * p_right[:, 1:2]
        mu2 = p_right[:, 0:1] * p_left[:, 2:3]
        mu3 = p_right[:, 0:1] * p_right[:, 2:3]
        leaf_probs = torch.cat([mu0, mu1, mu2, mu3], dim=-1)  # [batch, 4]

        return torch.matmul(leaf_probs, self.leaf_weights)


class RBFKernelRidge(nn.Module):
    """Reproducing Kernel Hilbert Space (RKHS) RBF Kernel Ridge Regressor.

    Inductive Bias:
        Local isotropic Euclidean smoothness:
        k(x, s) = exp(-gamma * ||x - s||_2^2).
        Strictly invariant under orthogonal rotations R in O(d) when support points are rotated:
        ||R x - R s||_2^2 = ||x - s||_2^2.
    """

    def __init__(
        self,
        support_vectors: Float[Tensor, "support in_dim"],
        dual_coeffs: Float[Tensor, "support out_dim"],
        gamma: Float[Tensor, "*scalar"] | float,
        bias: Float[Tensor, "*out_dim"],
    ) -> None:
        super().__init__()
        self.support_vectors = nn.Parameter(support_vectors.clone())
        self.dual_coeffs = nn.Parameter(dual_coeffs.clone())
        if isinstance(gamma, (float, int)):
            self.gamma = nn.Parameter(torch.tensor([float(gamma)], dtype=torch.float32))
        else:
            self.gamma = nn.Parameter(gamma.clone())
        self.bias = nn.Parameter(bias.clone())

    def forward(self, x: Float[Tensor, "batch in_dim"]) -> Float[Tensor, "batch out_dim"]:
        # Squared Euclidean distances: ||x_i - s_j||^2 = ||x_i||^2 + ||s_j||^2 - 2 x_i s_j^T
        x_sq = torch.sum(x**2, dim=-1, keepdim=True)  # [batch, 1]
        s_sq = torch.sum(self.support_vectors**2, dim=-1, keepdim=True).T  # [1, support]
        dists_sq = torch.clamp(
            x_sq + s_sq - 2.0 * torch.matmul(x, self.support_vectors.T),
            min=0.0,
        )
        kernel_sim = torch.exp(-self.gamma * dists_sq)  # [batch, support]
        return torch.matmul(kernel_sim, self.dual_coeffs) + self.bias


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron (MLP) with adaptive representation learning.

    Inductive Bias:
        Hierarchical linear projections interleaved with point-wise non-linearities:
        f(x) = W_2 ReLU(W_1 x + b_1) + b_2.
        Learns continuous task-specific coordinate rotations and manifold alignments via gradients.
    """

    def __init__(
        self,
        w1: Float[Tensor, "in_dim hidden_dim"],
        b1: Float[Tensor, "*hidden_dim"],
        w2: Float[Tensor, "hidden_dim out_dim"],
        b2: Float[Tensor, "*out_dim"],
    ) -> None:
        super().__init__()
        self.w1 = nn.Parameter(w1.clone())
        self.b1 = nn.Parameter(b1.clone())
        self.w2 = nn.Parameter(w2.clone())
        self.b2 = nn.Parameter(b2.clone())

    def forward(self, x: Float[Tensor, "batch in_dim"]) -> Float[Tensor, "batch out_dim"]:
        h = F.relu(torch.matmul(x, self.w1) + self.b1)
        return torch.matmul(h, self.w2) + self.b2
