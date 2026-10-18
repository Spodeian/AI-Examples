"""Production PyTorch 2.x implementation for Category 01 Topic 04:
04_deterministic_vs_stochastic_and_mixture_density_networks
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class DeterministicRegressor(nn.Module):
    """Deterministic Conditional Mean Regressor (MSE Point Estimator).

    Inductive Bias:
        Assumes unimodal conditional Gaussian distribution with constant variance.
        Minimizing MSE converges to the conditional expectation E[Y|X=x], which
        fails on multimodal bifurcations or one-to-many inverse problems.
    """

    def __init__(
        self,
        w_h: Float[Tensor, "in_dim hidden_dim"],
        b_h: Float[Tensor, "*hidden_dim"],
        w_det: Float[Tensor, "hidden_dim out_dim"],
        b_det: Float[Tensor, "*out_dim"],
    ) -> None:
        super().__init__()
        self.w_h = nn.Parameter(w_h.clone())
        self.b_h = nn.Parameter(b_h.clone())
        self.w_det = nn.Parameter(w_det.clone())
        self.b_det = nn.Parameter(b_det.clone())

    def forward(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> Float[Tensor, "batch out_dim"]:
        hidden = F.relu(torch.matmul(x, self.w_h) + self.b_h)
        return torch.matmul(hidden, self.w_det) + self.b_det


class MixtureDensityNetwork(nn.Module):
    """Mixture Density Network (Bishop 1994).

    Inductive Bias:
        Parameterizes the complete conditional density p(y|x) as a mixture of Gaussians:
            p(y|x) = sum_{k=1}^K pi_k(x) * N(y; mu_k(x), sigma_k^2(x))
        Can represent arbitrary multimodal, skewed, or heteroskedastic conditional distributions.
    """

    def __init__(
        self,
        w_h: Float[Tensor, "in_dim hidden_dim"],
        b_h: Float[Tensor, "*hidden_dim"],
        w_pi: Float[Tensor, "hidden_dim num_components"],
        b_pi: Float[Tensor, "*num_components"],
        w_mu: Float[Tensor, "hidden_dim num_components"],
        b_mu: Float[Tensor, "*num_components"],
        w_sigma: Float[Tensor, "hidden_dim num_components"],
        b_sigma: Float[Tensor, "*num_components"],
    ) -> None:
        super().__init__()
        self.w_h = nn.Parameter(w_h.clone())
        self.b_h = nn.Parameter(b_h.clone())
        self.w_pi = nn.Parameter(w_pi.clone())
        self.b_pi = nn.Parameter(b_pi.clone())
        self.w_mu = nn.Parameter(w_mu.clone())
        self.b_mu = nn.Parameter(b_mu.clone())
        self.w_sigma = nn.Parameter(w_sigma.clone())
        self.b_sigma = nn.Parameter(b_sigma.clone())

    def forward(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> tuple[
        Float[Tensor, "batch num_components"],
        Float[Tensor, "batch num_components"],
        Float[Tensor, "batch num_components"],
    ]:
        hidden = F.relu(torch.matmul(x, self.w_h) + self.b_h)
        pi_logits = torch.matmul(hidden, self.w_pi) + self.b_pi
        pi = F.softmax(pi_logits, dim=-1)
        mu = torch.matmul(hidden, self.w_mu) + self.b_mu
        log_sigma = torch.matmul(hidden, self.w_sigma) + self.b_sigma
        sigma = torch.exp(log_sigma)
        return pi, mu, sigma

    def log_prob(
        self,
        x: Float[Tensor, "batch in_dim"],
        y: Float[Tensor, "batch out_dim"],
    ) -> Float[Tensor, "*batch"]:
        hidden = F.relu(torch.matmul(x, self.w_h) + self.b_h)
        pi_logits = torch.matmul(hidden, self.w_pi) + self.b_pi
        log_pi = F.log_softmax(pi_logits, dim=-1)

        mu = torch.matmul(hidden, self.w_mu) + self.b_mu
        log_sigma = torch.matmul(hidden, self.w_sigma) + self.b_sigma
        sigma = torch.exp(log_sigma)

        # Broadcast diff: [B, 1] - [B, K] -> [B, K]
        diff = y - mu
        log_gaussian = (
            -0.5 * math.log(2.0 * math.pi)
            - log_sigma
            - 0.5 * (diff / sigma).pow(2)
        )
        return torch.logsumexp(log_pi + log_gaussian, dim=-1)


class QuantileRegressor(nn.Module):
    """Direct Multi-Quantile Regressor with Pinball Loss (Koenker & Bassett 1978).

    Inductive Bias:
        Non-parametric distribution representation through ordered quantiles tau.
        Avoids explicit distributional form assumptions while capturing heteroskedasticity.
    """

    def __init__(
        self,
        w_h: Float[Tensor, "in_dim hidden_dim"],
        b_h: Float[Tensor, "*hidden_dim"],
        w_quantile: Float[Tensor, "hidden_dim num_quantiles"],
        b_quantile: Float[Tensor, "*num_quantiles"],
        quantiles: Float[Tensor, "*num_quantiles"] | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.w_h = nn.Parameter(w_h.clone())
        self.b_h = nn.Parameter(b_h.clone())
        self.w_quantile = nn.Parameter(w_quantile.clone())
        self.b_quantile = nn.Parameter(b_quantile.clone())
        if quantiles is None:
            quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        elif not isinstance(quantiles, Tensor):
            quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.register_buffer("quantiles", quantiles)

    def forward(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> Float[Tensor, "batch num_quantiles"]:
        hidden = F.relu(torch.matmul(x, self.w_h) + self.b_h)
        return torch.matmul(hidden, self.w_quantile) + self.b_quantile

    @staticmethod
    def pinball_loss(
        y_pred: Float[Tensor, "batch num_quantiles"],
        y_true: Float[Tensor, "batch out_dim"],
        quantiles: Float[Tensor, "*num_quantiles"],
    ) -> Float[Tensor, "batch num_quantiles"]:
        u = y_true - y_pred
        return torch.maximum(quantiles * u, (quantiles - 1.0) * u)
