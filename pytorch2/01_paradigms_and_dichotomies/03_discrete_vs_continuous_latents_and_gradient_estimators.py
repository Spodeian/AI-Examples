"""Production PyTorch 2.x implementation for Category 01 Topic 03:
03_discrete_vs_continuous_latents_and_gradient_estimators
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class GaussianReparameterization(nn.Module):
    """Continuous Pathwise Reparameterization (Kingma & Welling 2013).

    Inductive Bias:
        Smooth Euclidean latent space with low variance gradient estimator.
        Gradient propagates directly through coordinates: z = mu + sigma * eps.
    """

    def __init__(
        self,
        w_mu: Float[Tensor, "in_dim latent_dim"],
        b_mu: Float[Tensor, "*latent_dim"],
        w_logvar: Float[Tensor, "in_dim latent_dim"],
        b_logvar: Float[Tensor, "*latent_dim"],
        w_dec: Float[Tensor, "latent_dim out_dim"],
        b_dec: Float[Tensor, "*out_dim"],
    ) -> None:
        super().__init__()
        self.w_mu = nn.Parameter(w_mu.clone())
        self.b_mu = nn.Parameter(b_mu.clone())
        self.w_logvar = nn.Parameter(w_logvar.clone())
        self.b_logvar = nn.Parameter(b_logvar.clone())
        self.w_dec = nn.Parameter(w_dec.clone())
        self.b_dec = nn.Parameter(b_dec.clone())

    def encode(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> tuple[Float[Tensor, "batch latent_dim"], Float[Tensor, "batch latent_dim"]]:
        mu = torch.matmul(x, self.w_mu) + self.b_mu
        logvar = torch.matmul(x, self.w_logvar) + self.b_logvar
        return mu, logvar

    def reparameterize(
        self,
        mu: Float[Tensor, "batch latent_dim"],
        logvar: Float[Tensor, "batch latent_dim"],
        eps: Float[Tensor, "batch latent_dim"] | None = None,
    ) -> Float[Tensor, "batch latent_dim"]:
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_divergence(
        mu: Float[Tensor, "batch latent_dim"],
        logvar: Float[Tensor, "batch latent_dim"],
    ) -> Float[Tensor, "*batch"]:
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    def decode(
        self, z: Float[Tensor, "batch latent_dim"]
    ) -> Float[Tensor, "batch out_dim"]:
        return torch.matmul(z, self.w_dec) + self.b_dec

    def forward(
        self,
        x: Float[Tensor, "batch in_dim"],
        eps: Float[Tensor, "batch latent_dim"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch latent_dim"],
        Float[Tensor, "batch out_dim"],
        Float[Tensor, "*batch"],
    ]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, eps=eps)
        out = self.decode(z)
        kl = self.kl_divergence(mu, logvar)
        return z, out, kl


class GumbelSoftmaxRelaxation(nn.Module):
    """Discrete Relaxed Concrete / Gumbel-Softmax (Jang et al. 2017; Maddison et al. 2017).

    Inductive Bias:
        Approximates non-differentiable categorical argmax sampling via smooth temperature
        relaxation: y = softmax((logits + gumbel_noise) / tau).
    """

    def __init__(
        self,
        w_logits: Float[Tensor, "in_dim latent_dim"],
        b_logits: Float[Tensor, "*latent_dim"],
        w_dec: Float[Tensor, "latent_dim out_dim"],
        b_dec: Float[Tensor, "*out_dim"],
        temperature: Float[Tensor, "*scalar"] | float = 1.0,
    ) -> None:
        super().__init__()
        self.w_logits = nn.Parameter(w_logits.clone())
        self.b_logits = nn.Parameter(b_logits.clone())
        self.w_dec = nn.Parameter(w_dec.clone())
        self.b_dec = nn.Parameter(b_dec.clone())
        if isinstance(temperature, (float, int)):
            self.temperature = nn.Parameter(
                torch.tensor([float(temperature)], dtype=torch.float32)
            )
        else:
            self.temperature = nn.Parameter(temperature.clone())

    def encode(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> Float[Tensor, "batch latent_dim"]:
        return torch.matmul(x, self.w_logits) + self.b_logits

    def sample(
        self,
        logits: Float[Tensor, "batch latent_dim"],
        gumbel_noise: Float[Tensor, "batch latent_dim"] | None = None,
        hard: bool = False,
    ) -> Float[Tensor, "batch latent_dim"]:
        if gumbel_noise is None:
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(u))
        y_soft = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        if hard:
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def decode(
        self, y: Float[Tensor, "batch latent_dim"]
    ) -> Float[Tensor, "batch out_dim"]:
        return torch.matmul(y, self.w_dec) + self.b_dec

    def forward(
        self,
        x: Float[Tensor, "batch in_dim"],
        gumbel_noise: Float[Tensor, "batch latent_dim"] | None = None,
        hard: bool = False,
    ) -> tuple[Float[Tensor, "batch latent_dim"], Float[Tensor, "batch out_dim"]]:
        logits = self.encode(x)
        y = self.sample(logits, gumbel_noise=gumbel_noise, hard=hard)
        out = self.decode(y)
        return y, out


class BernoulliReinforceEstimator:
    """Score function / REINFORCE gradient estimator for discrete Bernoulli latents."""

    @staticmethod
    def log_prob(
        probs: Float[Tensor, "batch latent_dim"],
        b: Float[Tensor, "batch latent_dim"],
    ) -> Float[Tensor, "*batch"]:
        p_safe = probs.clamp(1e-7, 1.0 - 1e-7)
        return torch.sum(b * torch.log(p_safe) + (1.0 - b) * torch.log(1.0 - p_safe), dim=-1)

    @staticmethod
    def score_gradient(
        probs: Float[Tensor, "batch latent_dim"],
        b: Float[Tensor, "batch latent_dim"],
        reward: Float[Tensor, "*batch"],
        baseline: Float[Tensor, "*batch"] | float = 0.0,
    ) -> Float[Tensor, "batch latent_dim"]:
        # d/d_logits log p(b) = b - sigmoid(logits) = b - probs
        advantage = reward - baseline
        if isinstance(advantage, Tensor) and advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)
        return advantage * (b - probs)

    @staticmethod
    def vimco_baseline(
        rewards: Float[Tensor, "batch num_samples"],
    ) -> Float[Tensor, "batch num_samples"]:
        # Leave-one-out baseline: c_i = 1/(K-1) * sum_{j != i} r_j
        num_samples = rewards.shape[-1]
        if num_samples <= 1:
            return torch.zeros_like(rewards)
        sum_r = torch.sum(rewards, dim=-1, keepdim=True)
        return (sum_r - rewards) / float(num_samples - 1)
