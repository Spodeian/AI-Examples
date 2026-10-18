"""Production PyTorch 2.x implementation for Category 01 Topic 01:
01_generative_vs_discriminative_classifiers
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def fit_gda_parameters(
    x: Float[Tensor, "batch dim"],
    y: Tensor,
    num_classes: int,
    reg_lambda: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fits Gaussian Discriminant Analysis (GDA) with shared covariance and derives equivalent affine parameters."""
    num_samples, dim = x.shape
    means = torch.zeros(num_classes, dim, dtype=x.dtype, device=x.device)
    priors = torch.zeros(num_classes, dtype=x.dtype, device=x.device)

    for c in range(num_classes):
        mask = y == c
        count = float(mask.sum().item())
        priors[c] = count / float(num_samples)
        means[c] = x[mask].mean(dim=0)

    # Compute shared empirical covariance across classes
    centered_x = x.clone()
    for c in range(num_classes):
        mask = y == c
        centered_x[mask] = x[mask] - means[c]

    shared_cov = (centered_x.T @ centered_x) / float(num_samples)
    reg_cov = shared_cov + reg_lambda * torch.eye(dim, dtype=x.dtype, device=x.device)
    cov_inv = torch.linalg.inv(reg_cov)

    # Derive equivalent linear parameters: w_c = Sigma^{-1} mu_c, b_c = -0.5 * mu_c^T Sigma^{-1} mu_c + ln(pi_c)
    w_derived = torch.zeros(dim, num_classes, dtype=x.dtype, device=x.device)
    b_derived = torch.zeros(num_classes, dtype=x.dtype, device=x.device)

    for c in range(num_classes):
        mu_c = means[c]
        sigma_inv_mu = cov_inv @ mu_c
        w_derived[:, c] = sigma_inv_mu
        b_derived[c] = -0.5 * torch.dot(mu_c, sigma_inv_mu) + torch.log(priors[c] + 1e-12)

    return means, reg_cov, priors, w_derived, b_derived


class GaussianDiscriminantAnalysis:
    """Generative classifier modeling joint p(x, y) = p(y) p(x | y) with shared covariance."""

    def __init__(
        self,
        means: Tensor,
        shared_cov: Tensor,
        priors: Tensor,
        w_derived: Tensor,
        b_derived: Tensor,
    ) -> None:
        self.means = means
        self.shared_cov = shared_cov
        self.priors = priors
        self.w_derived = w_derived
        self.b_derived = b_derived

    def predict_logits(
        self,
        x: Float[Tensor, "batch dim"],
    ) -> Float[Tensor, "batch num_classes"]:
        """Evaluates linear discriminant logits x @ w_derived + b_derived."""
        return x @ self.w_derived + self.b_derived

    def predict_proba(
        self,
        x: Float[Tensor, "batch dim"],
    ) -> Float[Tensor, "batch num_classes"]:
        """Evaluates posterior class probabilities p(y | x) via Softmax."""
        logits = self.predict_logits(x)
        return torch.softmax(logits, dim=-1)

    def marginal_log_likelihood(
        self,
        x: Float[Tensor, "batch dim"],
    ) -> Tensor:
        """Evaluates marginal log-likelihood ln p(x) = LSE_c(ln p(y=c) + ln p(x | y=c))."""
        batch, dim = x.shape
        num_classes = self.means.shape[0]
        cov_inv = torch.linalg.inv(self.shared_cov)
        sign, logdet = torch.linalg.slogdet(self.shared_cov)
        logdet_term = -0.5 * (dim * math.log(2 * math.pi) + float(logdet.item()))

        class_log_joint = torch.zeros(batch, num_classes, dtype=x.dtype, device=x.device)
        for c in range(num_classes):
            diff = x - self.means[c]
            mahalanobis = torch.sum((diff @ cov_inv) * diff, dim=-1)
            log_p_x_given_c = logdet_term - 0.5 * mahalanobis
            class_log_joint[:, c] = torch.log(self.priors[c] + 1e-12) + log_p_x_given_c

        return torch.logsumexp(class_log_joint, dim=-1)


class DiscriminativeClassifier(nn.Module):
    """Discriminative classifier modeling conditional posterior p(y | x) directly via linear softmax."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(in_features, num_classes))
        self.bias = nn.Parameter(torch.empty(num_classes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / (self.in_features**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: Float[Tensor, "batch in_features"],
    ) -> Float[Tensor, "batch num_classes"]:
        return x @ self.weight + self.bias

    def predict_proba(
        self,
        x: Float[Tensor, "batch in_features"],
    ) -> Float[Tensor, "batch num_classes"]:
        return torch.softmax(self.forward(x), dim=-1)
