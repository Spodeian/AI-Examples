"""Generate deterministic reference safetensors checkpoint for Category 01 Topic 01:
01_generative_vs_discriminative_classifiers
"""

from __future__ import annotations

from pathlib import Path

import torch

import core


def fit_gaussian_discriminant_analysis(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    reg_lambda: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fits Gaussian Discriminant Analysis (GDA) with shared covariance and derives equivalent affine parameters.

    Returns:
        means: [num_classes, dim]
        shared_cov: [dim, dim]
        priors: [num_classes]
        w_derived: [dim, num_classes]
        b_derived: [num_classes]
    """
    n, d = x.shape
    means = torch.zeros(num_classes, d, dtype=x.dtype, device=x.device)
    priors = torch.zeros(num_classes, dtype=x.dtype, device=x.device)

    for c in range(num_classes):
        mask = y == c
        count = float(mask.sum().item())
        priors[c] = count / float(n)
        means[c] = x[mask].mean(dim=0)

    # Compute shared empirical covariance
    centered_x = x.clone()
    for c in range(num_classes):
        mask = y == c
        centered_x[mask] = x[mask] - means[c]

    shared_cov = (centered_x.T @ centered_x) / float(n)
    # Ridge regularization for strictly well-conditioned inversion
    reg_cov = shared_cov + reg_lambda * torch.eye(d, dtype=x.dtype, device=x.device)
    cov_inv = torch.linalg.inv(reg_cov)

    # Derive equivalent linear parameters: w_c = Sigma^{-1} mu_c, b_c = -0.5 * mu_c^T Sigma^{-1} mu_c + ln(pi_c)
    w_derived = torch.zeros(d, num_classes, dtype=x.dtype, device=x.device)
    b_derived = torch.zeros(num_classes, dtype=x.dtype, device=x.device)

    for c in range(num_classes):
        mu_c = means[c]
        sigma_inv_mu = cov_inv @ mu_c
        w_derived[:, c] = sigma_inv_mu
        b_derived[c] = -0.5 * torch.dot(mu_c, sigma_inv_mu) + torch.log(priors[c] + 1e-12)

    return means, reg_cov, priors, w_derived, b_derived


def generate_topic_01_cat01_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    in_dim = 6
    num_classes = 3

    # Generate synthetic training partition for GDA parameter estimation
    n_train = 60
    train_x = torch.randn(n_train, in_dim, generator=gen, dtype=torch.float32)
    # 20 samples per class
    train_y = torch.tensor([0] * 20 + [1] * 20 + [2] * 20, dtype=torch.long)

    # Add class-conditional separation to train_x
    class_shifts = torch.tensor(
        [
            [1.5, -1.0, 0.5, 0.0, -0.5, 1.0],
            [-1.0, 1.5, -0.5, 1.0, 0.0, -0.5],
            [0.0, -0.5, 1.0, -1.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    for c in range(num_classes):
        mask = train_y == c
        train_x[mask] += class_shifts[c]

    # Fit generative Gaussian Discriminant Analysis
    means, reg_cov, priors, w_gda, b_gda = fit_gaussian_discriminant_analysis(
        train_x, train_y, num_classes=num_classes, reg_lambda=1e-3
    )

    # Discriminative linear projection weights (anchor)
    weight = torch.randn(in_dim, num_classes, generator=gen, dtype=torch.float32)
    bias = torch.randn(num_classes, generator=gen, dtype=torch.float32)

    test_input = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
    # test_output satisfies canonical cross-framework linear projection parity
    test_output = test_input @ weight + bias

    # GDA generative logits on test_input
    gda_logits = test_input @ w_gda + b_gda

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "gda_means": means.detach(),
        "gda_covariance": reg_cov.detach(),
        "gda_priors": priors.detach(),
        "gda_weight": w_gda.detach(),
        "gda_bias": b_gda.detach(),
        "gda_logits": gda_logits.detach(),
    }

    metadata = {
        "topic": "01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers",
        "description": "Generative Gaussian Discriminant Analysis (GDA) vs Discriminative Softmax Logistic Regression",
        "batch_size": str(batch_size),
        "in_dim": str(in_dim),
        "num_classes": str(num_classes),
    }

    out_path = Path("weights/01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core.parity.save_reference_checkpoint(
        path=out_path,
        weights_dict=weights_dict,
        input_tensor=test_input,
        output_tensor=test_output.detach(),
        metadata=metadata,
    )
    print(f"Successfully generated golden checkpoint at {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_topic_01_cat01_weights()
