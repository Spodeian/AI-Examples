"""Generate golden weights and reference I/O tensors for Topic 04:
04_deterministic_vs_stochastic_and_mixture_density_networks.

Pillars:
1. Deterministic Conditional Expectation (MSE Point Estimator)
2. Mixture Density Network (MDN: mixing coefficients pi, component means mu, scales sigma)
3. Implicit / Direct Quantile Estimator (Pinball Loss at quantiles tau = [0.1, 0.5, 0.9])
4. Stable Log-Likelihood computation via LogSumExp
5. Canonical Linear Anchor for cross-track test harness
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

import core


def generate_topic_04_cat01_weights(output_path: Path) -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    in_dim = 4
    hidden_dim = 8
    num_components = 3  # K=3
    out_dim = 1
    num_quantiles = 3

    # 1. Test input batch and target batch
    x = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)
    sample_y = torch.randn(batch_size, out_dim, generator=gen, dtype=torch.float32)

    # 2. Canonical linear reference anchor
    weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    test_output = torch.matmul(x, weight) + bias

    # 3. Feature backbone (shared ReLU MLP)
    w_h = torch.randn(in_dim, hidden_dim, generator=gen, dtype=torch.float32)
    b_h = torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    hidden = F.relu(torch.matmul(x, w_h) + b_h)  # [B, H]

    # 4. Deterministic Point Estimator Head
    w_det = torch.randn(hidden_dim, out_dim, generator=gen, dtype=torch.float32)
    b_det = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    y_det = torch.matmul(hidden, w_det) + b_det  # [B, 1]

    # 5. Mixture Density Network Head
    # Mixing logits -> pi in simplex
    w_pi = torch.randn(hidden_dim, num_components, generator=gen, dtype=torch.float32)
    b_pi = torch.randn(num_components, generator=gen, dtype=torch.float32)
    pi_logits = torch.matmul(hidden, w_pi) + b_pi
    pi = F.softmax(pi_logits, dim=-1)  # [B, K]

    # Component means
    w_mu = torch.randn(hidden_dim, num_components, generator=gen, dtype=torch.float32)
    b_mu = torch.randn(num_components, generator=gen, dtype=torch.float32)
    mu = torch.matmul(hidden, w_mu) + b_mu  # [B, K]

    # Component log-scales -> positive sigma
    w_sigma = torch.randn(hidden_dim, num_components, generator=gen, dtype=torch.float32) * 0.1
    b_sigma = torch.randn(num_components, generator=gen, dtype=torch.float32) * 0.1
    log_sigma = torch.matmul(hidden, w_sigma) + b_sigma
    sigma = torch.exp(log_sigma)  # [B, K]

    # 6. Stable Gaussian Mixture Log-Likelihood:
    # log p(y|x) = LogSumExp_k [ log(pi_k) - log(sigma_k) - 0.5*log(2*pi) - 0.5 * ((y - mu_k) / sigma_k)^2 ]
    log_pi = F.log_softmax(pi_logits, dim=-1)  # [B, K]
    diff = sample_y - mu  # [B, K] broadcasted over out_dim=1
    log_gaussian = -0.5 * math.log(2.0 * math.pi) - log_sigma - 0.5 * (diff / sigma).pow(2)  # [B, K]
    mdn_log_probs = torch.logsumexp(log_pi + log_gaussian, dim=-1)  # [B]

    # 7. Quantile Estimator Head (tau = [0.1, 0.5, 0.9])
    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    w_quantile = torch.randn(hidden_dim, num_quantiles, generator=gen, dtype=torch.float32)
    b_quantile = torch.randn(num_quantiles, generator=gen, dtype=torch.float32)
    y_quantiles = torch.matmul(hidden, w_quantile) + b_quantile  # [B, Q]

    # Pinball Loss: max(tau * (y - q), (tau - 1) * (y - q))
    u = sample_y - y_quantiles  # [B, Q]
    pinball_loss = torch.maximum(quantiles * u, (quantiles - 1.0) * u)  # [B, Q]

    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "w_h": w_h.detach(),
        "b_h": b_h.detach(),
        "w_det": w_det.detach(),
        "b_det": b_det.detach(),
        "w_pi": w_pi.detach(),
        "b_pi": b_pi.detach(),
        "w_mu": w_mu.detach(),
        "b_mu": b_mu.detach(),
        "w_sigma": w_sigma.detach(),
        "b_sigma": b_sigma.detach(),
        "quantiles": quantiles.detach(),
        "w_quantile": w_quantile.detach(),
        "b_quantile": b_quantile.detach(),
        "sample_y": sample_y.detach(),
        "hidden": hidden.detach(),
        "y_det": y_det.detach(),
        "pi": pi.detach(),
        "mu": mu.detach(),
        "sigma": sigma.detach(),
        "mdn_log_probs": mdn_log_probs.detach(),
        "y_quantiles": y_quantiles.detach(),
        "pinball_loss": pinball_loss.detach(),
    }

    core.parity.save_reference_checkpoint(
        path=output_path,
        weights_dict=weights_dict,
        input_tensor=x,
        output_tensor=test_output,
        metadata={
            "topic": "04_deterministic_vs_stochastic_and_mixture_density_networks",
            "category": "01_paradigms_and_dichotomies",
            "batch_size": str(batch_size),
            "in_dim": str(in_dim),
            "hidden_dim": str(hidden_dim),
            "num_components": str(num_components),
            "out_dim": str(out_dim),
            "num_quantiles": str(num_quantiles),
        },
    )

    print(f"Generated Topic 04 Category 01 weights at: {output_path}")
    print(f"Input shape: {x.shape}")
    print(f"y_det shape: {y_det.shape}")
    print(f"pi shape: {pi.shape}")
    print(f"mu shape: {mu.shape}")
    print(f"sigma shape: {sigma.shape}")
    print(f"mdn_log_probs shape: {mdn_log_probs.shape}")
    print(f"y_quantiles shape: {y_quantiles.shape}")


if __name__ == "__main__":
    out_file = (
        Path(__file__).resolve().parent.parent
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "04_deterministic_vs_stochastic_and_mixture_density_networks.safetensors"
    )
    generate_topic_04_cat01_weights(out_file)
