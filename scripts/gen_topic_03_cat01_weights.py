"""Generate golden weights and reference I/O tensors for Topic 03:
03_discrete_vs_continuous_latents_and_gradient_estimators.

Pillars:
1. Continuous Pathwise Reparameterization (Gaussian VAE: z = mu + sigma * eps)
2. Discrete Relaxed Gumbel-Softmax (Concrete distribution with temperature tau)
3. Score Function Bernoulli Estimator (REINFORCE log-derivative)
4. Canonical Linear Anchor for cross-track test harness
"""

from pathlib import Path

import torch
import torch.nn.functional as F

import core


def generate_topic_03_cat01_weights(output_path: Path) -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    in_dim = 4
    latent_dim = 4
    out_dim = 2

    # 1. Test input batch
    x = torch.randn(batch_size, in_dim, generator=gen, dtype=torch.float32)

    # 2. Canonical linear reference anchor (for core.parity.load_reference_checkpoint)
    weight = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    test_output = torch.matmul(x, weight) + bias

    # 3. Continuous Pathwise Reparameterization Parameters
    w_mu = torch.randn(in_dim, latent_dim, generator=gen, dtype=torch.float32)
    b_mu = torch.randn(latent_dim, generator=gen, dtype=torch.float32)
    w_logvar = torch.randn(in_dim, latent_dim, generator=gen, dtype=torch.float32)
    b_logvar = torch.randn(latent_dim, generator=gen, dtype=torch.float32)

    mu = torch.matmul(x, w_mu) + b_mu  # [B, latent_dim]
    logvar = torch.matmul(x, w_logvar) + b_logvar  # [B, latent_dim]
    std = torch.exp(0.5 * logvar)

    # Deterministic standard normal noise
    eps = torch.randn(batch_size, latent_dim, generator=gen, dtype=torch.float32)
    continuous_z = mu + std * eps  # [B, latent_dim]

    # Gaussian analytical KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_divergence = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B]

    # Linear decoder
    w_dec = torch.randn(latent_dim, out_dim, generator=gen, dtype=torch.float32)
    b_dec = torch.randn(out_dim, generator=gen, dtype=torch.float32)
    continuous_out = torch.matmul(continuous_z, w_dec) + b_dec  # [B, out_dim]

    # 4. Discrete Relaxed Gumbel-Softmax Parameters
    w_logits = torch.randn(in_dim, latent_dim, generator=gen, dtype=torch.float32)
    b_logits = torch.randn(latent_dim, generator=gen, dtype=torch.float32)
    temperature = torch.tensor([1.0], dtype=torch.float32)

    logits = torch.matmul(x, w_logits) + b_logits  # [B, latent_dim]

    # Deterministic standard Gumbel noise from uniform u: g = -log(-log(u))
    uniform_u = torch.rand(batch_size, latent_dim, generator=gen, dtype=torch.float32).clamp(
        1e-6, 1.0 - 1e-6
    )
    gumbel_noise = -torch.log(-torch.log(uniform_u))
    gumbel_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)  # [B, latent_dim]
    gumbel_out = torch.matmul(gumbel_soft, w_dec) + b_dec  # [B, out_dim]

    # 5. Bernoulli REINFORCE Score Function Parameters
    w_bernoulli = torch.randn(in_dim, latent_dim, generator=gen, dtype=torch.float32)
    b_bernoulli = torch.randn(latent_dim, generator=gen, dtype=torch.float32)
    bernoulli_logits = torch.matmul(x, w_bernoulli) + b_bernoulli
    bernoulli_probs = torch.sigmoid(bernoulli_logits)  # [B, latent_dim]

    # Assemble weights dict
    weights_dict = {
        "weight": weight.detach(),
        "bias": bias.detach(),
        "w_mu": w_mu.detach(),
        "b_mu": b_mu.detach(),
        "w_logvar": w_logvar.detach(),
        "b_logvar": b_logvar.detach(),
        "eps": eps.detach(),
        "continuous_z": continuous_z.detach(),
        "kl_divergence": kl_divergence.detach(),
        "w_dec": w_dec.detach(),
        "b_dec": b_dec.detach(),
        "continuous_out": continuous_out.detach(),
        "w_logits": w_logits.detach(),
        "b_logits": b_logits.detach(),
        "temperature": temperature.detach(),
        "gumbel_noise": gumbel_noise.detach(),
        "gumbel_soft": gumbel_soft.detach(),
        "gumbel_out": gumbel_out.detach(),
        "w_bernoulli": w_bernoulli.detach(),
        "b_bernoulli": b_bernoulli.detach(),
        "bernoulli_probs": bernoulli_probs.detach(),
    }

    core.parity.save_reference_checkpoint(
        path=output_path,
        weights_dict=weights_dict,
        input_tensor=x,
        output_tensor=test_output,
        metadata={
            "topic": "03_discrete_vs_continuous_latents_and_gradient_estimators",
            "category": "01_paradigms_and_dichotomies",
            "batch_size": str(batch_size),
            "in_dim": str(in_dim),
            "latent_dim": str(latent_dim),
            "out_dim": str(out_dim),
        },
    )

    print(f"Generated Topic 03 Category 01 weights at: {output_path}")
    print(f"Input shape: {x.shape}")
    print(f"Continuous Out shape: {continuous_out.shape}")
    print(f"Gumbel Soft shape: {gumbel_soft.shape}")
    print(f"KL Divergence shape: {kl_divergence.shape}")


if __name__ == "__main__":
    out_file = (
        Path(__file__).resolve().parent.parent
        / "weights"
        / "01_paradigms_and_dichotomies"
        / "03_discrete_vs_continuous_latents_and_gradient_estimators.safetensors"
    )
    generate_topic_03_cat01_weights(out_file)
