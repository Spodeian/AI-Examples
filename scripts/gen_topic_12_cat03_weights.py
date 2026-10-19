"""Generate deterministic reference safetensors checkpoint for Category 03 Topic 12:
12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
from safetensors.torch import save_file

import core


def compute_highway(x: torch.Tensor, w_h: torch.Tensor, b_h: torch.Tensor, w_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    h = torch.relu(x @ w_h + b_h)
    t = torch.sigmoid(x @ w_t + b_t)
    return t * h + (1.0 - t) * x


def compute_glu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # x: [B, D], w: [D, 2*D], b: [2*D]
    z = x @ w + b
    d = z.shape[-1] // 2
    a = z[..., :d]
    gate = torch.sigmoid(z[..., d:])
    return a * gate


def compute_hard_concrete(
    log_alpha: torch.Tensor,
    beta: float = 0.67,
    gamma: float = -0.1,
    zeta: float = 1.1,
    u: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # u in (0, 1)
    logit_u = torch.log(u) - torch.log(1.0 - u)
    s = torch.sigmoid((logit_u + log_alpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    g = torch.clamp(s_bar, min=0.0, max=1.0)
    # Expected L0 penalty: P(g > 0) = sigmoid(log_alpha - beta * log(-gamma / zeta))
    offset = beta * math.log(-gamma / zeta)
    l0_prob = torch.sigmoid(log_alpha - offset)
    return g, l0_prob


def compute_sparsemax(z: torch.Tensor) -> torch.Tensor:
    # z: [B, K]
    # Sort descending
    z_sorted, _ = torch.sort(z, descending=True, dim=-1)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    z_cumsum = torch.cumsum(z_sorted, dim=-1)
    # Condition: 1 + k * z_k > cumsum
    support = (1.0 + k * z_sorted) > z_cumsum
    # k_star is the number of support elements (last True index + 1)
    k_star = torch.sum(support.to(dtype=k.dtype), dim=-1, keepdim=True)
    # Tau = (cumsum_k* - 1) / k*
    tau = (torch.gather(z_cumsum, -1, (k_star - 1).long()) - 1.0) / k_star
    return torch.clamp(z - tau, min=0.0)


def compute_pondernet_halting(
    lambdas: torch.Tensor,
    lambda_p: float = 0.3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # lambdas: [B, N], hazard rate at each step n in (0, 1)
    B, N = lambdas.shape
    p = torch.zeros_like(lambdas)
    cum_unhalted = torch.ones(B, 1, device=lambdas.device)
    for n in range(N):
        p[:, n : n + 1] = lambdas[:, n : n + 1] * cum_unhalted
        cum_unhalted = cum_unhalted * (1.0 - lambdas[:, n : n + 1])

    # Expected steps
    steps = torch.arange(1, N + 1, device=lambdas.device, dtype=lambdas.dtype).unsqueeze(0)
    expected_steps = torch.sum(p * steps, dim=-1)

    # Geometric prior: p*_n = lambda_p * (1 - lambda_p)^(n - 1)
    geom_factors = (1.0 - lambda_p) ** (steps - 1.0)
    p_star = lambda_p * geom_factors  # [1, N]
    # KL(p || p*)
    kl = torch.sum(p * (torch.log(p.clamp(min=eps)) - torch.log(p_star.clamp(min=eps))), dim=-1)
    return p, expected_steps, kl


def compute_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # x: [B, D], gamma: [B, D], beta: [B, D]
    return gamma * x + beta


def generate_weights() -> None:
    core.setup_env(seed=42)
    gen = torch.Generator().manual_seed(42)

    batch_size = 8
    dim = 4

    # 1. Linear Anchor
    weight = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    bias = torch.randn(dim, generator=gen, dtype=torch.float32)
    test_input = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    test_output = test_input @ weight + bias

    # 2. Highway Network
    w_h = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    b_h = torch.randn(dim, generator=gen, dtype=torch.float32)
    w_t = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    b_t = torch.randn(dim, generator=gen, dtype=torch.float32)
    highway_x = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    expected_highway = compute_highway(highway_x, w_h, b_h, w_t, b_t)

    # 3. Gated Linear Unit (GLU)
    w_glu = torch.randn(dim, 2 * dim, generator=gen, dtype=torch.float32)
    b_glu = torch.randn(2 * dim, generator=gen, dtype=torch.float32)
    glu_x = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    expected_glu = compute_glu(glu_x, w_glu, b_glu)

    # 4. Hard Concrete Stochastic Gates
    log_alpha = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    hard_concrete_u = torch.rand(batch_size, dim, generator=gen, dtype=torch.float32) * 0.98 + 0.01
    expected_hard_concrete_g, expected_hard_concrete_l0 = compute_hard_concrete(
        log_alpha, beta=0.67, gamma=-0.1, zeta=1.1, u=hard_concrete_u
    )

    # 5. Sparsemax Exact-Zero Simplex Projection
    # Generate logits with some large negatives so Sparsemax actually zeroes them out
    sparsemax_logits = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32) * 2.0
    sparsemax_logits[:, -1] = -5.0  # Force exact zero
    expected_sparsemax = compute_sparsemax(sparsemax_logits)

    # 6. PonderNet Halting Hazards
    raw_hazards = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    ponder_lambdas = torch.sigmoid(raw_hazards)
    expected_ponder_p, expected_ponder_steps, expected_ponder_kl = compute_pondernet_halting(
        ponder_lambdas, lambda_p=0.3
    )

    # 7. FiLM (Feature-wise Linear Modulation)
    film_x = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    film_gamma = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    film_beta = torch.randn(batch_size, dim, generator=gen, dtype=torch.float32)
    expected_film = compute_film(film_x, film_gamma, film_beta)

    tensors = {
        "weight": weight,
        "bias": bias,
        "test_input": test_input,
        "test_output": test_output,
        # Highway
        "w_h": w_h,
        "b_h": b_h,
        "w_t": w_t,
        "b_t": b_t,
        "highway_x": highway_x,
        "expected_highway": expected_highway,
        # GLU
        "w_glu": w_glu,
        "b_glu": b_glu,
        "glu_x": glu_x,
        "expected_glu": expected_glu,
        # Hard Concrete
        "log_alpha": log_alpha,
        "hard_concrete_u": hard_concrete_u,
        "expected_hard_concrete_g": expected_hard_concrete_g,
        "expected_hard_concrete_l0": expected_hard_concrete_l0,
        # Sparsemax
        "sparsemax_logits": sparsemax_logits,
        "expected_sparsemax": expected_sparsemax,
        # PonderNet
        "ponder_lambdas": ponder_lambdas,
        "expected_ponder_p": expected_ponder_p,
        "expected_ponder_steps": expected_ponder_steps,
        "expected_ponder_kl": expected_ponder_kl,
        # FiLM
        "film_x": film_x,
        "film_gamma": film_gamma,
        "film_beta": film_beta,
        "expected_film": expected_film,
    }

    out_dir = Path("weights/03_building_blocks_and_losses")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting.safetensors"
    save_file(tensors, out_path)
    print(f"Saved {len(tensors)} tensors to {out_path}")


if __name__ == "__main__":
    generate_weights()
