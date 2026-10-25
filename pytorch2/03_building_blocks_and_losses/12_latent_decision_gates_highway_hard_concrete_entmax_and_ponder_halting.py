"""PyTorch 2.x implementation of Category 03 Topic 12:
Latent Decision Gates: Highway Networks, Hard Concrete L0, Sparsemax, PonderNet Halting, and FiLM.
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file


class LinearAnchor(nn.Module):
    """Linear anchor module: Y = X @ W + b."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


def highway_forward(
    x: torch.Tensor,
    w_h: torch.Tensor,
    b_h: torch.Tensor,
    w_t: torch.Tensor,
    b_t: torch.Tensor,
) -> torch.Tensor:
    """Highway network layer: y = t * relu(x @ W_h + b_h) + (1 - t) * x."""
    h = torch.relu(x @ w_h + b_h)
    t = torch.sigmoid(x @ w_t + b_t)
    return t * h + (1.0 - t) * x


def glu_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Gated Linear Unit: z = x @ W + b, y = z[..., :d] * sigmoid(z[..., d:])."""
    z = x @ w + b
    d = z.shape[-1] // 2
    a = z[..., :d]
    gate = torch.sigmoid(z[..., d:])
    return a * gate


def hard_concrete_forward(
    log_alpha: torch.Tensor,
    beta: float = 0.67,
    gamma: float = -0.1,
    zeta: float = 1.1,
    u: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard Concrete stretched-and-rectified gate with closed-form expected L0 penalty."""
    logit_u = torch.log(u) - torch.log(1.0 - u)
    s = torch.sigmoid((logit_u + log_alpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    g = torch.clamp(s_bar, min=0.0, max=1.0)
    offset = beta * math.log(-gamma / zeta)
    l0_prob = torch.sigmoid(log_alpha - offset)
    return g, l0_prob


def sparsemax_forward(z: torch.Tensor) -> torch.Tensor:
    """Exact-zero projection of logits onto the probability simplex."""
    z_sorted, _ = torch.sort(z, descending=True, dim=-1)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    z_cumsum = torch.cumsum(z_sorted, dim=-1)
    support = (1.0 + k * z_sorted) > z_cumsum
    k_star = torch.sum(support.to(dtype=k.dtype), dim=-1, keepdim=True)
    tau = (torch.gather(z_cumsum, -1, (k_star - 1).long()) - 1.0) / k_star
    return torch.clamp(z - tau, min=0.0)


def pondernet_halting_forward(
    lambdas: torch.Tensor,
    lambda_p: float = 0.3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PonderNet geometric halting hazard formulation with expected steps and KL prior."""
    B, N = lambdas.shape
    p = torch.zeros_like(lambdas)
    cum_unhalted = torch.ones(B, 1, device=lambdas.device)
    for n in range(N):
        p[:, n : n + 1] = lambdas[:, n : n + 1] * cum_unhalted
        cum_unhalted = cum_unhalted * (1.0 - lambdas[:, n : n + 1])

    steps = torch.arange(1, N + 1, device=lambdas.device, dtype=lambdas.dtype).unsqueeze(0)
    expected_steps = torch.sum(p * steps, dim=-1)

    geom_factors = (1.0 - lambda_p) ** (steps - 1.0)
    p_star = lambda_p * geom_factors
    kl = torch.sum(p * (torch.log(p.clamp(min=eps)) - torch.log(p_star.clamp(min=eps))), dim=-1)
    return p, expected_steps, kl


def film_forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Feature-wise Linear Modulation: y = gamma * x + beta."""
    return gamma * x + beta


def verify_topic_12_pytorch() -> None:
    weights_path = Path("weights/03_building_blocks_and_losses/12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting.safetensors")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights checkpoint: {weights_path}")

    tensors = load_file(str(weights_path))
    print(f"Loaded {len(tensors)} tensors from {weights_path.name}")

    # 1. Linear Anchor Parity
    anchor = LinearAnchor(4, 4)
    with torch.no_grad():
        anchor.weight.copy_(tensors["weight"])
        anchor.bias.copy_(tensors["bias"])
    pred_linear = anchor(tensors["test_input"])
    err_linear = (pred_linear - tensors["test_output"]).abs().max().item()
    assert err_linear < 1e-5, f"Linear anchor error {err_linear:.2e} exceeds 1e-5"

    # 2. Highway Network Parity
    pred_highway = highway_forward(
        tensors["highway_x"], tensors["w_h"], tensors["b_h"], tensors["w_t"], tensors["b_t"]
    )
    err_highway = (pred_highway - tensors["expected_highway"]).abs().max().item()
    assert err_highway < 1e-5, f"Highway error {err_highway:.2e} exceeds 1e-5"

    # 3. GLU Parity
    pred_glu = glu_forward(tensors["glu_x"], tensors["w_glu"], tensors["b_glu"])
    err_glu = (pred_glu - tensors["expected_glu"]).abs().max().item()
    assert err_glu < 1e-5, f"GLU error {err_glu:.2e} exceeds 1e-5"

    # 4. Hard Concrete Parity
    pred_hc_g, pred_hc_l0 = hard_concrete_forward(
        tensors["log_alpha"], beta=0.67, gamma=-0.1, zeta=1.1, u=tensors["hard_concrete_u"]
    )
    err_hc_g = (pred_hc_g - tensors["expected_hard_concrete_g"]).abs().max().item()
    err_hc_l0 = (pred_hc_l0 - tensors["expected_hard_concrete_l0"]).abs().max().item()
    assert err_hc_g < 1e-5, f"Hard concrete g error {err_hc_g:.2e} exceeds 1e-5"
    assert err_hc_l0 < 1e-5, f"Hard concrete L0 error {err_hc_l0:.2e} exceeds 1e-5"

    # 5. Sparsemax Parity
    pred_sparsemax = sparsemax_forward(tensors["sparsemax_logits"])
    err_sparsemax = (pred_sparsemax - tensors["expected_sparsemax"]).abs().max().item()
    assert err_sparsemax < 1e-5, f"Sparsemax error {err_sparsemax:.2e} exceeds 1e-5"

    # 6. PonderNet Halting Parity
    pred_ponder_p, pred_ponder_steps, pred_ponder_kl = pondernet_halting_forward(
        tensors["ponder_lambdas"], lambda_p=0.3
    )
    err_p = (pred_ponder_p - tensors["expected_ponder_p"]).abs().max().item()
    err_steps = (pred_ponder_steps - tensors["expected_ponder_steps"]).abs().max().item()
    err_kl = (pred_ponder_kl - tensors["expected_ponder_kl"]).abs().max().item()
    assert err_p < 1e-5, f"PonderNet p error {err_p:.2e} exceeds 1e-5"
    assert err_steps < 1e-5, f"PonderNet steps error {err_steps:.2e} exceeds 1e-5"
    assert err_kl < 1e-5, f"PonderNet KL error {err_kl:.2e} exceeds 1e-5"

    # 7. FiLM Parity
    pred_film = film_forward(tensors["film_x"], tensors["film_gamma"], tensors["film_beta"])
    err_film = (pred_film - tensors["expected_film"]).abs().max().item()
    assert err_film < 1e-5, f"FiLM error {err_film:.2e} exceeds 1e-5"

    print(
        f"All 7 PyTorch operations verified! Anchor: {err_linear:.2e}, Highway: {err_highway:.2e}, "
        f"GLU: {err_glu:.2e}, HardConcrete: {err_hc_g:.2e}/{err_hc_l0:.2e}, Sparsemax: {err_sparsemax:.2e}, "
        f"PonderNet: {err_p:.2e}/{err_steps:.2e}/{err_kl:.2e}, FiLM: {err_film:.2e}"
    )


def export_golden() -> None:
    """Regenerates golden reference weights deterministically (seed 42)."""
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


    generate_weights()


if __name__ == "__main__":
    import sys

    if "--export-golden" in sys.argv[1:]:
        export_golden()

    verify_topic_12_pytorch()
