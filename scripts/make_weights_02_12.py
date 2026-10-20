"""Generates reference SafeTensors checkpoint for Category 02 Topic 12:
12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.safetensors
"""

from __future__ import annotations
from pathlib import Path
import torch
import safetensors.torch


def hmc_leapfrog_step(
    theta: torch.Tensor,
    p: torch.Tensor,
    Sigma_inv: torch.Tensor,
    eps: float = 0.1,
    L: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs L leapfrog steps on quadratic potential U(theta) = 0.5 * theta^T Sigma_inv theta."""
    # Grad U(theta) = Sigma_inv @ theta
    cur_theta = theta.clone()
    cur_p = p.clone()

    for _ in range(L):
        # Half step for momentum
        grad_u = Sigma_inv @ cur_theta
        cur_p = cur_p - 0.5 * eps * grad_u
        # Full step for position
        cur_theta = cur_theta + eps * cur_p
        # Half step for momentum
        grad_u_new = Sigma_inv @ cur_theta
        cur_p = cur_p - 0.5 * eps * grad_u_new

    # Acceptance probability
    # H = U(theta) + 0.5 * p^T p
    h_init = 0.5 * torch.dot(theta, Sigma_inv @ theta) + 0.5 * torch.dot(p, p)
    h_prop = 0.5 * torch.dot(cur_theta, Sigma_inv @ cur_theta) + 0.5 * torch.dot(cur_p, cur_p)
    accept_ratio = torch.clamp(torch.exp(h_init - h_prop), max=1.0)

    return cur_theta, cur_p, accept_ratio


def compute_svgd_velocity(
    particles: torch.Tensor,    # [M, D]
    grad_logp: torch.Tensor,    # [M, D]
    h: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes SVGD velocity field phi(x) = (1/M) sum_j [k(x_j, x) grad log p(x_j) + grad_{x_j} k(x_j, x)]."""
    M, D = particles.shape
    diff = particles.unsqueeze(1) - particles.unsqueeze(0)  # [M, M, D] where diff[i, j] = x_i - x_j
    dist_sq = torch.sum(diff ** 2, dim=-1)                  # [M, M]
    K = torch.exp(-0.5 * dist_sq / (h ** 2))                # [M, M]

    # Drive term: (1/M) * K @ grad_logp -> [M, D]
    drive = (K @ grad_logp) / M

    # Repulsive term: grad_{x_j} k(x_j, x_i) = -(x_j - x_i)/h^2 * k(x_j, x_i) = (x_i - x_j)/h^2 * k(x_j, x_i)
    # Sum over j: (1/M) sum_j diff[i, j] / h^2 * K[i, j]
    repulsive = torch.sum(diff * K.unsqueeze(-1), dim=1) / (M * (h ** 2))

    phi = drive + repulsive
    return phi, K


def generate_checkpoint():
    torch.manual_seed(42)

    # 1. Target potential: 2D Gaussian with correlation
    Sigma_target = torch.tensor([[1.0, 0.6], [0.6, 1.0]], dtype=torch.float32)
    Sigma_inv = torch.linalg.inv(Sigma_target)

    theta_init = torch.tensor([1.2, -0.8], dtype=torch.float32)
    p_init = torch.tensor([0.5, 0.3], dtype=torch.float32)

    hmc_theta_prop, hmc_p_prop, hmc_accept_ratio = hmc_leapfrog_step(
        theta_init, p_init, Sigma_inv, eps=0.1, L=5
    )

    # 2. SGLD step
    eps_sgld = 0.05
    theta_sgld = torch.tensor([0.8, -0.5], dtype=torch.float32)
    grad_u = Sigma_inv @ theta_sgld
    grad_logp_sgld = -grad_u
    noise_sgld = torch.tensor([0.02, -0.015], dtype=torch.float32) # precomputed thermal noise
    theta_sgld_next = theta_sgld + 0.5 * eps_sgld * grad_logp_sgld + (eps_sgld ** 0.5) * noise_sgld

    # 3. SVGD particles
    M, D = 8, 2
    particles_init = torch.tensor([
        [-1.5, -1.2],
        [-0.8, -0.6],
        [-0.2, 0.1],
        [0.4, 0.5],
        [1.1, 0.9],
        [1.6, 1.4],
        [-1.0, 0.8],
        [0.7, -1.1],
    ], dtype=torch.float32)

    grad_logp_svgd = -(particles_init @ Sigma_inv.T)
    svgd_velocity, svgd_kernel_gram = compute_svgd_velocity(particles_init, grad_logp_svgd, h=1.0)
    particles_next = particles_init + 0.1 * svgd_velocity

    # Golden linear mapping for test_cross_track_parity.py
    D_core = 4
    test_input = torch.randn(8, D_core)
    weight = torch.randn(D_core, D_core)
    bias = torch.randn(D_core)
    test_output = test_input @ weight + bias

    checkpoint_data = {
        "weight": weight.clone().contiguous(),
        "bias": bias.clone().contiguous(),
        "test_input": test_input.clone().contiguous(),
        "test_output": test_output.clone().contiguous(),
        "Sigma_target": Sigma_target.clone().contiguous(),
        "Sigma_inv": Sigma_inv.clone().contiguous(),
        "theta_init": theta_init.clone().contiguous(),
        "p_init": p_init.clone().contiguous(),
        "hmc_theta_prop": hmc_theta_prop.clone().contiguous(),
        "hmc_p_prop": hmc_p_prop.clone().contiguous(),
        "hmc_accept_ratio": hmc_accept_ratio.unsqueeze(0).clone().contiguous(),
        "theta_sgld": theta_sgld.clone().contiguous(),
        "grad_logp_sgld": grad_logp_sgld.clone().contiguous(),
        "noise_sgld": noise_sgld.clone().contiguous(),
        "theta_sgld_next": theta_sgld_next.clone().contiguous(),
        "particles_init": particles_init.clone().contiguous(),
        "grad_logp_svgd": grad_logp_svgd.clone().contiguous(),
        "svgd_velocity": svgd_velocity.clone().contiguous(),
        "svgd_kernel_gram": svgd_kernel_gram.clone().contiguous(),
        "particles_next": particles_next.clone().contiguous(),
    }

    out_dir = Path("weights/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.safetensors"
    safetensors.torch.save_file(checkpoint_data, out_path)
    print("Saved Topic 12 reference checkpoint to", out_path)


if __name__ == "__main__":
    generate_checkpoint()
