"""MCMC, Hamiltonian Monte Carlo, SGLD, and Stein Variational Gradient Descent in PyTorch 2.x.

Features:
- Hamiltonian Monte Carlo (HMC) with symplectic leapfrog integrator and Metropolis acceptance
- Stochastic Gradient Langevin Dynamics (SGLD) with exact thermal noise scaling sqrt(eps)
- Stein Variational Gradient Descent (SVGD) RBF kernel repulsive and driving dynamics
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch


def hmc_leapfrog(
    theta: torch.Tensor,
    p: torch.Tensor,
    Sigma_inv: torch.Tensor,
    eps: float = 0.1,
    L: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs L symplectic leapfrog integration steps on quadratic potential."""
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

    h_init = 0.5 * torch.dot(theta, Sigma_inv @ theta) + 0.5 * torch.dot(p, p)
    h_prop = 0.5 * torch.dot(cur_theta, Sigma_inv @ cur_theta) + 0.5 * torch.dot(cur_p, cur_p)
    accept_ratio = torch.clamp(torch.exp(h_init - h_prop), max=1.0)

    return cur_theta, cur_p, accept_ratio


def sgld_step(
    theta: torch.Tensor,
    grad_logp: torch.Tensor,
    noise: torch.Tensor,
    eps: float = 0.05,
) -> torch.Tensor:
    """Executes single SGLD step: theta + 0.5 * eps * grad_logp + sqrt(eps) * noise."""
    return theta + 0.5 * eps * grad_logp + (eps ** 0.5) * noise


def svgd_velocity_field(
    particles: torch.Tensor,
    grad_logp: torch.Tensor,
    h: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes SVGD velocity field with drive and repulsive forces."""
    M, D = particles.shape
    diff = particles.unsqueeze(1) - particles.unsqueeze(0) # [M, M, D]
    dist_sq = torch.sum(diff ** 2, dim=-1)
    K = torch.exp(-0.5 * dist_sq / (h ** 2))

    drive = (K @ grad_logp) / M
    repulsive = torch.sum(diff * K.unsqueeze(-1), dim=1) / (M * (h ** 2))
    phi = drive + repulsive
    return phi, K


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    Sigma_inv = chk["Sigma_inv"]
    theta_init = chk["theta_init"]
    p_init = chk["p_init"]
    golden_theta_prop = chk["hmc_theta_prop"]
    golden_p_prop = chk["hmc_p_prop"]
    golden_accept = chk["hmc_accept_ratio"][0].item()

    # 1. HMC Leapfrog Parity
    theta_prop, p_prop, accept = hmc_leapfrog(theta_init, p_init, Sigma_inv, eps=0.1, L=5)
    diff_theta = torch.max(torch.abs(theta_prop - golden_theta_prop)).item()
    diff_p = torch.max(torch.abs(p_prop - golden_p_prop)).item()
    diff_acc = abs(accept.item() - golden_accept)
    assert diff_theta < 1e-5, f"HMC proposed theta mismatch: {diff_theta:.2e}"
    assert diff_p < 1e-5, f"HMC proposed p mismatch: {diff_p:.2e}"
    assert diff_acc < 1e-5, f"HMC acceptance ratio mismatch: {diff_acc:.2e}"

    # 2. SGLD Parity
    theta_sgld = chk["theta_sgld"]
    grad_logp_sgld = chk["grad_logp_sgld"]
    noise_sgld = chk["noise_sgld"]
    golden_sgld_next = chk["theta_sgld_next"]
    sgld_next = sgld_step(theta_sgld, grad_logp_sgld, noise_sgld, eps=0.05)
    diff_sgld = torch.max(torch.abs(sgld_next - golden_sgld_next)).item()
    assert diff_sgld < 1e-5, f"SGLD step mismatch: {diff_sgld:.2e}"

    # 3. SVGD Parity
    particles_init = chk["particles_init"]
    grad_logp_svgd = chk["grad_logp_svgd"]
    golden_velocity = chk["svgd_velocity"]
    golden_gram = chk["svgd_kernel_gram"]
    golden_part_next = chk["particles_next"]

    velocity, gram = svgd_velocity_field(particles_init, grad_logp_svgd, h=1.0)
    diff_vel = torch.max(torch.abs(velocity - golden_velocity)).item()
    diff_gram = torch.max(torch.abs(gram - golden_gram)).item()
    part_next = particles_init + 0.1 * velocity
    diff_part = torch.max(torch.abs(part_next - golden_part_next)).item()
    assert diff_vel < 1e-5, f"SVGD velocity mismatch: {diff_vel:.2e}"
    assert diff_gram < 1e-5, f"SVGD Gram matrix mismatch: {diff_gram:.2e}"
    assert diff_part < 1e-5, f"SVGD next particles mismatch: {diff_part:.2e}"

    print("PyTorch Topic 12 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
