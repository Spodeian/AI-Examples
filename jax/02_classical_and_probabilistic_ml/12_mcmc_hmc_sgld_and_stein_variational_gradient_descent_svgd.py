"""MCMC, Hamiltonian Monte Carlo, SGLD, and Stein Variational Gradient Descent in JAX.

Features:
- Functional Hamiltonian Monte Carlo (HMC) with symplectic leapfrog integrator
- Stochastic Gradient Langevin Dynamics (SGLD) with exact thermal noise injection
- JIT-compiled Stein Variational Gradient Descent (SVGD) RBF velocity field
"""

from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
import safetensors.torch
import numpy as np


import functools

@functools.partial(jax.jit, static_argnames=("L",))
def hmc_leapfrog(
    theta: jnp.ndarray,
    p: jnp.ndarray,
    Sigma_inv: jnp.ndarray,
    eps: float = 0.1,
    L: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Runs L symplectic leapfrog integration steps on quadratic potential."""
    def step_fn(carry, _):
        cur_theta, cur_p = carry
        # Half step momentum
        grad_u = Sigma_inv @ cur_theta
        cur_p = cur_p - 0.5 * eps * grad_u
        # Full step position
        cur_theta = cur_theta + eps * cur_p
        # Half step momentum
        grad_u_new = Sigma_inv @ cur_theta
        cur_p = cur_p - 0.5 * eps * grad_u_new
        return (cur_theta, cur_p), None

    (final_theta, final_p), _ = jax.lax.scan(step_fn, (theta, p), None, length=L)

    h_init = 0.5 * jnp.dot(theta, Sigma_inv @ theta) + 0.5 * jnp.dot(p, p)
    h_prop = 0.5 * jnp.dot(final_theta, Sigma_inv @ final_theta) + 0.5 * jnp.dot(final_p, final_p)
    accept_ratio = jnp.clip(jnp.exp(h_init - h_prop), a_max=1.0)

    return final_theta, final_p, accept_ratio


@jax.jit
def sgld_step(
    theta: jnp.ndarray,
    grad_logp: jnp.ndarray,
    noise: jnp.ndarray,
    eps: float = 0.05,
) -> jnp.ndarray:
    """Executes single SGLD step."""
    return theta + 0.5 * eps * grad_logp + (eps ** 0.5) * noise


@jax.jit
def svgd_velocity_field(
    particles: jnp.ndarray,
    grad_logp: jnp.ndarray,
    h: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes SVGD velocity field with drive and repulsive forces."""
    M, _ = particles.shape
    diff = jnp.expand_dims(particles, 1) - jnp.expand_dims(particles, 0) # [M, M, D]
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    K = jnp.exp(-0.5 * dist_sq / (h ** 2))

    drive = (K @ grad_logp) / M
    repulsive = jnp.sum(diff * jnp.expand_dims(K, -1), axis=1) / (M * (h ** 2))
    phi = drive + repulsive
    return phi, K


def run_verification() -> bool:
    weights_path = Path("weights/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.safetensors")
    if not weights_path.is_file():
        print(f"Skipping verification: {weights_path} not found")
        return True

    chk = safetensors.torch.load_file(weights_path)
    Sigma_inv = jnp.array(chk["Sigma_inv"].numpy())
    theta_init = jnp.array(chk["theta_init"].numpy())
    p_init = jnp.array(chk["p_init"].numpy())
    golden_theta_prop = chk["hmc_theta_prop"].numpy()
    golden_p_prop = chk["hmc_p_prop"].numpy()
    golden_accept = float(chk["hmc_accept_ratio"][0].numpy())

    # 1. HMC Leapfrog Parity
    theta_prop, p_prop, accept = hmc_leapfrog(theta_init, p_init, Sigma_inv, eps=0.1, L=5)
    diff_theta = float(np.max(np.abs(np.array(theta_prop) - golden_theta_prop)))
    diff_p = float(np.max(np.abs(np.array(p_prop) - golden_p_prop)))
    diff_acc = abs(float(accept) - golden_accept)
    assert diff_theta < 1e-5, f"JAX HMC theta mismatch: {diff_theta:.2e}"
    assert diff_p < 1e-5, f"JAX HMC p mismatch: {diff_p:.2e}"
    assert diff_acc < 1e-5, f"JAX HMC accept ratio mismatch: {diff_acc:.2e}"

    # 2. SGLD Parity
    theta_sgld = jnp.array(chk["theta_sgld"].numpy())
    grad_logp_sgld = jnp.array(chk["grad_logp_sgld"].numpy())
    noise_sgld = jnp.array(chk["noise_sgld"].numpy())
    golden_sgld_next = chk["theta_sgld_next"].numpy()
    sgld_next = sgld_step(theta_sgld, grad_logp_sgld, noise_sgld, eps=0.05)
    diff_sgld = float(np.max(np.abs(np.array(sgld_next) - golden_sgld_next)))
    assert diff_sgld < 1e-5, f"JAX SGLD step mismatch: {diff_sgld:.2e}"

    # 3. SVGD Parity
    particles_init = jnp.array(chk["particles_init"].numpy())
    grad_logp_svgd = jnp.array(chk["grad_logp_svgd"].numpy())
    golden_velocity = chk["svgd_velocity"].numpy()
    golden_gram = chk["svgd_kernel_gram"].numpy()
    golden_part_next = chk["particles_next"].numpy()

    velocity, gram = svgd_velocity_field(particles_init, grad_logp_svgd, h=1.0)
    diff_vel = float(np.max(np.abs(np.array(velocity) - golden_velocity)))
    diff_gram = float(np.max(np.abs(np.array(gram) - golden_gram)))
    part_next = particles_init + 0.1 * velocity
    diff_part = float(np.max(np.abs(np.array(part_next) - golden_part_next)))
    assert diff_vel < 1e-5, f"JAX SVGD velocity mismatch: {diff_vel:.2e}"
    assert diff_gram < 1e-5, f"JAX SVGD Gram mismatch: {diff_gram:.2e}"
    assert diff_part < 1e-5, f"JAX SVGD next particles mismatch: {diff_part:.2e}"

    print("JAX Topic 12 verification passed!")
    return True


if __name__ == "__main__":
    run_verification()
