"""Generate 8-cell educational notebook for Topic 12:
12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.ipynb
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 12. MCMC, Hamiltonian Monte Carlo, SGLD & SVGD

- **Category:** 02. Classical & Probabilistic ML
- **Lineage & Citations:** Simon Duane et al. (1987) *Hybrid Monte Carlo*; Radford M. Neal (2011) *MCMC using Hamiltonian dynamics*; Max Welling & Yee Whye Teh (2011) *Bayesian Learning via Stochastic Gradient Langevin Dynamics*; Qiang Liu & Dilin Wang (2016) *Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm*.
- **Purpose:** Unifies gradient-guided Bayesian posterior exploration, massive-scale mini-batch sampling, and deterministic particle variational inference via Hamiltonian Monte Carlo (HMC), Stochastic Gradient Langevin Dynamics (SGLD), and Stein Variational Gradient Descent (SVGD).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use HMC with large un-tuned step sizes $\\epsilon$ on ill-conditioned or discontinuous energy landscapes; symplectic error explodes, collapsing acceptance rates to 0.0. Do not use SGLD with constant non-decaying learning rates, as mini-batch noise prevents asymptotic convergence to the true posterior. Do not apply SVGD with fixed small kernel bandwidths in high dimensions without median-heuristic adaptation, or particles will suffer repulsive decay and mode collapse."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Hamiltonian Monte Carlo & Symplectic Leapfrog Integration
Phase space Hamiltonian $H(\\theta, p) = U(\\theta) + \\frac{1}{2} p^\\top \\mathbf{M}^{-1} p$ where $U(\\theta) = -\\log p(\\theta, \\mathcal{D})$:
$$\\frac{d\\theta}{dt} = \\mathbf{M}^{-1} p, \\quad \\frac{dp}{dt} = -\\nabla_\\theta U(\\theta)$$
Symplectic leapfrog integration preserves phase volume and satisfies time-reversibility:
$$p\\left(t + \\frac{\\epsilon}{2}\\right) = p(t) - \\frac{\\epsilon}{2} \\nabla_\\theta U(\\theta(t))$$
$$\\theta(t + \\epsilon) = \\theta(t) + \\epsilon \\mathbf{M}^{-1} p\\left(t + \\frac{\\epsilon}{2}\\right)$$
$$p(t + \\epsilon) = p\\left(t + \\frac{\\epsilon}{2}\\right) - \\frac{\\epsilon}{2} \\nabla_\\theta U(\\theta(t + \\epsilon))$$
Metropolis acceptance ratio: $\\alpha = \\min(1, \\exp(-H(\\theta^*, p^*) + H(\\theta, p)))$.

### 2. Stochastic Gradient Langevin Dynamics (SGLD)
Subsamples $N$ data points with mini-batches $\\mathcal{S}_t$ of size $n$, injecting thermal Gaussian noise:
$$\\theta_{t+1} = \\theta_t + \\frac{\\epsilon_t}{2} \\left[ \\nabla_\\theta \\log p(\\theta_t) + \\frac{N}{n} \\sum_{i \\in \\mathcal{S}_t} \\nabla_\\theta \\log p(x_i | \\theta_t) \\right] + \\eta_t, \\quad \\eta_t \\sim \\mathcal{N}(0, \\epsilon_t \\mathbf{I})$$

### 3. Stein Variational Gradient Descent (SVGD)
Drives $M$ particles along optimal velocity field $\\phi^*(x) = \\frac{1}{M} \\sum_{j=1}^M [k(x_j, x) \\nabla \\log p(x_j) + \\nabla_{x_j} k(x_j, x)]$:
$$\\phi^*(x) = \\frac{1}{M} \\sum_{j=1}^M \\left[ k(x_j, x) \\nabla_{x_j} \\log p(x_j) + \\frac{x - x_j}{h^2} k(x_j, x) \\right]$$

**Complexity:** HMC: $\\mathcal{O}(L \\cdot D)$ per proposal; SGLD: $\\mathcal{O}(n \\cdot D)$ per step; SVGD: $\\mathcal{O}(M^2 D)$ pairwise interaction."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import safetensors.torch
from pathlib import Path
import core

env = core.setup_env(seed=42)

# Load Reference Checkpoint
ckpt_path = Path("../../weights/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.safetensors")
ckpt = safetensors.torch.load_file(ckpt_path) if ckpt_path.exists() else {}

Sigma_inv = ckpt.get("Sigma_inv", torch.linalg.inv(torch.tensor([[1.0, 0.6], [0.6, 1.0]])))
theta_init = ckpt.get("theta_init", torch.tensor([1.2, -0.8]))
p_init = ckpt.get("p_init", torch.tensor([0.5, 0.3]))
particles_init = ckpt.get("particles_init", torch.randn(8, 2))
grad_logp_svgd = ckpt.get("grad_logp_svgd", -(particles_init @ Sigma_inv.T))

print(f"Setup complete on {env.device}: HMC theta={theta_init.tolist()}, Particles={particles_init.shape}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class MCMCToolkit:
    @staticmethod
    def hmc_leapfrog(theta: torch.Tensor, p: torch.Tensor, Sigma_inv: torch.Tensor, eps: float = 0.1, L: int = 5):
        cur_t, cur_p = theta.clone(), p.clone()
        for _ in range(L):
            cur_p = cur_p - 0.5 * eps * (Sigma_inv @ cur_t)
            cur_t = cur_t + eps * cur_p
            cur_p = cur_p - 0.5 * eps * (Sigma_inv @ cur_t)
        h_init = 0.5 * torch.dot(theta, Sigma_inv @ theta) + 0.5 * torch.dot(p, p)
        h_prop = 0.5 * torch.dot(cur_t, Sigma_inv @ cur_t) + 0.5 * torch.dot(cur_p, cur_p)
        alpha = torch.clamp(torch.exp(h_init - h_prop), max=1.0)
        return cur_t, cur_p, alpha

    @staticmethod
    def sgld_step(theta: torch.Tensor, grad_logp: torch.Tensor, noise: torch.Tensor, eps: float = 0.05):
        return theta + 0.5 * eps * grad_logp + (eps ** 0.5) * noise

    @staticmethod
    def svgd_velocity(particles: torch.Tensor, grad_logp: torch.Tensor, h: float = 1.0):
        M, D = particles.shape
        diff = particles.unsqueeze(1) - particles.unsqueeze(0)  # [M, M, D]
        dist_sq = torch.sum(diff ** 2, dim=-1)
        K = torch.exp(-0.5 * dist_sq / (h ** 2))
        drive = (K @ grad_logp) / M
        repulsive = torch.sum(diff * K.unsqueeze(-1), dim=1) / (M * (h ** 2))
        return drive + repulsive, K"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """# Quantitative Invariants & Precision Verification
theta_prop, p_prop, accept_ratio = MCMCToolkit.hmc_leapfrog(theta_init, p_init, Sigma_inv, eps=0.1, L=5)
vel_svgd, K_svgd = MCMCToolkit.svgd_velocity(particles_init, grad_logp_svgd, h=1.0)

# Invariant 1: HMC acceptance ratio must be bounded in [0, 1]
assert 0.0 <= accept_ratio.item() <= 1.0, f"Invalid acceptance ratio {accept_ratio.item()}"

# Invariant 2: Symplectic energy conservation: H_prop - H_init must be small for eps=0.1
h_init = 0.5 * torch.dot(theta_init, Sigma_inv @ theta_init) + 0.5 * torch.dot(p_init, p_init)
h_prop = 0.5 * torch.dot(theta_prop, Sigma_inv @ theta_prop) + 0.5 * torch.dot(p_prop, p_prop)
delta_h = torch.abs(h_prop - h_init).item()
assert delta_h < 0.15, f"Symplectic leapfrog energy drift too large: {delta_h:.4f}"

# Invariant 3: SVGD Kernel Gram matrix is symmetric positive definite with unit diagonal
assert torch.allclose(torch.diag(K_svgd), torch.ones(len(particles_init)), atol=1e-5), "K diagonal != 1"
assert torch.allclose(K_svgd, K_svgd.T, atol=1e-5), "K is not symmetric"

if "hmc_theta_prop" in ckpt:
    assert torch.allclose(theta_prop, ckpt["hmc_theta_prop"], atol=1e-5), "HMC parity failed"
    assert torch.allclose(vel_svgd, ckpt["svgd_velocity"], atol=1e-5), "SVGD parity failed"

print(f"Verification passed: HMC accept={accept_ratio.item():.4f}, Delta H={delta_h:.4f}, K_00={K_svgd[0, 0].item():.1f}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = """from core.benchmark import benchmark_fn, format_benchmark_table

b_hmc = benchmark_fn(lambda: MCMCToolkit.hmc_leapfrog(theta_init, p_init, Sigma_inv, eps=0.1, L=5), iters=30, name="HMC Leapfrog (L=5)")
b_sgld = benchmark_fn(lambda: MCMCToolkit.sgld_step(theta_init, -(Sigma_inv @ theta_init), torch.randn(2), eps=0.05), iters=30, name="SGLD Single Step")
b_svgd = benchmark_fn(lambda: MCMCToolkit.svgd_velocity(particles_init, grad_logp_svgd, h=1.0), iters=30, name="SVGD Velocity (M=8, D=2)")
print(format_benchmark_table([b_hmc, b_sgld, b_svgd]))"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Non-Symplectic Euler vs Symplectic Leapfrog Integration
# Explicit Euler breaks phase space volume conservation, causing exponential Hamiltonian explosion!
t_euler, p_euler = theta_init.clone(), p_init.clone()
t_leap, p_leap = theta_init.clone(), p_init.clone()
eps_test = 0.2

for _ in range(30):
    # Non-symplectic forward Euler (simultaneous update from old states)
    grad_u = Sigma_inv @ t_euler
    new_t = t_euler + eps_test * p_euler
    new_p = p_euler - eps_test * grad_u
    t_euler, p_euler = new_t, new_p

    # Symplectic Leapfrog
    p_leap = p_leap - 0.5 * eps_test * (Sigma_inv @ t_leap)
    t_leap = t_leap + eps_test * p_leap
    p_leap = p_leap - 0.5 * eps_test * (Sigma_inv @ t_leap)

h_euler = (0.5 * torch.dot(t_euler, Sigma_inv @ t_euler) + 0.5 * torch.dot(p_euler, p_euler)).item()
h_leap = (0.5 * torch.dot(t_leap, Sigma_inv @ t_leap) + 0.5 * torch.dot(p_leap, p_leap)).item()

print(f"Initial Hamiltonian:        {h_init:.4f}")
print(f"Non-Symplectic Euler H(30): {h_euler:.4f} (Catastrophic energy explosion!)")
print(f"Symplectic Leapfrog H(30):  {h_leap:.4f} (Bounded stable oscillation)")
assert h_euler > h_leap * 5.0, "Non-symplectic Euler must exhibit severe energy drift" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Phase Space Volume Conservation:** Symplectic leapfrog integrators preserve Poincaré invariants and phase volume, restricting energy drift to bounded $\\mathcal{O}(\\epsilon^2)$ oscillations.
2. **Thermal Mini-batch Scaling:** SGLD balances injected Gaussian noise against subsampling gradient variance, achieving scalable Bayesian inference on massive datasets without rejection steps.
3. **Stein Variational Force Balance:** SVGD operates via a dual-force mechanism: the drive force guides particles toward modes while the kernel repulsive gradient maintains non-collapsed posterior coverage.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.md](../../markdown/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.md)
- **PyTorch 2.x Module:** [12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.py](../../pytorch2/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.py)
- **Functional JAX Module:** [12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.py](../../jax/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.py)
- **Native Rust Candle Track:** [12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.rs](../../candle/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.rs)
- **Hardware WGSL Kernel:** [12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.rs](../../kernels/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.rs)
- **Golden Reference Checkpoint:** [12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.safetensors](../../weights/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous: 11. Factor Graphs, Belief Propagation & CAVI](11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.ipynb) | [Category Index](../README.md) | [Next: 13. Tractable Probabilistic Circuits, SPN, Copulas, EVT & SBI](13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/02_classical_and_probabilistic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
