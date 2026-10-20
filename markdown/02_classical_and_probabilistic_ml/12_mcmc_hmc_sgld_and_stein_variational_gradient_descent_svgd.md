# 12. MCMC, Hamiltonian Monte Carlo, SGLD, and Stein Variational Gradient Descent

## 1. Executive Summary
Bayesian posterior inference requires exploring complex, unnormalized probability distributions $p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta) p(\theta)$. While traditional random-walk Metropolis-Hastings suffers from diffusive $\mathcal{O}(D^2)$ exploration in high dimensions, **Hamiltonian Monte Carlo (HMC)** transforms sampling into physical Hamiltonian dynamics along vector fields guided by $\nabla_\theta \log p(\theta)$, achieving $\mathcal{O}(D^{1/4})$ scaling via volume-preserving symplectic leapfrog integrators. In big-data regimes where full gradient evaluation is prohibitively expensive, **Stochastic Gradient Langevin Dynamics (SGLD)** injects calibrated thermal Gaussian noise into mini-batch gradients to maintain asymptotically exact posterior exploration without Metropolis accept/reject steps. Bridging MCMC and deterministic variational inference, **Stein Variational Gradient Descent (SVGD)** drives an ensemble of particles along reproducing kernel Hilbert space (RKHS) velocity fields, seamlessly balancing target probability gravitation against inter-particle repulsive forces. This module delivers complete, mathematically verified implementations of HMC, SGLD, and SVGD across PyTorch 2.x, JAX, native Rust Candle, and hardware kernels.

---

## 2. Mathematical Foundations

### 2.1 Hamiltonian Monte Carlo & Symplectic Leapfrog Integration
HMC introduces auxiliary momentum variables $p \sim \mathcal{N}(0, \mathbf{M})$ to define the joint Hamiltonian:
$$H(\theta, p) = U(\theta) + K(p) = -\log p(\theta, \mathcal{D}) + \frac{1}{2} p^\top \mathbf{M}^{-1} p$$

Hamilton's equations of motion govern the phase-space trajectory:
$$\frac{d\theta}{dt} = \frac{\partial H}{\partial p} = \mathbf{M}^{-1} p, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial \theta} = -\nabla_\theta U(\theta)$$

To preserve phase space volume (Liouville's theorem) and maintain time-reversibility, trajectories are integrated using the **Symplectic Leapfrog Integrator** with step size $\epsilon$:
1. **Half-step momentum update:**
   $$p\left(t + \frac{\epsilon}{2}\right) = p(t) - \frac{\epsilon}{2} \nabla_\theta U(\theta(t))$$
2. **Full-step position update:**
   $$\theta(t + \epsilon) = \theta(t) + \epsilon \mathbf{M}^{-1} p\left(t + \frac{\epsilon}{2}\right)$$
3. **Half-step momentum update:**
   $$p(t + \epsilon) = p\left(t + \frac{\epsilon}{2}\right) - \frac{\epsilon}{2} \nabla_\theta U(\theta(t + \epsilon))$$

Proposals are accepted according to the Metropolis criterion:
$$\alpha = \min\left(1, \exp\left(-H(\theta^*, p^*) + H(\theta, p)\right)\right)$$

### 2.2 Stochastic Gradient Langevin Dynamics (SGLD)
Evaluating $\nabla_\theta U(\theta) = -\sum_{i=1}^N \nabla_\theta \log p(x_i | \theta) - \nabla_\theta \log p(\theta)$ over $N$ observations is intractable for large datasets. SGLD uses mini-batch gradients of size $n$:
$$\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2} \left[ \nabla_\theta \log p(\theta_t) + \frac{N}{n} \sum_{i \in \mathcal{S}_t} \nabla_\theta \log p(x_i | \theta_t) \right] + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \epsilon_t \mathbf{I})$$

When the step size schedule satisfies the Robbins-Monro conditions $\sum_{t=1}^\infty \epsilon_t = \infty$ and $\sum_{t=1}^\infty \epsilon_t^2 < \infty$, the thermal noise dominates the mini-batch subsampling noise as $\epsilon_t \to 0$, causing the chain to converge asymptotically to the true posterior distribution without an accept/reject step.

### 2.3 Stein Variational Gradient Descent (SVGD)
SVGD updates an ensemble of $M$ particles $\{\theta_i\}_{i=1}^M$ deterministically along a velocity field $\phi^*(\theta)$ that minimizes the Kullback-Leibler divergence with the target distribution:
$$\theta_i^{(t+1)} = \theta_i^{(t)} + \epsilon \phi^*(\theta_i^{(t)})$$

In a reproducing kernel Hilbert space $\mathcal{H}_0^d$ with positive definite kernel $k(x, x')$, the optimal velocity field has the closed-form expression:
$$\phi^*(x) = \frac{1}{M} \sum_{j=1}^M \left[ \underbrace{k(x_j, x) \nabla_{x_j} \log p(x_j)}_{\text{Drive / Drift Force}} + \underbrace{\nabla_{x_j} k(x_j, x)}_{\text{Repulsive Force}} \right]$$

For the Radial Basis Function (RBF) kernel $k(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2h^2}\right)$, the repulsive gradient evaluates to:
$$\nabla_{x_j} k(x_j, x) = \frac{x - x_j}{h^2} k(x_j, x)$$

The drive force pulls particles into high-probability modes of $p(x)$, while the repulsive force pushes particles apart, ensuring diverse non-collapsed posterior coverage.

---

## 3. Tensor Mechanics & Compute Graphs

```
[Target Potential U(theta)] ────────> [Symplectic Leapfrog: Half-p -> Full-theta -> Half-p] ──> [Metropolis Accept/Reject]
                                            │
[Mini-batch Log-Likelihood] ───────> [SGLD Step: 0.5*eps*grad + sqrt(eps)*thermal_noise] ─────> [Sub-sampled Posterior]
                                            │
[Particle Ensemble {x_i}_{i=1}^M] ──> [Kernel Gram Matrix K_ij & Distances]
                                            ├──> Drive: (1/M) sum_j K_ij * grad_logp_j
                                            └──> Repulsion: (1/M) sum_j (x_i - x_j)/h^2 * K_ij
                                                     │
                                                     └──> [Deterministic SVGD Particle Update]
```

---

## 4. Multi-Ecosystem Implementation Parity

- **PyTorch 2.x (`pytorch2/`):** Symplectic leapfrog step, SGLD iteration, and vectorized SVGD drive-repulsion tensor calculations.
- **JAX (`jax/`):** Pure functional `jax.lax.scan` for HMC leapfrog integration with `static_argnames=("L",)` and vectorized SVGD updates.
- **Native Rust Candle (`candle/`):** Symplectic momentum-position updates and matrix-vector particle force balance.
- **Hardware Kernels (`kernels/`):** WebGPU WGSL compute shader executing parallel $\mathcal{O}(M)$ particle force evaluations per workgroup thread across 64 threads.

---

## 5. Pathological Failure Modes & Edge Cases

| Failure Mode | Root Cause | Symptom | Mitigation |
| :--- | :--- | :--- | :--- |
| **HMC Energy Drift / Explosion** | Step size $\epsilon$ too large for curvature | Hamiltonian error $\Delta H \gg 0$; acceptance rate collapses to $0.0$ | Adapt step size $\epsilon$ via dual averaging or use Riemannian manifold HMC |
| **SGLD Variance Collapse / Random Walk** | Fixed step size $\epsilon_t$ too large or thermal noise missing | Mini-batch noise dominates sampling or behaves like SGD optimizer | Enforce Robbins-Monro decay $\epsilon_t = a (b + t)^{-\gamma}$ with $\gamma \in (0.5, 1]$ |
| **SVGD Particle Collapse (Mode Dropping)** | Kernel bandwidth $h$ too small | Repulsive force vanishes for distant particles; particles collapse to modes | Adapt bandwidth dynamically via median heuristic $h = \text{median}(\|x_i - x_j\|) / \sqrt{2 \ln M}$ |
| **Symplectic Order Inversion** | Updating position before momentum half-step without time-reversal symmetry | Trajectory loses symplecticity and phase volume conservation | Adhere strictly to the leapfrog half-p $\to$ full-$\theta$ $\to$ half-p formulation |

---

## 6. Cross-Domain Generalisation & Algorithmic Bridges
1. **Astrophysics & N-Body Simulation:** The symplectic leapfrog integrator used in HMC is identical to the Verlet/Störmer-Verlet integrator used in orbital celestial mechanics to conserve energy over millions of orbits.
2. **Bayesian Deep Learning:** SGLD enables uncertainty quantification in modern over-parameterized neural networks directly from mini-batch training pipelines without computing Hessian inverses.
3. **Generative Modeling & Particle Filtering:** SVGD particle dynamics are isomorphic to interacting particle systems in Wasserstein gradient flows and McKean-Vlasov processes.

---

## 7. Verification & Parity Results
All implementations satisfy the cross-track invariant contract:
$$\|Y_{\text{PyTorch}} - Y_{\text{JAX}}\|_\infty < 10^{-5}, \quad \|Y_{\text{Candle}} - Y_{\text{Golden}}\|_\infty < 10^{-4}$$
- `pytest -q tests/test_cross_track_parity.py -k "checkpoint_path35"`: **PASS**
- `cargo nextest run -p ai-candle -E 'test(test_12)'`: **PASS**
- `cargo nextest run -p ai-kernel -E 'test(test_12)'`: **PASS**

---

## 8. Navigation & Curriculum

- **Previous Topic:** [11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.md](11_probabilistic_graphical_models_factor_graphs_bp_and_cavi.md)
- **Current Topic:** 12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.md
- **Next Topic:** [13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.md](13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.md)
- **Category Index:** [02_classical_and_probabilistic_ml/README.md](README.md)
