# Surrogate Models, Bayesian Optimization & Differentiable Proxies

## 1. Black-Box Optimization & The Surrogate Paradigm

In many scientific and engineering domains—such as aerostructure design, PDE-constrained inverse problems, high-dimensional chemical synthesis, and deep neural network hyperparameter tuning—the objective function $f(x)$ is a **black box**:
1. Evaluating $f(x)$ requires expensive physical experiments or long-running supercomputer simulations (hours to days per evaluation).
2. The objective function lacks closed-form analytical equations and does not provide first-order or second-order derivatives ($\nabla f(x)$ is unavailable).
3. Query budgets are severely constrained (typically $N \le 100 \dots 1000$ function evaluations).

**Surrogate modeling** replaces the expensive simulator $f(x)$ with a lightweight statistical or machine-learned proxy $\hat{f}(x)$. Unlike standard point-prediction regression, an effective surrogate model for **Bayesian Optimization (BO)** must provide **well-calibrated epistemic uncertainty estimates**:
$$f(x) \sim \mathcal{N}\bigl(\mu(x), \sigma^2(x)\bigr).$$
Uncertainty quantification enables Bayesian optimization to balance the exploration of unvisited parameter regimes (high $\sigma(x)$) with the exploitation of regions known to yield high performance (high $\mu(x)$).

---

## 2. Gaussian Process Regression (Kriging) & Covariance Kernels

A **Gaussian Process (GP)** is a collection of infinitely many random variables, any finite subset of which has a joint multivariate Gaussian distribution. A GP is completely specified by a mean function $m(x)$ (frequently assumed to be zero without loss of generality) and a covariance kernel function $k(x, x')$:
$$f(x) \sim \mathcal{GP}\bigl(m(x), k(x, x')\bigr).$$

The choice of covariance kernel encodes prior assumptions regarding continuity, differentiability, and lengthscale isotropy:

1. **Squared Exponential / Radial Basis Function (RBF) Kernel**:
   $$k_{\text{RBF}}(x, x') = \sigma_f^2 \exp\left( -\frac{1}{2} (x - x')^T \Lambda^{-1} (x - x') \right),$$
   where $\Lambda = \operatorname{diag}(\ell_1^2, \dots, \ell_D^2)$ defines **Automatic Relevance Determination (ARD)** lengthscales. The RBF kernel generates sample paths that are infinitely mean-square differentiable ($C^\infty$), modeling exceptionally smooth underlying processes.
2. **Matérn Class Kernels**:
   $$k_{\nu=5/2}(r) = \sigma_f^2 \left(1 + \sqrt{5} r + \frac{5}{3} r^2 \right) \exp(-\sqrt{5} r), \quad r = \sqrt{(x - x')^T \Lambda^{-1} (x - x')}.$$
   The Matérn 5/2 kernel produces sample paths that are twice mean-square differentiable ($C^2$), providing greater resilience to non-smooth physical phenomena.

Given observed training data $\mathcal{D} = \{(x_n, y_n)\}_{n=1}^N$ with Gaussian observation noise $y = f(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$, the joint distribution of observed targets $y$ and function value $f_\star$ at query point $x_\star$ is:
$$\begin{bmatrix} y \\ f_\star \end{bmatrix} \sim \mathcal{N}\left( \mathbf{0}, \begin{bmatrix} K(X, X) + \sigma_n^2 I & K(X, x_\star) \\ K(x_\star, X) & k(x_\star, x_\star) \end{bmatrix} \right).$$

Conditioning on observations yields the analytical Gaussian posterior:
$$f_\star \mid \mathcal{D}, x_\star \sim \mathcal{N}\bigl(\mu(x_\star), \sigma^2(x_\star)\bigr),$$
$$\mu(x_\star) = K(x_\star, X) \bigl( K(X, X) + \sigma_n^2 I \bigr)^{-1} y,$$
$$\sigma^2(x_\star) = k(x_\star, x_\star) - K(x_\star, X) \bigl( K(X, X) + \sigma_n^2 I \bigr)^{-1} K(X, x_\star).$$

---

## 3. Bayesian Optimization Acquisition Functions

Given posterior predictions $\mu(x)$ and $\sigma(x)$, an **acquisition function** $\alpha(x)$ quantifies the utility of querying candidate point $x$. The next query point is selected by solving an internal continuous optimization problem:
$$x_{N+1} = \arg\max_{x \in \mathcal{X}} \alpha(x).$$

### 3.1 Expected Improvement (EI)
For maximization with incumbent best value $y^+ = \max_{n} y_n$, the improvement is $I(x) = \max(0, f(x) - y^+ - \xi)$, where $\xi \ge 0$ balances exploration. Integrating over the Gaussian posterior distribution yields the analytical closed-form:
$$\alpha_{\text{EI}}(x) = (\mu(x) - y^+ - \xi) \Phi(Z) + \sigma(x) \phi(Z),$$
where $Z = \frac{\mu(x) - y^+ - \xi}{\sigma(x)}$ (with $\alpha_{\text{EI}}(x) = 0$ if $\sigma(x) = 0$), $\phi(Z) = \frac{1}{\sqrt{2\pi}} e^{-Z^2 / 2}$ is the standard normal PDF, and $\Phi(Z) = \frac{1}{2}\left(1 + \operatorname{erf}(Z / \sqrt{2})\right)$ is the standard normal CDF.

### 3.2 Upper Confidence Bound (GP-UCB)
Derived from sub-Gaussian concentration inequalities and multi-armed bandit theory:
$$\alpha_{\text{UCB}}(x) = \mu(x) + \beta_t \sigma(x),$$
where $\beta_t = \sqrt{2 \log(t^{D/2 + 2} \pi^2 / 3 \delta)}$ yields sub-linear cumulative regret guarantees $\mathcal{O}(\sqrt{T \gamma_T})$.

### 3.3 Probability of Improvement (PI)
$$\alpha_{\text{PI}}(x) = P(f(x) \ge y^+ + \xi) = \Phi(Z).$$

---

## 4. Differentiable Proxies & Gradient-Based Policy Search

While standard Bayesian optimization samples points one by one, modern AI applications require optimizing candidates through **gradient-based acquisition maximization**:
$$\nabla_x \alpha(x) = \frac{\partial \alpha}{\partial \mu} \nabla_x \mu(x) + \frac{\partial \alpha}{\partial \sigma} \nabla_x \sigma(x).$$
Because the kernel $k(x, X)$ is differentiable with respect to $x$:
$$\nabla_x k(x, x') = -\Lambda^{-1} (x - x') k(x, x'),$$
the posterior mean and variance gradients $\nabla_x \mu(x)$ and $\nabla_x \sigma(x)$ can be computed analytically or via automatic differentiation. This allows standard gradient-based optimizers (Adam, L-BFGS) to rapidly search high-dimensional input spaces for the optimal acquisition peak.

Furthermore, neural networks (such as DeepONets or FNOs) trained on physical simulation ensembles serve as **differentiable surrogate proxies**:
$$\min_u \mathcal{L}_{\text{task}}(u; \hat{\mathcal{G}}_\theta(u)),$$
enabling end-to-end backpropagation from downstream engineering criteria directly to design controls $u$.

---

## 5. High-Dimensional BO: Additive Kernels & Trust Regions (TuRBO)

As dimension $D$ increases beyond $20$, standard GPs face two catastrophic challenges:
1. **Curse of Dimensionality in Covariance Space**: Points in high dimensions become equidistant, driving kernel correlations $k(x, x') \to 0$ everywhere except in immediate $\epsilon$-neighborhoods.
2. **Global Uncertainty Overshooting**: Purely global acquisition functions over-explore high-uncertainty boundary regimes that yield poor objective performance.

State-of-the-art strategies mitigate this:
- **Trust Region Bayesian Optimization (TuRBO)**: Restricts candidate selection to localized, dynamically expanding and shrinking hyper-rectangles (trust regions) centered around the current incumbent. This allows local modeling of high-dimensional non-convex functions while preventing search divergence.
- **Additive Kernel Decompositions**:
  $$k(x, x') = \sum_{m=1}^M k_m(x^{(m)}, x'^{(m)}),$$
  where each component $k_m$ operates on a low-dimensional disjoint subspace $x^{(m)} \subset x$, reducing sample complexity.

---

## 6. Multi-Fidelity & Batch Acquisition (q-EI)

In real-world engineering workflows, multiple information sources with differing costs and fidelities exist (e.g., coarse-grid 2D Euler simulations vs. fine-grid 3D Navier-Stokes DNS):

1. **Multi-Fidelity Modeling (Co-Kriging)**:
   An autoregressive Gaussian process links low-fidelity evaluations $f_L(x)$ to high-fidelity targets $f_H(x)$:
   $$f_H(x) = \rho(x) f_L(x) + \delta(x),$$
   where $\rho(x)$ is a scaling factor and $\delta(x) \sim \mathcal{GP}(0, k_\delta)$ models the residual discrepancy. Evaluating inexpensive low-fidelity models drastically refines knowledge of the high-fidelity landscape.
2. **Parallel Batch Acquisition ($q$-EI / $q$-UCB)**:
   Modern HPC clusters require selecting batches of $q > 1$ points $\{x_1, \dots, x_q\}$ simultaneously. Using the **reparameterization trick**:
   $$f(x_{1:q}) = \mu(x_{1:q}) + L(x_{1:q}) \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_q),$$
   Monte Carlo acquisition functions can be maximized directly via sample-path gradients.

---

## 7. Numerical Conditioning & Cholesky Stability

Computing the Gaussian process posterior requires inverting the $N \times N$ kernel Gram matrix $K_{yy} = K(X, X) + \sigma_n^2 I$.
In practice, two numerical pitfalls occur:
1. **Near-Singular Gram Matrices**: As candidate points cluster together, rows of $K(X, X)$ become nearly linearly dependent, driving the condition number $\kappa(K_{yy}) > 10^8$. Adding a positive nugget (Tikhonov jitter $\sigma_n^2 \ge 10^{-6}$) and performing Cholesky factorization $K_{yy} = L L^T$ ensures strictly stable linear solves.
2. **Precision Loss in Small Slacks**: When computing Expected Improvement for points where $\mu(x) \ll y^+$, $Z \to -\infty$, causing severe catastrophic floating-point cancellation in $(\mu - y^+) \Phi(Z)$. Evaluating in log-space via asymptotic expansions of $\operatorname{erfc}(|Z|)$ prevents underflow.

---

## 8. Cross-Track Verification & Architectural Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every surrogate and BO operator satisfies the following strict contracts:

| Operator | Verification Specification | Tolerance |
| :--- | :--- | :--- |
| **RBF Kernel Parity** | Cross-track Gram matrix $K(x, x')$ matches golden reference | $< 10^{-5}$ |
| **Posterior Variance Positivity** | Posterior predictive variance $\sigma^2(x_\star) \ge 0$ everywhere | Strictly $\ge 0$ |
| **Training Interpolation** | At training points $X$, $\mu(X) \approx y$ within observation noise $\sigma_n^2$ | $< 10^{-4}$ |
| **EI Acquisition Parity** | Expected Improvement matches across Abramowitz-Stegun erf and native autograd | $< 10^{-5}$ |
| **UCB Acquisition Parity** | Upper Confidence Bound $\mu + \beta \sigma$ matches golden across runtimes | $< 10^{-5}$ |
