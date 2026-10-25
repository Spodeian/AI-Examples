# Koopman Operators, SINDy Equation Discovery & 4D-Var Data Assimilation

## 1. Data-Driven Dynamical Systems: Global Linearization vs. Sparse Discovery

Nonlinear dynamical systems govern phenomena across physical domains, from climate and atmospheric fluid dynamics to orbital celestial mechanics and biological morphogenesis:
$$\dot{x}(t) = F(x(t)), \quad x(t) \in \mathcal{M} \subseteq \mathbb{R}^D.$$

When modeling such systems from observational data, two fundamental challenges arise:
1. **Geometric Non-linearity**: Local linearizations (Jacobians $J_F = \frac{\partial F}{\partial x}$) are valid only within infinitesimally small neighborhoods of fixed points or limit cycles.
2. **Missing Analytical Governing Laws**: In complex multi-physics settings (turbulent fluid flow, epidemiological contagion, macroeconomic feedback), the exact governing equations $F(x)$ are unknown.

Two revolutionary mathematical paradigms address these challenges:
- **Koopman Operator Theory**: Lifts finite-dimensional nonlinear state dynamics into an infinite-dimensional Hilbert space of scalar observables $\psi(x)$ where the evolution is **globally linear**.
- **Sparse Identification of Nonlinear Dynamics (SINDy)**: Discovers parsimonious, interpretable analytical differential equations directly from measurement data by combining candidate non-linear function libraries with sparse regression.
- **4D-Var Data Assimilation**: Optimally synthesizes physical dynamical models with sparse, noisy streaming observational data over a spatio-temporal assimilation window.

---

## 2. Koopman Operator Theory & Neural Lifting Architectures

Formulated by B. O. Koopman (1931), the **Koopman operator** $\mathcal{K}$ acts on scalar observation functions $g: \mathcal{M} \to \mathbb{C}$ of the state space:
$$\mathcal{K} g(x) = g(F(x)).$$
Crucially, while the underlying state dynamic $F$ is nonlinear, $\mathcal{K}$ is a **linear operator**:
$$\mathcal{K} \bigl(\alpha g_1(x) + \beta g_2(x)\bigr) = \alpha g_1(F(x)) + \beta g_2(F(x)) = \alpha \mathcal{K} g_1(x) + \beta \mathcal{K} g_2(x).$$

The spectral decomposition of $\mathcal{K}$ yields Koopman eigenvalues $\mu_j$ and eigenfunctions $\phi_j(x)$ satisfying:
$$\mathcal{K} \phi_j(x) = \phi_j(F(x)) = \mu_j \phi_j(x).$$
In the eigenfunction coordinate system, multi-step nonlinear time evolution collapses into simple scalar multiplications:
$$\phi_j(x(t)) = e^{\lambda_j t} \phi_j(x(0)), \quad \lambda_j = \ln(\mu_j) / \Delta t.$$

**Deep Koopman Networks** parameterize the coordinate transformation using an autoencoder framework:
1. **Encoder / Lifting Function**: $\psi = \phi_{\text{enc}}(x; \theta) \in \mathbb{R}^K$.
2. **Linear Koopman Evolution**: $\psi_{t+1} = \psi_t K_{\text{mat}}$, where $K_{\text{mat}} \in \mathbb{R}^{K \times K}$ is a learned finite-dimensional matrix representation of $\mathcal{K}$.
3. **Decoder / Projection Function**: $\hat{x} = \phi_{\text{dec}}(\psi; \theta) \in \mathbb{R}^D$.

Training minimizes three coupled objectives:
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{lin}} + \lambda_2 \mathcal{L}_{\text{pred}},$$
$$\mathcal{L}_{\text{recon}} = \|x - \phi_{\text{dec}}(\phi_{\text{enc}}(x))\|_2^2,$$
$$\mathcal{L}_{\text{lin}} = \|\phi_{\text{enc}}(x_{t+1}) - \phi_{\text{enc}}(x_t) K_{\text{mat}}\|_2^2,$$
$$\mathcal{L}_{\text{pred}} = \|x_{t+1} - \phi_{\text{dec}}(\phi_{\text{enc}}(x_t) K_{\text{mat}})\|_2^2.$$

---

## 3. SINDy: Sparse Identification of Nonlinear Dynamics

Brunton, Proctor, & Kutz (2016) introduced **SINDy** to extract human-interpretable analytical differential equations:
$$\dot{x}_d = \sum_{j=1}^P \xi_{j, d} \theta_j(x),$$
where $\Theta(X) \in \mathbb{R}^{B \times P}$ is a library of candidate non-linear functions (polynomials, trigonometric terms, rational functions) evaluated on the state history $X$:
$$\Theta(X) = \begin{bmatrix} \mathbf{1} & X & X^{(2)} & \dots & \sin(X) & \cos(X) \end{bmatrix}.$$

Because physical systems typically depend on only a few active terms (parsimony / Occam's razor), the coefficient matrix $\Xi \in \mathbb{R}^{P \times D}$ should be **sparse**:
$$\min_{\Xi} \|\dot{X} - \Theta(X) \Xi\|_F^2 + \lambda \|\Xi\|_1.$$

### Sequentially Thresholded Least Squares (STLS)
Rather than standard LASSO (which often biases active non-zero coefficients), SINDy solves the $\ell_0$ sparse regression problem via alternating iterative thresholding:
1. Compute the unconstrained least-squares fit:
   $$\Xi = \bigl(\Theta^T \Theta + \epsilon I\bigr)^{-1} \Theta^T \dot{X}.$$
2. Apply hard thresholding:
   $$\Xi_{j, d} = 0 \quad \text{if } |\Xi_{j, d}| < \lambda_{\text{threshold}}.$$
3. Re-estimate non-zero coefficients using ordinary least squares restricted only to the active support.
This sequence converges rapidly to the exact physical governing equations (e.g., discovering the Lorenz attractor or Navier-Stokes vorticity transport).

---

## 4. 4D-Var Data Assimilation & Adjoint Optimization

In numerical weather prediction, oceanography, and geophysics, neither purely physical PDE simulations nor noisy sensor measurements alone suffice. **Data Assimilation** synthesizes both sources to obtain the optimal estimate of the continuous physical state trajectory over time.

Given:
- A prior background state $x_b \in \mathbb{R}^D$ with background error covariance $B \in \mathbb{R}^{D \times D}$.
- A physical dynamical model $x_{k+1} = \mathcal{M}_k(x_k)$.
- Sparse, noisy observation measurements $y_k \in \mathbb{R}^P$ with observation error covariance $R \in \mathbb{R}^{P \times P}$ and observation operator $\mathcal{H}(x_k)$.

**Four-Dimensional Variational Data Assimilation (4D-Var)** formulates state estimation as a constrained optimization problem over the initial state $x_0$:
$$J(x_0) = \frac{1}{2} (x_0 - x_b)^T B^{-1} (x_0 - x_b) + \frac{1}{2} \sum_{k=0}^K \bigl(y_k - \mathcal{H}(\mathcal{M}^k(x_0))\bigr)^T R^{-1} \bigl(y_k - \mathcal{H}(\mathcal{M}^k(x_0))\bigr).$$

In the linearized (incremental) 4D-Var formulation, the gradient $\nabla_{x_0} J$ is computed via the continuous adjoint model integrating backward in time:
$$\nabla_{x_0} J = B^{-1} (x_0 - x_b) - \sum_{k=0}^K (M^k)^T H^T R^{-1} d_k,$$
where $d_k = y_k - H M^k x_0$ is the **innovation vector**, and $M$ and $H$ are the tangent linear operators of $\mathcal{M}$ and $\mathcal{H}$.

The analysis increment $\delta x_0 = x_{\text{analysis}} - x_b$ is the solution to the symmetric positive-definite Hessian system:
$$\left( B^{-1} + \sum_{k=0}^K (M^k)^T H^T R^{-1} H M^k \right) \delta x_0 = \sum_{k=0}^K (M^k)^T H^T R^{-1} d_k.$$

---

## 5. Differentiable Hybrid Assimilation: Neural 4D-Var

Traditional 4D-Var relies on hand-crafted tangent linear and adjoint PDE codebases that require millions of human engineering hours to develop and maintain.

**Neural 4D-Var** combines deep learning and data assimilation:
1. **Learned Tangent Linear Models**: Neural operators (FNO, DeepONet) replace explicit numerical solvers as the dynamical forward operator $\mathcal{M}_\theta$.
2. **Automatic Adjoints**: Automatic differentiation frameworks automatically compute exact vector-Jacobian products $(M^k)^T v$ through the neural surrogate.
3. **Learned Covariance Priors**: Deep generative models (Normalizing Flows, Diffusion Models) replace fixed diagonal Gaussian background matrices $B$ with rich, state-dependent non-Gaussian priors $p(x)$.

---

## 6. SINDy with Control (SINDYc) & Physical Invariance Constraints

When external actuation $u(t)$ affects the dynamical system:
$$\dot{x} = F(x, u),$$
**SINDYc** augments the candidate library with control features:
$$\Theta(X, U) = \begin{bmatrix} \Theta(X) & U & X \otimes U \end{bmatrix}.$$

To guarantee that discovered models conform to physical principles:
- **Skew-Symmetric Energy Conservation**: For conservative Hamiltonian systems, the linear coefficient matrix is constrained to be skew-symmetric ($\Xi_{\text{lin}} = -\Xi_{\text{lin}}^T$), certifying $\frac{\mathrm{d}}{\mathrm{d}t} \|x\|_2^2 = 0$.
- **Thermodynamic Dissipation**: Imposing negative semi-definiteness on dissipative modes ensures asymptotic stability.

---

## 7. Numerical Verification & Regularization Invariants

To ensure reproducible numerical parity across computational backends:
1. **Spectral Conditioning of Koopman Matrices**: To prevent exponential runaway ($\|K^t\| \to \infty$) or trivial decay ($\|K^t\| \to 0$), the spectral radius of $K_{\text{mat}}$ must be strictly constrained to the unit circle:
   $$\rho(K_{\text{mat}}) = 1 \iff |\lambda_j| = 1, \quad \forall j.$$
2. **Gram Matrix Regularization in SINDy**: When candidate terms $\theta_j(x)$ and $\theta_k(x)$ are collinear, the Gram matrix $\Theta^T \Theta$ becomes ill-conditioned. Adding a small Tikhonov parameter $\epsilon \ge 10^{-5}$ restores numerical stability.
3. **Hessian Invertibility in 4D-Var**: The 4D-Var Hessian $B^{-1} + H_M R^{-1} H_M^T$ is guaranteed to be strictly positive definite as long as background covariance $B \succ 0$ and observation covariance $R \succ 0$.

---

## 8. Cross-Track Verification & Architectural Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every Koopman, SINDy, and 4D-Var component satisfies the following strict contracts:

| Operator | Verification Specification | Tolerance |
| :--- | :--- | :--- |
| **Koopman Linear Parity** | Encoded evolution $\psi_{t+1} = \psi_t K$ matches across all runtimes | $< 10^{-5}$ |
| **SINDy Library Construction** | Multi-variate polynomial and trigonometric basis matches golden | $< 10^{-5}$ |
| **STLS Sparsity Support** | Hard thresholding zeros out noise while preserving true physical modes | Exact support |
| **4D-Var Innovation Parity** | Optimal analysis increment $\delta x$ solves Hessian system identically | $< 10^{-5}$ |
| **Symmetric Positive Definiteness** | 4D-Var assimilation Hessian $B^{-1} + H_M R^{-1} H_M^T \succ 0$ certified | Guaranteed |
