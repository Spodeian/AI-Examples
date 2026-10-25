# Differentiable Optimization: OptNet, KKT Conditions & Implicit Layers

## 1. Declarative vs. Procedural Neural Architectures

Standard deep networks define procedural mappings: layers sequentially execute predefined arithmetic operations ($z = \sigma(W x + b)$). While universal approximators, purely procedural layers struggle to respect hard physical, kinematic, or resource constraints (e.g., non-negativity, capacity limits, conservation of mass, portfolio allocations).

**Differentiable Optimization Layers** introduce declarative primitives into deep architectures:
$$z^\star(x) = \arg\min_{z \in \mathcal{C}(x)} f_0(z; x),$$
where the output $z^\star$ is implicitly defined as the minimizer of an optimization program over a feasible set $\mathcal{C}(x)$. Rather than treating optimization as an external post-processing step, differentiable optimization layers embed convex quadratic programs (QPs), second-order cone programs (SOCPs), and semidefinite programs (SDPs) directly as internal network layers.

Backpropagation through such declarative layers is accomplished not by unrolling optimization trajectories (which incurs $\mathcal{O}(T)$ memory and numerical instability), but by invoking the **Karush-Kuhn-Tucker (KKT)** optimality conditions and the **Implicit Function Theorem (IFT)**.

---

## 2. Quadratic Programs & KKT Optimality System

Consider a general convex Quadratic Program (QP) parameterized by input $x$:
$$\begin{aligned}
\min_{z \in \mathbb{R}^D} \quad & \frac{1}{2} z^T Q z + p^T z \\
\text{s.t.} \quad & A z = b, \\
& G z \le h,
\end{aligned}$$
where $Q \succ 0$ is symmetric positive definite ($D \times D$), $A \in \mathbb{R}^{M \times D}$ ($M < D$, full row rank), and $G \in \mathbb{R}^{K \times D}$.

The Lagrangian associated with the QP is:
$$\mathcal{L}_{\text{QP}}(z, \nu, \lambda) = \frac{1}{2} z^T Q z + p^T z + \nu^T (A z - b) + \lambda^T (G z - h),$$
where $\nu \in \mathbb{R}^M$ and $\lambda \in \mathbb{R}_{\ge 0}^K$ are the Lagrange multipliers for equality and inequality constraints.

The Karush-Kuhn-Tucker (KKT) conditions characterize the unique primal-dual optimum $(z^\star, \nu^\star, \lambda^\star)$:
1. **Stationarity**:
$$\nabla_z \mathcal{L}_{\text{QP}} = Q z^\star + p + A^T \nu^\star + G^T \lambda^\star = 0.$$
2. **Primal Feasibility**:
$$A z^\star = b, \quad G z^\star \le h.$$
3. **Dual Feasibility**:
$$\lambda^\star \ge 0.$$
4. **Complementary Slackness**:
$$\operatorname{diag}(\lambda^\star) (G z^\star - h) = 0 \iff \lambda_i^\star (g_i^T z^\star - h_i) = 0, \quad \forall i=1,\dots,K.$$

---

## 3. OptNet Implicit Differentiation via KKT Differentials

Let $\mathcal{L}(z^\star)$ be a scalar loss downstream of the optimization layer. To compute gradients $\frac{\partial \mathcal{L}}{\partial \theta}$ with respect to program parameters $\theta \in \{Q, p, A, b, G, h\}$, we differentiate the KKT optimality conditions at $(z^\star, \nu^\star, \lambda^\star)$.

Taking the total differential of the KKT conditions yields the linear system:
$$\begin{bmatrix}
Q & A^T & G^T \\
A & 0 & 0 \\
\operatorname{diag}(\lambda^\star) G & 0 & \operatorname{diag}(G z^\star - h)
\end{bmatrix}
\begin{bmatrix}
\mathrm{d} z \\
\mathrm{d} \nu \\
\mathrm{d} \lambda
\end{bmatrix}
=
-\begin{bmatrix}
\mathrm{d} Q z^\star + \mathrm{d} p + \mathrm{d} A^T \nu^\star + \mathrm{d} G^T \lambda^\star \\
\mathrm{d} A z^\star - \mathrm{d} b \\
\operatorname{diag}(\lambda^\star)(\mathrm{d} G z^\star - \mathrm{d} h)
\end{bmatrix}.$$

In the adjoint formulation, given incoming loss gradient $g_z = \nabla_{z^\star} \mathcal{L}$, we solve the symmetric adjoint KKT system:
$$\begin{bmatrix}
Q & A^T & G^T \\
A & 0 & 0 \\
G & 0 & -\operatorname{diag}(\lambda^\star)^{-1} \operatorname{diag}(s^\star)
\end{bmatrix}
\begin{bmatrix}
d_z \\
d_\nu \\
d_\lambda
\end{bmatrix}
=
\begin{bmatrix}
-g_z \\
0 \\
0
\end{bmatrix},$$
where $s^\star = h - G z^\star$ is the primal slack vector.

Once the adjoint solution $(d_z, d_\nu, d_\lambda)$ is obtained, the parameter gradients decouple into direct outer products:
$$\nabla_p \mathcal{L} = -d_z, \quad \nabla_b \mathcal{L} = -d_\nu, \quad \nabla_h \mathcal{L} = -d_\lambda,$$
$$\nabla_Q \mathcal{L} = -\frac{1}{2} (d_z (z^\star)^T + z^\star d_z^T), \quad \nabla_A \mathcal{L} = -d_\nu (z^\star)^T - \nu^\star d_z^T, \quad \nabla_G \mathcal{L} = -d_\lambda (z^\star)^T - \lambda^\star d_z^T.$$

---

## 4. Interior Point & Log-Barrier Smoothing

For pure equality-constrained QPs ($G=0$), the KKT system is linear and admits an exact, closed-form block inversion:
$$\begin{bmatrix} Q & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} z^\star \\ \nu^\star \end{bmatrix} = \begin{bmatrix} -p \\ b \end{bmatrix}.$$

When inequality constraints $G z \le h$ are present, complementary slackness introduces non-smooth combinatorial switching: constraints become active ($\lambda_i > 0, s_i = 0$) or inactive ($\lambda_i = 0, s_i > 0$). To enable smooth end-to-end backpropagation, two primary paradigms are used:

1. **Log-Barrier Interior Point Approximation**:
   Replace the hard inequality constraints with a parameterized logarithmic barrier:
   $$\min_z \frac{1}{2} z^T Q z + p^T z - \frac{1}{t} \sum_{i=1}^K \log(h_i - g_i^T z), \quad t > 0.$$
   The unconstrained objective is strictly convex inside the interior of the feasible polytope. The gradient is smooth everywhere in the interior:
   $$\nabla_z \phi_t(z) = Q z + p + \frac{1}{t} \sum_{i=1}^K \frac{g_i}{h_i - g_i^T z}.$$
   As $t \to \infty$, the trajectory of unconstrained minimizers converges to the exact constrained QP optimum.
2. **Primal-Dual Path Following**:
   Solve the perturbed KKT system $\lambda_i s_i = \mu = 1/t$ using predictor-corrector Newton steps.

---

## 5. Differentiable Convex Programs Beyond Quadratic Forms

The OptNet KKT framework generalizes to broader classes of convex and structured optimization:

1. **Cone Programs (DiffCP)**:
   $$\min_z c^T z \quad \text{s.t.} \quad A z + s = b, \quad s \in \mathcal{K},$$
   where $\mathcal{K}$ is a product of second-order cones (SOCP) or positive semidefinite cones (SDP). Gradients are derived via implicit differentiation of the homogeneous self-dual embedding.
2. **Optimal Transport (Sinkhorn Layers)**:
   Entropically regularized Wasserstein distances:
   $$\min_{P \in \mathcal{U}(r, c)} \langle P, M \rangle - \epsilon H(P).$$
   Differentiating the Sinkhorn fixed point via implicit function theorem computes transport sensitivities in closed form.
3. **Submodular & Combinatorial Relaxations**:
   Continuous multilinear extensions and proximal operators (e.g., differentiable sorting and top-$k$ via projection onto the permutahedron).

---

## 6. Energy-Based Models & Bilevel Learning

Differentiable optimization layers naturally formalize **bilevel optimization**:
$$\min_\theta \mathcal{L}(z^\star(\theta)) \quad \text{s.t.} \quad z^\star(\theta) = \arg\min_z E(z; \theta).$$

Applications include:
- **Model Predictive Control (MPC)**: Parameterizing cost weights $Q$ and terminal penalties in a linear-quadratic regulator (LQR) to learn robust control policies from trajectory demonstrations.
- **Physics-Informed Equilibrium Allocation**: Enforcing exact physical conservation laws (e.g., Kirchhoff's current laws, fluid continuity) as hard equality constraints $A z = b$ while a neural network predicts cost coefficients $p(x)$.
- **Constrained Deep Learning**: Imposing guaranteed budget, margin, or safety envelopes on neural network outputs.

---

## 7. Numerical Stability & Degeneracy Handling

In practical deployment, three numerical hazards must be addressed:
1. **Rank-Deficient Constraints**: If constraint matrix $A$ loses full row rank, or if active inequality constraints are linearly dependent, the KKT matrix becomes singular. Adding Tikhonov regularization:
   $$KKT_\epsilon = \begin{bmatrix} Q & A^T \\ A & -\epsilon I \end{bmatrix}, \quad \epsilon \sim 10^{-6},$$
   restores strict invertibility while preserving condition number.
2. **Strict Complementarity Violations**: At points where both $s_i = 0$ and $\lambda_i = 0$ (weakly active constraints), the directional derivative is non-unique. Interior-point barrier regularization strictly bounds $\lambda_i s_i = 1/t > 0$, guaranteeing a unique derivative everywhere.
3. **Cholesky & Schur-Complement Factorizations**: Instead of inverting the full $(D+M) \times (D+M)$ KKT matrix directly, modern solvers factorize $Q = L L^T$ and solve the reduced Schur complement $(A Q^{-1} A^T) \nu = A Q^{-1} p + b$, accelerating inference from $\mathcal{O}((D+M)^3)$ to $\mathcal{O}(D^3 + M^3)$.

---

## 8. Cross-Track Verification & Architectural Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute shaders, every implementation obeys the following strict invariants:

| Metric | Verification Specification | Tolerance |
| :--- | :--- | :--- |
| **Primal Equality Residual** | $\|A z^\star - b\|_\infty < 10^{-5}$ on equality QP solutions | $< 10^{-5}$ |
| **Stationarity Residual** | $\|Q z^\star + p + A^T \nu^\star\|_\infty < 10^{-5}$ at equilibrium | $< 10^{-5}$ |
| **Adjoint Linear Identity** | $KKT \cdot [d_z; d_\nu] = [-g_z; 0]$ satisfied within machine precision | $< 10^{-5}$ |
| **Log-Barrier Monotonicity** | Slack variables $s_i = h_i - g_i^T z^\star > 0$ strictly maintained | $> 0$ |
| **Cross-Platform Parity** | Primal solution $z^\star$, duals $\nu^\star$, and adjoint gradients match golden | $< 10^{-5}$ |
