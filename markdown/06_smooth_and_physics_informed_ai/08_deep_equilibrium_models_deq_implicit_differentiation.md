# Deep Equilibrium Models (DEQ) & Implicit Differentiation

## 1. Introduction to Implicit Layer Architectures

Traditional deep feedforward networks compute an explicit composition of discrete layers:
$$z_{k+1} = \sigma(W_k z_k + U_k x + b_k), \quad k=0, \dots, L-1.$$
While increasing depth $L$ provides greater representational capacity, it carries substantial memory overhead: standard automatic differentiation via backpropagation must store all intermediate activations $\{z_k\}_{k=1}^L$ during the forward pass to compute backward gradients.

Deep Equilibrium Models (DEQ) fundamentally reformulate depth by posing the network output as the fixed point $z^\star$ of an implicit nonlinear algebraic equation:
$$z^\star = f_\theta(z^\star, x).$$
Rather than computing $L$ discrete transformations, a DEQ conceptually runs an *infinitely deep, weight-tied feedforward network*. Crucially, computing the forward fixed point and calculating backward parameter gradients decouple entirely:
1. **Forward Pass**: Any black-box root-finding or fixed-point solver (Picard iteration, Broyden's method, Anderson acceleration) can be used to find $z^\star$ satisfying $z^\star - f_\theta(z^\star, x) = 0$.
2. **Backward Pass**: By invoking the **Implicit Function Theorem (IFT)**, gradients $\frac{\partial \mathcal{L}}{\partial \theta}$ and $\frac{\partial \mathcal{L}}{\partial x}$ are computed directly at the equilibrium point $z^\star$ without storing or backpropagating through any intermediate solver trajectories, granting exact $\mathcal{O}(1)$ training memory complexity.

---

## 2. Banach Fixed-Point Theorem & Contractive Mappings

To guarantee that the nonlinear equation $z = f_\theta(z, x)$ possesses a unique fixed point $z^\star$ that can be stably obtained by simple iteration, we appeal to the **Banach Fixed-Point Theorem**.

Let $(\mathcal{Z}, \|\cdot\|)$ be a complete metric space (a Banach space). An operator $T: \mathcal{Z} \to \mathcal{Z}$ is a **contraction mapping** if there exists a Lipschitz constant $\gamma \in [0, 1)$ such that:
$$\|T(z_1) - T(z_2)\| \le \gamma \|z_1 - z_2\|, \quad \forall z_1, z_2 \in \mathcal{Z}.$$

For a canonical parameterized DEQ cell:
$$f_\theta(z, x) = \tanh(z W_z + x W_x + b),$$
the Jacobian of $f_\theta$ with respect to $z$ is:
$$J_f(z) = \frac{\partial f_\theta(z, x)}{\partial z} = \operatorname{diag}\bigl(1 - \tanh^2(z W_z + x W_x + b)\bigr) W_z^T.$$
Since $|\tanh'(u)| \le 1$, the induced operator norm satisfies:
$$\|J_f(z)\|_2 \le \|W_z\|_2.$$
Hence, if the spectral norm of the weight matrix is certified to be strictly less than unity ($\|W_z\|_2 < 1$), the mapping $f_\theta(\cdot, x)$ is a strict contraction. Under this condition:
- A unique fixed point $z^\star$ exists for any input $x$.
- The naive Picard iteration $z_{k+1} = f_\theta(z_k, x)$ converges linearly at rate $\gamma = \|W_z\|_2$:
$$\|z_k - z^\star\| \le \frac{\gamma^k}{1 - \gamma} \|z_1 - z_0\|.$$

---

## 3. Anderson Acceleration for Nonlinear Fixed Points

While simple Picard iteration $z_{k+1} = f_\theta(z_k, x)$ is guaranteed to converge for contractive mappings, linear convergence with contractivity $\gamma \approx 0.95$ can require hundreds of function evaluations. 

**Anderson Acceleration (Type II)** dramatically accelerates fixed-point iteration by maintaining a sliding history of the last $m$ iterates $\{z_{k-j}\}_{j=0}^m$ and residuals $\{R_{k-j}\}_{j=0}^m$, where:
$$R_k = f_\theta(z_k, x) - z_k.$$

At step $k$, Anderson acceleration seeks an optimal affine combination of the past $m$ residuals to minimize the Euclidean norm of the residual:
$$\min_{\alpha \in \mathbb{R}^m} \left\| R_k - \sum_{j=1}^m \alpha_j (R_k - R_{k-j}) \right\|_2^2.$$
Defining the residual differences $\Delta R_j = R_{k - j + 1} - R_{k - j}$ and state differences $\Delta Z_j = z_{k - j + 1} - z_{k - j}$, the optimal coefficient vector $\alpha^\star = [\alpha_1, \dots, \alpha_m]^T$ is the least-squares solution to:
$$(\Delta R^T \Delta R + \epsilon I) \alpha^\star = \Delta R^T R_k.$$

The next iterate is then constructed with mixing parameter $\beta \in (0, 1]$:
$$z_{k+1} = (1 - \beta)\left( z_k - \sum_{j=1}^m \alpha_j^\star \Delta Z_j \right) + \beta \left( f_\theta(z_k, x) - \sum_{j=1}^m \alpha_j^\star \Delta R_j \right).$$
Anderson acceleration acts as a multi-secant quasi-Newton acceleration without explicitly forming or factorizing full Jacobian matrices, frequently reducing iteration counts by $3\times$ to $10\times$.

---

## 4. Implicit Differentiation via the Adjoint Method

To train a DEQ model by gradient descent, we consider a scalar loss function $\mathcal{L}(z^\star, y)$. The total derivative of $\mathcal{L}$ with respect to network parameters $\theta$ (e.g., $W_z, W_x, b$) must account for the implicit dependence of $z^\star$ on $\theta$:
$$\frac{\mathrm{d} \mathcal{L}}{\mathrm{d} \theta} = \frac{\partial \mathcal{L}}{\partial z^\star} \frac{\mathrm{d} z^\star}{\mathrm{d} \theta} + \frac{\partial \mathcal{L}}{\partial \theta}.$$

By definition, the fixed-point residual identity holds identically at equilibrium:
$$G(z^\star, \theta, x) = z^\star - f_\theta(z^\star, x) = 0.$$
Differentiating this identity totally with respect to $\theta$ yields:
$$\frac{\mathrm{d} G}{\mathrm{d} \theta} = \frac{\partial G}{\partial z^\star} \frac{\mathrm{d} z^\star}{\mathrm{d} \theta} + \frac{\partial G}{\partial \theta} = 0,$$
$$\left( I - \frac{\partial f_\theta(z^\star, x)}{\partial z^\star} \right) \frac{\mathrm{d} z^\star}{\mathrm{d} \theta} - \frac{\partial f_\theta(z^\star, x)}{\partial \theta} = 0.$$

Assuming $(I - J_f)$ is invertible (which holds whenever $\|J_f\| < 1$), we obtain:
$$\frac{\mathrm{d} z^\star}{\mathrm{d} \theta} = \left( I - J_f \right)^{-1} \frac{\partial f_\theta(z^\star, x)}{\partial \theta}.$$

Substituting this into the total gradient gives:
$$\frac{\mathrm{d} \mathcal{L}}{\mathrm{d} \theta} = \frac{\partial \mathcal{L}}{\partial z^\star} \left( I - J_f \right)^{-1} \frac{\partial f_\theta(z^\star, x)}{\partial \theta}.$$

By grouping terms via the adjoint vector $v^\star \in \mathbb{R}^{B \times D}$:
$$(I - J_f^T) v^\star = \left( \frac{\partial \mathcal{L}}{\partial z^\star} \right)^T \implies v^\star = (I - J_f^T)^{-1} \left( \frac{\partial \mathcal{L}}{\partial z^\star} \right)^T,$$
the parameter gradients become simple vector-Jacobian products:
$$\frac{\mathrm{d} \mathcal{L}}{\mathrm{d} \theta} = (v^\star)^T \frac{\partial f_\theta(z^\star, x)}{\partial \theta}.$$

---

## 5. Solving the Adjoint Linear System

The linear system $(I - J_f^T) v = \nabla_{z^\star} \mathcal{L}$ does not require explicit matrix inversion. Rewriting the system in fixed-point form:
$$v = \nabla_{z^\star} \mathcal{L} + J_f^T v,$$
we recognize that $v^\star$ is itself the fixed point of an affine mapping. Under contractivity $\|J_f\|_2 < 1$, this linear system can be solved efficiently using:
1. **Neumann Series Iteration**:
$$v^{(t+1)} = \nabla_{z^\star} \mathcal{L} + J_f^T v^{(t)}, \quad v^{(0)} = 0.$$
Expanding the series yields:
$$v^\star = \sum_{k=0}^\infty (J_f^T)^k \nabla_{z^\star} \mathcal{L} = (I - J_f^T)^{-1} \nabla_{z^\star} \mathcal{L}.$$
2. **Krylov Subspace Solvers**: Conjugate Gradient (for symmetric systems) or GMRES (Generalized Minimal Residual) can be executed using matrix-free vector-Jacobian products (VJP) provided by modern autograd engines.

For cell $f_\theta(z, x) = \tanh(u)$ with $u = z W_z + x W_x + b$, the transpose Jacobian action is:
$$J_f^T v = W_z \bigl( (1 - \tanh^2(u)) \odot v \bigr).$$
Thus, every iteration of the adjoint solver requires only a single vector-matrix multiplication with $W_z$, requiring zero intermediate activation storage.

---

## 6. Monotone Operator Equilibrium Networks (MON)

While enforcing $\|W_z\|_2 < 1$ guarantees contraction, it severely restricts network expressivity by precluding non-expansive or dissipative dynamics that arise in physical systems and deep representations.

**Monotone Operator Networks (MON)** lift the contraction constraint by modeling the equilibrium as the zero of a maximally monotone operator:
$$0 \in A(z) + B(z) + x.$$
In a linear-activation formulation, the operator is defined by a structured matrix:
$$W = -(W_{\text{sym}} + W_{\text{skew}}),$$
where $W_{\text{sym}} \succeq m I$ is symmetric positive definite (strongly monotone) and $W_{\text{skew}} = -W_{\text{skew}}^T$ is skew-symmetric.

The forward equilibrium $z^\star$ is computed using **operator splitting** (e.g., Peaceman-Rachford or Forward-Backward splitting):
$$z_{k+1} = \operatorname{Prox}_{\alpha \phi}\left( z_k - \alpha (z_k W + x W_x + b) \right),$$
where $\operatorname{Prox}$ denotes the proximal operator of a convex regularizer (yielding activations like ReLU, tanh, or soft-thresholding). 
Because monotone operators are closed under addition and resolvent mappings are firmly non-expansive, MONs guarantee existence and uniqueness of $z^\star$ across a vastly larger hypothesis class than strict contractive mappings.

---

## 7. Multi-Scale & Hierarchical DEQ Formulations

To scale implicit equilibrium models to high-resolution computer vision and long-sequence language modeling, modern architectures deploy:

1. **Multi-Scale Deep Equilibrium (MDEQ)**:
   Instead of maintaining a single latent resolution $z$, MDEQ simultaneously solves for equilibria across multiple spatial feature resolutions $\{z^{(1)}, z^{(2)}, \dots, z^{(S)}\}$:
   $$z^{(s)} = f_\theta^{(s)}\left( z^{(1)}, \dots, z^{(S)}, x \right), \quad s=1,\dots,S.$$
   Cross-scale feature fusion and residual convolutions are iteratively applied until all resolution streams reach mutual equilibrium.
2. **Jacobian Regularization**:
   To prevent ill-conditioning of the linear system $(I - J_f)$ during training, an explicit spectral penalty is added to the objective:
   $$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda_{\text{jac}} \|J_f(z^\star)\|_F^2.$$
   This penalty ensures that the singular values of $J_f$ remain bounded away from $1$, ensuring fast forward solver convergence and numerically stable backward adjoint solves.

---

## 8. Cross-Track Verification & Architectural Invariants

Across PyTorch 2.x, JAX, Candle, and custom WGSL compute kernels, every DEQ implementation satisfies the following strict numerical and structural contracts:

| Component | Invariant Specification | Tolerance |
| :--- | :--- | :--- |
| **Fixed Point Residual** | $\|z^\star - f_\theta(z^\star, x)\|_\infty < 10^{-5}$ after solver convergence | $< 10^{-5}$ |
| **Anderson Parity** | Multi-secant quasi-Newton acceleration matches across tensor runtimes | $< 10^{-5}$ |
| **IFT Adjoint Solve** | Adjoint vector $(I - J_f^T) v^\star = \nabla_{z^\star} \mathcal{L}$ via Neumann expansion | $< 10^{-5}$ |
| **Gradient Equivalence** | Implicit gradients match standard unrolled BPTT through $K \to \infty$ layers | $< 10^{-5}$ |
| **Memory Bound** | Memory complexity during training is certified $\mathcal{O}(1)$ with respect to depth | Exact |
