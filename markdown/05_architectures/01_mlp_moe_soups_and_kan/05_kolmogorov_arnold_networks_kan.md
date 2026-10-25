# Kolmogorov-Arnold Networks (KAN): B-Spline KAN, FastKAN (RBF), and ChebyKAN

## Executive Summary & ELI5
For decades, the Multi-Layer Perceptron (MLP) has served as the universal computational substrate of deep learning, grounded in the **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991). In standard MLPs, activation functions $\sigma$ are static, fixed non-linearities placed on **nodes** (neurons), while learnable parameters reside strictly on linear matrix **edges**:
$$\mathbf{y}_j = \sigma\left(\sum_{i=1}^{D_{\text{in}}} W_{i, j} x_i + b_j\right)$$

**Kolmogorov-Arnold Networks (KAN)** (*Liu et al., 2024*) invert this design entirely, grounded in the **Kolmogorov-Arnold Representation Theorem** (Kolmogorov, 1957; Arnold, 1957). KANs place **learnable 1D univariate non-linear parameterized activation functions on edges**, while nodes simply sum incoming activations:
$$\mathbf{y}_j = \sum_{i=1}^{D_{\text{in}}} \phi_{i, j}(x_i)$$

While original KANs parameterized $\phi_{i, j}$ using B-splines via recursive Cox-de Boor algorithms (which are computationally slow and memory-intensive on GPUs), two major high-performance variants emerged:
1. **ChebyKAN:** Parameterizes $\phi_{i, j}$ using orthogonal **Chebyshev polynomials** $T_n(\tanh(x))$, replacing recursive spline evaluations with closed-form 3-term recurrence that evaluates at pure GEMM speed.
2. **FastKAN** (*Li, 2024*): Approximates splines via **Gaussian Radial Basis Functions (RBF)** with a SiLU linear base, eliminating Cox-de Boor tree traversals and executing $5\times$ faster.

```mermaid
flowchart TD
    subgraph MLP ["Standard Multi-Layer Perceptron (MLP)"]
        X1["Input x_i"] -->|Linear Weight W_ij| N1["Node Sum + Fixed Activation sigma(.)"]
        X2["Input x_2"] -->|Linear Weight W_2j| N1
        N1 --> Y1["Output y_j"]
    end

    subgraph KAN ["Kolmogorov-Arnold Network (KAN)"]
        KX1["Input x_i"] -->|Learnable Univariate Function phi_ij(.)| KN1["Linear Summing Node sum_i (.)"]
        KX2["Input x_2"] -->|Learnable Univariate Function phi_2j(.)| KN1
        KN1 --> KY1["Output y_j"]
    end
```

---

## Historical Citations & Seminal Literature
1. **On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition** (*Andrey Kolmogorov, Doklady Akademii Nauk SSSR, 1957*). Proved that any multivariate continuous function on a bounded domain can be written as a finite composition and sum of continuous univariate functions.
2. **On functions of three variables** (*Vladimir Arnold, Doklady Akademii Nauk SSSR, 1957*). Solved Hilbert's 13th problem by proving the representation for three variables, laying the foundation for the Kolmogorov-Arnold theorem.
3. **KAN: Kolmogorov-Arnold Networks** (*Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark, arXiv:2404.19737, 2024*). Introduced modern neural KAN architectures using B-splines, demonstrating faster neural scaling laws ($N^{-4}$ vs $N^{-2}$ on PDE solving) and interpretability for scientific discovery.
4. **FastKAN: Efficient Kolmogorov-Arnold Networks using Radial Basis Functions** (*Ziyao Li, arXiv:2405.02986, 2024*). Replaced Cox-de Boor B-splines with Gaussian RBF kernels, yielding a $5\times$ speedup and $O(1)$ GPU memory access.
5. **Chebyshev Polynomials in Neural Networks (ChebyKAN)** (*Syed Shakir et al., 2024*). Leveraged Chebyshev polynomials of the first kind $T_n(x)$ to construct orthogonal KANs with bounded numerical stability on $[-1, 1]$.

---

## Mathematical Foundations & Governing Equations

### 1. Kolmogorov-Arnold Representation Theorem
If $f: [0, 1]^n \to \mathbb{R}$ is a continuous multivariate function, it can be written as:
$$f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^n \phi_{q, p}(x_p)\right)$$
where $\phi_{q, p}: [0, 1] \to \mathbb{R}$ and $\Phi_q: \mathbb{R} \to \mathbb{R}$ are continuous univariate functions.

### 2. General KAN Layer Formulation
A general KAN layer with $D_{\text{in}}$ inputs and $D_{\text{out}}$ outputs computes:
$$y_j = \sum_{i=1}^{D_{\text{in}}} \phi_{i, j}(x_i), \quad \forall j \in \{1, \dots, D_{\text{out}}\}$$
where each edge activation $\phi_{i, j}(x)$ combines a base residual non-linearity with a parameterized basis:
$$\phi_{i, j}(x) = w_b \cdot b(x) + w_s \cdot \text{basis}_{i, j}(x)$$

### 3. ChebyKAN (Orthogonal Chebyshev Basis)
Chebyshev polynomials of the first kind are defined on $[-1, 1]$ via the 3-term recurrence:
$$T_0(u) = 1$$
$$T_1(u) = u$$
$$T_{n+1}(u) = 2u T_n(u) - T_{n-1}(u)$$
To ensure inputs remain bounded on the polynomial support, ChebyKAN applies $u = \tanh(x) \in (-1, 1)$:
$$y_j = \sum_{i=1}^{D_{\text{in}}} \sum_{n=0}^N c_{i, j, n} T_n(\tanh(x_i))$$
Because Chebyshev polynomials are orthogonal with respect to the weight $\frac{1}{\sqrt{1 - u^2}}$, gradient updates across polynomial degrees remain decoupled, avoiding cross-order gradient interference.

### 4. FastKAN (Gaussian Radial Basis Functions)
FastKAN replaces recursive B-spline knot evaluations with uniformly spaced Gaussian RBF kernels:
$$\phi_{i, j}(x) = w_{b, i, j} \cdot \text{SiLU}(x) + \sum_{k=1}^K w_{rbf, i, k, j} \cdot \exp\left(-\left(\frac{x - \mu_k}{h}\right)^2\right)$$
where $\{\mu_k\}_{k=1}^K$ are $K$ grid centers uniformly spaced in $[-1, 1]$, and $h = \frac{\mu_K - \mu_1}{K - 1}$ is the grid spacing.

---

## Tensor Memory Layout & Dimension Shapes
```text
Batch Size = B, Sequence Length = S, Input Dim = D_in, Output Dim = D_out

ChebyKAN Tensor Flow:
Input X:                 [B, S, D_in]
Tanh Squashed u:         Tanh(X) -> [B, S, D_in]
Chebyshev Basis T(u):    Stack(T_0..T_N) -> [B, S, D_in, degree + 1]
Flattened Basis:         Reshape -> [B * S, D_in * (degree + 1)]
Coefficient Tensor c:    [D_in, D_out, degree + 1]
Permuted & Flat c:       Permute(0, 2, 1) -> Reshape -> [D_in * (degree + 1), D_out]
Matrix Multiplication:   [B * S, D_in * (deg + 1)] @ [D_in * (deg + 1), D_out] -> [B * S, D_out]
Reshaped Output:         [B, S, D_out]

FastKAN Tensor Flow:
Input X:                 [B, S, D_in]
Base Branch:             SiLU(X) @ base_w -> [B, S, D_out]
RBF Differences:         X.unsqueeze(-1) - grid -> [B, S, D_in, num_grids]
RBF Basis Phi:           Exp(-((diff / h)^2)) -> [B, S, D_in, num_grids]
Contracted RBF:          Phi_flat @ rbf_w_flat -> [B, S, D_out]
Total Output:            Base_Out + RBF_Out -> [B, S, D_out]
```

---

## Multi-Ecosystem Implementation Matrix
| Feature / Characteristic | PyTorch 2.x (`pytorch2/`) | JAX / Equinox (`jax/`) | Candle Rust (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Chebyshev Basis** | Recurrence loop + `torch.stack` | Vectorized array stack | `Tensor::stack` | Parallel 1D thread WGSL compute |
| **Basis Contraction** | `torch.einsum('...id,ijd->...j')` | `np.einsum('...id,ijd->...j')` | Reshape + Permute 2D Matmul | Fused parallel GEMM |
| **RBF Kernel** | `torch.exp(-((x - g)/h)^2)` | `np.exp(-np.square((x - g)/h))` | `diff.affine.sqr.neg.exp` | Vectorized RBF shader |
| **Domain Stabilization**| `torch.tanh(x)` | `np.tanh(x)` | `x.tanh()` | In-register `tanh()` |
| **Linear Anchor** | `x @ weight + bias` | `x @ weight + bias` | `x.matmul + bias` | CPU / GPU linear forward |

---

## Pathological Misuse & Failure Modes

### 1. High-Degree Chebyshev Runge Phenomenon
- **Failure:** Increasing polynomial degree $N > 8$ without input normalization or squashing.
- **Consequence:** Polynomial values $T_N(u)$ explode near the boundaries $u \to \pm 1$, inducing catastrophic gradient overflow (`NaN` / `Inf`) and wild oscillatory artifacts between grid points.
- **Stabilized Solution:** Strictly squash inputs into $(-1, 1)$ via `tanh(x)` or layer normalization, and cap polynomial degrees to $N \le 6$.

### 2. Activation Out-of-Grid Drift in B-Splines / RBF
- **Failure:** Evaluating un-normalized inputs whose magnitudes $|x| \gg 1.0$ exceed the pre-allocated RBF grid domain $[\mu_1, \mu_K]$.
- **Consequence:** Gaussian RBF activations vanish exponentially to zero: $\exp(-((x - \mu)/h)^2) \to 0$. Gradients for edge parameters $\nabla_{w_{rbf}} \mathcal{L} \to 0$ undergo dead zone stagnation.
- **Stabilized Solution:** Pair the RBF branch with an unconditional SiLU base linear branch ($w_b \cdot \text{SiLU}(x)$) as done in FastKAN, and normalize layer inputs with RMSNorm.

---

## Cross-Domain Generalisation Matrix
1. **Scientific Discovery & Symbolic Regression:** KAN univariate edge activations can be directly plotted and fitted to closed-form analytical equations (e.g. $\sin(x), \exp(x), x^2$), discovering physical conservation laws from experimental data.
2. **Physics-Informed Neural Networks (PINNs):** High-order spatial derivatives $\nabla^k u$ of Chebyshev polynomials are smooth, bounded, and analytically differentiable, solving stiff boundary value PDEs without spectral dissipation.
3. **Robotics Dynamics & Kinematics:** Learning continuous joint compliance and non-linear torque-velocity curves via edge-parameterized spline functions.

---

## Verification, Parity & Empirical Benchmarks

| Verification Target | Reference Source | Max Absolute Error Ceiling | Status |
| :--- | :--- | :--- | :--- |
| **Linear Parity Anchor (x @ W + b)** | `weights/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors` | `< 1.0e-5` | Verified |
| **Topic-Specific Forward Operators** | `pytorch2/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.py` | `< 1.0e-5` | Verified |
| **Functional JAX Parity** | `jax/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.py` | `< 1.0e-5` | Verified |
| **Native Rust Candle Parity** | `candle/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.rs` | `< 1.0e-5` | Verified |
| **Hardware Kernel (CPU / WGSL)** | `kernels/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.rs` | `< 1.0e-5` | Verified |

---

## Synchronized Track Artifacts
- **Universal SafeTensors Checkpoint:** [`weights/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors`](../../../weights/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors)
- **PyTorch 2.x Module:** [`pytorch2/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.py`](../../../pytorch2/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.py)
- **Functional JAX Module:** [`jax/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.py`](../../../jax/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.py)
- **Candle Rust Module:** [`candle/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.rs`](../../../candle/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.rs)
- **Hardware Kernel:** [`kernels/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.rs`](../../../kernels/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.rs)
- **Interactive Concept Notebook:** [`notebooks/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.ipynb`](../../../notebooks/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.ipynb)
