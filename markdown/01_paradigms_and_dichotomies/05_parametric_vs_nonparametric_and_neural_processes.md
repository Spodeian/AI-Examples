# Parametric vs. Non-Parametric & Neural Processes

**Category:** 01 Learning Paradigms & Taxonomic Dichotomies  
**Topic Index:** 05 / 12 in Category 01 (Topic 16 of 263 Total)  
**Status:** Production Ready  
**Parity Verification:** Transitive Parity Across PyTorch 2.x, JAX, Candle, CPU Kernels ($\|Y_A - Y_B\|_\infty < 10^{-5}$)

---

## 1. Executive Summary & Theoretical Lineage

A fundamental architectural dichotomy in statistical machine learning is the tension between **parametric** and **non-parametric** models:

1. **Parametric Models (Deep Neural Networks / MLPs):**
   * *McCulloch & Pitts (1943); Rosenblatt (1958)*.
   * Parameterized by a finite, fixed set of weights $\theta \in \mathbb{R}^P$. Once trained, memory and evaluation complexity are $\mathcal{O}(1)$ with respect to dataset size $N$.
   * **Limitation:** Capacity is strictly bounded. The model cannot dynamically condition on new, arbitrary context observations $\mathcal{D}_C = \{(x_c, y_c)\}_{c=1}^C$ at test time without performing gradient descent fine-tuning.

2. **Exact Non-Parametric Models (Gaussian Processes / Kernel Methods):**
   * *Fix & Hodges (1951)*; *Rasmussen & Williams (Gaussian Processes for Machine Learning, MIT Press 2006)*.
   * Retains all training observations $\mathcal{D}_N$. Prior distribution over functions $f \sim \mathcal{GP}(m(x), k(x, x'))$ yields exact Bayesian posterior predictions:
     $$\mu(x_*) = K(x_*, X) (K(X, X) + \sigma^2 I)^{-1} Y$$
   * **Limitation:** Cubically scaling computational complexity $\mathcal{O}(N^3)$ due to Cholesky matrix inversion and quadratic $\mathcal{O}(N^2)$ memory storage, rendering exact inference intractable for large datasets ($N > 10^4$).

3. **Semi-Parametric Neural Processes (CNP & ANP):**
   * *Marta Garnelo et al. (Conditional Neural Processes, ICML 2018)*; *Hyunjik Kim et al. (Attentive Neural Processes, ICLR 2019)*.
   * Bridges the dichotomy: uses deep neural networks to parameterize stochastic processes, mapping arbitrary context sets $\mathcal{D}_C$ to representations $r$ that condition predictions at target points $x_t$.
   * **Conditional Neural Process (CNP):** Uniform mean-pooling $r = \frac{1}{C}\sum_c e(x_c, y_c)$ enforces permutation invariance in $\mathcal{O}(C + T)$, but creates an information bottleneck that underfits complex context functions.
   * **Attentive Neural Process (ANP):** Replaces uniform pooling with **cross-attention** from target query $x_t$ to context keys $x_c$, scaling in $\mathcal{O}(C \cdot T)$ compute with $\mathcal{O}(1)$ parameter memory while resolving underfitting.

```
+-----------------------------------------------------------------------------------------+
|                  PARAMETRIC VS. NON-PARAMETRIC VS. NEURAL PROCESSES                     |
|                                                                                         |
|  [ Context Set D_C = {(x_c, y_c)} ] + [ Target Query x_t ]                              |
|          |                                                                              |
|          +---> Parametric MLP (Fixed Weights theta)                                     |
|          |       y_hat = MLP_theta(x_t)                                                 |
|          |       [O(1) Memory, O(1) Inference | Cannot adapt to D_C without retraining] |
|          |                                                                              |
|          +---> Non-Parametric Gaussian Process (RBF Kernel)                             |
|          |       mu(x_t) = K(x_t, X_c) * (K(X_c, X_c) + sigma^2 I)^(-1) * Y_c           |
|          |       [Infinite Capacity, Exact Bayesian | O(C^3) Cholesky Bottleneck]       |
|          |                                                                              |
|          +---> Semi-Parametric Attentive Neural Process (ANP)                           |
|                  Cross-Attention: A = Softmax( (x_t W_q)(x_c W_k)^T / sqrt(d) )         |
|                  y_hat = Decoder(x_t, A * Value(x_c, y_c))                              |
|                  [O(C * T) Inference, O(1) Weights | Dynamic In-Context Conditioning]   |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Architecture & Operator Invariants

### 2.1 Exact Gaussian Process Formulation
For context dataset $X_c \in \mathbb{R}^{C \times D_x}, Y_c \in \mathbb{R}^{C \times D_y}$ and target query points $X_t \in \mathbb{R}^{T \times D_x}$:
1. **RBF Covariance Matrix:**
   $$K_{c, ij} = \exp\left(-\frac{\|x_{c, i} - x_{c, j}\|_2^2}{2 \ell^2}\right)$$
2. **Dual Representation Coefficients:**
   $$\alpha = (K_{cc} + \sigma_n^2 I)^{-1} Y_c \in \mathbb{R}^{C \times D_y}$$
3. **Posterior Mean Prediction:**
   $$\mu(X_t) = K_{tc} \alpha \in \mathbb{R}^{T \times D_y}, \quad K_{tc, ij} = \exp\left(-\frac{\|x_{t, i} - x_{c, j}\|_2^2}{2 \ell^2}\right)$$

### 2.2 Attentive Neural Process Cross-Attention Mechanism
1. **Context Representation:**
   $$v_c = \text{ReLU}\left([x_c, y_c] W_{\text{enc}} + b_{\text{enc}}\right) \in \mathbb{R}^{C \times D_h}$$
2. **Target-Context Cross-Attention:**
   $$Q = X_t W_q \in \mathbb{R}^{T \times D_h}, \quad K = X_c W_k \in \mathbb{R}^{C \times D_h}, \quad V = v_c W_v \in \mathbb{R}^{C \times D_h}$$
   $$A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{D_h}}, \text{axis}=-1\right) \in \mathbb{R}^{T \times C}$$
3. **Target Representation & Decoding:**
   $$R_{\text{anp}} = A V \in \mathbb{R}^{T \times D_h}, \quad \hat{Y}_{\text{anp}} = [X_t, R_{\text{anp}}] W_{\text{dec}} + b_{\text{dec}} \in \mathbb{R}^{T \times D_y}$$

**Simplex Invariant:** The attention weights satisfy $\sum_{c=1}^C A_{t, c} = 1$ and $A_{t, c} > 0$ for all targets $t$.

---

## 3. Pathological Failure Contrast & Misuse Patterns

### 3.1 The Uniform Mean-Pooling Bottleneck (CNP vs. ANP)
* **Conditional Neural Process (CNP):** Aggregates context via uniform mean pooling $r = \frac{1}{C}\sum_{c=1}^C v_c$.
* **Failure Mode:** Every target query $x_t$ receives the exact same global summary vector $r$, regardless of whether $x_t$ is located adjacent to context point $x_1$ or $x_C$.
* On localized or non-stationary functions, uniform pooling washes out high-frequency spatial variation, causing severe **underfitting** at context locations ($\|y_c - \hat{y}(x_c)\| \gg 0$).
* **ANP Remedy:** Cross-attention assigns localized weights $A_{t, c} \approx 1$ when $x_t \approx x_c$, faithfully reconstructing observed context points while maintaining smooth extrapolation elsewhere.

---

## 4. Multi-Track Hardware & Computational Complexity

| Paradigm / Model | Parameter Count | Context Memory Scaling | Inference FLOPs Scaling | Hardware Bound |
| :--- | :--- | :--- | :--- | :--- |
| **Parametric MLP** | $\mathcal{O}(D_x D_h + D_h D_y)$ | $\mathcal{O}(1)$ | $\mathcal{O}(T \cdot D_h)$ | Compute Bound (GEMM) |
| **Gaussian Process** | $\mathcal{O}(1)$ ($\ell, \sigma_n^2$) | $\mathcal{O}(C^2)$ | $\mathcal{O}(C^3 + T \cdot C)$ | Memory & Cholesky Bound |
| **Attentive NP (ANP)** | $\mathcal{O}(D_h^2)$ | $\mathcal{O}(C \cdot D_h)$ | $\mathcal{O}(T \cdot C \cdot D_h)$ | Bandwidth & Attention Bound |
| **Conditional NP (CNP)** | $\mathcal{O}(D_h^2)$ | $\mathcal{O}(D_h)$ | $\mathcal{O}((C + T) \cdot D_h)$ | Memory-Bandwidth Bound |

---

## 5. Parity & Empirical Proofs

All four framework implementations produce numerically identical outputs against the seed-42 reference checkpoint (`05_parametric_vs_nonparametric_and_neural_processes.safetensors`):

| Operator Output | Target Tensor | PyTorch Parity | JAX Parity | Candle Parity | Kernel Parity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Parametric MLP Mean** | Golden `y_parametric` | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **Exact GP Posterior** | Golden `y_gp` | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **ANP Prediction** | Golden `y_anp` | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **ANP Cross-Attention** | Golden `attn_weights` | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |
| **CNP Prediction** | Golden `y_cnp` | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ |

---

## 6. Curriculum Navigation & Synchronized Artifacts

- **Goal-Oriented Index:** [INDEX_BY_GOAL.md](../../INDEX_BY_GOAL.md)
- **Interactive Notebook Explorer:** [05_parametric_vs_nonparametric_and_neural_processes.ipynb](../../notebooks/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.ipynb)
- **Production PyTorch 2.x Module:** [05_parametric_vs_nonparametric_and_neural_processes.py](../../pytorch2/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.py)
- **Functional JAX Module:** [05_parametric_vs_nonparametric_and_neural_processes.py](../../jax/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.py)
- **Native Rust Candle Crate:** [05_parametric_vs_nonparametric_and_neural_processes.rs](../../candle/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.rs)
- **Hardware CPU Kernel:** [05_parametric_vs_nonparametric_and_neural_processes_cpu.py](../../kernels/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes_cpu.py)
- **Golden Reference Checkpoint:** [05_parametric_vs_nonparametric_and_neural_processes.safetensors](../../weights/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.safetensors)
- **Previous Topic:** [04. Deterministic vs. Stochastic & Mixture Density Networks](04_deterministic_vs_stochastic_and_mixture_density_networks.md)
- **Next Topic:** [06. AI by Purpose: Prediction vs Prescription vs Control vs Combinatorial vs Symbolic](06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.md)
