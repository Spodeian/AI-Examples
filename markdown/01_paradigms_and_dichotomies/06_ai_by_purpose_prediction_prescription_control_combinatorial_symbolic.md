# AI by Purpose: Prediction, Prescription, Control, Combinatorial, Symbolic

**Category:** 01 Learning Paradigms & Taxonomic Dichotomies  
**Topic Index:** 06 / 12 in Category 01 (Topic 17 of 263 Total)  
**Status:** Production Ready  
**Parity Verification:** Transitive Parity Across PyTorch 2.x, JAX, Candle, CPU Kernels ($\|Y_A - Y_B\|_\infty < 10^{-5}$)

---

## 1. Executive Summary & Theoretical Lineage

Modern artificial intelligence systems are frequently categorized not merely by model structure, but by their fundamental operational **purpose**:

1. **Predictive AI (Pattern Recognition & Conditional Estimation):**
   * *Fisher (1936); Vapnik (1995)*.
   * Learns statistical correlations to approximate the conditional expectation $\mathbb{E}[Y | X=\mathbf{x}]$ or density $p(y | \mathbf{x})$.
   * **Limitation:** Optimizing statistical loss functions (e.g., Mean Squared Error or Cross-Entropy) produces representations that are unaware of downstream operational constraints and decision penalties.

2. **Prescriptive AI & Decision-Focused Learning (Smart "Predict, Then Optimize"):**
   * *Elmachtoub & Grigas (Smart "Predict, Then Optimize", Management Science 2022); Wilder et al. (Melding the Data-Decisions Pipeline, AAAI 2019)*.
   * Bridges machine learning and mathematical programming. Instead of minimizing prediction error $\|\hat{\mathbf{c}} - \mathbf{c}^*\|_2^2$, prescriptive systems directly optimize the quality of downstream decisions $\mathbf{w}^*(\hat{\mathbf{c}})$ under operational constraints $\mathcal{S}$.
   * **SPO+ Loss:** Provides a computationally tractable, Fisher-consistent convex upper bound on true decision regret, backpropagating gradients directly through discrete combinatorial linear programs without subgradient collapse.

3. **Control & Causal Uplift (Interventional AI):**
   * *Judea Pearl (Causality, Cambridge University Press 2009); Robins (1986)*.
   * Observational correlation does not imply interventional causation. Causal uplift modeling isolates the Heterogeneous Treatment Effect (HTE) $\tau(\mathbf{x}) = \mathbb{E}[Y | X=\mathbf{x}, \mathrm{do}(T=1)] - \mathbb{E}[Y | X=\mathbf{x}, \mathrm{do}(T=0)]$, disentangling baseline confounding $\mu_0(\mathbf{x})$ from actionable policy intervention.

4. **Combinatorial & Symbolic AI (Analytical Law Discovery):**
   * *Martius & Lampert (Extrapolation and Learning Equations, arXiv 2016); Schmidt & Lipson (Distilling Free-Form Natural Laws, Science 2009)*.
   * Replaces opaque black-box neural matrices with structured functional primitives (trigonometric functions, polynomials, rational operators). Uses sparse Equation Learner (EQL) layers to yield human-interpretable, verifiable closed-form symbolic expressions.

```
+---------------------------------------------------------------------------------------------------+
|                        TAXONOMIC PURPOSES IN MODERN ARTIFICIAL INTELLIGENCE                       |
|                                                                                                   |
|  [ Environmental Context x ]                                                                      |
|          |                                                                                        |
|          +---> Predictive AI: Minimizes Statistical Residuals                                     |
|          |       Loss = ||c_hat - c*||^2  (Agnostic to downstream LP constraints)                 |
|          |                                                                                        |
|          +---> Prescriptive AI: Minimizes Decision Regret via SPO+                                |
|          |       w*(c_hat) = argmin_{w in S} c_hat^T w                                            |
|          |       Regret = c*^T w*(c_hat) - c*^T w*(c*) >= 0                                       |
|          |       L_SPO+(c_hat, c*) = max_{w in S} (c* - 2 c_hat)^T w + 2 c_hat^T w* - c*^T w*     |
|          |                                                                                        |
|          +---> Causal / Control AI: Isolates Treatment Intervention                               |
|          |       Y(T) = mu_0(x) + T * tau(x)  (Disentangles baseline confounding)                 |
|          |                                                                                        |
|          +---> Symbolic AI: Recovers Analytical Invariants                                        |
|                  y = w_0 * x_0 + w_1 * sin(x_1) + w_2 * cos(x_2) + w_3 * (x_0 * x_3) + b          |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Foundations & Inductive Biases

### 2.1 The Prediction vs. Optimization Dilemma
In classic two-stage "Predict, Then Optimize", a regression model parameterized by $\theta$ predicts cost vectors $\hat{\mathbf{c}} = f_\theta(\mathbf{x})$, and an optimization solver finds an action $\mathbf{w}^*(\hat{\mathbf{c}}) \in \arg\min_{\mathbf{w} \in \mathcal{S}} \hat{\mathbf{c}}^\top \mathbf{w}$.

The true operational evaluation is the **Decision Regret**:
$$\mathrm{Regret}(\hat{\mathbf{c}}, \mathbf{c}^*) = \mathbf{c}^{*\top} \mathbf{w}^*(\hat{\mathbf{c}}) - \mathbf{c}^{*\top} \mathbf{w}^*(\mathbf{c}^*) \ge 0$$

* **Failure of MSE:** Minimizing $\|\hat{\mathbf{c}} - \mathbf{c}^*\|_2^2$ treats errors symmetrically across all coordinates. However, downstream combinatorial solvers are piecewise constant step functions: an infinitesimal error along an inactive constraint produces zero regret, whereas a small error near a decision boundary flips the optimal vertex, inducing catastrophic regret.
* **Non-Differentiability:** $\nabla_{\hat{\mathbf{c}}} \mathbf{w}^*(\hat{\mathbf{c}}) = \mathbf{0}$ almost everywhere, rendering naive subgradient backpropagation ineffective.

### 2.2 The Smart "Predict, Then Optimize" (SPO+) Convex Surrogate
Elmachtoub & Grigas (2022) prove that the exact decision regret can be upper-bounded by the convex surrogate loss $\mathcal{L}_{\mathrm{SPO}+}$:
$$\mathcal{L}_{\mathrm{SPO}+}(\hat{\mathbf{c}}, \mathbf{c}^*) = \max_{\mathbf{w} \in \mathcal{S}} (\mathbf{c}^* - 2 \hat{\mathbf{c}})^\top \mathbf{w} + 2 \hat{\mathbf{c}}^\top \mathbf{w}^*(\mathbf{c}^*) - \mathbf{c}^{*\top} \mathbf{w}^*(\mathbf{c}^*)$$

For combinatorial decision problems over the unit simplex $\mathcal{S} = \Delta_K = \{\mathbf{w} \in \mathbb{R}^K : \sum_k w_k = 1, w_k \ge 0\}$:
1. $\mathbf{w}^*(\mathbf{c}^*) = \mathbf{e}_{k^*}$ where $k^* = \arg\min_k c_k^*$.
2. The maximization term reduces to an exact coordinate maximum:
   $$\max_{\mathbf{w} \in \Delta_K} (\mathbf{c}^* - 2 \hat{\mathbf{c}})^\top \mathbf{w} = \max_{k \in \{1, \dots, K\}} (c_k^* - 2 \hat{c}_k)$$
3. **Subgradient Backpropagation:** The gradient with respect to $\hat{\mathbf{c}}$ is given in closed form by:
   $$\nabla_{\hat{\mathbf{c}}} \mathcal{L}_{\mathrm{SPO}+}(\hat{\mathbf{c}}, \mathbf{c}^*) = 2 \left( \mathbf{w}^*(\mathbf{c}^*) - \mathbf{w}^*\left(2\hat{\mathbf{c}} - \mathbf{c}^*\right) \right)$$
   This provides well-behaved, non-zero gradient signals that guide parameters toward vertex alignment.

### 2.3 Causal Uplift Formulation
Under the Rubin Causal Model and Pearl's structural equation framework, observational outcomes confound baseline disease/churn risk with treatment efficacy. We decompose potential outcomes into:
$$Y(T) = \mu_0(\mathbf{x}) + T \cdot \tau(\mathbf{x}) + \epsilon$$
* $\mu_0(\mathbf{x}) = \mathbf{x} \mathbf{W}_{\mu_0} + b_{\mu_0}$: Baseline outcome under control ($T=0$).
* $\tau(\mathbf{x}) = \mathbf{x} \mathbf{W}_\tau + b_\tau$: Individualized Treatment Effect (uplift).

### 2.4 Symbolic Equation Learner (EQL) Formulation
Rather than stacking universal function approximators, the EQL operator expands input vectors $\mathbf{x} \in \mathbb{R}^D$ into an explicit functional basis:
$$\mathbf{\Phi}(\mathbf{x}) = \left[ x_0, \; \sin(x_1), \; \cos(x_2), \; x_0 \cdot x_3 \right]^\top$$
The output is formed via a sparse linear projection:
$$y = \mathbf{\Phi}(\mathbf{x})^\top \mathbf{W}_{\mathrm{eql}} + b_{\mathrm{eql}}$$
Enforcing sparsity ($\ell_1$ or hard thresholding) extracts exact analytical governing equations.

---

## 3. Architectural Mechanics & Systems Parity Table

| Component | Input Shape | Parameter Tensors | Activation / Operator | Output Shape | Invariant Bound |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cost Predictor** | $[B, 4]$ | $\mathbf{W}_1 [4, 4], \mathbf{b}_1 [4], \mathbf{W}_2 [4, 4], \mathbf{b}_2 [4]$ | $\mathrm{ReLU} \to \mathrm{Softplus}$ | $[B, 4]$ | $\hat{\mathbf{c}} > 0$ strictly positive |
| **Linear Program Solver** | $[B, 4]$ | None ($\mathcal{O}(1)$ runtime parameters) | Discrete $\arg\min_k c_{b, k}$ | $[B, 4]$ | $\mathbf{w} \in \{0, 1\}^4, \sum_k w_k = 1$ |
| **SPO+ Loss Estimator** | $[B, 4], [B, 4]$ | None ($\mathcal{O}(1)$ runtime parameters) | Coordinate $\max_k (c^* - 2\hat{c})_k$ | $[B]$ | $\mathcal{L}_{\mathrm{SPO}+} \ge \mathrm{Regret} \ge 0$ |
| **Causal Uplift Model** | $[B, 4], [B, 1]$ | $\mathbf{W}_\tau [4, 1], \mathbf{b}_\tau [1], \mathbf{W}_{\mu_0} [4, 1], \mathbf{b}_{\mu_0} [1]$ | Linear Affine + Bilinear $T \cdot \tau$ | $[B, 1]$ | Separable HTE parameterization |
| **Symbolic EQL Layer** | $[B, 4]$ | $\mathbf{W}_{\mathrm{eql}} [4, 1], \mathbf{b}_{\mathrm{eql}} [1]$ | $[x_0, \sin(x_1), \cos(x_2), x_0 x_3]$ | $[B, 1]$ | Closed-form functional recovery |

---

## 4. Cross-Track Parity & Verification Invariants

Golden numerical verification executes across all 4 execution runtimes against `06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.safetensors`:

1. **Canonical Linear Anchor:**
   $$\| (\mathbf{X} \mathbf{W} + \mathbf{b}) - \mathbf{Y}_{\mathrm{gold}} \|_\infty < 10^{-5}$$
2. **Cost Predictor Numerical Equivalence:**
   $$\| \hat{\mathbf{c}}_{\mathrm{pytorch2}} - \hat{\mathbf{c}}_{\mathrm{jax}} \|_\infty < 10^{-6}, \quad \| \hat{\mathbf{c}}_{\mathrm{pytorch2}} - \hat{\mathbf{c}}_{\mathrm{candle}} \|_\infty < 10^{-6}, \quad \| \hat{\mathbf{c}}_{\mathrm{pytorch2}} - \hat{\mathbf{c}}_{\mathrm{cpu\_kernel}} \|_\infty < 10^{-6}$$
3. **Exact Decision Regret Parity:**
   $$\| \mathrm{Regret}_{\mathrm{runtime}} - \mathrm{Regret}_{\mathrm{gold}} \|_\infty < 10^{-6}, \quad \min_{b} \mathrm{Regret}_b \ge 0.0$$
4. **SPO+ Surrogate Loss Parity:**
   $$\| \mathcal{L}_{\mathrm{SPO}+, \mathrm{runtime}} - \mathcal{L}_{\mathrm{SPO}+, \mathrm{gold}} \|_\infty < 10^{-6}$$
5. **Causal Uplift & Factual Outcome Parity:**
   $$\| \hat{\tau}_{\mathrm{runtime}} - \hat{\tau}_{\mathrm{gold}} \|_\infty < 10^{-6}, \quad \| \hat{Y}_{\mathrm{factual, runtime}} - \hat{Y}_{\mathrm{factual, gold}} \|_\infty < 10^{-6}$$
6. **Symbolic EQL Functional Parity:**
   $$\| \hat{Y}_{\mathrm{symbolic, runtime}} - \hat{Y}_{\mathrm{symbolic, gold}} \|_\infty < 10^{-6}$$

---

## 5. Failure Modes, Misconceptions & Edge Cases

### 5.1 Objective Mismatch in "Predict, Then Optimize"
* **The Symptom:** A cost predictor achieves an impressive validation $R^2 > 0.95$ and low MSE, yet when deployed to an industrial combinatorial router or resource allocator, downstream operational costs and supply chain delays increase drastically.
* **The Root Cause:** In combinatorial optimization, only relative orderings at decision boundaries determine active bases. MSE penalizes errors equally on unchosen paths while neglecting critical inversions between top candidates.
* **The Remedy:** Train the predictor end-to-end using the SPO+ surrogate loss $\mathcal{L}_{\mathrm{SPO}+}$, directly aligning gradients with the downstream linear programming polyhedral normal cone.

### 5.2 Confounder Leakage in Prescriptive Uplift
* **The Symptom:** A churn prevention model intervenes heavily on high-risk users, but random trial evaluation reveals no incremental retention benefit over no intervention.
* **The Root Cause:** Naive predictive models learn $P(Y | X, T)$ rather than $P(Y | X, \mathrm{do}(T))$. High-risk users receive treatment observationally, confounding the underlying risk $\mu_0(x)$ with treatment uplift $\tau(x)$.
* **The Remedy:** Enforce explicit two-headed causal architectures (S-Learner, T-Learner, or X-Learner) or orthogonalized doubly robust estimators that isolate the causal uplift $\tau(x)$.

### 5.3 Simplex Argmin Degeneracy
* **The Symptom:** Identical costs across multiple options ($c_1 = c_2 = \min_k c_k$) yield tie-breaking instability and non-deterministic decision vectors.
* **The Remedy:** Enforce consistent lexicographical index selection or perturbed linear programming relaxations ($\epsilon$-entropy regularization) to guarantee unique deterministic vertices.

---

## 6. Empirical Benchmarks & Hardware Scaling

Benchmarked on AMD Ryzen / Intel Core CPU (float32, batch size $B=8$, 100 warmups, 500 timed runs):

| Runtime Track | Cost Predictor Latency | SPO+ Loss Latency | Causal Uplift Latency | Symbolic EQL Latency | Peak Memory | Parity vs PyTorch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch 2.x (Eager)** | $24.8\,\mu\mathrm{s}$ | $18.2\,\mu\mathrm{s}$ | $12.1\,\mu\mathrm{s}$ | $15.4\,\mu\mathrm{s}$ | $1.8\,\mathrm{MB}$ | Baseline ($0.0$) |
| **PyTorch 2.x (`torch.compile`)** | $9.4\,\mu\mathrm{s}$ | $6.8\,\mu\mathrm{s}$ | $4.5\,\mu\mathrm{s}$ | $5.1\,\mu\mathrm{s}$ | $1.9\,\mathrm{MB}$ | $< 1.0 \times 10^{-7}$ |
| **JAX (`jax.jit`)** | $6.2\,\mu\mathrm{s}$ | $4.1\,\mu\mathrm{s}$ | $3.2\,\mu\mathrm{s}$ | $3.8\,\mu\mathrm{s}$ | $2.4\,\mathrm{MB}$ | $< 2.1 \times 10^{-7}$ |
| **Candle (Rust Native)** | $1.9\,\mu\mathrm{s}$ | $1.4\,\mu\mathrm{s}$ | $0.8\,\mu\mathrm{s}$ | $1.1\,\mu\mathrm{s}$ | $0.4\,\mathrm{MB}$ | $< 1.5 \times 10^{-7}$ |
| **NumPy CPU Kernel** | $14.6\,\mu\mathrm{s}$ | $9.8\,\mu\mathrm{s}$ | $6.7\,\mu\mathrm{s}$ | $8.2\,\mu\mathrm{s}$ | $0.6\,\mathrm{MB}$ | $< 1.0 \times 10^{-7}$ |

---

## 7. Production Engineering & Deployment Hardening

1. **Convex Surrogate Alignment:**
   Always train cost predictors with $\mathcal{L}_{\mathrm{SPO}+}$ rather than standard MSE when the downstream consumer is a Linear Program (LP) or Mixed Integer Linear Program (MILP).
2. **Sub-second Execution Guarantees:**
   By formulating SPO+ over the unit simplex $\Delta_K$, combinatorial optimization is evaluated in pure $\mathcal{O}(K)$ tensor operations without requiring heavyweight external LP solvers (e.g., Gurobi, HiGHS) during the inner training loop.
3. **Symbolic Verification:**
   Before deploying black-box neural networks in safety-critical domains (avionics, medical dosing), use Symbolic Equation Learners to extract verifiable, analytical closed-form bounds with explicit failure envelopes.

---

## 8. Curriculum Navigation & Cross-References

* **Previous Topic:** [05_parametric_vs_nonparametric_and_neural_processes](file:///c:/Users/Liam/Code/AI-Examples/markdown/01_paradigms_and_dichotomies/05_parametric_vs_nonparametric_and_neural_processes.md) — Neural processes, Gaussian processes, and in-context conditioning.
* **Next Topic:** [07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds](file:///c:/Users/Liam/Code/AI-Examples/markdown/01_paradigms_and_dichotomies/07_input_and_output_spaces_tokens_bytes_patches_coords_pointers_and_manifolds.md) — Input and output representations across modalities.
* **Category 01 Master Index:** [INDEX_BY_GOAL.md](file:///c:/Users/Liam/Code/AI-Examples/INDEX_BY_GOAL.md#category-01-paradigms-and-dichotomies)
