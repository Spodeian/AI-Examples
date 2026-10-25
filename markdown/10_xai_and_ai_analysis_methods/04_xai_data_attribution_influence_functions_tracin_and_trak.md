# XAI Data Attribution: Influence Functions, TracIn, and TRAK

## 1. Mathematical Foundations & Lineage

While feature attribution explains which inputs features swayed a prediction, training data attribution answers the dual question: *which specific training examples caused the model to make this prediction?*

The lineage of data attribution methods comprises:
1. **Classical Deletion Diagnostics (Cook & Weisberg 1982)**: Cook's distance measured parameter shift $\hat{\theta} - \hat{\theta}_{-i}$ upon deleting sample $i$ in ordinary least squares.
2. **Influence Functions (Koh & Liang 2017)**: Formalized asymptotic robust statistics for deep neural networks, using implicit differentiation on empirical risk minimizers to estimate the infinitesimal influence of upweighting training sample $z_{\text{train}}$ without retraining.
3. **Data Shapley (Ghorbani & Zou 2019)**: Adapted cooperative game theory to value individual training examples based on their marginal contribution to validation accuracy across power-set subsets.
4. **TracIn (Pruthi et al. 2020)**: Decomposed training data influence along first-order gradient descent trajectories, calculating checkpoint inner products $\nabla \mathcal{L}_{\text{train}} \cdot \nabla \mathcal{L}_{\text{test}}$ without requiring expensive Hessian inversions.
5. **TRAK (Park et al. 2023)**: Scaled gradient-based attribution to billion-parameter models by combining linearized deep network representations with Johnson-Lindenstrauss random projections and empirical covariance normalizations.

---

## 2. Influence Functions Formulation & Inverse-HVP

Let $\hat{\theta} = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \mathcal{L}(z_i, \theta)$ be the empirical risk minimizer. If training instance $z_{\text{train}}$ is upweighted by $\epsilon$, the perturbed minimizer satisfies:
$$\left. \frac{d\hat{\theta}_{\epsilon, z_{\text{train}}}}{d\epsilon} \right|_{\epsilon=0} = - H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z_{\text{train}}, \hat{\theta})$$
where $H_{\hat{\theta}} = \frac{1}{N} \sum_{i=1}^N \nabla^2_\theta \mathcal{L}(z_i, \hat{\theta}) + \lambda I$ is the positive-definite damped empirical Hessian.

### Chain Rule on Test Loss
Applying the chain rule yields the change in test loss $\mathcal{L}(z_{\text{test}}, \hat{\theta})$:
$$\mathcal{I}_{\text{up, loss}}(z_{\text{train}}, z_{\text{test}}) = - \nabla_\theta \mathcal{L}(z_{\text{test}}, \hat{\theta})^T H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z_{\text{train}}, \hat{\theta})$$

To compute this efficiently, one first solves the linear system for the inverse Hessian-vector product (IHVP):
$$v_{\text{test}} = H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z_{\text{test}}, \hat{\theta})$$
and then computes the scalar inner product: $\mathcal{I} = - \nabla_\theta \mathcal{L}(z_{\text{train}}, \hat{\theta}) \cdot v_{\text{test}}$.

---

## 3. Tracing Gradient Descent (TracIn)

Influence functions assume that the model has reached a strict local minimum where $\nabla \mathcal{L} \approx 0$ and $H \succ 0$. In deep learning, training trajectories undergo non-convex dynamics and early stopping.

TracIn measures the cumulative reduction in test loss directly induced by gradient steps on training sample $z_{\text{train}}$ across checkpoints $t \in \{1, \dots, T\}$:
$$\text{TracIn}(z_{\text{train}}, z_{\text{test}}) = \sum_{t=1}^T \eta_t \left( \nabla_\theta \mathcal{L}(z_{\text{train}}, \theta_t) \cdot \nabla_\theta \mathcal{L}(z_{\text{test}}, \theta_t) \right)$$
where $\eta_t$ is the learning rate at checkpoint $t$. Positive TracIn scores denote *proponents* (examples that decreased test loss), while negative scores denote *opponents* (conflicting examples).

---

## 4. Scalable Data Attribution via Random Projections (TRAK)

For large neural networks where parameter count $D \gg 10^7$, storing and taking inner products of raw parameter gradient vectors is computationally prohibitive.

TRAK projects model gradients into a lower-dimensional feature space $\mathbb{R}^K$ ($K \ll D$) using a random Johnson-Lindenstrauss projection matrix $P \in \mathbb{R}^{D \times K}$:
$$\phi(z) = \nabla_\theta \mathcal{L}(z, \theta) P$$
The normalized attribution score between training example $i$ and test query $z_{\text{test}}$ is:
$$\text{TRAK}(z_i, z_{\text{test}}) = \frac{\phi(z_i) \cdot \phi(z_{\text{test}})}{\|\phi(z_i)\|_2 \|\phi(z_{\text{test}})\|_2 + \epsilon}$$

---

## 5. Multi-Ecosystem Implementation Matrix

| Operator | PyTorch 2.x (`pytorch2/`) | JAX (`jax/`) | Candle (`candle/`) | Hardware Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **`influence_score`** | `torch.linalg.solve` damped IHVP | `jnp.linalg.solve` damped IHVP | Gauss-Jordan linear solver | Vectorized Hessian-vector |
| **`tracin_trajectory_attribution`** | Checkpoint reduction + dot product | Elementwise sum over checkpoints | Dot product accumulator | `TRACIN_ACCUMULATION_WGSL` |
| **`trak_projected_attribution`** | Random matmul + cosine normalize | Matrix multiplication + norm | Vectorized matmul and normalize | Parallel JL projection |

---

## 6. Pathological Failure Modes & Diagnostic Postmortems

1. **Negative Curvature & LiSSA Divergence in Deep Networks**:
   - *Symptom*: Stochastic inverse-HVP solvers (LiSSA) explode towards $\pm \infty$ or diverge.
   - *Cause*: Intermediate checkpoints of deep neural networks possess negative Hessian eigenvalues ($\lambda_{\min}(H) < 0$), violating the positive-definite requirement for Neumann series inversion $(I - H)^k$.
   - *Fix*: Apply sufficient damping $\lambda \ge |\lambda_{\min}| + 10^{-3}$ or formulate influence via Conjugate Gradient on $H^T H + \lambda I$.
2. **First-Order TracIn Path Horizon Bias**:
   - *Symptom*: Checkpoints near the end of training dominate attribution scores while early representations are ignored.
   - *Cause*: Constant weighting when gradient magnitudes vary wildly across training phases.
   - *Fix*: Weight checkpoints by learning rates $\eta_t$ or evaluate normalized TracIn.
3. **Data Shapley Combinatorial Explosion**:
   - *Symptom*: Exact Data Shapley evaluation requires $2^N$ model retraining runs, which is impossible for $N > 20$.
   - *Cause*: Exact combinatorial formulation.
   - *Fix*: Use Monte Carlo Permutation Sampling with performance truncation (Truncated Monte Carlo Shapley).

---

## 7. Real-World Engineering Applications

1. **Training Set Poisoning & Mislabeled Data Auditing**: Flagging training samples with the highest negative influence on validation accuracy to automatically discover corrupted or adversarially mislabeled samples.
2. **Copyright & Intellectual Property Attribution**: Determining which licensed training documents contributed most to a generated code block or literary text.
3. **Curriculum Pruning & Active Learning**: Identifying redundant training examples whose removal would have negligible influence on test generalization.

---

## 8. Verification & Parity Protocols

Parity across PyTorch, JAX, and Candle is verified against `weights/10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak.safetensors`:
1. `linear_anchor`: Canonical multi-framework linear mapping.
2. `inf_ihvp_ref` & `inf_score_ref`: Inverse Hessian-vector product and influence scalar.
3. `tracin_score_ref`: TracIn trajectory inner product score.
4. `trak_scores_ref`: TRAK random-projected attribution vector across training instances.
