# Continual Learning: EWC, Synaptic Intelligence, LwF, and Dark Experience Replay

## 1. Mathematical Foundations & Lineage

Continual (lifelong) learning addresses the fundamental challenge of **catastrophic forgetting**: when a neural network sequentially trains on a sequence of tasks $\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_T$, standard stochastic gradient descent overwrites the weights optimal for earlier tasks.

Solutions are broadly categorized into three paradigms:
1. **Prior-Focused / Regularization (Kirkpatrick et al. 2017 EWC; Zenke et al. 2017 SI)**: Place quadratic penalties on parameter movements based on how critical each weight was to prior task performance, approximating the posterior distribution $p(\theta \mid \mathcal{D}_{1:T})$.
2. **Functional Regularization / Distillation (Li & Hoiem 2016 LwF)**: Constrain output logits via knowledge distillation on new task inputs, avoiding parameter-space penalties.
3. **Rehearsal & Dark Knowledge (Chaudhry et al. 2019 A-GEM; Buzzega et al. 2020 DER++)**: Maintain a tiny episodic memory of past examples and replay historical un-normalized logits ("dark knowledge") to retain both decision boundaries and calibration.

---

## 2. Elastic Weight Consolidation (EWC) & Fisher Information Geometry

From a Bayesian perspective, conditional probability after observing task $A$ and learning task $B$ follows:
$$\log p(\theta \mid \mathcal{D}_A, \mathcal{D}_B) = \log p(\mathcal{D}_B \mid \theta) + \log p(\theta \mid \mathcal{D}_A) - \log p(\mathcal{D}_B)$$

Because the posterior $p(\theta \mid \mathcal{D}_A)$ is intractable, EWC approximates it as a Gaussian centered at $\theta_A^*$ with precision given by the diagonal elements of the Fisher Information Matrix $F$:
$$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_B(\theta) + \sum_{i=1}^D \frac{\lambda}{2} F_i (\theta_i - \theta_{A, i}^*)^2$$

The diagonal empirical Fisher is:
$$F_i = \frac{1}{|\mathcal{D}_A|} \sum_{x \in \mathcal{D}_A} \left( \frac{\partial \log p_\theta(y \mid x)}{\partial \theta_i} \right)^2$$
Parameters with large Fisher values represent principal directions of task $A$'s manifold curvature and are heavily penalized if shifted.

---

## 3. Synaptic Intelligence & Online Trajectory Integration

Rather than estimating curvature offline via post-hoc Fisher computation, Synaptic Intelligence (SI; Zenke et al., 2017) continuously accumulates the path integral of parameter importance online along the optimization trajectory:
$$\omega_k = \sum_{t=1}^T g_k(t) \cdot \Delta \theta_k(t) = \int_{\theta(0)}^{\theta(T)} \frac{\partial \mathcal{L}}{\partial \theta_k} d\theta_k$$

The normalized parameter importance matrix is:
$$\Omega_k = \frac{\max(0, \omega_k)}{(\Delta \theta_k)^2 + \xi}$$
where $\Delta \theta_k = \theta_k(T) - \theta_k(0)$ and $\xi$ is a damping factor preventing division by zero. SI adds quadratic penalties $\frac{c}{2} \sum_k \Omega_k (\theta_k - \theta_k^*)^2$.

---

## 4. Learning without Forgetting (LwF) & Functional Regularization

LwF (Li & Hoiem, 2016) eliminates the need for past data or parameter importance matrices by treating the old model $\theta_{\text{old}}$ as a teacher for prior task heads.

Given current task inputs $x$, the distillation loss with temperature $T$ is:
$$\mathcal{L}_{\text{distill}} = T^2 \cdot \text{KL}\left( \text{softmax}\left(\frac{f_{\theta_{\text{old}}}(x)}{T}\right) \parallel \text{softmax}\left(\frac{f_{\theta_{\text{new}}}(x)}{T}\right) \right)$$
LwF preserves output response manifolds on the input distribution of the new task.

---

## 5. Dark Experience Replay (DER++) & Experience Buffer Memory

Buzzega et al. (2020) demonstrated that replaying hard labels from a replay buffer leads to severe overfitting on the tiny buffer. Instead, Dark Experience Replay (`DER++`) preserves the model's entire logit trajectory:
1. When sample $(x, y)$ is stored in buffer $\mathcal{M}$, store its current pre-softmax logits $z = f_\theta(x)$.
2. When training on current task minibatch, sample a rehearsal batch $(x_r, y_r, z_r) \sim \mathcal{M}$.
3. Minimize composite objective:
   $$\mathcal{L}_{\text{DER++}} = \mathcal{L}_{\text{current}} + \alpha \| f_\theta(x_r) - z_r \|_2^2 + \beta \text{CE}(f_\theta(x_r), y_r)$$
Matching historical logits $z_r$ regularizes gradient updates and retains feature geometry across intermediate representations.

---

## 6. Cross-Framework Implementations & Numerical Parity

Every algorithm in this topic is validated against golden weights in `05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay.safetensors` across PyTorch 2.x, JAX, and Candle:
1. `linear_anchor`: System-wide verification baseline.
2. `ewc_penalty`: Weighted quadratic parameter consolidation penalty.
3. `synaptic_intelligence_omega`: Online path integral normalization.
4. `lwf_distillation_loss`: Temperature-scaled relative entropy distillation.
5. `der_dark_replay_loss`: Combined MSE logit matching and cross-entropy rehearsal.

All implementations match with maximum absolute difference $< 10^{-5}$.

---

## 7. Hardware Acceleration & WGSL Compute Shaders

The elementwise EWC quadratic penalty $\frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$ scales with model parameter count $D$ and is accelerated on WebGPU via `EWC_QUADRATIC_PENALTY_WGSL`:
```wgsl
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= cfg.d) {
        return;
    }
    let diff = theta[i] - theta_star[i];
    penalty_vec[i] = 0.5 * cfg.lambda_ewc * fisher[i] * diff * diff;
}
```
Executing this kernel directly in GPU VRAM prevents pulling large parameter vectors back to host memory.

---

## 8. Performance Profiling, Anti-Patterns & Misuse Guardrails

### Performance Profiling
1. **Fisher Computation Overhead**: Evaluating the full empirical Fisher requires an additional forward/backward pass over the previous task dataset. Diagonal Fisher reduces storage from $\mathcal{O}(D^2)$ to $\mathcal{O}(D)$.
2. **Buffer Capacity in Replay**: DER++ achieves $>90\%$ retention with buffer sizes as small as $0.5\text{–}1\%$ of total dataset size due to the high information density of dark logits.

### Misuse Guardrails & Anti-Patterns
1. **Ground-Truth Fisher Bias**: Computing Fisher using ground-truth labels $\nabla_\theta \log p_\theta(y_{\text{true}} \mid x)$ instead of model-sampled labels $\nabla_\theta \log p_\theta(y \sim p_\theta \mid x)$ biases the Fisher towards high-loss outliers rather than model certainty.
2. **Replay Hard Label Overfitting**: Replaying only one-hot hard labels on small replay buffers leads to rapid memorization and feature distortion.
3. **Task-Recency Imbalance in EWC**: Summing independent Fisher penalties across many sequential tasks ($\sum_t F^{(t)}$) causes over-regularization where weights become completely frozen. Use online EWC running averages.
