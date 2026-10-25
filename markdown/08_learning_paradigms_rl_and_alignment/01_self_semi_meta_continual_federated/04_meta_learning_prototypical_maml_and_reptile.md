# Meta-Learning: Prototypical Networks, MAML, and Reptile

## 1. Mathematical Foundations & Lineage

Meta-learning ("learning to learn") aims to optimize a model's adaptation process across a distribution of tasks $p(\mathcal{T})$, enabling rapid generalization to novel tasks with scarce supervision (few-shot learning). The field diverges into three classical families:
1. **Metric-Based (Koch et al. 2015 Siamese; Vinyals et al. 2016 Matching Nets; Snell et al. 2017 ProtoNets)**: Learns a non-parametric metric space where queries are classified based on proximity to labeled support prototypes.
2. **Gradient-Based (Finn et al. 2017 MAML)**: Explicitly optimizes for a shared initialization $\theta$ that can adapt to any task $\mathcal{T}_i \sim p(\mathcal{T})$ in one or a few gradient steps via bilevel optimization.
3. **First-Order Meta-Learning (Nichol et al. 2018 Reptile; Finn et al. 2017 FOMAML)**: Eliminates expensive Hessian-vector products and second-order computation by stepping initialization in the direction of task-adapted parameters.

---

## 2. Metric-Based Meta-Learning & Prototypical Networks

In an $N$-way $K$-shot episodic setup, each task provides a support set $S = \{(x_i, y_i)\}_{i=1}^{N \times K}$ containing $K$ labeled examples for each of $N$ classes, and a query set $Q$.

Prototypical Networks compute an embedding representation $f_\phi(x) \in \mathbb{R}^D$ and represent each class $k \in \{1, \dots, N\}$ by its mean support prototype:
$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

Given a query instance $x$, distances to each prototype define the predictive softmax distribution:
$$P(y = k \mid x) = \frac{\exp\left( -d(f_\phi(x), c_k) \right)}{\sum_{j=1}^N \exp\left( -d(f_\phi(x), c_j) \right)}$$
where $d(\cdot, \cdot)$ is squared Euclidean distance $d(u, v) = \|u - v\|_2^2$. Snell et al. proved that Bregman divergences (including squared Euclidean distance) correspond to linear models acting on support embeddings, while cosine distance breaks the clustering geometry.

---

## 3. Gradient-Based Optimization & Model-Agnostic Meta-Learning (MAML)

MAML (Finn et al., 2017) treats meta-learning as finding an initial parameter vector $\theta$ that minimizes validation loss after inner-loop gradient adaptation:

### Inner Loop (Task Adaptation)
For task $\mathcal{T}_i$ with support loss $\mathcal{L}_{\mathcal{T}_i}^{\text{supp}}$:
$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{supp}}(\theta)$$

### Outer Loop (Meta-Optimization)
The meta-gradient updates the initial parameters across a batch of tasks using query losses $\mathcal{L}_{\mathcal{T}_i}^{\text{query}}$:
$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta'_i)$$

By chain rule, the meta-gradient requires computing the Jacobian of the inner loop:
$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta'_i) = \left( I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{supp}}(\theta) \right) \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta'_i)$$
which incorporates the Hessian matrix $\nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{supp}}(\theta)$.

---

## 4. First-Order Approximations & Reptile Dynamics

### First-Order MAML (FOMAML)
FOMAML approximates the inner Jacobian as identity $\frac{\partial \theta'_i}{\partial \theta} \approx I$, dropping second derivatives:
$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta'_i) \approx \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta'_i)$$

### Reptile
Nichol et al. (2018) showed that moving the initialization toward task-adapted parameters without explicit query-set gradient computation naturally aligns initializations with task manifolds:
1. Sample task $\mathcal{T}$.
2. Perform $m$ steps of standard SGD from $\theta$ yielding $\theta^{(m)}$.
3. Update initialization:
   $$\theta \leftarrow \theta + \epsilon \left( \theta^{(m)} - \theta \right)$$
Reptile optimizes for parameters $\theta$ from which Euclidean distance to task manifolds is minimized.

---

## 5. Task Conditioning & Episodic Optimization

Episodic training structures the data stream to mirror test-time few-shot evaluation:
- At each training step, an episodic batch samples $N$ classes, $K$ support examples, and $Q$ query examples.
- Non-episodic standard batching fails to train encoders for invariant metric spaces, as the model memorizes class identities rather than comparative relational geometries.

---

## 6. Cross-Framework Implementations & Numerical Parity

Every algorithm in this topic is validated against golden tensors in `04_meta_learning_prototypical_maml_and_reptile.safetensors` across PyTorch 2.x, JAX, and Candle:
1. `linear_anchor`: System-wide parity check.
2. `protonet_prototypes`: Mean prototype aggregation across shots.
3. `protonet_logits`: Negative squared Euclidean distances between query set and class prototypes.
4. `maml_inner_linear_step`: Closed-form analytic inner gradient descent step under MSE loss.
5. `reptile_outer_step`: Outer meta-interpolation update.

All implementations achieve transitive numerical parity $< 10^{-5}$.

---

## 7. Hardware Acceleration & WGSL Compute Shaders

The computation of query-to-prototype Euclidean distances $\mathcal{O}(Q \cdot N \cdot D)$ is parallelized on WebGPU via `PROTONET_EUCLIDEAN_DIST_WGSL`:
```wgsl
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ni = global_id.x;
    let qi = global_id.y;
    if (qi >= cfg.q || ni >= cfg.n) {
        return;
    }
    let q_offset = qi * cfg.d;
    let n_offset = ni * cfg.d;
    var sum_sq: f32 = 0.0;
    for (var di: u32 = 0u; di < cfg.d; di = di + 1u) {
        let diff = queries[q_offset + di] - prototypes[n_offset + di];
        sum_sq = sum_sq + diff * diff;
    }
    logits[qi * cfg.n + ni] = -sum_sq;
}
```
This 2D workgroup tile computes Euclidean distances with zero intermediate tensor allocations.

---

## 8. Performance Profiling, Anti-Patterns & Misuse Guardrails

### Performance Profiling
1. **Inner-Loop Unrolling Memory**: Unrolling $M$ steps of full second-order MAML stores $M$ computation graphs in VRAM. For deep networks, use FOMAML or implicit MAML (iMAML) to bound memory to $\mathcal{O}(1)$ graphs.
2. **ProtoNet Metric Simplicity**: ProtoNets run $10\times$ faster than MAML because inference requires zero test-time backpropagation—only feedforward distance evaluation.

### Misuse Guardrails & Anti-Patterns
1. **Accidental Detach in MAML**: Calling `.detach()` on adapted parameters $\theta'_i$ before computing query loss completely severs the computation graph, reducing meta-gradients to zero.
2. **Cosine Distance in ProtoNets**: Using cosine similarity instead of squared Euclidean distance in Prototypical Networks violates Bregman divergence guarantees and leads to substantial accuracy drops ($3\text{–}5\%$).
3. **Overfitting Support Set in MAML**: Setting inner learning rate $\alpha$ too high causes the model to memorize the tiny support set rather than learning generalizable features.
