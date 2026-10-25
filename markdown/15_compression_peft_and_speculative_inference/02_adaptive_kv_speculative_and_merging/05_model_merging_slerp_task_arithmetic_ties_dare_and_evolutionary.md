# Weight-Space Model Merging: SLERP, Task Arithmetic, TIES, DARE, and Evolutionary Merging

Training distinct foundation models for every domain is computationally prohibitive. Rather than training multi-task models from scratch or fine-tuning sequentially (which induces catastrophic forgetting), **weight-space model merging** combines multiple task-specialized checkpoints directly in parameter space without requiring access to original training data or gradient computation. This topic covers the modern taxonomy of weight-space composition: **Linear Task Arithmetic**, hyper-spherical interpolation (**SLERP**), interference-eliminating sign consensus (**TIES-Merging**), extreme stochastic pruning (**DARE**), and automated benchmark-driven architecture search (**Evolutionary Merging**).

---

## 1. Weight-Space Model Merging: Foundations & Linear Arithmetic

When fine-tuning from a common pretrained base checkpoint $\theta_{\text{base}}$, fine-tuned models lie in a shared low-loss basin. The update vector $\tau_m = \theta_m - \theta_{\text{base}}$ behaves as a modular **task vector** (Ilharco et al., 2023):

$$\theta_{\text{merged}} = \theta_{\text{base}} + \sum_{m=1}^M \lambda_m \tau_m$$

Task vectors support algebraic operations:
- **Task Addition**: Combining math reasoning ($\tau_{\text{math}}$) and Python coding ($\tau_{\text{code}}$).
- **Task Negation**: Subtracting toxic or biased behavior ($\theta_{\text{base}} - \lambda \tau_{\text{toxic}}$).
- **Domain Analogy**: $\theta_{\text{German-Medical}} \approx \theta_{\text{German}} + (\theta_{\text{English-Medical}} - \theta_{\text{English}})$.

| Merging Paradigm | Geometric Space | Interference Mechanism | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **Linear Arithmetic** | Euclidean Flat Space | Naive parameter summation | Simple 2-model composition |
| **SLERP** | Hyperspherical Manifold | Geodesic arc interpolation | High-angle checkpoint fusion |
| **TIES-Merging** | Sparsified Sign Space | Magnitude trimming & sign consensus | Multi-task ($M \ge 3$) without interference |
| **DARE** | Stochastic Bernoulli Space | Extreme dropout ($p \in [0.7, 0.9]$) + rescaling | Merging $10+$ instruction/chat models |
| **Evolutionary Merging** | Combinatorial Graph Space | CMA-ES optimization across layers | Automated Open LLM Leaderboard models |

---

## 2. Spherical Linear Interpolation (SLERP): Curvature & Norm Preservation

In high-dimensional weight spaces, naive linear interpolation (LERP) $(1-t)\theta_A + t\theta_B$ cuts through the interior of the sphere, causing parameter magnitude shrinkage:
$$\|(1-t)v_0 + tv_1\| < \|v_0\|$$

This norm collapse flattens activation variances, degrading downstream perplexity. **SLERP (Shoemake, 1985)** traverses the spherical geodesic surface, preserving parameter magnitude and curvature:

```
          v0 ●
            / \
           /   \  Geodesic Arc (SLERP preserves norm)
          /  ●  \
         /  SLERP\
        /    │    \
       /     │ LERP (Norm shrinks!)
      /      ▼      \
     ●───────────────● v1
   Origin
```

### Mathematical Formulation
Given unit vectors $u_0 = v_0 / \|v_0\|$ and $u_1 = v_1 / \|v_1\|$:
$$\theta = \arccos(\text{clamp}(u_0 \cdot u_1, -1, 1))$$
$$\text{SLERP}(v_0, v_1, t) = \frac{\sin((1-t)\theta)}{\sin(\theta)} v_0 + \frac{\sin(t\theta)}{\sin(\theta)} v_1$$

When $\theta \approx 0$ or $\sin(\theta) < 10^{-6}$, the geodesic degenerates to standard LERP.

---

## 3. TIES-Merging: Trimming, Electing Sign, and Disjoint Merging

As the number of fine-tuned models $M$ increases, task vectors suffer from two catastrophic failure modes:
1. **Redundant Parameter Accumulation**: Small non-zero parameter drifts act as high-frequency noise.
2. **Sign Disagreement (Interference)**: Conflicting gradient signs between tasks cancel each other out ($+0.1 + (-0.1) = 0.0$), destroying both capabilities.

**TIES-Merging (Yadav et al., NeurIPS 2023)** resolves interference in three stages:

### Stage 1: Trim (Sparsification)
Retain only the top $k\%$ largest magnitude values in each task vector $\tau_m$:
$$\hat{\tau}_{m, j} = \begin{cases} \tau_{m, j} & \text{if } |\tau_{m, j}| \ge \text{threshold}_k \\ 0 & \text{otherwise} \end{cases}$$

### Stage 2: Elect Sign (Consensus)
Compute majority directional agreement across tasks for each coordinate $j$:
$$\gamma_j = \text{sgn}\left(\sum_{m=1}^M \hat{\tau}_{m, j}\right)$$

### Stage 3: Disjoint Merge (Selective Averaging)
Discard parameters whose sign conflicts with the consensus $\gamma_j$:
$$\tilde{\tau}_{m, j} = \begin{cases} \hat{\tau}_{m, j} & \text{if } \text{sgn}(\hat{\tau}_{m, j}) == \gamma_j \\ 0 & \text{otherwise} \end{cases}$$
Average the retained non-zero entries across participating tasks:
$$\theta_{\text{merged}} = \theta_{\text{base}} + \lambda \frac{1}{|A_j|} \sum_{m \in A_j} \tilde{\tau}_{m, j}$$

---

## 4. DARE (Drop And REscale): Zero-Shot Extreme Sparsification

Yu et al. (ICML 2024) uncovered a stunning property of fine-tuned task vectors: up to $90\%$ of delta parameters can be dropped completely at random without loss of capability, provided surviving elements are rescaled.

### Formulation
For each parameter coordinate with drop probability $p \in [0.7, 0.9]$:
$$\tilde{\tau}_{m, j} = \frac{1}{1 - p} \cdot \text{Bernoulli}(1 - p) \cdot \tau_{m, j}$$

By combining DARE with TIES-merging (**DARE-TIES**), dozens of specialized models (coding, medical, roleplay, translation) can be merged into a single multi-expert foundation model with zero degradation.

---

## 5. Evolutionary Model Merging & Layer Stacking (Frankenmerging)

Rather than applying uniform scalar weights across all layers, **Evolutionary Model Merging (Akiba et al., Sakana AI 2024)** explores two macro-search spaces using CMA-ES:

1. **Parameter Merging Space**: Optimizes per-layer interpolation weights $\vec{\lambda} \in \mathbb{R}^L$.
2. **Layer Stacking (Frankenmerging)**: Slices and interleaves layers from different models (e.g. taking layers $0\dots 16$ from Model A, layers $8\dots 24$ from Model B, and layers $16\dots 32$ from Model C) to create expanded architectures (e.g. 7B $\to$ 11B).

Automated fitness evaluation against target benchmarks (GSM8k, HumanEval, ARC) discovers non-intuitive parameter alignments that outperform human manual tuning.

---

## 6. Multi-Framework Parity & Verification (PyTorch, JAX, Candle)

All model merging transformations maintain exact mathematical equivalence across PyTorch, JAX, and Candle:

```rust
// Candle TIES-Merging Verification
let t_out = topic::ties_merge(&w["base_w"], &w["deltas"], 0.5, 0.6)?;
common::assert_max_diff(&t_out, &w["ties_out"], 1e-5)?;

// Candle SLERP Verification
let s_out = topic::slerp_merge(&w["v0"], &w["v1"], 0.3)?;
common::assert_max_diff(&s_out, &w["slerp_out"], 1e-5)?;
```

Maximum absolute error across all tracks is bounded below $10^{-5}$.

---

## 7. GPU/WebGPU Kernel Optimizations

Multi-task vector accumulation across memory buffers is executed using a parallel 1D workgroup compute shader in WGSL:

```wgsl
@compute @workgroup_size(64, 1, 1)
fn task_arithmetic_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.total_elements) { return; }
    var val = base[idx];
    for (var k = 0u; k < u.num_tasks; k++) {
        val += weights[k] * deltas[k * u.total_elements + idx];
    }
    merged[idx] = val;
}
```

Coalesced memory reads stream task deltas directly into FP32 registers.

---

## 8. Best Practices, Interference Diagnosis, and Production Assembly

1. **Check Base Checkpoint Lineage**: Merging models derived from different base architectures or different base pretraining checkpoints (e.g. Mistral + LLaMA) causes catastrophic activation divergence; always verify identical parent weights.
2. **Trim Redundant Perturbations**: Always apply TIES or DARE pruning when merging $>2$ task vectors to prevent gradient sign collision.
3. **Layer-Wise Weight Profiling**: Early transformer layers (token representations) require smaller task perturbation weights ($\lambda \approx 0.2$), whereas middle-to-late MLP layers absorb high task weights ($\lambda \approx 0.6-0.8$).
4. **Evaluate Cross-Task Transfer**: Measure safety and perplexity degradation on unrelated held-out benchmarks after merging to ensure general reasoning capabilities remain uncompromised.
