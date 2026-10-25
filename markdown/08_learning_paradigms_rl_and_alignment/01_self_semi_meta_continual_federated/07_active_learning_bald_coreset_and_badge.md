# Active Learning: BALD, Core-Set, and BADGE

## 1. Mathematical Foundations & Lineage

Active Learning (AL) optimizes the acquisition of labeled training data from an unlabeled pool $\mathcal{U}$ given a labeling budget $B \ll |\mathcal{U}|$. Rather than randomly sampling, active selection algorithms prioritize points that maximize expected model improvement.

The field has evolved through three primary mathematical regimes:
1. **Uncertainty & Information Theory (Lewis & Gale 1994; Houlsby et al. 2011 BALD)**: Measures epistemic uncertainty (model parameter ambiguity) rather than aleatoric noise (inherent sensor or label noise) via mutual information.
2. **Geometric & Density Coverage (Sener & Savarese 2018 Core-Set)**: Solves the $k$-center problem on latent feature embeddings to guarantee that every unlabeled point lies within an $\epsilon$-cover of a labeled point.
3. **Hybrid Gradient Manifolds (Ash et al. 2020 BADGE)**: Unifies uncertainty and diversity by embedding points as hypothetical loss gradients in the final network layer, subsequently sampling diverse directions via $k$-means++.

---

## 2. Information-Theoretic Active Learning & BALD

Given model parameters $\omega$ with posterior $p(\omega \mid \mathcal{D})$, Bayesian Active Learning by Disagreement (BALD) selects points that maximize mutual information between the predicted class $y$ and the parameters $\omega$:
$$\mathbb{I}(y; \omega \mid x, \mathcal{D}) = \mathbb{H}(y \mid x, \mathcal{D}) - \mathbb{E}_{\omega \sim p(\omega \mid \mathcal{D})} \left[ \mathbb{H}(y \mid x, \omega) \right]$$

Using $M$ Monte Carlo dropout or ensemble samples $\omega^{(1)}, \dots, \omega^{(M)}$:
1. Mean class probability:
   $$\bar{p}_c(x) = \frac{1}{M} \sum_{m=1}^M p(y=c \mid x, \omega^{(m)})$$
2. Entropy of the expected prediction:
   $$\mathbb{H}(\bar{p}) = - \sum_{c=1}^C \bar{p}_c(x) \log \bar{p}_c(x)$$
3. Expected entropy of individual predictions:
   $$\mathbb{E}[\mathbb{H}(p)] = \frac{1}{M} \sum_{m=1}^M \left( - \sum_{c=1}^C p_c^{(m)}(x) \log p_c^{(m)}(x) \right)$$
$$\text{BALD}(x) = \mathbb{H}(\bar{p}) - \mathbb{E}[\mathbb{H}(p)]$$

BALD approaches zero when the model is confident or when predictions are uniformly random across all models (aleatoric noise). It peaks exclusively when individual models make confident but conflicting predictions (epistemic uncertainty).

---

## 3. Geometric Diversity & Core-Set k-Center Greedy

Sener & Savarese (2018) prove that the generalization error of active learning is bounded by the Core-Set coverage radius $\delta$:
$$\min_{S \subseteq \mathcal{U}, |S|=B} \max_{u \in \mathcal{U}} \min_{s \in S \cup \mathcal{L}} \| \phi(u) - \phi(s) \|_2$$

Because the $k$-center problem is NP-hard, the standard polynomial-time 2-approximation algorithm is $k$-Center Greedy:
1. Initialize active batch $S = \emptyset$.
2. For each candidate $u \in \mathcal{U}$, compute minimum distance to the current labeled and selected set:
   $$d(u) = \min_{l \in \mathcal{L} \cup S} \| \phi(u) - \phi(l) \|_2$$
3. Select point with maximum distance:
   $$u^* = \arg\max_{u \in \mathcal{U} \setminus S} d(u)$$
4. Add $u^*$ to $S$ and update distances until budget $B$ is reached.

---

## 4. Hybrid Uncertainty & Gradient Embeddings (BADGE)

BADGE (Ash et al., 2020) combines uncertainty and diversity by embedding unlabeled points in the space of gradients with respect to the output layer parameters $W \in \mathbb{R}^{C \times D}$.

For an unlabeled sample $x$ with penultimate activation $z(x) \in \mathbb{R}^D$ and model output distribution $\hat{p} \in \Delta^C$:
- Hypothetical pseudo-label: $\hat{y} = \arg\max_c \hat{p}_c$.
- Gradient of cross-entropy loss wrt $W$:
  $$g_x = (\hat{p} - e_{\hat{y}}) \otimes z(x) \in \mathbb{R}^{C \times D}$$
The norm $\|g_x\|_2 = \|\hat{p} - e_{\hat{y}}\|_2 \|z(x)\|_2$ captures classification uncertainty (zero if confident, large if uncertain), while the outer product direction captures feature diversity. Diverse points are selected using $k$-means++ initialization on $\{g_x\}$.

---

## 5. Query Batching & Diversity vs Uncertainty Tradeoffs

Pure uncertainty sampling (e.g. max entropy) frequently queries redundant clusters of identical difficult samples. Core-Set guarantees diversity but wastes labeling capacity on uninformative outliers in low-density feature space. BADGE and BALD+CoreSet hybrids achieve Pareto optimality on the uncertainty-diversity frontier.

---

## 6. Cross-Framework Implementations & Numerical Parity

Every component in this topic is validated across PyTorch 2.x, JAX, and Candle against seed-42 golden tensors in `07_active_learning_bald_coreset_and_badge.safetensors`:
1. `linear_anchor`: Multi-framework linear mapping verification.
2. `bald_score`: Epistemic mutual information calculation over MC predictions.
3. `coreset_min_distances`: $k$-Center greedy minimum Euclidean distance evaluations.
4. `badge_gradient_embeddings`: Hypothetical output layer gradient outer-product embedding.

All implementations achieve transitive numerical parity $< 10^{-5}$.

---

## 7. Hardware Acceleration & WGSL Compute Shaders

The computation of BADGE gradient embeddings $\mathcal{O}(B \cdot C \cdot D)$ is parallelized on WebGPU via `BADGE_GRADIENT_EMBEDDING_WGSL`:
```wgsl
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let di = global_id.x;
    let bi = global_id.y;
    if (bi >= cfg.b || di >= cfg.d) {
        return;
    }
    let p_offset = bi * cfg.c;
    var max_p: f32 = -1.0;
    var argmax: u32 = 0u;
    for (var ci: u32 = 0u; ci < cfg.c; ci = ci + 1u) {
        let p = probs[p_offset + ci];
        if (p > max_p) {
            max_p = p;
            argmax = ci;
        }
    }
    let f_val = feats[bi * cfg.d + di];
    let out_b_offset = bi * cfg.c * cfg.d;
    for (var ci: u32 = 0u; ci < cfg.c; ci = ci + 1u) {
        var y: f32 = 0.0;
        if (ci == argmax) {
            y = 1.0;
        }
        let diff = probs[p_offset + ci] - y;
        embeddings[out_b_offset + ci * cfg.d + di] = diff * f_val;
    }
}
```
This kernel computes outer-product gradient representations in a single GPU pass without memory reallocation.

---

## 8. Performance Profiling, Anti-Patterns & Misuse Guardrails

### Performance Profiling
1. **Core-Set Distance Caching**: In $k$-Center greedy selection of $B$ points from $N$ candidates, update minimum distances in $\mathcal{O}(N)$ per step rather than recomputing the full $N \times |S|$ distance matrix from scratch ($\mathcal{O}(N \cdot |S|)$).
2. **BADGE Dimension Scaling**: For high-dimensional classifiers (large $C$ and $D$), apply random Johnson-Lindenstrauss projections to $g_x$ before running $k$-means++.

### Misuse Guardrails & Anti-Patterns
1. **Raw Predictive Entropy on Noisy Datasets**: Querying points by raw entropy $H(\bar{p})$ squanders annotation budgets on inherently ambiguous/corrupt samples (aleatoric noise); **BALD** explicitly isolates epistemic disagreement.
2. **Outlier Sensitivity in Core-Set**: Standard Core-Set without outlier filtering queries noisy outliers that lie far away in embedding space but contribute zero generalizable information.
3. **Omitting Penultimate Features in BADGE**: Computing BADGE using only the probability vector $(\hat{p} - e_{\hat{y}})$ discards all geometric feature representation diversity.
