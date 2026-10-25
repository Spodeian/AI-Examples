# Adaptive Compute: Early Exit, Token Merging (ToMe), and FrugalGPT Cascades

Standard deep learning architectures execute a static computational budget for every input token and prompt regardless of intrinsic difficulty. A trivial greeting ("Hello!") traverses the exact same number of transformer layers, feed-forward matrix multiplications, and attention heads as a complex mathematical proof. **Adaptive compute** dynamically allocates execution cycles based on query complexity: early exit terminates depth-wise processing when confidence saturates, Token Merging (ToMe) compresses sequence length by fusing redundant tokens, and FrugalGPT cascades route prompts across hierarchical model tiers to slash API expenditure.

---

## 1. Adaptive Compute Foundations: Dynamic Inference Allocation

Adaptive inference operates across three distinct architectural axes: **depth-wise** (layer exit), **sequence-wise** (token reduction), and **model-wise** (cascade routing):

| Adaptive Strategy | Operational Axis | Core Mechanism | Speedup / Savings | Key Papers & Frameworks |
| :--- | :--- | :--- | :--- | :--- |
| **Early Exit** | Depth (Layers) | Internal classification heads & entropy stopping | $1.5\times - 3\times$ latency reduction | DeeBERT, FastBERT, BranchyNet |
| **Token Merging (ToMe)** | Sequence Length ($N$) | Bipartite soft matching on key/token embeddings | $2\times - 3\times$ throughput increase | Bolya et al. (ICLR 2023) |
| **FrugalGPT Cascades** | System / Model Tier | Sequential confidence-guided query escalation | $50\% - 90\%$ cost reduction | Chen et al. (NeurIPS 2023) |
| **Dynamic Routing (MoE)** | Width (Parameters) | Top-$k$ gating across specialized expert FFNs | Constant FLOPs with $10\times$ capacity | Switch Transformer, Mixtral |

---

## 2. Early Exit Architectures (DeeBERT, FastBERT, BranchyNet)

In standard transformer inference, intermediate representations often converge to the correct decision long before reaching the final layer. Early exit equips intermediate layers with auxiliary classification or LM prediction heads:

Given intermediate hidden states $h^{(l)} \in \mathbb{R}^{B \times d}$ at layer $l$:
$$z^{(l)} = h^{(l)} W_{\text{exit}}^{(l)} + b_{\text{exit}}^{(l)}, \quad p^{(l)} = \text{softmax}(z^{(l)})$$

### Exit Decision Metrics
1. **Maximum Probability (Confidence)**:
   $$\text{exit}^{(l)} = \mathbb{I}\left(\max_{c} p_c^{(l)} \ge \tau\right)$$
2. **Shannon Entropy**:
   $$H(p^{(l)}) = -\sum_{c} p_c^{(l)} \log p_c^{(l)}, \quad \text{exit}^{(l)} = \mathbb{I}\left(H(p^{(l)}) \le \eta\right)$$

If the exit condition is satisfied, execution halts immediately and $p^{(l)}$ is emitted as the final prediction, bypassing remaining layers $l+1 \dots L$.

---

## 3. Token Merging (ToMe) & Bipartite Soft Matching

Rather than pruning tokens (which destroys information) or retraining with specialized sparse attention patterns, **Token Merging (ToMe)** reduces token count dynamically within pretrained vision and language transformers without retraining.

```
Initial Tokens: [ x_0,  x_1,  x_2,  x_3,  x_4,  x_5,  x_6,  x_7 ]
                 │     │     │     │     │     │     │     │
Partition:      Set A (Even): {x_0, x_2, x_4, x_6}   Set B (Odd): {x_1, x_3, x_5, x_7}
                        │                                    │
Bipartite Sim:          └──────── Cosine Sim Matrix ─────────┘
                                         │
Top-r Merge:            Select top r pairs (e.g. x_2 ──► x_3, x_6 ──► x_7)
                                         │
Weighted Average:       x_3' = (s_3 x_3 + s_2 x_2) / (s_3 + s_2)
                        x_7' = (s_7 x_7 + s_6 x_6) / (s_7 + s_6)
                                         │
Output Tokens:          [ x_1, x_3', x_5, x_7', x_0, x_4 ] (Length reduced by r)
```

### Mathematical Formulation
1. **Bipartite Split**: Partition sequence of length $N$ into sets $A$ and $B$ of size $\approx N/2$.
2. **Cosine Similarity**: For each token $a_i \in A$ and $b_j \in B$:
   $$S_{ij} = \frac{a_i \cdot b_j}{\|a_i\|_2 \|b_j\|_2}$$
3. **Best Match**: Identify best candidate in $B$ for each $a_i$: $\text{best}(i) = \arg\max_j S_{ij}$.
4. **Top-$r$ Edge Selection**: Select the $r$ indices in $A$ with largest cosine similarities to their respective best matches in $B$.
5. **Size-Weighted Fusion**: Merge selected $a_i$ into target $b_{\text{best}(i)}$:
   $$b' = \frac{s_b b + s_a a_i}{s_b + s_a}, \quad s_b' = s_b + s_a$$
   where $s$ denotes the token size tracking vector (initialized to all ones).

---

## 4. FrugalGPT & Multi-Tier Model Cascades

Commercial foundation model deployments frequently process heterogeneous queries ranging from trivial extraction to multi-step logical deduction. Sending all requests to frontier models (e.g. GPT-4o, Claude 3.5 Sonnet) incurs astronomical operational expenses. **FrugalGPT (Chen et al., 2023)** orchestrates a dynamic model cascade:

### Cascade Pipeline
Given ordered model tiers $\mathcal{M} = [M_1, M_2, \dots, M_K]$ with monotonically increasing capability and invocation cost $C_1 < C_2 < \dots < C_K$:
1. Query is first evaluated by lightweight model $M_1$.
2. Confidence score $c_1$ (e.g. generation perplexity, consistency, or auxiliary scorer) is assessed against threshold $\tau_1$.
3. If $c_1 \ge \tau_1$, accept response and exit; cost incurred is $C_1$.
4. Otherwise, escalate to $M_2$, incurring cumulative cost $C_1 + C_2$.
5. Repeat until acceptance or final tier $M_K$.

### Expected Query Cost
$$\mathbb{E}[C] = C_1 + (1 - \pi_1) C_2 + (1 - \pi_1)(1 - \pi_2) C_3 + \dots$$
where $\pi_k = P(c_k \ge \tau_k \mid \text{escalated to } M_k)$. In practice, $70\%-85\%$ of typical user traffic resolves at tier 1 or 2, reducing aggregate cost by up to $90\%$ while matching top-tier accuracy.

---

## 5. Quality-Cost-Latency Optimization Trade-offs

Adaptive compute operates on a continuous Pareto frontier governed by user-defined threshold hyperparameters ($\tau_{\text{exit}}, r_{\text{merge}}, \tau_{\text{cascade}}$):

```
Accuracy (%)
  │                     ● Full Dense Baseline (100% cost, 1.0x speed)
  │                 ● FrugalGPT Cascade (20% cost, 98.5% accuracy)
  │             ● Early Exit (60% latency, 97.8% accuracy)
  │         ● ToMe (50% FLOPs, 96.5% accuracy)
  │     ● Aggressive Early Exit (30% latency, 89.2% accuracy)
  │
  └────────────────────────────────────────────────────────── Latency / Cost
```

Key trade-off dimensions:
- **Batching Irregularity**: Early exit introduces non-uniform sequence termination within a batch. Solutions include dynamic sample compaction or asynchronous stream execution.
- **Attention KV Re-indexing**: In ToMe, merging tokens shifts sequence positions; attention masks must properly handle merged token sizes $s$.
- **Router Calibration**: Cascade thresholds must be calibrated on validation splits using isotonic regression to ensure well-calibrated confidence estimates.

---

## 6. Multi-Framework Parity & Verification (PyTorch, JAX, Candle)

All three adaptive compute mechanisms maintain exact numerical equivalence across frameworks:

```rust
// Candle ToMe Bipartite Merge
let (x_merged, merged_sizes) = topic::tome_bipartite_merge(&w["x_tokens"], 2, &w["init_sizes"])?;
common::assert_max_diff(&x_merged, &w["x_merged"], 1e-5)?;

// Candle FrugalGPT Cascade
let (routed_idx, total_cost) = topic::frugal_cascade_route(&w["confidences"], &w["costs"], &w["thresholds"])?;
common::assert_max_diff(&routed_idx, &w["routed_idx"], 1e-5)?;
```

Every numerical tensor (confidences, size-weighted token embeddings, routing selections, cumulative costs) matches reference values to $< 10^{-5}$ maximum absolute deviation.

---

## 7. GPU/WebGPU Kernel Optimizations

Computing bipartite cosine similarities between sets $A$ and $B$ is accelerated using a parallel 2D compute shader in WGSL:

```wgsl
@compute @workgroup_size(16, 16, 1)
fn bipartite_sim_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; // index in A
    let j = gid.y; // index in B
    let b = gid.z; // batch index
    if (b >= u.batch || i >= u.len_a || j >= u.len_b) { return; }

    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    for (var d = 0u; d < u.dim; d++) {
        let va = a_tokens[(b * u.len_a + i) * u.dim + d];
        let vb = b_tokens[(b * u.len_b + j) * u.dim + d];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    sim_matrix[(b * u.len_a + i) * u.len_b + j] = dot / (max(sqrt(norm_a), 1e-8) * max(sqrt(norm_b), 1e-8));
}
```

This 2D tile layout achieves peak memory bandwidth by coalescing vector dot-product reductions in GPU registers.

---

## 8. Best Practices, Pitfalls, and Production Deployment

1. **Avoid Hard Deletions in Attention**: When tokens are merged via ToMe, weight their self-attention contributions by token size $s$ ($QK^\top / \sqrt{d} + \log s$) to preserve total attention mass.
2. **Prevent Cascade Oscillation**: Implement strict timeout limits on tier 1 and tier 2 models; if an early model times out or encounters network partition, fail-forward immediately to the higher tier.
3. **Calibrate Exit Thresholds**: Static threshold values drift across task distributions. Regularly re-evaluate early exit $\tau$ under domain shift using conformal prediction to guarantee bounded error rates.
4. **Profile Paged KV Footprint**: Pairing ToMe with paged KV caching reduces memory footprint by $r \times d_{\text{head}} \times 2$ bytes per merged token per layer.
