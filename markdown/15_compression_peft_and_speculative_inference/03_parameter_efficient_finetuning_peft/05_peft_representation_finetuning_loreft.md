# Representation Fine-Tuning (ReFT) and Low-Rank Linear Subspace Interventions (LoReFT)

Traditional Parameter-Efficient Fine-Tuning (PEFT) methods—such as LoRA, DoRA, and OFT—modify model weights ($W = W_0 + \Delta W$). However, weights are merely indirect causal mechanisms that produce representations. Inspired by mechanistic interpretability and causal abstraction, **Representation Fine-Tuning (ReFT)** and specifically **Low-Rank Linear Subspace ReFT (LoReFT)** intervene directly on hidden representations ($h \mapsto \Phi(h)$) during the forward pass. By operating on compact orthogonal subspaces, LoReFT achieves state-of-the-art adaptation using up to **$50\times$ fewer parameters than LoRA**.

---

## 1. Representation Fine-Tuning: Intervening on Hidden States

In neural language models, rich semantic concepts, task instructions, and factual knowledge are encoded linearly in distributed activation spaces (the *Linear Representation Hypothesis*). 

Rather than perturbing dense weight tensors across hundreds of matrix operations, ReFT intervenes directly on hidden activations at designated layers and token positions:

```
Token Embeddings ──► [ Layer 1 ] ──► h^(1) ──► ... ──► [ Layer L ] ──► h^(L) ────► Final Logits
                                                          │
                                                          ▼
                                                  [ LoReFT Intervention ]
                                                          │
                                                          ▼
                                                      h_adapted^(L)
```

| Parameterization Dimension | Standard LoRA | LoReFT (Wu et al., 2024) |
| :--- | :--- | :--- |
| **Intervention Target** | Weights ($W_0 + \frac{\alpha}{r} B A$) | Hidden states ($h \in \mathbb{R}^d$) |
| **Number of Tuned Matrices** | At least 2 per projection ($B, A$) | Subspace projector $R$, map $W$, bias $b$ |
| **Token Selectivity** | Uniform across all tokens in sequence | Highly selective (e.g., last token or prompt suffix) |
| **Trainable Parameter Ratio** | $\approx 0.1\%$ | **$\approx 0.002\%$ ($10\times - 50\times$ smaller)** |
| **Inference Overhead** | Absorbed via weight merging | Minimal vector projection $\mathcal{O}(r \cdot d)$ |

---

## 2. LoReFT: Low-Rank Linear Subspace ReFT

Let $h \in \mathbb{R}^d$ be the hidden activation vector at a specific layer and token position. **LoReFT (Wu et al., Stanford 2024)** defines a parameterized linear subspace intervention $\Phi(h)$:

$$\Phi(h) = h + R^\top \left( W h + b - R h \right)$$

where:
- $R \in \mathbb{R}^{r \times d}$ is an orthonormal matrix ($R R^\top = I_r$) defining an $r$-dimensional linear subspace.
- $W \in \mathbb{R}^{r \times d}$ is a learned linear transformation mapping full hidden states into the subspace.
- $b \in \mathbb{R}^r$ is a learned subspace bias/offset vector.
- $r \ll d$ is the intervention rank (typically $r \in \{1, 2, 4\}$).

### Intuitive Mechanics:
1. $R h \in \mathbb{R}^r$ projects the existing representation $h$ onto the low-rank subspace.
2. $W h + b \in \mathbb{R}^r$ computes the desired task-steered state in that subspace.
3. $\Delta s = (W h + b) - R h \in \mathbb{R}^r$ measures the necessary subspace shift.
4. $R^\top \Delta s \in \mathbb{R}^d$ embeds the subspace shift back into the original $d$-dimensional representation space.
5. If $W = R$ and $b = 0$, then $\Phi(h) = h + R^\top(0) \equiv h$, establishing an exact **identity intervention** at initialization.

---

## 3. Mathematical Derivations & Subspace Orthogonality

Let $P = R^\top R \in \mathbb{R}^{d \times d}$. Since $R R^\top = I_r$:
$$P^2 = (R^\top R)(R^\top R) = R^\top (R R^\top) R = R^\top I_r R = R^\top R = P$$
$$P^\top = (R^\top R)^\top = R^\top R = P$$
Thus, $P$ is an orthogonal projector onto the row space of $R$, and $I - P$ is the projector onto its orthogonal complement.

Rewriting the LoReFT intervention:
$$\Phi(h) = h - R^\top R h + R^\top (W h + b) = (I - R^\top R) h + R^\top (W h + b)$$

This demonstrates that LoReFT **perfectly leaves the orthogonal complement $(I - P)h$ untouched**, altering only the tiny $r$-dimensional subspace spanned by $R$!

---

## 4. Position-Selective Interventions

Unlike weight-based adapters that apply identically to every token in the context window, representations vary dynamically across token positions. LoReFT exploits this by selectively intervening only on positions where task information is processed:

1. **Prefix Intervention:** Intervene on the first $k$ tokens to steer prompt semantics.
2. **Suffix / Last-Token Intervention:** Intervene on the final prompt token or generated token positions, where reasoning representations concentrate before emission:
   $$h_{\text{out}}[b, s] = h[b, s] + \text{mask}[b, s] \cdot \Delta h[b, s]$$
where $\text{mask}[b, s] \in \{0, 1\}$.

This position selectivity allows LoReFT to drastically reduce computation while maintaining high steerability.

---

## 5. Efficiency Comparison: LoReFT vs LoRA, DoRA, and Full Tuning

On a LLaMA-3-8B benchmark ($d = 4096$, $L = 32$ layers):

| Metric | Full Fine-Tuning | LoRA ($r=8$) | DoRA ($r=8$) | LoReFT ($r=2$, 4 positions) |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters per Intervention** | $33.5 \times 10^6$ / layer | $65,536$ / layer | $69,632$ / layer | $2 \times (2 \cdot 4096) + 2 = 16,386$ total |
| **Total Model Checkpoint** | $16.0\text{ GB}$ | $18.4\text{ MB}$ | $19.2\text{ MB}$ | **$< 0.5\text{ MB}$** |
| **Parameter Reduction vs LoRA**| Baseline | $1.0\times$ | $-6\%$ | **$37.5\times$ smaller!** |
| **Mathematical Target** | Dense weights | Matrix decomposition | Direction + Norm | Orthogonal Subspace |
| **Selective Tokens?** | No | No | No | **Yes** |

---

## 6. Mechanistic Interpretability & Linear Representations

Why can a rank-1 or rank-2 subspace intervention steer complex behaviors in an 8-billion parameter model?
- **Linear Representation Hypothesis:** High-level concepts (e.g. sentiment, truthfulness, syntactic agreement, formality) are represented as 1-dimensional directions or low-dimensional linear subspaces in activation space.
- **Causal Abstraction:** By projecting activations onto these conceptual directions and steering them, LoReFT directly manipulates the decision boundary without causing collateral damage to unrelated pre-trained knowledge.

---

## 7. Hardware & Kernel Acceleration: Fused LoReFT Interventions

In production serving, LoReFT can be implemented as an in-place fused kernel right after transformer layer normalization or residual addition:

```wgsl
// WGSL Fused LoReFT In-Place Intervention
@compute @workgroup_size(16, 16, 1)
fn loreft_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    let col = gid.y;
    if (token_idx >= u.batch * u.seq_len || col >= u.dim) { return; }

    let base_idx = token_idx * u.dim + col;
    let m = mask[token_idx];
    if (m == 0.0) {
        h_out[base_idx] = h[base_idx];
        return;
    }

    var delta_out: f32 = 0.0;
    for (var r = 0u; r < u.rank; r++) {
        var proj_w: f32 = b_vec[r];
        var proj_r: f32 = 0.0;
        for (var k = 0u; k < u.dim; k++) {
            let h_val = h[token_idx * u.dim + k];
            proj_w += h_val * w_mat[r * u.dim + k];
            proj_r += h_val * r_mat[r * u.dim + k];
        }
        let delta_s = proj_w - proj_r;
        delta_out += delta_s * r_mat[r * u.dim + col];
    }
    h_out[base_idx] = h[base_idx] + m * delta_out;
}
```

---

## 8. Implementation Guidelines & Cross-Framework Parity

To maintain transitive numerical parity ($< 10^{-5}$) across PyTorch, JAX, and Candle:

1. **Pre-Computed Orthonormal Basis $R$:** Ensure $R \in \mathbb{R}^{r \times d}$ is strictly orthonormal ($R R^\top = I_r$) via QR decomposition. Exporting the pre-orthogonalized matrix into the safetensors checkpoint enables zero-overhead inference in lightweight Candle environments.
2. **Associative Projection Batching:** Evaluate $h W^\top + b$ and $h R^\top$ as 2D batched matrix multiplications, avoiding 3D-to-2D dimension mismatches.
3. **Seed-42 Golden Parity:** Validate all representations ($h, R, W, b, \text{mask}, \Delta h, h_{\text{out}}$) across PyTorch, JAX, and Candle.
