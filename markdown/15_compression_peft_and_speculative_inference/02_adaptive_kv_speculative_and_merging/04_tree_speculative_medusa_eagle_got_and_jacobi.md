# Tree Speculative Decoding: Medusa, EAGLE, Graph-of-Thoughts, and Jacobi Lookahead

Linear speculative decoding evaluates a single sequence of draft candidates. If the target model rejects the second token, all subsequent tokens are immediately discarded, bottlenecking average acceptance length. **Tree-based speculative decoding (Medusa, EAGLE)** escapes this linear trap by drafting a branching tree of candidates verified simultaneously in a single parallel target forward pass using specialized 2D tree attention masks. Complementary paradigms like **Jacobi fixed-point iteration** and **Graph-of-Thoughts (GoT)** reframe generation as parallel state relaxation and non-linear graph traversal, boosting decoding acceleration to $2.5\times - 4\times$.

---

## 1. Beyond Sequential Drafting: Tree-Based Speculation & Parallel Fixed Points

In linear drafting, predicting $\gamma$ tokens yields only one chance per position. In tree-based speculation, branching paths exponentially expand the probability of finding an accepted continuation:

| Paradigm | Candidate Structure | Auxiliary Model Required | Verification Mechanism | Acceptance Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Speculative** | 1D Chain $[x_1 \dots x_\gamma]$ | Separate small LLM | Standard causal attention | Moderate ($\beta \sim 65\%$) |
| **Medusa (Cai et al.)** | Tree of $K$ heads | Zero (auxiliary FFN heads) | 2D Tree causal mask | High ($\beta \sim 80\%$) |
| **EAGLE / EAGLE-2** | Feature-extrapolated tree | Lightweight transformer layer | Tree mask on top-layer features | State-of-the-art ($\beta > 85\%$) |
| **Jacobi / Lookahead** | Parallel $n$-token grid | Zero (fixed-point iterations) | Nonlinear Jacobi relaxation | High for low entropy text |
| **Graph-of-Thoughts** | Directed Acyclic Graph (DAG) | Reasoning controller | Graph verification & voting | Peak for complex reasoning |

---

## 2. Medusa Architecture: Auxiliary Multi-Head Drafting Without Secondary Models

Standard speculative decoding requires hosting and scheduling a separate draft model in VRAM, causing memory bus contention and deployment complexity. **Medusa (Cai et al., 2024)** augments the frozen base target model with $K$ lightweight parameter-efficient heads $\{H_1, H_2, \dots, H_K\}$ trained on top of the final transformer hidden state $h_t$:

$$z_t^{(k)} = h_t W_{\text{medusa}}^{(k)} + b_{\text{medusa}}^{(k)}, \quad k \in \{1, \dots, K\}$$

Head $k$ directly predicts the token distribution at offset $t + k$:
$$\tilde{x}_{t+k} \sim \text{softmax}(z_t^{(k)})$$

Top-$s_k$ predictions from each head are combined into a Cartesian tree of candidate sequences (e.g. $s_1=2, s_2=2, s_3=1 \implies 4$ candidate trajectories) with zero secondary model overhead.

---

## 3. EAGLE & EAGLE-2: Context-Aware Feature Extrapolation

While Medusa uses only the final hidden state, **EAGLE (Li et al., 2024)** discovers that drafting conditioned on token embeddings alone suffers from semantic blindness. EAGLE feeds the target model's penultimate hidden states into a single lightweight transformer decoder layer:

```
Target Model Layer L-1 ──► [ Feature Extrapolator ] ──► Extrapolated Feature Vectors
                                     ▲
Base Token Embeddings  ──────────────┘
                                     │
                             [ Prediction Head ]
                                     │
                           Branching Candidate Tree
```

EAGLE-2 dynamically adjusts tree topology (number of branches per node) conditioned on draft confidence, achieving $>85\%$ token acceptance with negligible compute overhead.

---

## 4. Tree Attention Masks & Parallel Candidate Verification

To verify an arbitrary candidate tree in a single target model forward pass, modern serving engines flatten tree nodes into a single sequence and enforce tree ancestry via a custom 2D causal attention mask $M \in \{0, -\infty\}^{N_{\text{nodes}} \times N_{\text{nodes}}}$:

```
Tree Topology:
       Node 0 (Root / Current Token)
        ├── Node 1 (Head 1, Top 1) ──► Node 3 (Head 2, Top 1)
        └── Node 2 (Head 1, Top 2) ──► Node 4 (Head 2, Top 2)
```

### Mask Construction Rule
For query node $i$ and key node $j$:
$$M_{ij} = \begin{cases} 0.0 & \text{if } j \in \text{Ancestors}(i) \lor j = i \\ -\infty & \text{otherwise} \end{cases}$$

Under this mask, node 3 attends exclusively to $\{0, 1, 3\}$, and node 4 attends to $\{0, 2, 4\}$. Cross-branch attention is strictly forbidden ($M_{3, 2} = -\infty$). The target model evaluates all branches simultaneously in $\mathcal{O}(1)$ target latency.

---

## 5. Jacobi / Lookahead Decoding & Graph-of-Thoughts (GoT)

### Jacobi Fixed-Point Iteration (Lookahead Decoding)
Instead of autoregressive recurrence $x_{t} = f(x_{<t})$, Jacobi iteration initializes future $n$ token guesses $X^{(0)} = [x_1^{(0)}, \dots, x_n^{(0)}]$ and updates all positions simultaneously:
$$X^{(k+1)} = \sigma(X^{(k)} W + B)$$

Until convergence:
$$\|X^{(k+1)} - X^{(k)}\|_\infty < \epsilon$$

In low-entropy repetitive or structured code generation, Jacobi iteration converges in 2–3 parallel steps for 6–10 tokens.

### Graph-of-Thoughts (GoT)
GoT extends tree speculation to Directed Acyclic Graphs (DAGs), enabling operations impossible in linear streams: branching thoughts, aggregating multiple reasoning paths via graph union, and evaluating feedback loops with reflexive verification.

---

## 6. Multi-Framework Parity & Verification (PyTorch, JAX, Candle)

All tree speculation components (Medusa multi-heads, 2D tree causal mask generation, longest accepted path extraction, and Jacobi fixed-point step) are verified across PyTorch, JAX, and Candle:

```rust
// Candle Tree Attention Mask
let t_mask = topic::build_tree_attention_mask(&w["parent_indices"])?;
common::assert_max_diff(&t_mask, &w["tree_mask"], 1e-5)?;

// Candle Longest Path Selection
let (p_mask, p_len) = topic::tree_longest_accepted_path(
    &w["candidate_tokens"], &w["target_tokens"], &w["parent_indices"]
)?;
common::assert_max_diff(&p_mask, &w["path_mask"], 1e-5)?;
```

Transitive numerical parity matches seed-42 golden references to $< 10^{-5}$.

---

## 7. Hardware Kernel Acceleration: 2D Tree Causal Masks in WGSL

Parallel generation of 2D tree causal attention masks is accelerated using a WebGPU compute shader:

```wgsl
@compute @workgroup_size(16, 16, 1)
fn tree_mask_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= u.num_nodes || j >= u.num_nodes) { return; }

    var is_ancestor = false;
    var curr = i;
    for (var step = 0u; step < u.num_nodes; step++) {
        if (curr == j) { is_ancestor = true; break; }
        if (curr == 0u) { break; }
        curr = u32(parent_indices[curr]);
    }

    tree_mask[i * u.num_nodes + j] = select(-1e9, 0.0, is_ancestor);
}
```

---

## 8. Production Engineering: Branch Pruning, Memory Contiguity, and Dynamic Tree Topologies

1. **Optimal Tree Topologies**: Flattening wide trees ($N > 64$ nodes) degrades parallel self-attention efficiency. Empirically, a depth-4 tree with 16–32 total nodes maximizes expected accepted length while maintaining $T_{\text{verify}} \approx T_{\text{single}}$.
2. **KV Cache Slot Recycling**: Tree nodes rejected by the target model must be rolled back immediately. Maintain a tree-to-page table mapping to avoid freeing active prompt blocks.
3. **Draft Head Distillation**: Medusa heads should be trained using combined hard cross-entropy and soft teacher KL divergence to ensure head agreement.
4. **Nonlinear Systems**: Pair Jacobi lookahead decoding with greedy n-gram draft initialization to slash the number of relaxation iterations to convergence.
