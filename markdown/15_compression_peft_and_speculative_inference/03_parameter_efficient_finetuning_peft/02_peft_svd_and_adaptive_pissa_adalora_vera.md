# PEFT with SVD and Adaptive Allocation: PiSSA, AdaLoRA, and VeRA

Standard LoRA initializes the adapter with a random Gaussian matrix $A \sim \mathcal{N}(0, \sigma^2)$ and a zero matrix $B = 0$. Consequently, fine-tuning begins by exploring arbitrary noise directions while the model's true dominant representational directions remain frozen in $W_0$. Furthermore, assigning a static uniform rank across all layers wastes parameter capacity on saturated layers while starving bottleneck layers. This topic explores advanced SVD-grounded adaptation: **PiSSA** initializes adapters directly with the principal singular directions of the base weights, **AdaLoRA** dynamically allocates parameter budgets across layers via singular value pruning, and **VeRA** shares frozen random projections across all layers to reduce trainable parameters to tiny scaling vectors.

---

## 1. SVD-Grounded Parameter Adaptation: Beyond Random Initialization

The Eckart-Young-Mirsky theorem establishes that the truncated Singular Value Decomposition (SVD) of rank $r$ is the optimal low-rank Frobenius approximation of a matrix. SVD-based adaptation techniques leverage this spectral property:

| PEFT Strategy | Initialization | Parameter Allocation | Trainable Parameter Count | Key Papers |
| :--- | :--- | :--- | :--- | :--- |
| **Standard LoRA** | Random Gaussian $A$, $B = 0$ | Static uniform rank $r$ across layers | $2 \times L \times d \times r$ | Hu et al. (ICLR 2022) |
| **PiSSA** | SVD Principal components of $W_0$ | Principal subspace tuning, residual frozen | $2 \times L \times d \times r$ | Meng et al. (NeurIPS 2024) |
| **AdaLoRA** | SVD parameterization $P \Lambda Q$ | Dynamic budget allocation via importance | $\le 2 \times L \times d \times \bar{r}$ | Zhang et al. (ICLR 2023) |
| **VeRA** | Frozen shared random matrices $A, B$ | Vector scaling $d, b$ per layer only | $L \times (r + d)$ ($>99\%$ reduction!) | Kopiczko et al. (ICLR 2024) |

---

## 2. PiSSA (Principal Singular values and Singular vectors Adaptation)

In standard LoRA, the adapter update $\Delta W = B A$ must build semantic representation capacity from scratch. **PiSSA (Meng et al., 2024)** decomposes base weight $W_0 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ using truncated SVD:

$$W_0 = U \Sigma V^\top = \underbrace{\sum_{i=1}^r \sigma_i u_i v_i^\top}_{\text{Principal Components (Top } r)} + \underbrace{\sum_{i=r+1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_i u_i v_i^\top}_{\text{Residual Matrix } W_{\text{res}}}$$

### Initialization Rule
$$B_{\text{init}} = U_{:r} \sqrt{\Sigma_{:r}} \in \mathbb{R}^{d_{\text{in}} \times r}, \quad A_{\text{init}} = \sqrt{\Sigma_{:r}} V_{:r}^\top \in \mathbb{R}^{r \times d_{\text{out}}}$$
$$W_{\text{res}} = W_0 - B_{\text{init}} A_{\text{init}}$$

### Optimization Dynamics
During training, $W_{\text{res}}$ is frozen while $A$ and $B$ are fine-tuned:
$$h = x W_{\text{res}} + (x B) A$$
At step 0, $W_{\text{res}} + B A \equiv W_0$ exactly. Because optimization directly perturbs the dominant singular directions responsible for core capabilities, PiSSA converges up to $4\times$ faster and achieves higher downstream accuracy than LoRA.

---

## 3. AdaLoRA: Dynamic Budget Allocation and SVD Pruning

Standard LoRA applies an identical rank (e.g. $r=8$) to every transformer block, ignoring layer sensitivity variations (e.g. middle layers undergo greater representational changes than initial token embeddings).

**AdaLoRA (Zhang et al., 2023)** parameterizes the incremental update via explicit SVD decomposition:
$$\Delta W = P \Lambda Q$$
where $P \in \mathbb{R}^{d_{\text{in}} \times r}$ and $Q \in \mathbb{R}^{r \times d_{\text{out}}}$ maintain orthogonality penalties, and $\Lambda \in \mathbb{R}^{r \times r}$ is a diagonal matrix containing singular values $\lambda_i$.

### Importance-Based Pruning
AdaLoRA scores the importance of the $i$-th singular triplet $(P_{:, i}, \lambda_i, Q_{i, :})$ using magnitude and gradient magnitude:
$$S_i = |\lambda_i \cdot \nabla_{\lambda_i} \mathcal{L}|$$

Throughout training, triplets with low $S_i$ are zeroed out via binary masks. Surviving triplets are dynamically redistributed, allocating higher effective rank to critical attention and MLP layers while pruning redundant capacity from shallow layers.

---

## 4. VeRA: Frozen Shared Random Projections & Vector-Only Adaptation

When deploying hundreds of fine-tuned adapter checkpoints (e.g. per-user or per-task customization in SaaS environments), storing $2 \times d \times r$ parameters per layer across hundreds of models consumes massive HBM.

**VeRA (Vector-based Random Matrix Adaptation, Kopiczko et al., 2024)** freezes a single pair of random matrices $A_{\text{shared}} \in \mathbb{R}^{d_{\text{in}} \times r}$ and $B_{\text{shared}} \in \mathbb{R}^{r \times d_{\text{out}}}$ across the **entire network**:

```
Layer l Input ──► [ Frozen W0 ] ─────────────────────────► Base Output
       │                                                         ▲
       └────────► [ Frozen A_shared ] ──► ( · d_l )              │
                                                │                │
                                      [ Frozen B_shared ] ──► ( · b_l )
```

### Forward Formulation
$$\Delta W_l = \text{diag}(b_l) B_{\text{shared}} \text{diag}(d_l) A_{\text{shared}}$$
The only trainable parameters per layer are:
- $d_l \in \mathbb{R}^r$ (rank scaling vector)
- $b_l \in \mathbb{R}^{d_{\text{out}}}$ (feature scaling vector)

Total trainable parameters shrink from $2 \times L \times d \times r$ (LoRA) to $L \times (r + d)$ (VeRA), slashing adapter checkpoint storage by $>95\%$ (e.g. a 7B model adapter shrinks from $20\text{ MB}$ to $<1\text{ MB}$).

---

## 5. Comparison Matrix: Storage, Computation, and Gradient Mechanics

| Metric | Full Fine-Tuning | LoRA | PiSSA | AdaLoRA | VeRA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trainable Params (7B, $r=8$)** | $7 \times 10^9$ ($100\%$) | $\sim 2 \times 10^7$ ($0.28\%$) | $\sim 2 \times 10^7$ ($0.28\%$) | $\le 2 \times 10^7$ ($0.15\%$) | $\sim 1.5 \times 10^5$ ($0.002\%$) |
| **Checkpoint Size** | $14\text{ GB}$ | $40\text{ MB}$ | $40\text{ MB}$ | $\sim 25\text{ MB}$ | **$< 1\text{ MB}$** |
| **Initialization Cost** | $0$ | $0$ | One-time SVD | SVD init + orthogonality | $0$ (seed generated) |
| **Convergence Speed** | Baseline ($1\times$) | $1.2\times$ | **$2\times - 4\times$** | $1.5\times$ | $1.0\times$ |
| **Inference Latency** | Baseline ($1.0\times$) | $1.0\times$ (folded) | $1.0\times$ (folded) | $1.0\times$ (folded) | $1.0\times$ (folded) |

---

## 6. Multi-Framework Parity & Verification (PyTorch, JAX, Candle)

All three SVD/adaptive PEFT implementations achieve transitive numerical equivalence across PyTorch, JAX, and Candle:

```rust
// Candle PiSSA forward pass verification
let p_out = topic::pissa_forward(&w["x"], &w["w_res"], &w["b_pissa"], &w["a_pissa"])?;
common::assert_max_diff(&p_out, &w["pissa_out"], 1e-5)?;

// Candle VeRA vector-scaled forward pass
let v_out = topic::vera_forward(&w["x"], &w["w0"], &w["a_shared"], &w["b_shared"], &w["d_vec"], &w["b_vec"])?;
common::assert_max_diff(&v_out, &w["vera_out"], 1e-5)?;
```

Maximum absolute error across all tracks is bounded below $10^{-5}$.

---

## 7. GPU/WebGPU Kernel Optimizations

The VeRA forward pass $h = x W_0 + ((x A_{\text{shared}} \odot d) B_{\text{shared}}) \odot b$ is accelerated in WebGPU WGSL by keeping the frozen shared matrices $A$ and $B$ in read-only storage while vector scaling $d$ and $b$ execute in thread registers:

```wgsl
@compute @workgroup_size(16, 16, 1)
fn vera_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= u.batch || col >= u.d_out) { return; }

    var base_val: f32 = 0.0;
    for (var i = 0u; i < u.d_in; i++) {
        base_val += x[row * u.d_in + i] * w0[i * u.d_out + col];
    }

    var adapter_val: f32 = 0.0;
    for (var r = 0u; r < u.rank; r++) {
        var x_a: f32 = 0.0;
        for (var i = 0u; i < u.d_in; i++) {
            x_a += x[row * u.d_in + i] * a_shared[i * u.rank + r];
        }
        adapter_val += (x_a * d_vec[r]) * b_shared[r * u.d_out + col];
    }
    y[row * u.d_out + col] = base_val + adapter_val * b_vec[col];
}
```

---

## 8. Best Practices, SVD Truncation Errors, and Multi-Task Memory Footprints

1. **PiSSA SVD Computation**: Truncated SVD should be performed in double precision (`float64`) on CPU or GPU before quantizing weights to FP16 to avoid numerical precision loss on small singular values.
2. **AdaLoRA Orthogonality Penalty**: Enforce orthogonality loss $\mathcal{L}_{\text{orth}} = \|P^\top P - I\|_F^2 + \|Q Q^\top - I\|_F^2$ during training to ensure $\lambda_i$ accurately reflects true singular value magnitude.
3. **VeRA Shared Seed**: In multi-task serving, store only the pseudorandom generator seed for $A_{\text{shared}}$ and $B_{\text{shared}}$. Worker nodes regenerate the shared matrices deterministically on startup, transmitting only $(d_l, b_l)$ across the network.
4. **Adapter Folding**: For all three paradigms, the final weight matrix $W_{\text{deploy}} = W_{\text{base}} + \Delta W$ can be pre-folded before serving, incurring zero inference cost.
