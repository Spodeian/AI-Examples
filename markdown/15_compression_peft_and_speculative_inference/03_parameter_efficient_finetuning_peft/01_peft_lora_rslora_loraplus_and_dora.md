# Parameter-Efficient Fine-Tuning: LoRA, rsLoRA, LoRA+, and DoRA

Full parameter fine-tuning of multi-billion parameter foundation models requires updating and storing complete optimizer states (e.g. 8 bytes per parameter in AdamW), quickly exhausting GPU memory during training and requiring hundreds of gigabytes per task checkpoint. **Parameter-Efficient Fine-Tuning (PEFT)** freezes the base foundation model and trains a tiny fraction ($<0.1\%$) of parameters. This topic covers the core evolution of low-rank adaptation: classic **LoRA**, rank-invariant scaling (**rsLoRA**), asymmetric feature learning rate dynamics (**LoRA+**), and magnitude-direction decoupled adaptation (**DoRA**).

---

## 1. Parameter-Efficient Fine-Tuning: The LoRA Paradigm

Aghajanyan et al. (2020) demonstrated that pre-trained language models reside on an intrinsically low-dimensional parameter manifold: adaptation to downstream tasks can be accomplished within a subspace of very small intrinsic rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$.

| Adapter Architecture | Parameter Update $\Delta W$ | Scaling Factor | Primary Advantage | Key Papers |
| :--- | :--- | :--- | :--- | :--- |
| **Standard LoRA** | $B A$ | $\alpha / r$ | Zero inference latency (foldable), $<0.1\%$ params | Hu et al. (ICLR 2022) |
| **rsLoRA** | $B A$ | $\alpha / \sqrt{r}$ | Preserves learning dynamics as rank $r$ scales | Kalajdzievski (2023) |
| **LoRA+** | $B A$ with $\eta_B \gg \eta_A$ | $\alpha / r$ | Accelerates feature representation learning | Hayou et al. (ICML 2024) |
| **DoRA** | $m \odot \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c} - W_0$ | Dynamic norm | Matches/exceeds full fine-tuning accuracy | Liu et al. (ICML 2024) |

---

## 2. Mathematical Formulation: Standard LoRA & Weight Updates

For a frozen base weight matrix $W_0 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, LoRA parameterizes the update as low-rank matrices $A \in \mathbb{R}^{d_{\text{in}} \times r}$ and $B \in \mathbb{R}^{r \times d_{\text{out}}}$ where $r \ll \min(d_{\text{in}}, d_{\text{out}})$:

$$h = x W_0 + \frac{\alpha}{r} (x A) B$$

### Initialization Protocol
- $A \sim \mathcal{N}(0, \sigma^2)$ (random Gaussian)
- $B = 0$ (exact zero initialization)
At step 0, $\Delta W = B A = 0$, guaranteeing that model output is strictly unmodified at the start of training.

### Zero-Latency Deployment (Weight Folding)
At inference time, the adapter can be permanently merged into the base weights:
$$W_{\text{deploy}} = W_0 + \frac{\alpha}{r} A B$$
yielding **zero additional latency, memory overhead, or kernel launches**.

---

## 3. Rank Stabilization: The rsLoRA Scaling Law

In standard LoRA, the scaling factor is $\gamma = \alpha / r$. Kalajdzievski (2023) proved that as rank $r$ increases to capture complex tasks, the expected gradient magnitude of $\Delta W$ collapses at rate $\mathcal{O}(1/r)$:

$$\|\nabla_{\Delta W} \mathcal{L}\|_F \propto \frac{\alpha}{r} \implies \lim_{r \to \infty} \|\nabla_{\Delta W} \mathcal{L}\|_F = 0$$

To maintain constant gradient norm independent of rank, **rsLoRA (Rank-Stabilized LoRA)** introduces square-root scaling:
$$\gamma_{\text{rsLoRA}} = \frac{\alpha}{\sqrt{r}}$$

$$\begin{aligned}
r = 4 &: \quad \text{LoRA scale: } \alpha/4, \quad \text{rsLoRA scale: } \alpha/2 \\
r = 64 &: \quad \text{LoRA scale: } \alpha/64, \quad \text{rsLoRA scale: } \alpha/8
\end{aligned}$$
rsLoRA unleashes the representational power of larger ranks ($r \ge 32$) without vanishing gradient updates.

---

## 4. Learning Rate Disparity & Asymmetric Optimization (LoRA+)

Hayou et al. (ICML 2024) analyzed the disparate functional roles of $A$ and $B$:
- Matrix $A$ maps input activations from high-dimensional space into the low-rank bottleneck (feature extraction).
- Matrix $B$ maps bottleneck representations to output dimensions (linear readout).

Under standard uniform learning rates ($\eta_A = \eta_B = \eta$), matrix $A$ updates too slowly relative to matrix $B$. **LoRA+** enforces asymmetric learning rates:
$$\eta_B = \lambda \cdot \eta_A, \quad \text{where } \lambda \in [8, 16]$$

This ratio accelerates feature alignment, yielding $1.5\times - 2\times$ faster convergence and $1\% - 2\%$ higher task benchmark accuracy with zero parameter or architectural modifications.

---

## 5. Weight-Decomposed Low-Rank Adaptation (DoRA)

Liu et al. (ICML 2024) uncovered a fundamental discrepancy between Full Fine-Tuning (FT) and LoRA:
- **Full Fine-Tuning** alters both parameter **magnitude** and parameter **direction** with negative correlation (small magnitude changes accompany large directional rotations).
- **Standard LoRA** couples magnitude and direction updates proportionally, limiting capacity.

**DoRA (Weight-Decomposed Low-Rank Adaptation)** explicitly decouples weights into magnitude vector $m \in \mathbb{R}^{d_{\text{out}}}$ and directional matrix $V \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:

$$W = m \odot \frac{V}{\|V\|_c}$$
where $\|V\|_c = \left[\|V_{:, 1}\|_2, \dots, \|V_{:, d_{\text{out}}}\|_2\right]$ is the column-wise $\ell_2$ norm.

### DoRA Forward Equation
Initial magnitude is set to base column norms $m = \|W_0\|_c$. The directional matrix is updated via low-rank LoRA:
$$V = W_0 + \frac{\alpha}{r} A B$$
$$W_{\text{dora}} = m \odot \frac{W_0 + \frac{\alpha}{r} A B}{\|W_0 + \frac{\alpha}{r} A B\|_c}$$

DoRA consistently matches or exceeds full fine-tuning performance across reasoning, vision-language, and instruction tasks.

---

## 6. Multi-Framework Parity & Verification (PyTorch, JAX, Candle)

All PEFT forward formulations are cross-verified against seed-42 golden references across PyTorch, JAX, and Candle:

```rust
// Candle DoRA Forward Verification
let d_out = topic::dora_forward(&w["x"], &w["w0"], &w["a"], &w["b"], &w["m"], 4.0, 2)?;
common::assert_max_diff(&d_out, &w["dora_out"], 1e-5)?;

// Candle rsLoRA Forward Verification
let rs_out = topic::rslora_forward(&w["x"], &w["w0"], &w["a"], &w["b"], 4.0, 2)?;
common::assert_max_diff(&rs_out, &w["rslora_out"], 1e-5)?;
```

Maximum absolute error across all tracks is strictly $< 10^{-5}$.

---

## 7. GPU/WebGPU Kernel Optimizations

The low-rank forward pass $h = x W_0 + \gamma (x A) B$ can be fused into a single compute pass in WebGPU WGSL:

```wgsl
@compute @workgroup_size(16, 16, 1)
fn lora_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
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
            x_a += x[row * u.d_in + i] * a[i * u.rank + r];
        }
        adapter_val += x_a * b[r * u.d_out + col];
    }
    y[row * u.d_out + col] = base_val + u.scale * adapter_val;
}
```

This avoids writing intermediate tensor $(x A)$ of shape $[B, r]$ to HBM.

---

## 8. Best Practices, Rank Selection, and Multi-Adapter Serving

1. **Target Modules**: Applying LoRA to all linear layers ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) at low rank ($r=8$) consistently outperforms targeting attention only at high rank ($r=64$).
2. **rsLoRA for High Rank**: If task complexity demands $r \ge 32$ (e.g. math or code synthesis), switch scaling from $\alpha/r$ to $\alpha/\sqrt{r}$.
3. **Multi-Tenant Serving (S-LoRA / Punica)**: Keep base weights frozen in GPU memory; dynamically route batched requests to distinct adapter weights using segmented gather-scatter GEMM.
4. **DoRA for Precision Tasks**: For tasks demanding fine-grained numerical calibration (e.g. instruction following, medical diagnosis), prefer DoRA over standard LoRA.
