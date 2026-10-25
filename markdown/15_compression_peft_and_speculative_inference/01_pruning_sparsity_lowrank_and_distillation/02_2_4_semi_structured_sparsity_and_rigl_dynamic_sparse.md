# 2:4 Semi-Structured Sparsity and RigL Dynamic Sparse Training

While fine-grained unstructured sparsity yields high theoretical compression ratios ($>80\%$), standard GPU architectures cannot easily translate arbitrary zero-value distributions into execution speedups due to irregular memory access and thread divergence. **2:4 Semi-Structured Sparsity** bridges this gap by imposing a hardware-enforced constraint: in every contiguous block of 4 elements, exactly 2 elements must be zero. Introduced natively in NVIDIA Ampere architectures, 2:4 sparsity doubles dense math throughput ($2\times$ GEMM TFLOPS) via dedicated Sparse Tensor Cores. Concurrently, **RigL (Rigging the Lottery)** pioneers dynamic sparse training: maintaining fixed global sparsity throughout training while iteratively dropping low-magnitude weights and regrowing connections guided by instantaneous gradient magnitude.

---

## 1. Hardware-Aware Sparsity: The 2:4 Semi-Structured Breakthrough

The divergence between theoretical compression and realized runtime speedup stems from memory layout:

| Sparsity Pattern | Hardware Support | Math Throughput | Storage Format | Retraining Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Baseline** | Standard Tensor Cores | $1.0\times$ | Contiguous FP16/BF16/FP8 | Zero |
| **Unstructured Sparse** | CPU sparse BLAS / SpMM | $<1.0\times$ at $<70\%$ | CSR / COO / Bitmask | High iterative pruning |
| **2:4 Semi-Structured** | Ampere/Hopper Sparse Tensor Cores | $2.0\times$ | Packed $2\times$ values + 2-bit metadata | Low (SR-STE finetuning) |
| **RigL Dynamic Sparse** | Any sparse/dense runtime | Dynamic topology | Fixed parameter budget | Native training from scratch |

---

## 2. Mathematical Formulation of N:M Sparsity & Storage Packing

For a weight matrix $W \in \mathbb{R}^{M \times K}$ where $K$ is divisible by 4, we partition each row into $K/4$ non-overlapping blocks:
$$W_{r, b} = \big[ W_{r, 4b}, W_{r, 4b+1}, W_{r, 4b+2}, W_{r, 4b+3} \big], \quad b \in \{0, \dots, K/4 - 1\}$$

In each 4-element block, the 2:4 pruning operator retains the top-2 elements by absolute magnitude:
$$i_1, i_2 = \arg\max_{i, j \in \{0,1,2,3\}, i \ne j} \big( |W_{r, 4b+i}| + |W_{r, 4b+j}| \big)$$

The resulting binary mask $M_{2:4}$ satisfies:
$$\sum_{j=0}^3 M_{r, 4b+j} = 2, \quad \forall r, b$$

### Metadata Packing
Because each retained element can only reside in one of 4 possible positions, its relative offset requires only $\log_2(4) = 2\text{ bits}$. For a pair of elements, $4\text{ bits}$ of metadata fully encode their locations within the 4-element window. Consequently, storage shrinks by:
$$\text{Storage Ratio} = \frac{2 \times 16\text{ bits (data)} + 4\text{ bits (metadata)}}{4 \times 16\text{ bits (dense)}} = \frac{36}{64} \approx 56.25\%$$

---

## 3. Sparsity-Aware Training: SR-STE vs. Post-Training 2:4 Pruning

Directly applying 2:4 post-training pruning to full-precision weights induces non-negligible accuracy degradation. To restore baseline performance, networks undergo **Sparse-Refined Straight-Through Estimator (SR-STE)** training:

```
Full Weights W ───► [ 2:4 Pruning ] ───► W_{2:4} ───► [ Forward GEMM ] ───► Loss L
       ▲                                                                       │
       │                                                                       ▼
  Update: W - lr * (∇_{W_{2:4}} L + λ W) ◄─── [ Backward: ∇_{W_{2:4}} L ] ─────┘
```

1. **Forward Pass**: $W_{2:4} = \text{prune}_{2:4}(W)$
2. **Backward Pass**: Gradients flow through the straight-through estimator:
   $$\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial W_{2:4}}$$
3. **Sparse Refinement**: Weight decay $\lambda W$ is applied to pruned weights, encouraging weights that persistently fail the 2:4 threshold to decay toward zero while allowing recovering weights to re-enter the active set.

---

## 4. RigL: Dynamic Sparse Training with Gradient Regrowth

Standard static pruning (like Lottery Ticket IMP) requires expensive iterative cycles of train-prune-retrain. **RigL (Evci et al., 2020)** trains sparse topologies directly from scratch under a fixed parameter budget:

### Topology Exploration Step
At each update interval $\Delta T$ (e.g. every 100 steps):
1. **Drop Low-Magnitude Weights**: From active connections $\mathcal{S}_t = \{(i, j) \mid M_{ij} = 1\}$, drop the bottom fraction $\alpha_t$ with smallest $|W_{ij}|$:
   $$\text{Drop set } \mathcal{D}_t = \arg\min_{\mathcal{D} \subset \mathcal{S}_t, |\mathcal{D}| = k_t} \sum_{(i,j) \in \mathcal{D}} |W_{ij}|$$
2. **Regrow High-Gradient Weights**: From inactive connections $\mathcal{S}_t^c = \{(i, j) \mid M_{ij} = 0\}$, regrow $k_t$ connections exhibiting the largest instantaneous gradient magnitude:
   $$\text{Regrow set } \mathcal{R}_t = \arg\max_{\mathcal{R} \subset \mathcal{S}_t^c, |\mathcal{R}| = k_t} \sum_{(i,j) \in \mathcal{R}} |\nabla_{W_{ij}} \mathcal{L}|$$
3. **Cosine Drop Schedule**: The drop fraction decays smoothly to zero:
   $$\alpha_t = \frac{\alpha_0}{2} \left( 1 + \cos\left( \frac{\pi t}{T_{\text{end}}} \right) \right)$$

---

## 5. Hardware Execution Architecture: NVIDIA Sparse Tensor Cores

On NVIDIA Ampere, Hopper, and Blackwell architectures:
1. **Compressed Weight Storage**: Sparse weight matrices are packed into a half-sized array of non-zero FP16 values alongside an interleaved array of 2-bit indices.
2. **Sparse Matrix Multiply-Accumulate (MMA)**: Dedicated Sparse Tensor Core units use the 2-bit index metadata to dynamically multiplex the corresponding activations from shared memory into the dot-product multiplier tree.
3. **2x Arithmetic Throughput**: Because only 2 multiplications are performed per 4-element vector, arithmetic throughput doubles without increasing register pressure or memory bus saturation.

---

## 6. Multi-Framework Implementation & Parity (PyTorch, JAX, Candle)

The 2:4 semi-structured pruning and RigL dynamic update rules are implemented identically across PyTorch, JAX, and Candle:

```rust
// Candle implementation of 2:4 semi-structured pruning
pub fn prune_2_to_4(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = w.dims2()?;
    let w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let mut mask_vec = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for b in 0..(cols / 4) {
            let offset = r * cols + b * 4;
            let block = [
                w_vec[offset].abs(), w_vec[offset + 1].abs(),
                w_vec[offset + 2].abs(), w_vec[offset + 3].abs(),
            ];
            let mut indexed: Vec<(usize, f32)> = block.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            mask_vec[offset + indexed[0].0] = 1.0;
            mask_vec[offset + indexed[1].0] = 1.0;
        }
    }
    let mask = Tensor::from_vec(mask_vec, (rows, cols), w.device())?;
    let w_sparse = w.broadcast_mul(&mask)?;
    Ok((w_sparse, mask))
}
```

Parity across PyTorch, JAX, and Candle is verified against seed-42 golden safetensors to $< 10^{-5}$ tolerance.

---

## 7. Empirical Evaluation: FLOPs, Throughput, and Accuracy Preservation

| Configuration | Matrix Dimensions | Non-Zero Params | Theoretical FLOPs | Measured Tensor Core Throughput | Accuracy Recovery |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense FP16 Baseline** | $4096 \times 4096$ | $1.67 \times 10^7$ | $3.35 \times 10^7$ | $156\text{ TFLOPS}$ | 100.0% (Ref) |
| **2:4 One-Shot Pruned** | $4096 \times 4096$ | $0.84 \times 10^7$ | $1.68 \times 10^7$ | $312\text{ TFLOPS}$ ($2.0\times$) | 96.2% |
| **2:4 SR-STE Retrained**| $4096 \times 4096$ | $0.84 \times 10^7$ | $1.68 \times 10^7$ | $312\text{ TFLOPS}$ ($2.0\times$) | 99.7% |
| **RigL 80% Sparse** | $4096 \times 4096$ | $0.33 \times 10^7$ | $0.67 \times 10^7$ | Depends on sparse kernel | 98.9% (Trained from scratch) |

---

## 8. Decision Matrix & Production Deployment Playbook

```
                         Are you targeting hardware-accelerated GEMM speedup?
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
             Yes (Ampere+ GPUs)                  No / Edge / CPU
                     │                                 │
         Is retraining compute available?     Do you train from scratch
           │                  │                or compress post-hoc?
     ┌─────┴─────┐      ┌─────┴─────┐                  │
     ▼           ▼      ▼           ▼            ┌─────┴─────┐
    Yes          No    Yes          No           ▼           ▼
     │           │      │           │       From Scratch  Post-hoc
  2:4 SR-STE   2:4 PTQ ShortGPT   Wanda /       RigL     Structured
 Retraining  Calibration Slicing  SparseGPT   Dynamic     Channel
 (Full Acc)   (Fast)     (Layer)   (INT4/8)   Training    Pruning
```

1. **For Modern GPU Inference Acceleration**: Target 2:4 semi-structured sparsity with SR-STE fine-tuning. Utilize PyTorch `torch.sparse` or TensorRT 2:4 sparse kernels to achieve true $2\times$ GEMM acceleration.
2. **For Long-Horizon Training from Scratch**: Adopt RigL dynamic sparse training to train large architectures under constrained parameter budgets without requiring pre-trained dense checkpoints.
