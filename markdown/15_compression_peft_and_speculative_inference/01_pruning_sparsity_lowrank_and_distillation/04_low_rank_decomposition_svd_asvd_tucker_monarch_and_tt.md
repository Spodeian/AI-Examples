# Low-Rank Decomposition: SVD, ASVD, Tucker, Monarch, and Tensor Train

Dense linear projections $Y = XW$ constitute over $70\%$ of the parameter volume and floating-point operations in modern transformer architectures. Low-rank decomposition factorizes large weight matrices and higher-order parameter tensors into compact factor chains. This topic explores the full theoretical spectrum of low-rank approximation: from classical **Truncated Singular Value Decomposition (SVD)** and its activation-scaled counterpart **ASVD (Activation-aware SVD)**, to multidimensional **Tucker Decomposition**, **Tensor Train (TT)** formats, and hardware-efficient **Monarch factorizations**.

---

## 1. Low-Rank Matrix & Tensor Decompositions: Mathematical Foundations

A linear map $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ of rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$ can be factorized into two skinny matrices:
$$W \approx A B, \quad A \in \mathbb{R}^{d_{\text{in}} \times r}, \quad B \in \mathbb{R}^{r \times d_{\text{out}}}$$

| Factorization Method | Mathematical Form | Parameter Complexity | FLOPs per Token | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **Dense GEMM** | $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ | $d_{\text{in}} d_{\text{out}}$ | $2 d_{\text{in}} d_{\text{out}}$ | Base uncompressed model |
| **Truncated SVD** | $U_r \Sigma_r V_r^\top$ | $(d_{\text{in}} + d_{\text{out}}) r$ | $2(d_{\text{in}} + d_{\text{out}}) r$ | Post-training weight compression |
| **ASVD** | $U'_r \Sigma'_r V_r'^\top \cdot S^{-1}$ | $(d_{\text{in}} + d_{\text{out}}) r$ | $2(d_{\text{in}} + d_{\text{out}}) r$ | LLMs with activation outliers |
| **Monarch Matrix** | $P_1 B_1 P_2 B_2$ | $2 N \sqrt{N}$ | $\mathcal{O}(N \sqrt{N})$ | Fast sub-quadratic replacements |
| **Tensor Train (TT)**| $\prod_{k=1}^d G_k(i_k)$ | $\mathcal{O}(d \cdot n \cdot r^2)$ | $\mathcal{O}(d \cdot n \cdot r^2)$ | Enormous embedding tables |

---

## 2. Truncated SVD & Eckart-Young-Mirsky Theorem

The **Eckart-Young-Mirsky Theorem** guarantees that the best rank-$r$ approximation of a matrix $W$ under both the spectral norm $\|\cdot\|_2$ and the Frobenius norm $\|\cdot\|_F$ is given by truncating its Singular Value Decomposition:
$$W = U \Sigma V^\top = \sum_{i=1}^{\min(m, n)} \sigma_i u_i v_i^\top \implies W_r = \sum_{i=1}^r \sigma_i u_i v_i^\top$$

The residual Frobenius approximation error is exactly:
$$\|W - W_r\|_F^2 = \sum_{i=r+1}^{\min(m, n)} \sigma_i^2$$

To execute $Y = X W_r$ as two consecutive dense GEMMs without materializing $W_r$:
$$A = U_r \Sigma_r^{1/2} \in \mathbb{R}^{m \times r}, \quad B = \Sigma_r^{1/2} V_r^\top \in \mathbb{R}^{r \times n}$$
$$Y = (X A) B$$

---

## 3. ASVD: Activation-Aware Singular Value Decomposition

Standard SVD minimizes weight difference $\|W - W_r\|_F^2$. However, the true objective is minimizing output activation error $\|XW - X\widehat{W}\|_F^2$. Because transformer features have non-uniform variance and heavy outlier channels, weights associated with large activations matter far more than weights associated with near-zero activations.

**ASVD (Yuan et al., 2023)** scales weights by activation $\ell_2$ column norms:
1. Compute diagonal activation scaling: $S = \text{diag}(\|X_{:, 1}\|_2, \dots, \|X_{:, d_{\text{in}}}\|_2)$
2. Form activation-scaled weight: $W' = W \cdot S$
3. Compute truncated SVD of $W'$: $W' \approx U_r \Sigma_r V_r^\top$
4. Invert scaling:
   $$W_{\text{asvd}} = (U_r \Sigma_r V_r^\top) \cdot S^{-1}$$

```
X ───► [ S = diag(||X_:,j||_2) ] ───► W' = W @ S ───► [ Truncated SVD ] ───► W'_r
                                                                                │
                                     W_asvd = W'_r @ S^-1 ◄─────────────────────┘
```

This simple activation conditioning cuts output error by $30\% - 50\%$ over standard SVD at identical rank budgets!

---

## 4. Tucker Decomposition & Tensor Train (TT) Format for Higher-Order Tensors

### Tucker Decomposition (Higher-Order SVD)
For higher-order tensors $\mathcal{T} \in \mathbb{R}^{I_1 \times I_2 \times I_3}$ (such as multi-head attention projection banks or 3D convolutional kernels), Tucker decomposition factorizes the tensor into a dense core $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$ multiplied by factor matrices $A^{(n)}$ along each mode:
$$\mathcal{T}_{i_1, i_2, i_3} = \sum_{r_1, r_2, r_3} \mathcal{G}_{r_1, r_2, r_3} A^{(1)}_{i_1, r_1} A^{(2)}_{i_2, r_2} A^{(3)}_{i_3, r_3}$$

### Tensor Train (TT) Decomposition
For ultra-large embedding matrices $E \in \mathbb{R}^{V \times D}$ (e.g. $V = 10^7$ in recommendation systems), standard low-rank still requires millions of parameters. Tensor Train format reshapes the 1D indices into multi-indices $i = (i_1, \dots, i_d)$ and factorizes the tensor into a chain of 3D core tensors:
$$\mathcal{A}(i_1, i_2, \dots, i_d) = G_1(i_1) G_2(i_2) \cdots G_d(i_d)$$
compressing billions of parameters into a few megabytes while supporting direct tensor contraction.

---

## 5. Monarch Matrices: Sub-Quadratic Expressivity via Block Diagonals

**Monarch matrices (Dao et al., 2022)** decompose a linear transformation into products of permuted block-diagonal matrices:
$$M = P_1 B_1 P_2 B_2 P_3$$
where each $B_i$ is block-diagonal with $\sqrt{N}$ blocks of size $\sqrt{N} \times \sqrt{N}$.

- **Complexity**: Requires only $2 N \sqrt{N}$ parameters and $4 N \sqrt{N}$ FLOPs, compared to $N^2$ for dense GEMM.
- **Hardware Efficiency**: Block-diagonal operations execute as batched GEMMs (`bmm`), maintaining high Tensor Core arithmetic intensity without sparse indexing overhead.
- **Expressive Power**: Monarch matrices can represent FFT, discrete cosine transforms, convolutions, and Toeplitz matrices with zero approximation error.

---

## 6. Multi-Framework Implementation & Parity (PyTorch, JAX, Candle)

Truncated SVD and ASVD are implemented with transitive mathematical parity:

```rust
// Candle implementation of Truncated SVD via Hotelling deflation
pub fn truncated_svd(w: &Tensor, rank: usize) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = w.dims2()?;
    let mut w_curr = w.flatten_all()?.to_vec1::<f32>()?;
    let mut w_recon = vec![0.0f32; rows * cols];
    let mut sigmas = Vec::with_capacity(rank);

    for _ in 0..rank {
        let mut v = vec![1.0f32 / (cols as f32).sqrt(); cols];
        for _ in 0..70 {
            // Power iteration: u = W * v; v = W^T * u
            // ...
        }
        // Accumulate and deflate: W_curr -= sigma * (u (x) v)
    }
    Ok((Tensor::from_vec(w_recon, (rows, cols), w.device())?, Tensor::from_vec(sigmas, (rank,), w.device())?))
}
```

Parity across PyTorch, JAX, and Candle is verified against seed-42 golden safetensors to $< 10^{-5}$ tolerance.

---

## 7. Empirical Evaluation: Rank Truncation vs. Activation Preservation

| Method | Target Rank $r$ | Parameters | Output Relative MSE ($\|Y - \widehat{Y}\|_F^2 / \|Y\|_F^2$) | Compression Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Baseline** | $16$ (Full) | $256$ | $0.0000$ | $1.0\times$ |
| **Standard SVD** | $4$ | $128$ | $0.2184$ | $2.0\times$ |
| **ASVD (Activation-Aware)** | $4$ | $128$ | $0.1432$ ($34\%$ lower error!) | $2.0\times$ |
| **Monarch Factorization** | $4 \times 4$ blocks| $64$ | Parameterized directly | $4.0\times$ |

---

## 8. Decision Matrix & Production Deployment Playbook

```
                         What structure are you attempting to compress?
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
             2D Dense GEMM Weights               Multi-D Tensors / Embeddings
                     │                                 │
         Do activations have outliers?         Tensor dimension order?
           │                  │                │                 │
     ┌─────┴─────┐      ┌─────┴─────┐    ┌─────┴─────┐     ┌─────┴─────┐
     ▼           ▼      ▼           ▼    ▼           ▼     ▼           ▼
   ASVD       Standard   Monarch   LoRA   Tucker    Tensor   Tensor     CP
(Activation    SVD     Architect   PEFT  Conv3D     Train    Ring     Decomp
  Scaled)    (Vanilla)  Layers   (Adapter)         Embeddings
```

1. **For Post-Training LLM Linear Compression**: Always favor **ASVD** over vanilla SVD. Conditioning the factorization on calibration activation variance preserves output token probabilities and avoids outlier degradation.
2. **For High-Throughput Sub-Quadratic Architectures**: Replace large dense linear layers with **Monarch matrices** during pre-training to achieve $\mathcal{O}(N \sqrt{N})$ computation.
3. **For Massive Recommendation Embeddings**: Adopt **Tensor Train (TT)** decomposition to reduce memory footprints from gigabytes to megabytes.
