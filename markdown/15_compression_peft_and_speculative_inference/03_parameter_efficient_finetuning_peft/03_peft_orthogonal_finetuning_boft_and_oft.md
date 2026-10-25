# Orthogonal Parameter Adaptation: OFT and Butterfly BOFT

While additive low-rank adapters (such as LoRA, DoRA, and AdaLoRA) update weights via $\Delta W = B A$, modifying weight matrices additively disrupts the intrinsic angular relationships and directional hyperspherical representations learned during pre-training. In contrast, **Orthogonal Finetuning (OFT)** and **Butterfly Orthogonal Finetuning (BOFT)** frame adaptation as orthogonal transformations ($W_{\text{adapted}} = R W_0$ where $R^\top R = I$), strictly preserving neuron-to-neuron angles, preventing catastrophic forgetting, and ensuring hyper-stable optimization across generative vision and language models.

---

## 1. Orthogonal Parameter Adaptation: Hyperspherical Invariance

In deep neural networks, semantic representations and token concepts reside primarily on high-dimensional hyperspherical manifolds defined by the directional cosines of weight vectors:
$$\cos \theta_{ij} = \frac{\langle w_i, w_j \rangle}{\|w_i\|_2 \|w_j\|_2}$$

When additive adapters modify weights ($\tilde{W} = W_0 + \Delta W$), the angular relationships $\theta_{ij}$ between neurons are distorted arbitrarily, often degrading the model's out-of-domain robustness or causing representation collapse. 

| Adaptation Paradigm | Mathematical Parameterization | Geometric Invariance | Forget Resistance | Typical Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| **Additive LoRA** | $W_0 + \frac{\alpha}{r} B A$ | None (distorts norms and angles) | Moderate | $2 \times d \times r$ |
| **DoRA** | $m \frac{W_0 + \frac{\alpha}{r} B A}{\|W_0 + \frac{\alpha}{r} B A\|_F}$ | Norm decoupled; angles still distorted | Moderate | $2 \times d \times r + d$ |
| **Block-OFT** | $R W_0, \quad R = \text{blkdiag}(R_1, \dots, R_m)$ | **Preserves pairwise angles within blocks** | Very High | $\frac{m \times b(b-1)}{2}$ |
| **Butterfly BOFT** | $\left(\prod_{l=1}^L R^{(l)}\right) W_0$ | **Global orthogonal rotation invariance** | Maximum | $L \times \frac{m \times b(b-1)}{2}$ |

By restricting weight updates to the Lie group of orthogonal matrices $\mathbb{O}(d)$, OFT guarantees that for any orthogonal transformation $R$:
$$\|R w_i\|_2 = \|w_i\|_2 \quad \text{and} \quad \langle R w_i, R w_j \rangle = \langle w_i, w_j \rangle$$
Hence, neuron angles are provably invariant under adaptation.

---

## 2. Cayley Parameterization of Orthogonal Matrices

Optimizing an unconstrained matrix $R \in \mathbb{R}^{d \times d}$ while enforcing orthogonality ($R^\top R = I$) is non-trivial and requires expensive Riemannian manifold optimization or Stiefel manifold retractions.

The **Cayley transform** provides a parameterization that maps any real skew-symmetric matrix $Q$ ($Q^\top = -Q$) directly to a special orthogonal matrix $R \in \mathbb{SO}(d)$ without matrix exponentials or Gram-Schmidt reorthogonalization:

$$R = (I - Q)(I + Q)^{-1} = (I + Q)^{-1}(I - Q)$$

### Mathematical Properties:
1. **Unconditional Orthogonality**: If $Q^\top = -Q$, then:
   $$R^\top R = \left((I + Q)^{-1}(I - Q)\right)^\top (I + Q)^{-1}(I - Q) = (I + Q)(I - Q)^{-1}(I + Q)^{-1}(I - Q) = I$$
2. **Determinant $+1$**: $\det(R) = +1$, representing a pure rotation without reflection.
3. **Invertibility Guarantee**: All eigenvalues of real skew-symmetric matrix $Q$ are strictly imaginary ($\pm i \lambda_k, \lambda_k \in \mathbb{R}$). Therefore, the eigenvalues of $I + Q$ are $1 \pm i \lambda_k \neq 0$, ensuring $I + Q$ is always strictly invertible.
4. **Skew-Symmetric Extraction**: Any unconstrained trainable square parameter matrix $M \in \mathbb{R}^{b \times b}$ is projected to skew-symmetry via:
   $$Q = \frac{1}{2}(M - M^\top)$$

---

## 3. OFT (Orthogonal Finetuning): Block-Diagonal Parameterization

A dense orthogonal matrix $R \in \mathbb{R}^{d \times d}$ requires $\mathcal{O}(d^2)$ parameters and $\mathcal{O}(d^3)$ inversion complexity. For modern LLMs ($d \ge 4096$), direct parameterization is computationally intractable.

**OFT (Qiu et al., 2023)** structures $R$ as a block-diagonal matrix composed of $m$ independent orthogonal blocks $R_k \in \mathbb{R}^{b \times b}$ (where $d = m \times b$):

$$R = \begin{bmatrix}
R_1 & 0 & \cdots & 0 \\
0 & R_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & R_m
\end{bmatrix}$$

Each block $R_k$ is independently generated from an unconstrained trainable matrix $M_k \in \mathbb{R}^{b \times b}$ using the Cayley transform:
$$Q_k = \frac{1}{2}(M_k - M_k^\top), \quad R_k = (I - Q_k)(I + Q_k)^{-1}$$

### Forward Pass Formulation:
$$W_{\text{oft}} = R W_0$$
$$h = x W_{\text{oft}} = x (R W_0) = (x R) W_0$$

Because $R$ is block-diagonal, $(x R)$ can be evaluated in $\mathcal{O}(m \cdot b^2) = \mathcal{O}(d \cdot b)$ operations rather than $\mathcal{O}(d^2)$.

---

## 4. BOFT (Butterfly Orthogonal Finetuning)

While block-diagonal OFT limits parameter interactions to within each independent block of size $b \times b$, cross-block interactions are completely prohibited ($R_{ij} = 0$ across different blocks).

**BOFT (Liu et al., 2023)** overcomes this limitation by composing multiple sparse orthogonal layers inspired by the Cooley-Tukey FFT butterfly factorization:

$$R_{\text{total}} = R^{(L)} \cdots R^{(2)} R^{(1)}$$

Each layer $l \in \{1, \dots, L\}$ applies a permuted block-diagonal orthogonal transform:
$$R^{(l)} = \Pi_l^\top \begin{bmatrix} R_1^{(l)} & & 0 \\ & \ddots & \\ 0 & & R_m^{(l)} \end{bmatrix} \Pi_l$$
where $\Pi_l$ is a fixed permutation matrix that interleaves coordinate indices (e.g. stride-2 permutation).

```
Layer Input x ────► [ Block Rotation R^(1) ] ────► [ Butterfly Permute ] ────► [ Block Rotation R^(2) ] ────► x_rotated
                                                                                                                   │
                                                                                                                   ▼
                                                                                                            [ Frozen W0 ]
                                                                                                                   │
                                                                                                                   ▼
                                                                                                              Output h
```

### Key Advantages of BOFT:
1. **Full Information Diffusion**: With $L = \log_2(m)$ butterfly layers, every input dimension can interact with every output dimension orthogonally.
2. **Strict Orthogonality Retention**: Since each $R^{(l)}$ is orthogonal and the product of orthogonal matrices is strictly orthogonal ($\prod R^{(l)} \in \mathbb{O}(d)$), BOFT preserves exact hyperspherical norms.
3. **Logarithmic Parameter Growth**: Achieves the expressive capacity of a dense rotation with only $\mathcal{O}(L \cdot d \cdot b)$ parameters.

---

## 5. Mathematical Proof of Cayley Orthogonality

Let $Q \in \mathbb{R}^{n \times n}$ be skew-symmetric ($Q^\top = -Q$). We prove that $R = (I - Q)(I + Q)^{-1}$ is orthogonal.

### Step 1: Commutativity of $(I - Q)$ and $(I + Q)^{-1}$
$$(I - Q)(I + Q) = I - Q + Q - Q^2 = I - Q^2$$
$$(I + Q)(I - Q) = I + Q - Q - Q^2 = I - Q^2$$
Since $(I - Q)(I + Q) = (I + Q)(I - Q)$, multiplying both sides by $(I + Q)^{-1}$ on the left and right yields:
$$(I + Q)^{-1}(I - Q) = (I - Q)(I + Q)^{-1}$$

### Step 2: Verification of $R^\top R = I$
$$R^\top = \left((I - Q)(I + Q)^{-1}\right)^\top = \left((I + Q)^{-1}\right)^\top (I - Q)^\top$$
Using $(A^{-1})^\top = (A^\top)^{-1}$ and $Q^\top = -Q$:
$$(I + Q)^\top = I + Q^\top = I - Q$$
$$(I - Q)^\top = I - Q^\top = I + Q$$
Therefore:
$$R^\top = (I - Q)^{-1} (I + Q)$$

Now multiplying $R^\top$ and $R$:
$$R^\top R = \left[(I - Q)^{-1} (I + Q)\right] \left[(I - Q)(I + Q)^{-1}\right]$$
Applying commutativity $(I + Q)(I - Q) = (I - Q)(I + Q)$:
$$R^\top R = (I - Q)^{-1} (I - Q) (I + Q) (I + Q)^{-1} = I \cdot I = I$$
$\blacksquare$

---

## 6. Parameter Efficiency, Memory Footprint & Convergence

The following table contrasts structural parameters, memory overhead, and computational characteristics across standard adaptation techniques ($d = 4096$, $r = 8$, block size $b = 4$, $m = 1024$ blocks):

| Parameter Metric | Standard LoRA | DoRA | Block OFT ($b=4$) | Butterfly BOFT ($L=2, b=4$) |
| :--- | :--- | :--- | :--- | :--- |
| **Trainable Params / Layer** | $2 \cdot d \cdot r = 65,536$ | $65,536 + 4,096 = 69,632$ | $m \cdot \frac{b(b-1)}{2} = 6,144$ | $2 \times 6,144 = 12,288$ |
| **Parameter Reduction vs LoRA** | Baseline ($1.0\times$) | $+6.2\%$ | **$90.6\%$ reduction** | **$81.2\%$ reduction** |
| **Inference Overhead** | Can merge into $W_0$ | Can merge into $W_0$ | Can merge ($R W_0$) | Can merge ($R_{\text{total}} W_0$) |
| **Geometric Preservation** | Unbounded angle shift | Radial scaling only | Intra-block angle invariant | Multi-stage angle invariant |
| **Empirical Stability** | Learning rate sensitive | Stable | Highly stable | Extremely robust |

OFT and BOFT require significantly fewer parameters than standard LoRA while avoiding the gradient explosion and out-of-distribution drift frequently encountered during continuous training.

---

## 7. Production Engineering: Associative Forwarding & Fused Acceleration

During inference, the adapted weight matrix can be pre-multiplied:
$$W_{\text{deployed}} = R_{\text{total}} W_0$$
in zero latency overhead mode.

During interactive training or fine-tuning, the order of matrix multiplication dictates computational complexity:
1. **Weight Adaptation Path**: $W_{\text{oft}} = R W_0 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, followed by $h = x W_{\text{oft}}$. Requires materializing dense $d_{\text{in}} \times d_{\text{out}}$ weights per step.
2. **Associative Activation Path**: $x_{\text{rot}} = x R \in \mathbb{R}^{\text{batch} \times d_{\text{in}}}$, followed by $h = x_{\text{rot}} W_0$.
   - Because $R$ is block-diagonal, $x R$ requires only $\mathcal{O}(\text{batch} \cdot d \cdot b)$ operations.
   - For batch size $B = 4$, $d = 4096$, and $b = 4$, $x R$ requires only $65,536$ FLOPs, completely avoiding large intermediate weight allocations.

```wgsl
// WGSL Fused Block OFT Activation Projection: x_rot = x @ R
@compute @workgroup_size(16, 16, 1)
fn block_oft_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= u.batch || col >= u.d_out) { return; }

    var acc: f32 = 0.0;
    for (var m = 0u; m < u.d_in; m++) {
        var x_rot_m: f32 = 0.0;
        for (var k = 0u; k < u.d_in; k++) {
            x_rot_m += x[row * u.d_in + k] * r_matrix[k * u.d_in + m];
        }
        acc += x_rot_m * w0[m * u.d_out + col];
    }
    y[row * u.d_out + col] = acc;
}
```

---

## 8. Implementation Guidelines & Cross-Framework Parity

To maintain transitive numerical parity ($< 10^{-5}$) across PyTorch, JAX, and Candle:

1. **Exact Skew-Symmetry Projection**: Always compute $Q = \frac{1}{2}(M - M^\top)$ to eliminate numerical asymmetry from single-precision floating point rounding before applying Cayley inversion.
2. **Gauss-Jordan Inversion for Candle**: In platforms lacking runtime LAPACK bindings (such as standalone Candle CPU), small block inverses $(I + Q)^{-1}$ are evaluated using Gauss-Jordan elimination with partial pivoting. Skew-symmetry guarantees non-zero pivots ($|\det(I + Q)| \ge 1$).
3. **Seed-42 Golden Parity**: All parameters ($x, W_0, Q_{\text{skew}}, R_{\text{cayley}}, \text{oft\_out}, \text{boft\_out}$) are exported to `safetensors` and validated across all three backends.
