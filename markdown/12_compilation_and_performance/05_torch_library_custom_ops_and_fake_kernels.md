# Torch Library Custom Ops and Fake Kernels

## 1. Mathematical Foundations & Extensible Compiler Lineage

Deep learning systems frequently rely on specialized operations not covered by standard tensor library primitives (e.g., custom flash attention variants, sparse matrix-vector products, domain-specific quantization routines, or hardware-specific accelerators).
1. **The Graph Break Dilemma**: In eager PyTorch, calling external C++ or CUDA extensions through `torch.autograd.Function` works seamlessly. However, when compiled under `torch.compile`, unmodeled external functions trigger catastrophic graph breaks, segmenting the computation graph and disabling cross-operator fusion.
2. **`torch.library` Unified Dispatch**: Introduced in PyTorch 2.x, `torch.library` provides an explicit, type-safe API to define custom operators, specify parameter schemas, and attach multiple backend implementations (CPU, CUDA, Meta/Fake, Autograd, and Triton).
3. **Fake Tensors & Abstract Interpretation (Cousot & Cousot 1977)**: Tracing compilers must infer output tensor shapes, strided layouts, and data types symbolically during ahead-of-time graph capture without allocating physical storage or executing device kernels. **Fake kernels** (or Meta kernels) implement this abstract semantics.

---

## 2. The `torch.library` Registration Architecture

```
  Custom Operator Definition (Schema: namespace::op_name)
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Meta / Fake Kernel Registration (@op.register_fake)     │ ──► Used by Dynamo & AOTAutograd for Shape Inference
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Autograd Registration (register_autograd / VJP formula) │ ──► Used by AOTAutograd for Backward Graph Extraction
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Hardware Kernel Implementations (CPU, CUDA, Triton)    │ ──► Executed at Runtime by Inductor Engine
└─────────────────────────────────────────────────────────┘
```

1. **Explicit Schema Specification**: Every custom operator declares its schema in LibTorch syntax:
   `"custom_namespace::soft_shrink(Tensor x, float lambd) -> Tensor"`
2. **Mutations and Aliasing Annotations**: Schemas explicitly state whether an argument is mutated (`Tensor(a!)`) or borrowed, allowing the memory planner to safely reuse buffers.

---

## 3. Fake Tensors and Meta-Kernel Formulation

During graph capture, TorchDynamo replaces real GPU tensors with `FakeTensor` instances that carry:
- Shape expressions (concrete or symbolic dynamic symbols like $s_0, s_1$)
- Strides and memory layout (contiguous, channels-last)
- Data type (`torch.float32`, `torch.bfloat16`)
- Device placement (`cuda:0`, `cpu`)

A **Meta-Kernel** computes output metadata purely as a mathematical function of input metadata:

$$\text{Meta}_{\text{shrink}}(\text{shape}(\mathbf{x}), \text{dtype}(\mathbf{x}), \text{device}(\mathbf{x})) = (\text{shape}(\mathbf{x}), \text{dtype}(\mathbf{x}), \text{device}(\mathbf{x}))$$

For convolution operations with kernel size $K$ and stride $S$:

$$\text{Meta}_{\text{conv1d}}(B, C, L, K, S) = \left(B, C, \left\lfloor \frac{L - K}{S} \right\rfloor + 1\right)$$

The meta-kernel allocates no storage (`torch.empty(out_shape, device="meta")`), enabling rapid compilation of massive billion-parameter models.

---

## 4. Algorithmic Formulation of Custom Operations

### A. Custom L1 Proximal Soft-Shrinkage Operator
The proximal operator for the $L_1$ norm (used in sparse coding, Lasso, and wavelets) is defined as:

$$\operatorname{shrink}(x, \lambda) = \operatorname{sign}(x) \cdot \max(|x| - \lambda, 0)$$

Its subgradient with respect to input $x$ is:

$$\frac{\partial \operatorname{shrink}(x, \lambda)}{\partial x} = \begin{cases} 1 & \text{if } |x| > \lambda \\ 0 & \text{if } |x| \le \lambda \end{cases}$$

### B. Row-wise 8-Bit Asymmetric Affine Quantization
Maps continuous activations to discrete unsigned 8-bit integers $[0, 255]$:

$$s_i = \max\left(\frac{\max_j x_{i, j} - \min_j x_{i, j}}{255}, \epsilon\right)$$

$$z_i = \operatorname{clamp}\left(\operatorname{round}\left(-\frac{\min_j x_{i, j}}{s_i}\right), 0, 255\right)$$

$$q_{i, j} = \operatorname{clamp}\left(\operatorname{round}\left(\frac{x_{i, j}}{s_i}\right) + z_i, 0, 255\right)$$

$$\hat{x}_{i, j} = (q_{i, j} - z_i) \cdot s_i$$

### C. Depthwise 1D Convolution
Applies independent spatial filtering across each channel without cross-channel summation:

$$y_{b, c, t} = \operatorname{bias}_c + \sum_{k=0}^{K-1} x_{b, c, t + k} \cdot w_{c, k}$$

---

## 5. Autograd Adjoint Formulation for Custom Operators

When registering reverse-mode autodiff for a custom op:
```python
@torch.library.register_autograd("custom::soft_shrink")
def soft_shrink_autograd(ctx, x, lambd):
    ctx.lambd = lambd
    ctx.save_for_backward(x)
    return custom_soft_shrink(x, lambd)

def soft_shrink_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    mask = (torch.abs(x) > ctx.lambd).to(grad_output.dtype)
    return grad_output * mask, None
```
AOTAutograd integrates `soft_shrink_backward` directly into the backward FX graph, allowing Inductor to fuse the backward mask evaluation with subsequent gradient calculations.

---

## 6. Numerical Invariants and Parity Contracts

- **Threshold Invariance**: For all $|x| \le \lambda$, $\operatorname{shrink}(x, \lambda) \equiv 0.0$.
- **Quantization Round-Trip Bound**: The reconstruction error is bounded by:
  $$\max_{i, j} |x_{i, j} - \hat{x}_{i, j}| \le \frac{s_i}{2}$$
- **Parity Tolerance**: Across PyTorch 2.x, JAX, and Candle, maximum absolute error between references and implementations must remain $< 10^{-5}$.

---

## 7. PyTorch 2.x, JAX, and Candle Architectural Parity

| Feature | PyTorch 2.x `torch.library` | JAX Custom Primitives | Rust Candle |
| :--- | :--- | :--- | :--- |
| **Op Registration** | `@torch.library.custom_op` | `jax.core.Primitive(name)` | `candle::CustomOp1` / `CustomOp2` trait |
| **Shape Inference** | `@op.register_fake` / Meta kernel | `primitive.def_abstract_eval` | `fn eval(&self, &Tensor) -> Result<Tensor>` |
| **Autograd Rule** | `register_autograd` / VJP rule | `jax.interpreters.ad.defvjp` | Reverse pass `backward` method on struct |
| **Batching / Vectorization** | `torch.vmap` dispatch key | `jax.interpreters.batching.defvectorized` | Native tensor broadcasting |
| **Backend Lowering** | TorchInductor Triton codegen | XLA / MLIR translation rule | Custom Wasm / WGSL / CPU kernels |

---

## 8. Production Diagnostics, Schema Validation, and Failure Modes

1. **Schema Mismatches**: Omitting mutable tags (e.g. declaring `Tensor x` when mutating in-place) causes silent memory corruption during graph scheduling.
2. **Missing Meta Kernels**: Calling an unregistered custom op inside `torch.compile` raises `torch._dynamo.exc.Unsupported: call_function ... without fake kernel`.
3. **Symbolic Shape Leakage**: Attempting to convert symbolic dimension variables into concrete Python integers (`int(shape[0])`) inside a meta-kernel triggers dynamic shape tracing errors. Use `torch.sym_max` or symbolic arithmetic helpers instead.
