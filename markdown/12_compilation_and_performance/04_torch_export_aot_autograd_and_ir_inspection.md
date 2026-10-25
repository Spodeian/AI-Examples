# Torch Export, AOTAutograd, and IR Inspection

## 1. Mathematical Foundations & Hermetic Export Lineage

Deploying deep learning models outside the CPython runtime (e.g., in embedded edge devices, high-throughput C++ inference servers, or gaming engines) requires capturing complete, self-contained computational graphs:
1. **TorchScript Limitations (2018–2022)**: Relied on AST parsing or tracing with concrete tensors. It attempted to preserve Python semantics (like classes and dictionary mutations) into an ad-hoc virtual machine, leading to silent tracing bugs, type inference failures, and brittle runtime deployments.
2. **`torch.export` & Sound Capture (PyTorch 2.x)**: Replaces TorchScript with a mathematically sound graph capture mechanism. By enforcing that the model contains no side effects and operates purely on tensor arguments with explicit dynamic shape constraints, `torch.export` produces an `ExportedProgram` guaranteed to evaluate identically across execution backends.
3. **Core ATen Operator Dialect**: Reduces PyTorch's 2,000+ public API operators into a closed, minimal set of roughly 180 functional, side-effect-free primitives known as **Core ATen IR**. High-level composite operations (e.g., `torch.nn.functional.cross_entropy`, `torch.layer_norm`) are decomposed into these standardized building blocks.

---

## 2. The Core ATen Operator Dialect Hierarchy

```
  High-Level PyTorch Frontend (torch.nn, torch.special)
                   │
                   ▼  (Symbolic Capture / Decomposition)
┌────────────────────────────────────────────────────────┐
│ Canonical ATen Dialect (torch.ops.aten.*)               │
└────────────────────────────────────────────────────────┘
                   │
                   ▼  (Core ATen Decomposition Pass)
┌────────────────────────────────────────────────────────┐
│ Core ATen Functional IR (~180 Primitives)              │
│ - Purely functional (no in-place mutations)            │
│ - Explicit broadcasting & type promotion               │
└────────────────────────────────────────────────────────┘
                   │
                   ▼  (Backend Lowering)
  AOTInductor C++ / Triton  |  StableHLO  |  Native Executable
```

1. **Functional Purity**: Core ATen eliminates in-place mutations (e.g., `x.add_()`) by replacing them with functional equivalents (`torch.ops.aten.add(x, y)`), making data dependency analysis trivial.
2. **Explicit Metadata**: Every intermediate tensor node carries explicit rank, dtype, dynamic shape expressions, and device metadata.

---

## 3. Mathematical Decompositions into Core ATen Primitives

High-level composite operations are compiled by breaking them down into primitive mathematical steps:

### A. Core ATen Softmax Decomposition
Given logits $\mathbf{x} \in \mathbb{R}^{B \times D}$ along dimension $d$:

$$m_i = \max_{j} x_{i, j} \quad \implies \quad \mathbf{m} = \operatorname{amax}(\mathbf{x}, \dim=-1, \text{keepdim}=\text{True})$$

$$e_{i, j} = \exp(x_{i, j} - m_i) \quad \implies \quad \mathbf{e} = \exp(\mathbf{x} - \mathbf{m})$$

$$s_i = \sum_{j} e_{i, j} \quad \implies \quad \mathbf{s} = \operatorname{sum}(\mathbf{e}, \dim=-1, \text{keepdim}=\text{True})$$

$$y_{i, j} = \frac{e_{i, j}}{s_i} \quad \implies \quad \mathbf{y} = \operatorname{div}(\mathbf{e}, \mathbf{s})$$

### B. Core ATen Layer Normalization Decomposition
Given activations $\mathbf{x} \in \mathbb{R}^{B \times D}$, scale $\boldsymbol{\gamma}$, and shift $\boldsymbol{\beta}$:

$$\mu_i = \frac{1}{D} \sum_{j=1}^D x_{i, j} \quad \implies \quad \boldsymbol{\mu} = \operatorname{mean}(\mathbf{x}, \dim=-1, \text{keepdim}=\text{True})$$

$$\sigma_i^2 = \frac{1}{D} \sum_{j=1}^D (x_{i, j} - \mu_i)^2 \quad \implies \quad \boldsymbol{\sigma}^2 = \operatorname{mean}((\mathbf{x} - \boldsymbol{\mu})^2, \dim=-1, \text{keepdim}=\text{True})$$

$$\hat{x}_{i, j} = \frac{x_{i, j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \quad \implies \quad \hat{\mathbf{x}} = (\mathbf{x} - \boldsymbol{\mu}) \odot \operatorname{rsqrt}(\boldsymbol{\sigma}^2 + \epsilon)$$

$$\mathbf{y} = \hat{\mathbf{x}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

---

## 4. AOTAutograd: Ahead-of-Time Reverse-Mode Automatic Differentiation

In eager execution, PyTorch creates dynamic autograd engine nodes (`Node`, `AccumulateGrad`) at runtime during every forward pass. **AOTAutograd** shifts autodiff entirely to compile time:
1. **Tracing Joint Forward-Backward Graphs**: AOTAutograd intercepts PyTorch's dispatch keys (`AutogradCPU`, `AutogradCUDA`), symbolically propagating dual numbers / tangents through the forward pass.
2. **Saved Tensors for Backward**: The compiler explicitly tracks which forward intermediate tensors must be preserved in memory for the backward pass (`ctx.save_for_backward`), enabling optimal memory reuse and activation checkpointing.
3. **Partitioned Subgraphs**: AOTAutograd partitions the joint graph into two isolated functional FX graphs:
   - `forward_graph(*inputs)` $\to$ `(*outputs, *saved_tensors)`
   - `backward_graph(*saved_tensors, *grad_outputs)` $\to$ `(*grad_inputs)`

---

## 5. Reverse-Mode Vector-Jacobian Product (VJP) Calculus: Affine + GeLU

Consider an Affine transformation followed by Gaussian Error Linear Unit (Hendrycks & Gimpel 2016):

$$\mathbf{z} = \mathbf{x} \mathbf{W} + \mathbf{b}, \quad \mathbf{a} = \operatorname{GeLU}(\mathbf{z}) = \frac{1}{2} \mathbf{z} \left(1 + \operatorname{erf}\left(\frac{\mathbf{z}}{\sqrt{2}}\right)\right)$$

Given incoming upstream adjoint $\mathbf{g}_a = \frac{\partial \mathcal{L}}{\partial \mathbf{a}} \in \mathbb{R}^{B \times D_{\text{out}}}$:

$$\frac{d \operatorname{GeLU}(z)}{dz} = \frac{1}{2} \left(1 + \operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)\right) + \frac{z}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)$$

Applying the chain rule, the adjoint with respect to pre-activation $\mathbf{z}$ is:

$$\mathbf{g}_z = \mathbf{g}_a \odot \frac{d \operatorname{GeLU}(\mathbf{z})}{d\mathbf{z}}$$

Propagating backwards through the linear transformation yields parameter and input gradients:

$$\nabla_{\mathbf{W}} = \mathbf{x}^T \mathbf{g}_z \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$$

$$\nabla_{\mathbf{b}} = \sum_{i=1}^B \mathbf{g}_{z, i} \in \mathbb{R}^{D_{\text{out}}}$$

$$\nabla_{\mathbf{x}} = \mathbf{g}_z \mathbf{W}^T \in \mathbb{R}^{B \times D_{\text{in}}}$$

AOTAutograd compiles both forward and backward equations into a single static intermediate representation, allowing fusion across the forward-backward boundary.

---

## 6. Numerical Invariants and Parity Contracts

Across PyTorch 2.x, JAX, and Rust Candle, all Core ATen decompositions and AOT backward adjoints must adhere to strict numerical guarantees:
- **Max Absolute Error**: Transitive difference between un-decomposed composite operators and Core ATen decomposed operators must not exceed $10^{-5}$:
  $$\max |\mathbf{y}_{\text{core}} - \mathbf{y}_{\text{composite}}| < 10^{-5}$$
- **VJP Analytic Parity**: Analytical gradient calculations must match automatic reverse-mode tangents to within machine precision $\epsilon_{\text{mach}}$.

---

## 7. PyTorch 2.x, JAX, and Candle Architectural Parity

| Feature | PyTorch 2.x `torch.export` / AOTAutograd | JAX (`jax.export` / `jax.vjp`) | Rust Candle |
| :--- | :--- | :--- | :--- |
| **Export Construct** | `torch.export.ExportedProgram` | StableHLO artifact (`jax.export.export`) | Serialized SafeTensors + Static Graph |
| **Dialect Reduction** | Core ATen Operator Set (~180 ops) | StableHLO primitive set | Explicit minimal tensor kernels |
| **Autodiff Timing** | Ahead-of-Time graph extraction (`aot_autograd`) | Ahead-of-Time functional reverse mode (`jax.vjp`) | Ahead-of-Time or Manual VJP structs |
| **Saved Tensors** | Tracked explicitly in `graph_signature` | Stored in closure environment of residual | Explicit Rust tuple bindings |
| **Target Runtime** | AOTInductor C++ Library / ExecuTorch | IREE / XLA / OpenXLA Runtime | Standalone Rust binary (`no_std` capable) |

---

## 8. Production Deployment, AOTInductor, and Failure Modes

1. **AOTInductor C++ Code Generation**: `torch._inductor.aoti_compile_and_package` compiles an `ExportedProgram` into a standalone shared library (`.so` or `.dll`) with a pure C interface (`AOTIModelContainerRunner`), completely removing Python from production inference.
2. **Failure Modes & Non-Exportable Code**:
   - *Data-dependent mutations*: mutating Python dictionaries or lists inside forward pass.
   - *Unsound dynamic shapes*: using Python integers without `torch.export.Dim` dynamic shape declarations.
   - *Side effects*: writing to disk, mutating global variables, or interacting with network sockets.
3. **IR Inspection Tools**:
   - `print(exported_program.graph_module)`: Inspects high-level FX Python code.
   - `exported_program.graph.print_tabular()`: Displays tabular opcode, target, arguments, and metadata for every node.
   - `exported_program.schema`: Validates inputs, outputs, mutation signatures, and lifted tensor attributes.
