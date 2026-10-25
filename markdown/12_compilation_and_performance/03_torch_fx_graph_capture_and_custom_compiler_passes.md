# Torch FX Graph Capture and Custom Compiler Passes

## 1. Mathematical Foundations & Graph IR Lineage

Modern deep learning frameworks construct computations as Directed Acyclic Graphs (DAGs) operating on tensors. Rather than executing each high-level tensor operation eagerly:
1. **Static Single Assignment (SSA) IR**: Every intermediate value is assigned exactly once, enabling clear dataflow dependency analysis, dead code elimination, and common subexpression elimination (CSE).
2. **Symbolic Execution (Reed et al. 2021)**: By tracing operations with symbolic proxy variables rather than concrete numerical tensors, the compiler records the exact sequence of functional invocations without performing expensive tensor arithmetic.
3. **Torch.FX Node Taxonomy**: The FX intermediate representation decomposes arbitrary PyTorch execution into exactly six node opcodes:
   - `placeholder`: Represents function arguments or inputs to the graph.
   - `get_attr`: Retrieves a parameter, buffer, or submodule attribute from the enclosing `torch.nn.Module`.
   - `call_function`: Direct invocation of a free function (e.g., `torch.add`, `operator.mul`).
   - `call_method`: Invocation of a method on an FX node (e.g., `x.relu()`, `x.matmul(w)`).
   - `call_module`: Invocation of a nested `torch.nn.Module` forward method.
   - `output`: The terminal node defining the return value(s) of the graph.

---

## 2. Torch.FX Architecture: Symbolic Tracing, GraphModule, and Code Generation

The Torch.FX pipeline operates across three foundational primitives:
```
  torch.nn.Module (Eager Python Model)
             │
             ▼
┌──────────────────────────────────────────┐
│ torch.fx.symbolic_trace / Tracer         │ ── Traces via Proxy objects
└──────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ torch.fx.Graph (Ordered Node List)       │ ── In-place graph rewrites, pattern matching
└──────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ torch.fx.GraphModule (Python CodeGen)    │ ── Generates valid Python bytecode / source
└──────────────────────────────────────────┘
```

1. **Proxy Objects & Graph Construction**: `torch.fx.Tracer` records operations invoked on `Proxy` instances, appending new `Node` records to the active `Graph` data structure while preserving topological ordering.
2. **Re-compilation via CodeGen**: When graph transformations are complete, `GraphModule.recompile()` generates valid Python source code reflecting the mutated node list and compiles it into a runnable Python code object.
3. **Preserving Model State**: State dictionaries, parameters, and buffers remain attached to the root `GraphModule`, allowing direct parameter mutation, quantization weight updates, or parameter elimination.

---

## 3. Algebraic Optimization Passes: Linear and BatchNorm Folding

In inference pipelines, sequential linear or convolutional layers followed by Batch Normalization (Ioffe & Szegedy 2015) can be algebraically folded into a single affine transformation with zero inference-time normalization overhead.

Given input $\mathbf{x} \in \mathbb{R}^{B \times D_{\text{in}}}$, linear weights $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, bias $\mathbf{b} \in \mathbb{R}^{D_{\text{out}}}$, and BatchNorm parameters (running mean $\boldsymbol{\mu}$, running variance $\boldsymbol{\sigma}^2$, scale $\boldsymbol{\gamma}$, shift $\boldsymbol{\beta}$, and stability constant $\epsilon$):

$$\mathbf{z} = \mathbf{x} \mathbf{W} + \mathbf{b}$$

$$\mathbf{y} = \frac{\mathbf{z} - \boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

Expanding $\mathbf{z}$:

$$\mathbf{y} = \left( (\mathbf{x} \mathbf{W} + \mathbf{b}) - \boldsymbol{\mu} \right) \odot \frac{\boldsymbol{\gamma}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} + \boldsymbol{\beta}$$

$$\mathbf{y} = \mathbf{x} \left( \mathbf{W} \operatorname{diag}\left( \frac{\boldsymbol{\gamma}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} \right) \right) + \left( (\mathbf{b} - \boldsymbol{\mu}) \odot \frac{\boldsymbol{\gamma}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} + \boldsymbol{\beta} \right)$$

Defining folded weights and bias:

$$\mathbf{W}_{\text{fold}} = \mathbf{W} \odot \left( \frac{\boldsymbol{\gamma}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} \right)^T$$

$$\mathbf{b}_{\text{fold}} = (\mathbf{b} - \boldsymbol{\mu}) \odot \frac{\boldsymbol{\gamma}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} + \boldsymbol{\beta}$$

The sequential two-layer sequence collapses into $\mathbf{y} = \mathbf{x} \mathbf{W}_{\text{fold}} + \mathbf{b}_{\text{fold}}$, eliminating one memory write, one memory read, and three elementwise vector operations per forward pass.

---

## 4. Horizontal GEMM Fusion Passes: Multi-Head Projection Fusion

In Transformer multi-head self-attention (Vaswani et al. 2017), independent query ($\mathbf{W}_Q$), key ($\mathbf{W}_K$), and value ($\mathbf{W}_V$) projections are applied to the same hidden representation $\mathbf{x} \in \mathbb{R}^{B \times D}$:

$$\mathbf{Q} = \mathbf{x} \mathbf{W}_Q + \mathbf{b}_Q, \quad \mathbf{K} = \mathbf{x} \mathbf{W}_K + \mathbf{b}_K, \quad \mathbf{V} = \mathbf{x} \mathbf{W}_V + \mathbf{b}_V$$

Executing three separate General Matrix Multiplications (GEMMs) incurs $3\times$ kernel launch overhead and underutilizes GPU systolic tensor cores when batch size is small.

An FX custom compiler pass matches nodes sharing the same input $\mathbf{x}$ and fuses them horizontally into a single concatenated GEMM:

$$\mathbf{W}_{\text{fused}} = [\mathbf{W}_Q \;\|\; \mathbf{W}_K \;\|\; \mathbf{W}_V] \in \mathbb{R}^{D \times 3D_{\text{proj}}}$$

$$\mathbf{b}_{\text{fused}} = [\mathbf{b}_Q \;\|\; \mathbf{b}_K \;\|\; \mathbf{b}_V] \in \mathbb{R}^{3D_{\text{proj}}}$$

$$\mathbf{Y}_{\text{fused}} = \mathbf{x} \mathbf{W}_{\text{fused}} + \mathbf{b}_{\text{fused}}$$

A downstream slice or split node then partitions $\mathbf{Y}_{\text{fused}}$ into $\mathbf{Q}, \mathbf{K}, \mathbf{V}$, reducing launch overhead from 3 kernels to 1 kernel.

---

## 5. Custom Compiler Passes with `Transformer` and Subgraph Rewriting

Torch.FX provides two primary methodologies for graph rewriting:
1. **`torch.fx.Transformer`**: A visitor pattern subclassed to intercept each node opcode (`call_function`, `call_module`). Returns replacement nodes or rewrites arguments during traversal.
2. **`torch.fx.subgraph_rewriter.replace_pattern`**: Declarative graph pattern replacement that identifies subgraphs matching a reference Python function and replaces them with an optimized target function.

Example linear-activation fusion using pattern matching:
```python
def pattern(x, w, b):
    return torch.relu(torch.matmul(x, w) + b)

def replacement(x, w, b):
    return torch.ops.aten.relu_fused_linear(x, w, b)

replace_pattern(traced_module, pattern, replacement)
traced_module.graph.lint()
traced_module.recompile()
```

---

## 6. Numerical Invariants and Cross-Framework Parity

To ensure mathematical soundness across graph transformations, all compiler passes must satisfy strict equivalence bounds:
- **Transitive Parity Bound**: The folded graph $\mathcal{G}_{\text{opt}}(\mathbf{x})$ and unoptimized graph $\mathcal{G}(\mathbf{x})$ must adhere to:
  $$\max_{i} |\mathcal{G}_{\text{opt}}(\mathbf{x})_i - \mathcal{G}(\mathbf{x})_i| < 10^{-5}$$
- **Linear-BatchNorm Equivalence**: Evaluated against float32 golden safetensors generated from seed 42.
- **Horizontal Fusion Invariance**: Concatenation of weight columns preserves exact dot-product linearity.

---

## 7. PyTorch 2.x, JAX, and Candle Architectural Parity

| Feature | PyTorch 2.x FX | JAX (`jax.core.ClosedCall`) | Rust Candle |
| :--- | :--- | :--- | :--- |
| **Graph Representation** | `torch.fx.Graph` / `GraphModule` | `jaxpr` (Jax Expressions) | Explicit Rust computation graph / AST |
| **Node Types** | 6 Opcodes (`placeholder`, `call_fn`, etc.) | Primitive `eqn` binding input/output `vars` | Native struct enum representations |
| **Rewriting Mechanism** | `Transformer`, `replace_pattern` | Jaxpr interpreters (`core.eval_jaxpr`) | Match statements over tensor nodes |
| **Code Generation** | Python code string compilation (`compile()`) | XLA / HLO compilation pipeline | Rust binary compilation / Wasm |
| **In-place Mutation** | Supported via `graph.inserting_after()` | Immutable functional transformations | Immutable Tensor borrows |

---

## 8. Production Diagnostics, Linting, and Failure Modes

1. **Graph Linting with `graph.lint()`**: Verifies that every node's dependencies are topologically sorted, all users are valid, and output node references exist.
2. **Handling Dynamic Control Flow**: If an FX tracer encounters dynamic `if` statements or loops depending on tensor values, tracing fails with `TraceError`. Workarounds include custom `Tracer` classes with symbolic tracing overrides or delegating dynamic sections to TorchDynamo.
3. **Dead Code Elimination**: After fusing or removing nodes, call `graph.eliminate_dead_code()` to purge unreachable nodes and prevent memory leaks.
4. **Parameter Synchronization**: When replacing `call_module` nodes (such as folding BatchNorm into Linear), unused submodules must be deleted via `delattr(root, submodule_name)` to prevent state dictionary desynchronization.
