# 02. Tensors, Views, Strides & Memory Layouts

## 1. Executive Architectural Brief & Core Invariants

Tensors in modern deep learning frameworks are multi-dimensional coordinate views projected over a flat, one-dimensional contiguous block of physical memory (the `Storage`). Understanding stride mechanics, view transformations, and physical memory layouts is essential for writing zero-copy high-performance algorithms, eliminating hidden memory allocations, and preventing silent numerical corruption from overlapping in-place writes.

```
+-----------------------------------------------------------------------------------+
|                        TENSOR STORAGE & VIEW TAXONOMY                             |
|                                                                                   |
|  APL (Iverson 1962)  -->  C/Fortran Strides  -->  PyTorch & NumPy Tensor Views     |
|  [Array Theory]           [Row/Col Major]         [Storage + Offset + Strides]    |
|                                                               |                   |
|                                                               v                   |
|                                    +------------------------------------------+   |
|                                    | 02. Tensors, Views, Strides & Layouts    |   |
|                                    +------------------------------------------+   |
|                                         /                  |                 \    |
|                                        v                   v                  v   |
|                              03. Autograd Tape    12. Compilation     17. Triton  |
+-----------------------------------------------------------------------------------+
```

### Core Invariant Table

| Metric / Invariant | Mathematical Expression | Target Bound / Invariant | Framework Mechanism |
| :--- | :--- | :--- | :--- |
| **Linear Address Mapping** | $\text{addr}(i) = \text{base} + \left(\text{offset} + \sum_{k=0}^{D-1} i_k s_k\right) \cdot \text{sizeof}(T)$ | Exact bijective index for contiguous tensors | `tensor.stride()`, `tensor.storage_offset()` |
| **Zero-Copy Invariant** | $\text{ptr}(\text{view}) == \text{ptr}(\text{source})$ | True ($0$ bytes allocated) | `torch.as_strided`, `narrow`, `slice` |
| **Contiguity Criterion** | $s_{D-1} = 1 \land s_k = s_{k+1} \cdot d_{k+1}$ | C-contiguous (Row-Major) | `tensor.is_contiguous()` |
| **Non-Overlapping Memory** | $\forall i \ne j \implies \sum_k i_k s_k \ne \sum_k j_k s_k$ | Mandatory for safe in-place writes | Overlap detector / clone guard |
| **Contraction Parity** | $\|Y_{\text{einsum}} - Y_{\text{GEMM}}\|_\infty$ | $< 10^{-5}$ | `torch.einsum(..., optimize=True)` |

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> 1. **In-Place Mutation on Broadcast / Strided Views:** Never invoke in-place operations (`.add_()`, `*=`, `scatter_`) on tensors produced via `.expand()` or `as_strided` with overlapping strides ($s_k = 0$ or stride step $<$ window length); doing so silently corrupts multiple coordinate locations simultaneously.
> 2. **Calling `.view()` on Transposed Tensors:** After `.transpose()` or `.permute()`, the memory layout becomes non-contiguous. Calling `.view()` raises a fatal `RuntimeError: view size is not compatible with input tensor's size and stride`. Use `.reshape()` or `.contiguous()` instead.
> 3. **Unparenthesized Multi-Operand Einsum:** Evaluating multi-tensor contractions `einsum("ab,bc,cd,de->ae")` without greedy/optimal path search materializes gigantic $O(N^4)$ intermediate tensors, triggering immediate Out-Of-Memory (`OOM`) crashes.

---

## 2. Mathematical Formalism & First-Principles Derivation

### Continuous to Discrete Storage Mapping
A tensor $A \in \mathbb{R}^{d_0 \times d_1 \times \dots \times d_{D-1}}$ is an abstract multidimensional array. The underlying physical computer memory is a 1D contiguous byte buffer $\mathcal{B}$.

The mapping from multidimensional logical index coordinate $\mathbf{i} = (i_0, i_1, \dots, i_{D-1})$ to linear scalar storage offset $L(\mathbf{i})$ is defined as:

$$L(\mathbf{i}) = \text{offset} + \sum_{k=0}^{D-1} i_k \cdot s_k, \quad \text{where } 0 \le i_k < d_k$$

where:
- $\text{offset} \in \mathbb{N}_0$ is the storage offset (the starting element in the underlying storage buffer).
- $\mathbf{s} = (s_0, s_1, \dots, s_{D-1}) \in \mathbb{Z}^D$ is the stride tuple. The stride $s_k$ specifies the number of physical memory elements skipped to advance by 1 in dimension $k$.

### Canonical Row-Major (C-Contiguous) vs Column-Major (Fortran) Strides

1. **Row-Major (C-Contiguous):** The last index varies fastest in linear memory:
   $$s_{D-1} = 1, \quad s_k = \prod_{m=k+1}^{D-1} d_m = s_{k+1} \cdot d_{k+1}$$

2. **Column-Major (Fortran-Contiguous):** The first index varies fastest in linear memory:
   $$s_0 = 1, \quad s_k = \prod_{m=0}^{k-1} d_m = s_{k-1} \cdot d_{k-1}$$

### Einstein Summation & Contraction Graphs
Einstein summation specifies tensor contractions by implicit summation over repeated indices:

$$C_{ik} = \sum_{j} A_{ij} B_{jk} \iff C_{ik} = A_{ij} B_{jk} \quad (\text{GEMM})$$

For an $N$-tensor contraction, the order in which index pairs are contracted dictates the intermediate memory footprint and computational complexity. For three tensors $A_{ij}, B_{jk}, C_{kl}$:
- Left-to-Right: $(A B) C \implies \mathcal{O}(I J K + I K L)$ FLOPs, intermediate size $I \times K$.
- Right-to-Left: $A (B C) \implies \mathcal{O}(J K L + I J L)$ FLOPs, intermediate size $J \times L$.
Optimal path selection (via dynamic programming or greedy graph search) prevents intermediate memory bloat.

---

## 3. Invertibility, Conditioning & Numerical Invariants

### Invertibility of Stride Transformations
A stride transformation is invertible (i.e. every physical element corresponds to a unique logical coordinate) if and only if the linear mapping $L(\mathbf{i})$ is strictly injective over the domain $\prod_{k=0}^{D-1} [0, d_k - 1]$:

$$\forall \mathbf{i} \ne \mathbf{j} \implies L(\mathbf{i}) \ne L(\mathbf{j})$$

### The Overlap Theorem
If there exists any dimension $k$ such that $s_k = 0$ with $d_k > 1$ (e.g. from `tensor.expand()`), or if the strides satisfy:

$$\exists k, m \text{ such that } s_k < s_m \cdot d_m \quad (\text{with overlapping sliding windows})$$

then the tensor view possesses overlapping memory addresses. Any write operation to $L(\mathbf{i})$ simultaneously alters $L(\mathbf{j})$, violating algebraic independence.

---

## 4. Cross-Framework Comparative Reference

| Operation | PyTorch 2.x Track (`pytorch2/`) | JAX / Functional Track (`jax/`) | Rust Candle Track (`candle/`) | Fused Kernel (`kernels/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Inspect Strides** | `tensor.stride()`, `tensor.storage_offset()` | `arr.strides` (NumPy/CPU) | `tensor.stride()`, `storage_offset()` | Pointer arithmetic offset |
| **Zero-Copy View** | `torch.as_strided(x, size, stride)` | `jnp.lib.stride_tricks.sliding_window_view` | `tensor.narrow()`, `slice()` | Offset loop in register |
| **Contiguous Check** | `tensor.is_contiguous()` | Implicitly managed by XLA | `tensor.is_contiguous()` | Row-major assertion |
| **Layout Conversion** | `x.to(memory_format=torch.channels_last)` | Handled by XLA layout optimization | `tensor.contiguous()?` | Fused strided read |
| **Optimized Einsum** | `torch.einsum("...", ..., optimize=True)` | `jnp.einsum("...", ...)` | `tensor.matmul()` composition | Implicit GEMM loop |

---

## 5. Hardware & Compilation Dynamics

### Memory Hierarchy & Cache Coalescing
Modern GPUs and CPUs transfer memory between DRAM and caches in 64-byte or 128-byte cache lines.
- **Coalesced Memory Access:** When consecutive threads in a GPU warp access adjacent memory addresses ($s_{D-1} = 1$), a single 64-byte memory transaction services all 32 threads.
- **Strided / Non-Coalesced Access:** When threads access memory with stride $s > 1$ (e.g. reading columns of a row-major matrix), each thread requires a distinct cache line, dropping effective memory bandwidth by up to $32\times$.

```
Coalesced (Stride = 1):
Thread:    [ T0 ][ T1 ][ T2 ][ T3 ][ T4 ][ T5 ][ T6 ][ T7 ] ... [ T31 ]
DRAM Line: [================== 64-byte Burst ==================]  (1 Transaction)

Non-Coalesced (Stride = 16):
Thread:    [ T0 ]--------[ T1 ]--------[ T2 ]--------[ T3 ] ...
DRAM Line: [ 64B ]       [ 64B ]       [ 64B ]       [ 64B ]     (32 Transactions!)
```

### Channels-Last (`NHWC`) on TensorCores
In vision workloads, standard PyTorch uses `NCHW` layout. However, NVIDIA TensorCores require input channels to be contiguous in memory (i.e. `NHWC` or `torch.channels_last`) to vectorize matrix multiply-accumulate operations across $8$- or $16$-element channel vectors. Converting models to `channels_last` yields a $1.2\times$ to $1.8\times$ speedup on modern GPUs.

---

## 6. Failure Modes, Pathology Lab & Anti-Patterns

### Pathology Table

| ID | Failure Mode | Root Cause | Observable Symptom | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **FAT-01** | Silent Overlap Data Corruption | In-place write on broadcast or sliding-window view | Unintended values written to multiple unrelated elements | Never mutate in-place on views; enforce `.clone()` before writing |
| **FAT-02** | Non-Contiguous `.view()` Crash | Calling `.view()` on a transposed/permuted tensor | `RuntimeError: view size is not compatible with input tensor's size and stride` | Use `.reshape()` or `.contiguous().view()` |
| **FAT-03** | Intermediate Einsum Memory Explosion | Unoptimized multi-tensor contraction path | CUDA Out-Of-Memory (`OOM`) during forward pass | Specify `optimize=True` in `torch.einsum` or use `opt_einsum` |
| **FAT-04** | Strided PyTorch Inductor Graph Break | Dynamic `as_strided` calls with non-constant strides | Inductor compiler falls back to eager Python runtime | Use static shapes or native `torch.nn.functional.unfold` |

### Concrete Misuse Snippet
```python
# ANTI-PATTERN: In-place write on overlapping strided view
x = torch.zeros(4, dtype=torch.float32)
# Stride 0 view broadcasts 1 element across 4 coordinates
view_overlap = torch.as_strided(x, size=(4,), stride=(0,))

# In-place write corrupts all 4 elements simultaneously
view_overlap[0] += 5.0
print(x)  # Unexpectedly outputs tensor([5., 5., 5., 5.]) instead of [5., 0., 0., 0.]!
```

---

## 7. Quantitative Evaluation & Verification Harness

### Verification Harness Specification
Every implementation of the Tensors, Views & Strides track must pass the following assertions:

```python
# 1. Linear Indexing Equivalence
calc_idx = storage_offset + sum(c * s for c, s in zip(coord, tensor.stride()))
assert tensor[coord].item() == tensor.storage()[calc_idx]

# 2. Zero-Copy Invariant
window = extract_sliding_windows(x, window_size=4, step=2)
assert x.data_ptr() == window.data_ptr()

# 3. Einsum vs GEMM Parity
out_einsum = torch.einsum("bi,ij,bj->b", x, w, y)
out_gemm = (x @ w * y).sum(dim=-1)
torch.testing.assert_close(out_einsum, out_gemm, atol=1e-5, rtol=1e-5)
```

---

## 8. Related Nodes & Architectural Lineage

### Lineage Graph
```
                        01. Linear Neuron & Invertible Transforms
                                          |
                                          v
                    02. Tensors, Views, Strides & Memory Layouts
                                          |
         +--------------------------------+--------------------------------+
         |                                                                 |
         v                                                                 v
03. Autograd Mechanics & Wirtinger Calculus                   12. Compilation & Fusion
         |                                                                 |
         +--------------------------------+--------------------------------+
                                          |
                                          v
                     17. Custom Triton / Hardware Kernels
```

### Relative Navigation
- **Master Curriculum Index:** [README.md](../../README.md)
- **Goal-Oriented Index:** [INDEX_BY_GOAL.md](../../INDEX_BY_GOAL.md)
- **Interactive Notebook Explorer:** [02_tensors_views_strides_and_memory_layouts.ipynb](../../notebooks/00_foundations/02_tensors_views_strides_and_memory_layouts.ipynb)
- **PyTorch 2.x Production Module:** [02_tensors_views_strides_and_memory_layouts.py](../../pytorch2/00_foundations/02_tensors_views_strides_and_memory_layouts.py)
- **Functional JAX Module:** [02_tensors_views_strides_and_memory_layouts.py](../../jax/00_foundations/02_tensors_views_strides_and_memory_layouts.py)
- **Native Rust Candle Crate:** [02_tensors_views_strides_and_memory_layouts.rs](../../candle/00_foundations/02_tensors_views_strides_and_memory_layouts.rs)
- **CPU Fused Kernel:** [02_tensors_views_strides_and_memory_layouts_cpu.py](../../kernels/00_foundations/02_tensors_views_strides_and_memory_layouts_cpu.py)
- **Golden Reference Checkpoint:** [02_tensors_views_strides_and_memory_layouts.safetensors](../../weights/00_foundations/02_tensors_views_strides_and_memory_layouts.safetensors)
- **Previous Topic:** [01. Linear Neuron & Invertible Transforms](01_linear_neuron_and_invertible_transforms.md)
- **Next Topic:** 03. Autograd Mechanics, Complex Wirtinger & Custom Functions *(Upcoming in Category 00)*
