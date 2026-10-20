"""Generate 8-cell educational notebook for Category 05 Topic 05:
05_kolmogorov_arnold_networks_kan.ipynb in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 05. Kolmogorov-Arnold Networks (KAN): B-Spline KAN, FastKAN & ChebyKAN

- **Category:** 05. Architectures / 01. MLP, MoE, Soups & KAN
- **Lineage & Citations:** Andrey Kolmogorov & Vladimir Arnold (1957) *Kolmogorov-Arnold Representation Theorem*; Ziming Liu et al. (arXiv 2024) *KAN: Kolmogorov-Arnold Networks*; Ziyao Li (arXiv 2024) *FastKAN: Efficient Kolmogorov-Arnold Networks using Radial Basis Functions*; Syed Shakir et al. (2024) *ChebyKAN: Chebyshev Polynomial KAN*.
- **Purpose:** Replaces static node-activations and linear edge-weights with learnable 1D univariate non-linear parameterized activation functions on edges, achieving improved scaling laws and interpretable symbolic representations.
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp](01_residual_mlp_mlp_mixer_and_gmlp.ipynb), [03_building_blocks_and_losses/01_activation_functions_zoo](../../03_building_blocks_and_losses/01_activation_functions_zoo.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth](06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.ipynb), [06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort](../../06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Do not use raw unconstrained high-degree Chebyshev polynomials ($N > 6$) without input squashing (e.g. $\\tanh(x)$), as boundary inputs $|x| > 1.0$ induce severe Runge phenomenon polynomial explosion ($|T_N(x)| \\gg 10^3$) and catastrophic gradient overflow (`NaN`). Do not replace dense linear MLPs with KANs in high-throughput autoregressive LLMs where hardware tensor cores natively optimize large dense matrix multiplications."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Kolmogorov-Arnold Representation Theorem
Any continuous multivariate function on a bounded domain can be written as:
$$f(\\mathbf{x}) = \\sum_{q=0}^{2n} \\Phi_q\\left(\\sum_{p=1}^n \\phi_{q, p}(x_p)\\right)$$
KAN layers generalize this by placing learnable univariate functions on network edges:
$$y_j = \\sum_{i=1}^{D_{\\text{in}}} \\phi_{i, j}(x_i)$$

### 2. ChebyKAN (Orthogonal Chebyshev Basis)
Chebyshev polynomials $T_n(u)$ are defined recursively on support $[-1, 1]$:
$$T_0(u) = 1, \\quad T_1(u) = u, \\quad T_{n+1}(u) = 2u T_n(u) - T_{n-1}(u)$$
With input squashing $u = \\tanh(x) \\in (-1, 1)$, ChebyKAN computes:
$$y_j = \\sum_{i=1}^{D_{\\text{in}}} \\sum_{n=0}^N c_{i, j, n} T_n(\\tanh(x_i))$$

### 3. FastKAN (Gaussian Radial Basis Functions)
FastKAN replaces Cox-de Boor B-splines with Gaussian RBF kernels and a SiLU linear base:
$$\\phi_{i, j}(x) = w_{b, i, j} \\cdot \\text{SiLU}(x) + \\sum_{k=1}^K w_{rbf, i, k, j} \\cdot \\exp\\left(-\\left(\\frac{x - \\mu_k}{h}\\right)^2\\right)$$
This yields $\\mathcal{O}(D_{\\text{in}} \\cdot D_{\\text{out}} \\cdot K)$ FLOPs without recursive spline tree overhead."""
    cells.append(nbf.v4.new_markdown_cell(c2))

    # Cell 3: Code
    c3 = """import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from pathlib import Path
import core

env = core.setup_env(seed=42)

# Load Universal Golden SafeTensors Checkpoint
ckpt_rel = "weights/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, S, D = x.shape
print(f"Loaded KAN checkpoint on {env.device}: X={x.shape}, D={D}, ChebyOrder=4, FastKANGrids=8")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class ChebyKAN(nn.Module):
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features, self.out_features, self.degree = in_features, out_features, degree
        self.c = nn.Parameter(torch.empty(in_features, out_features, degree + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(x)
        T = [torch.ones_like(u), u]
        for _ in range(2, self.degree + 1):
            T.append(2.0 * u * T[-1] - T[-2])
        return torch.einsum("...id,ijd->...j", torch.stack(T, dim=-1), self.c)

class FastKAN(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_grids: int = 8, h: float = 0.2857143):
        super().__init__()
        self.in_features, self.out_features, self.h = in_features, out_features, h
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, num_grids))
        self.base_w = nn.Parameter(torch.empty(in_features, out_features))
        self.rbf_w = nn.Parameter(torch.empty(in_features, num_grids, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.silu(x) @ self.base_w
        diff = x.unsqueeze(-1) - self.grid
        phi = torch.exp(-torch.pow(diff / self.h, 2.0))
        return base + torch.einsum("...ig,igj->...j", phi, self.rbf_w)"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """cheby = ChebyKAN(in_features=D, out_features=D, degree=4)
fastkan = FastKAN(in_features=D, out_features=D, num_grids=8, h=float(weights["fastkan_denominator"]))
with torch.no_grad():
    cheby.c.copy_(weights["cheby_c"])
    fastkan.base_w.copy_(weights["fastkan_base_w"])
    fastkan.rbf_w.copy_(weights["fastkan_rbf_w"])
    fastkan.grid.copy_(weights["fastkan_grid"])

# 1. ChebyKAN Output Parity
out_cheby = cheby(x)
err_cheby = float(torch.max(torch.abs(out_cheby.detach() - weights["chebykan_output"])))
assert err_cheby < 1e-5, f"ChebyKAN parity error {err_cheby:.2e} exceeds 1e-5"

# 2. FastKAN Output Parity
out_fast = fastkan(x)
err_fast = float(torch.max(torch.abs(out_fast.detach() - weights["fastkan_output"])))
assert err_fast < 1e-5, f"FastKAN parity error {err_fast:.2e} exceeds 1e-5"

# 3. Orthogonality Property Check on Chebyshev Polynomials
u_grid = torch.linspace(-0.99, 0.99, 1000)
T_0 = torch.ones_like(u_grid)
T_1 = u_grid
T_2 = 2.0 * u_grid * T_1 - T_0
weight = 1.0 / torch.sqrt(1.0 - u_grid.pow(2.0))
inner_01 = float(torch.sum(T_0 * T_1 * weight) / 1000.0)
inner_12 = float(torch.sum(T_1 * T_2 * weight) / 1000.0)
assert abs(inner_01) < 0.05 and abs(inner_12) < 0.05, "Chebyshev polynomials must be orthogonal!"
print(f"Parity Verified: ChebyKAN={err_cheby:.2e}, FastKAN={err_fast:.2e}")
print(f"Orthogonality Invariant: <T0, T1>={inner_01:.4f}, <T1, T2>={inner_12:.4f}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = r"""import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_cheby = benchmark_fn(lambda: cheby(x), iters=50, name="ChebyKAN (Degree 4)")
b_fast = benchmark_fn(lambda: fastkan(x), iters=50, name="FastKAN (8 RBF Grids)")
print(format_benchmark_table([b_cheby, b_fast]))

# Visualize Chebyshev polynomials T_0 .. T_4 over [-1, 1]
u_vals = torch.linspace(-1.0, 1.0, 100)
T_plot = [torch.ones_like(u_vals), u_vals]
for _ in range(2, 5):
    T_plot.append(2.0 * u_vals * T_plot[-1] - T_plot[-2])

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=100)
for n in range(5):
    ax[0].plot(u_vals.numpy(), T_plot[n].numpy(), label=f"$T_{n}(u)$", lw=1.8)
ax[0].set_title("Chebyshev Polynomial Basis $T_0 \\dots T_4$")
ax[0].set_xlabel("Support $u = \\tanh(x)$")
ax[0].legend(loc="upper center", ncol=3)
ax[0].grid(True, linestyle="--", alpha=0.5)

# FastKAN RBF Basis activation profiles
diff_plot = u_vals.unsqueeze(-1) - fastkan.grid
phi_plot = torch.exp(-torch.pow(diff_plot / fastkan.h, 2.0))
ax[1].plot(u_vals.numpy(), phi_plot.numpy(), lw=1.5)
ax[1].set_title("FastKAN Gaussian RBF Kernels")
ax[1].set_xlabel("Input Domain $x$")
ax[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Runge Phenomenon & Gradient Explosion without Tanh Squashing
# Un-squashed high-degree polynomial on un-normalized inputs |x| > 1 causes severe divergence!

x_unbounded = torch.tensor([2.5, 3.5, 4.0])

# 1. Pathological: Evaluating polynomial directly without tanh squashing
T_bad = [torch.ones_like(x_unbounded), x_unbounded]
for _ in range(2, 7):  # Degree 6
    T_bad.append(2.0 * x_unbounded * T_bad[-1] - T_bad[-2])
poly_unbounded = T_bad[-1]

# 2. Correct (ChebyKAN): Tanh squashes domain to (-1, 1), guaranteeing |T_n(u)| <= 1.0
u_safe = torch.tanh(x_unbounded)
T_safe = [torch.ones_like(u_safe), u_safe]
for _ in range(2, 7):
    T_safe.append(2.0 * u_safe * T_safe[-1] - T_safe[-2])
poly_safe = T_safe[-1]

print(f"Un-squashed Degree-6 Magnitude (Exploded): {poly_unbounded.max().item():.2f}")
print(f"Tanh-Squashed Degree-6 Magnitude (Bounded):  {poly_safe.max().item():.4f}")
assert poly_unbounded.max() > 1000.0, "Unbounded high-degree polynomial must explode!"
assert poly_safe.abs().max() <= 1.0001, "Chebyshev polynomials on [-1, 1] must never exceed 1.0!" """
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Edge-Parameterized Non-Linearity:** KANs shift non-linear activation learning from nodes to edges, replacing fixed point-wise functions with adaptive 1D functions.
2. **Chebyshev Efficiency:** ChebyKAN eliminates B-spline Cox-de Boor tree traversals by evaluating orthogonal Chebyshev polynomials with bounded support.
3. **Domain Support Invariant:** Polynomial basis functions on $[-1, 1]$ strictly require squashing ($\tanh$) or normalization to avoid Runge phenomenon boundary explosion.

---
### Cross-Domain Generalisation Matrix
- **Physics-Informed Neural Networks (PINNs):** Smooth higher-order derivatives of Chebyshev basis functions enable exact boundary-value PDE solution without numerical dissipation.
- **Symbolic Regression & SciML:** Extracting closed-form analytical equations directly from trained KAN edge functions for physical law discovery.
- **Robotics & Adaptive Control:** Approximating highly non-linear joint friction and hysteresis curves with orthogonal spline bases.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [05_kolmogorov_arnold_networks_kan.md](../../markdown/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.md)
- **PyTorch 2.x Module:** [05_kolmogorov_arnold_networks_kan.py](../../pytorch2/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.py)
- **Functional JAX Module:** [05_kolmogorov_arnold_networks_kan.py](../../jax/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.py)
- **Native Rust Candle Track:** [05_kolmogorov_arnold_networks_kan.rs](../../candle/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.rs)
- **Hardware WGSL Kernel:** [05_kolmogorov_arnold_networks_kan.rs](../../kernels/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.rs)
- **Universal Golden SafeTensors:** [05_kolmogorov_arnold_networks_kan.safetensors](../../weights/05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 04. Soup of Experts](04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.ipynb) | [Category Index](../README.md) | [Next Topic: 06. Lossless Deliberation & RevNet](06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "05_kolmogorov_arnold_networks_kan.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
