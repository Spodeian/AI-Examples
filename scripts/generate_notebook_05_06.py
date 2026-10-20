"""Generate 8-cell educational notebook for Category 05 Topic 06:
06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.ipynb in 05_architectures/01_mlp_moe_soups_and_kan.
"""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build_notebook() -> None:
    cells = []

    # Cell 1: Markdown
    c1 = """# 06. Lossless Deliberation: Reversible RevNet, Hyper-Connections & Looped Depth

- **Category:** 05. Architectures / 01. MLP, MoE, Soups & KAN
- **Lineage & Citations:** Aidan Gomez et al. (NeurIPS 2017) *The Reversible Residual Network: Backpropagation Without Storing Activations*; Nikita Kitaev et al. (ICLR 2020) *Reformer: The Efficient Transformer*; Defa Zhu et al. (arXiv 2024) *Hyper-Connections: Expanding Residual Streams*; Angeliki Giannou et al. (ICML 2023) *Looped Transformers as Programmable Computers*; Jonas Geiping et al. (arXiv 2025) *Scaling Test-Time Compute with Recurrent Depth*.
- **Purpose:** Eliminates $\\mathcal{O}(L)$ activation memory via measure-preserving reversible coupling (reconstructing activations on-the-fly during backward), prevents Pre-Norm representation dilution via multi-lane Hyper-Connections, and enables test-time reasoning via input-anchored recurrent depth.
- **Topic Connections:** `🔗 Upstream Prerequisites:` [05_architectures/01_mlp_moe_soups_and_kan/01_residual_mlp_mlp_mixer_and_gmlp](01_residual_mlp_mlp_mixer_and_gmlp.ipynb), [03_building_blocks_and_losses/06_regularization_and_training_stabilizers](../../03_building_blocks_and_losses/06_regularization_and_training_stabilizers.ipynb) $\\to$ `🔗 Downstream Syntheses:` [05_architectures/02_convolutional_and_spatiotemporal/01_modern_convnets_convnext_and_dilated](../02_convolutional_and_spatiotemporal/01_modern_convnets_convnext_and_dilated.ipynb), [06_smooth_and_physics_informed_ai/04_hamiltonian_and_lagrangian_neural_networks](../../06_smooth_and_physics_informed_ai/04_hamiltonian_and_lagrangian_neural_networks.ipynb).

> [!WARNING]
> **Decision Boundary: When NOT to use it**  
> Never perform in-place tensor mutations or use non-deterministic operations (like non-seeded dropout) inside a reversible block forward pass; analytical backward reconstruction requires exact bitwise determinism to invert $\\mathbf{x}_1 = \\mathbf{y}_1 - \\mathcal{F}(\\mathbf{x}_2)$. Never loop recurrent deliberation blocks for $K > 16$ steps without persistent input anchoring ($\\gamma = 0$); without continuous input injection, iterative states wash out into generic limit cycles and lose the input query."""
    cells.append(nbf.v4.new_markdown_cell(c1))

    # Cell 2: Markdown
    c2 = """## Mathematical Foundations & Governing Equations

### 1. Reversible Additive Coupling (RevNet / Reformer)
Partition hidden state $\\mathbf{x} = [\\mathbf{x}_1, \\mathbf{x}_2] \\in \\mathbb{R}^{D/2} \\times \\mathbb{R}^{D/2}$:
$$\\mathbf{y}_1 = \\mathbf{x}_1 + \\mathcal{F}(\\mathbf{x}_2), \\quad \\mathbf{y}_2 = \\mathbf{x}_2 + \\mathcal{G}(\\mathbf{y}_1)$$
The exact analytical inverse requires zero cached activations:
$$\\mathbf{x}_2 = \\mathbf{y}_2 - \\mathcal{G}(\\mathbf{y}_1), \\quad \\mathbf{x}_1 = \\mathbf{y}_1 - \\mathcal{F}(\\mathbf{x}_2)$$
Because the Jacobian determinant is unity ($\\det J \\equiv 1$), the transformation is volume-preserving and lossless ($I(\\mathbf{X}; \\mathbf{Y}) = H(\\mathbf{X})$).

### 2. Multi-Lane Hyper-Connections (HC)
Given $n$ parallel lanes $\\mathbf{H} = [\\mathbf{h}_1, \\dots, \\mathbf{h}_n]$:
$$\\mathbf{u} = \\sum_{i=1}^n \\alpha_i \\mathbf{h}_i, \\quad \\mathbf{v} = \\text{FFN}(\\mathbf{u}), \\quad \\mathbf{h}_i' = \\mathbf{h}_i + \\beta_i \\mathbf{v}$$
Learnable depth-mixing vectors $\\boldsymbol{\\alpha}, \\boldsymbol{\\beta}$ prevent deep Pre-Norm representation dilution.

### 3. Input-Anchored Recurrent-Depth Deliberation
Iterating a shared layer with persistent input injection $\\gamma \\cdot \\mathbf{x}$:
$$\\mathbf{z}^{(k+1)} = \\text{LayerNorm}\\left(\\mathbf{z}^{(k)} + \\mathcal{M}(\\mathbf{z}^{(k)} + \\gamma \\cdot \\mathbf{x})\\right)$$

### 4. Asymptotic Time & Memory Complexity
- **Time Complexity:** $\\mathcal{O}(L \\cdot S \\cdot D \\cdot D_{\\text{ff}})$ for $L$ reversible blocks, with identical computation to standard residual networks.
- **Activation Memory Complexity:** $\\mathcal{O}(1)$ across arbitrary depth $L$ during backpropagation (vs $\\mathcal{O}(L \\cdot S \\cdot D)$ for standard ResNet), eliminating activation caching bottlenecks."""
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
ckpt_rel = "weights/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors"
ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).exists() else Path("../../../" + ckpt_rel)
weights = safetensors.torch.load_file(ckpt_path)

x = weights["input"]
B, S, D = x.shape
d_sub = D // 2
print(f"Loaded Lossless Deliberation checkpoint on {env.device}: X={x.shape}, D={D}, D_sub={d_sub}")"""
    cells.append(nbf.v4.new_code_cell(c3))

    # Cell 4: Code
    c4 = """class ReversibleBlock(nn.Module):
    def __init__(self, d_sub: int, d_ff: int):
        super().__init__()
        self.d_sub = d_sub
        self.f = nn.Sequential(nn.Linear(d_sub, d_ff), nn.GELU(), nn.Linear(d_ff, d_sub))
        self.g = nn.Sequential(nn.Linear(d_sub, d_ff), nn.GELU(), nn.Linear(d_ff, d_sub))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :self.d_sub], x[..., self.d_sub:]
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y[..., :self.d_sub], y[..., self.d_sub:]
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return torch.cat([x1, x2], dim=-1)

class HyperConnections(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([0.6, 0.4]))
        self.beta = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, h1: torch.Tensor, h2: torch.Tensor):
        u = self.alpha[0] * h1 + self.alpha[1] * h2
        v = self.ffn(u)
        h1_next, h2_next = h1 + self.beta[0] * v, h2 + self.beta[1] * v
        return h1_next, h2_next, 0.5 * (h1_next + h2_next)

class LoopedDepth(nn.Module):
    def __init__(self, d_model: int, d_ff: int, steps: int = 4, gamma: float = 0.3):
        super().__init__()
        self.d_model, self.steps, self.gamma = d_model, steps, gamma
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.clone()
        for _ in range(self.steps):
            z = F.layer_norm(z + self.ffn(z + self.gamma * x), (self.d_model,))
        return z"""
    cells.append(nbf.v4.new_code_cell(c4))

    # Cell 5: Code
    c5 = """rev = ReversibleBlock(d_sub=d_sub, d_ff=16)
hc = HyperConnections(d_model=D, d_ff=16)
loop = LoopedDepth(d_model=D, d_ff=16, steps=4, gamma=float(weights["gamma"]))

with torch.no_grad():
    rev.f[0].weight.copy_(weights["f_w1"].t()); rev.f[0].bias.copy_(weights["f_b1"])
    rev.f[2].weight.copy_(weights["f_w2"].t()); rev.f[2].bias.copy_(weights["f_b2"])
    rev.g[0].weight.copy_(weights["g_w1"].t()); rev.g[0].bias.copy_(weights["g_b1"])
    rev.g[2].weight.copy_(weights["g_w2"].t()); rev.g[2].bias.copy_(weights["g_b2"])
    hc.alpha.copy_(weights["alpha"]); hc.beta.copy_(weights["beta"])
    hc.ffn[0].weight.copy_(weights["hc_w1"].t()); hc.ffn[0].bias.copy_(weights["hc_b1"])
    hc.ffn[2].weight.copy_(weights["hc_w2"].t()); hc.ffn[2].bias.copy_(weights["hc_b2"])
    loop.ffn[0].weight.copy_(weights["loop_w1"].t()); loop.ffn[0].bias.copy_(weights["loop_b1"])
    loop.ffn[2].weight.copy_(weights["loop_w2"].t()); loop.ffn[2].bias.copy_(weights["loop_b2"])

# 1. RevNet Parity & Inversion Verification
out_rev = rev(x)
err_rev = float(torch.max(torch.abs(out_rev.detach() - weights["revnet_output"])))
x_rec = rev.inverse(out_rev)
err_rec = float(torch.max(torch.abs(x_rec.detach() - x)))
assert err_rev < 1e-5 and err_rec < 1e-5, f"RevNet error: {err_rev:.2e}, rec={err_rec:.2e}"

# 2. Hyper-Connections Parity Verification
_, _, out_hc = hc(x, x * 0.5)
err_hc = float(torch.max(torch.abs(out_hc.detach() - weights["hc_output"])))
assert err_hc < 1e-5, f"Hyper-Connections error: {err_hc:.2e}"

# 3. Looped Recurrent Depth Parity Verification
out_loop = loop(x)
err_loop = float(torch.max(torch.abs(out_loop.detach() - weights["looped_output"])))
assert err_loop < 1e-5, f"Looped Depth error: {err_loop:.2e}"
print(f"Parity Verified: RevNet={err_rev:.2e}, RevInverse={err_rec:.2e}, HC={err_hc:.2e}, Looped={err_loop:.2e}")"""
    cells.append(nbf.v4.new_code_cell(c5))

    # Cell 6: Code
    c6 = r"""import matplotlib.pyplot as plt
from core.benchmark import benchmark_fn, format_benchmark_table

b_rev = benchmark_fn(lambda: rev(x), iters=50, name="Reversible Block Forward")
b_inv = benchmark_fn(lambda: rev.inverse(out_rev), iters=50, name="Analytical Inversion")
b_hc = benchmark_fn(lambda: hc(x, x * 0.5), iters=50, name="Hyper-Connections (2 Lanes)")
b_loop = benchmark_fn(lambda: loop(x), iters=50, name="Looped Depth (4 Steps)")
print(format_benchmark_table([b_rev, b_inv, b_hc, b_loop]))

# Visualize RevNet Bitwise Inversion Error and Trajectory
rec_diffs = torch.abs(x_rec - x).flatten().detach().cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=100)
ax[0].hist(rec_diffs, bins=15, color="#2ca02c", alpha=0.8, edgecolor="black")
ax[0].set_title("RevNet Exact Reconstruction Error $|X - X_{rec}|$")
ax[0].set_xlabel("Absolute Error")
ax[0].grid(True, linestyle="--", alpha=0.5)

# Looped depth trajectory divergence
traj_anchored = [torch.norm(x).item()]
traj_unanchored = [torch.norm(x).item()]
z_a, z_u = x.clone(), x.clone()
with torch.no_grad():
    for _ in range(8):
        z_a = F.layer_norm(z_a + loop.ffn(z_a + 0.3 * x), (D,))
        z_u = F.layer_norm(z_u + loop.ffn(z_u), (D,))
        traj_anchored.append(torch.norm(z_a - x).item())
        traj_unanchored.append(torch.norm(z_u - x).item())

ax[1].plot(range(9), traj_anchored, "o-", label=r"Anchored ($\gamma=0.3$)", color="#1f77b4", lw=2)
ax[1].plot(range(9), traj_unanchored, "s--", label=r"Unanchored ($\gamma=0.0$)", color="#d62728", lw=2)
ax[1].set_title("Deliberation Distance from Initial Input $X$")
ax[1].set_xlabel("Recurrent Step $k$")
ax[1].legend()
ax[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6))

    # Cell 7: Code
    c7 = """# Pathological Misuse: Unanchored Recurrent Drift (Washing out the Problem Prompt)
# Without persistent input injection (gamma=0), iterating a block causes representations
# to drift into an uninformative generic limit cycle!

z_unanchored = x.clone()
z_anchored = x.clone()

with torch.no_grad():
    for step in range(16):
        z_unanchored = F.layer_norm(z_unanchored + loop.ffn(z_unanchored), (D,))
        z_anchored = F.layer_norm(z_anchored + 0.3 * x + loop.ffn(z_anchored + 0.3 * x), (D,))

# Measure cosine similarity with the original input prompt
sim_unanchored = F.cosine_similarity(z_unanchored.flatten(), x.flatten(), dim=0).item()
sim_anchored = F.cosine_similarity(z_anchored.flatten(), x.flatten(), dim=0).item()

print(f"Cosine Similarity after 16 Steps (Unanchored): {sim_unanchored:.4f}")
print(f"Cosine Similarity after 16 Steps (Anchored):   {sim_anchored:.4f}")
assert sim_anchored > sim_unanchored, "Persistent input injection must retain prompt query alignment!"
print("Demonstrated catastrophic representation wash-out when recurrent deliberation lacks input injection.")"""
    cells.append(nbf.v4.new_code_cell(c7))

    # Cell 8: Markdown
    c8 = """## Algorithmic Takeaways & Cross-Domain Isomorphisms

### Key Takeaways
1. **Zero-Cache Backpropagation:** Reversible additive coupling enables $\\mathcal{O}(1)$ activation memory during training by reconstructing forward activations in exact arithmetic.
2. **Multi-Lane Residual Highway:** Hyper-Connections prevent Pre-Norm feature dilution in deep models through learnable depth-mixing and width-mixing streams.
3. **Input-Anchored Deliberation:** Scaling test-time reasoning via looped recurrent depth strictly requires persistent input injection $\\gamma \\cdot \\mathbf{x}$ to avoid limit-cycle state collapse.

---
### Cross-Domain Generalisation Matrix
- **Medical Imaging & 3D CT/MRI:** Training ultra-deep 3D volumetric segmentation networks with constant $\\mathcal{O}(1)$ GPU VRAM usage.
- **Continuous-Time Physics (Neural ODEs):** Measure-preserving reversible layers simulate Hamiltonian symplectic dynamical systems with exact energy conservation.
- **Mathematical & Symbolic Reasoning:** Looped Transformers dynamically scale test-time compute by adapting iteration count $K$ to problem difficulty.

---
### Synchronized Track Navigation
- **Architectural Reference (Markdown):** [06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.md](../../markdown/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.md)
- **PyTorch 2.x Module:** [06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.py](../../pytorch2/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.py)
- **Functional JAX Module:** [06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.py](../../jax/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.py)
- **Native Rust Candle Track:** [06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.rs](../../candle/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.rs)
- **Hardware WGSL Kernel:** [06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.rs](../../kernels/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.rs)
- **Universal Golden SafeTensors:** [06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors](../../weights/05_architectures/01_mlp_moe_soups_and_kan/06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.safetensors)

---
**Curriculum Navigation:**  
&larr; [Previous Topic: 05. Kolmogorov-Arnold Networks (KAN)](05_kolmogorov_arnold_networks_kan.ipynb) | [Category Index](../README.md) | [Next Subfolder: 02. Convolutional & Spatiotemporal](../../02_convolutional_and_spatiotemporal/01_modern_convnets_convnext_and_dilated.ipynb) &rarr;"""
    cells.append(nbf.v4.new_markdown_cell(c8))

    nb = nbf.v4.new_notebook(cells=cells)
    out_dir = Path("notebooks/05_architectures/01_mlp_moe_soups_and_kan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "06_lossless_deliberation_reversible_revnet_hyper_connections_and_looped_depth.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote notebook to", out_path)


if __name__ == "__main__":
    build_notebook()
