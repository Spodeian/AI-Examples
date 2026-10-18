"""Script to build Topic 02 Category 01 notebook and update Topic 01 notebook navigation link."""

import json
from pathlib import Path


def update_topic_01_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "01_generative_vs_discriminative_classifiers.ipynb"
    )
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cell_8 = nb["cells"][7]
    new_source = []
    for line in cell_8["source"]:
        if "Next: 02. Tree vs Kernel vs Neural Classifiers (Upcoming in Category 01)" in line:
            new_line = line.replace(
                "Next: 02. Tree vs Kernel vs Neural Classifiers (Upcoming in Category 01)",
                "[Next: 02. Tree vs. Kernel vs. Neural Inductive Biases](02_tree_vs_kernel_vs_neural_inductive_biases.ipynb)",
            )
            new_source.append(new_line)
        else:
            new_source.append(line)
    cell_8["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated Topic 01 notebook navigation link.")


def create_topic_02_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "02_tree_vs_kernel_vs_neural_inductive_biases.ipynb"
    )

    cell_1_md = """# Tree vs. Kernel vs. Neural Inductive Biases

[![Track](https://img.shields.io/badge/Track-PyTorch_2.x_%7C_JAX_%7C_Candle_%7C_Kernels-blue.svg)](../../README.md)
[![Parity](https://img.shields.io/badge/Parity-Transitive_%3C_1e--5-success.svg)](../../tests/test_cross_track_parity.py)

### Historical Lineage & Seminal Citations
- Breiman (2001, Random Forests, Machine Learning): Recursive axis-aligned step partitions.
- Schölkopf & Smola (2002, Learning with Kernels, MIT Press): Isotropic RKHS metric smoothness.
- Grinsztajn, Oyallon & Varoquaux (2022, Tabular Deep Learning, NeurIPS): Coordinate rotation sensitivity.

### Problem Statement & Purpose
Investigates the three foundational geometric inductive biases: axis-aligned partitioning (trees), local isotropic metric smoothness (RBF kernels), and adaptive representation learning (MLPs).

### Decision Boundary: When NOT to Use
> [!WARNING]
> Do NOT use axis-aligned trees when features undergo coordinate rotation $X R$; trees produce an inefficient staircase approximation. Use RBF kernels or neural MLPs for rotation equivariance.

### Topic Connections
🔗 Upstream: [01. Generative vs. Discriminative](01_generative_vs_discriminative_classifiers.ipynb) | 🔗 Downstream: [02/07. Differentiable Trees](../../notebooks/02_classical_and_probabilistic_ml/07_differentiable_trees_node_and_nam.ipynb)"""

    cell_2_md = """## Mathematical Foundations & First-Principles Derivations

### 1. Differentiable Soft Decision Tree
For depth $D_{\\text{tree}} = 2$ with $3$ nodes and $4$ leaves, routing uses one-hot matrix $W_{\\text{route}} \\in \\mathbb{R}^{3 \\times D}$:
$$\\mathbf{p}_{\\text{right}} = \\sigma\\left(\\beta (X W_{\\text{route}}^\\top - \\tau)\\right), \\quad \\mathbf{p}_{\\text{left}} = 1 - \\mathbf{p}_{\\text{right}}$$
$$\\mu(x) = [p_{l,0} p_{l,1}, \\; p_{l,0} p_{r,1}, \\; p_{r,0} p_{l,2}, \\; p_{r,0} p_{r,2}], \\quad \\hat{y}_{\\text{tree}}(x) = \\mu(x) W_{\\text{leaf}}$$

### 2. RBF Kernel Ridge Regression & Orthogonal Invariance
For support vectors $S \\in \\mathbb{R}^{M \\times D}$ and dual coefficients $\\alpha \\in \\mathbb{R}^{M \\times C}$:
$$K_{ij} = \\exp\\left(-\\gamma \\|x_i - s_j\\|_2^2\\right), \\quad \\hat{y}_{\\text{kernel}}(x) = K(x, S) \\alpha + b$$
**Theorem:** For orthogonal $R \\in \\mathcal{O}(D)$ ($R^\\top R = I$), $\\|R x - R s\\|_2^2 = \\|x - s\\|_2^2 \\implies K(R x, R S) = K(x, S)$.

### 3. Complexity & Roofline Bounds
- **Soft Tree:** $\\mathcal{O}(B (N_{\\text{nodes}} D + N_{\\text{leaves}} C))$ FLOPs, memory-bandwidth bound.
- **RBF Kernel:** $\\mathcal{O}(B M D + B M C)$ FLOPs, memory-bandwidth bound.
- **Neural MLP:** $\\mathcal{O}(B D H + B H C)$ FLOPs, compute bound (GEMM)."""

    cell_3_code = """import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path.cwd()
while not (repo_root / "core").is_dir() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import core  # noqa: E402

core.setup_env(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = repo_root / "weights" / "01_paradigms_and_dichotomies" / "02_tree_vs_kernel_vs_neural_inductive_biases.safetensors"
weights, ref_in, ref_out, _ = core.parity.load_reference_checkpoint(weights_path, device=device)
print(f"Device: {device} | Dim: {ref_in.shape[-1]} | Checkpoint: {weights_path.name}")"""

    cell_4_code = """from jaxtyping import Float
from torch import Tensor


class SoftDecisionTree(nn.Module):
    def __init__(self, w_route: Float[Tensor, "nodes dim"], tau: Float[Tensor, "*nodes"], beta: float, w_leaf: Float[Tensor, "leaves classes"]) -> None:
        super().__init__()
        self.w_route, self.tau, self.beta, self.w_leaf = nn.Parameter(w_route), nn.Parameter(tau), beta, nn.Parameter(w_leaf)

    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch classes"]:
        p_r = torch.sigmoid(self.beta * (x @ self.w_route.T - self.tau))
        p_l = 1.0 - p_r
        mu = torch.cat([p_l[:, 0:1]*p_l[:, 1:2], p_l[:, 0:1]*p_r[:, 1:2], p_r[:, 0:1]*p_l[:, 2:3], p_r[:, 0:1]*p_r[:, 2:3]], dim=-1)
        return mu @ self.w_leaf


class RBFKernelRidge(nn.Module):
    def __init__(self, s: Float[Tensor, "supp dim"], alpha: Float[Tensor, "supp classes"], gamma: float, bias: Float[Tensor, "*classes"]) -> None:
        super().__init__()
        self.s, self.alpha, self.gamma, self.bias = nn.Parameter(s), nn.Parameter(alpha), gamma, nn.Parameter(bias)

    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch classes"]:
        d_sq = torch.clamp(torch.sum(x**2, -1, keepdim=True) + torch.sum(self.s**2, -1, keepdim=True).T - 2.0 * (x @ self.s.T), min=0.0)
        return torch.exp(-self.gamma * d_sq) @ self.alpha + self.bias


class MLPClassifier(nn.Module):
    def __init__(self, w1: Float[Tensor, "dim hidden"], b1: Float[Tensor, "*hidden"], w2: Float[Tensor, "hidden classes"], b2: Float[Tensor, "*classes"]) -> None:
        super().__init__()
        self.w1, self.b1, self.w2, self.b2 = nn.Parameter(w1), nn.Parameter(b1), nn.Parameter(w2), nn.Parameter(b2)

    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch classes"]:
        return F.relu(x @ self.w1 + self.b1) @ self.w2 + self.b2


tree_model = SoftDecisionTree(weights["tree_routing_weights"], weights["tree_thresholds"], float(weights["tree_temperature"].item()), weights["tree_leaf_weights"]).to(device)
kernel_model = RBFKernelRidge(weights["kernel_support"], weights["kernel_dual_coeffs"], float(weights["kernel_gamma"].item()), weights["kernel_bias"]).to(device)
mlp_model = MLPClassifier(weights["mlp_w1"], weights["mlp_b1"], weights["mlp_w2"], weights["mlp_b2"]).to(device)
print("SoftDecisionTree, RBFKernelRidge, and MLPClassifier initialized.")"""

    cell_5_code = """# 1. 4-Way Transitive Parity against golden checkpoint (< 1e-5)
assert float(torch.max(torch.abs(tree_model(ref_in) - weights["tree_output"])).item()) < 1e-5
assert float(torch.max(torch.abs(kernel_model(ref_in) - weights["kernel_output"])).item()) < 1e-5
assert float(torch.max(torch.abs(mlp_model(ref_in) - weights["mlp_output"])).item()) < 1e-5

# 2. Probability Simplex Invariant
p_r = torch.sigmoid(tree_model.beta * (ref_in @ tree_model.w_route.T - tree_model.tau))
p_l = 1.0 - p_r
mu = torch.cat([p_l[:, 0:1]*p_l[:, 1:2], p_l[:, 0:1]*p_r[:, 1:2], p_r[:, 0:1]*p_l[:, 2:3], p_r[:, 0:1]*p_r[:, 2:3]], dim=-1)
assert torch.allclose(mu.sum(-1), torch.ones(ref_in.shape[0], device=device), atol=1e-6)

# 3. Orthogonal Rotation Equivariance Check on RBF Kernel
angle = math.pi / 4.0
cos_a, sin_a = math.cos(angle), math.sin(angle)
rot_mat = torch.tensor([[cos_a, -sin_a, 0.0, 0.0], [sin_a, cos_a, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
rot_kernel = RBFKernelRidge(kernel_model.s @ rot_mat, kernel_model.alpha, kernel_model.gamma, kernel_model.bias).to(device)
rot_diff = float(torch.max(torch.abs(rot_kernel(ref_in @ rot_mat) - kernel_model(ref_in))).item())
assert rot_diff < 1e-5, f"RBF rotation equivariance error {rot_diff:.2e} exceeds 1e-5"
print("All mathematical invariants (Transitive Parity + Simplex Axiom + Rotation Equivariance) verified.")"""

    cell_6_code = """# Multi-Track Micro-Benchmark Table
def run_tree():
    return tree_model(ref_in)

def run_kernel():
    return kernel_model(ref_in)

def run_mlp():
    return mlp_model(ref_in)

track_fns = {
    "Soft Decision Tree": run_tree,
    "RBF Kernel Ridge": run_kernel,
    "Neural MLP": run_mlp,
}
bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=8)
print(core.benchmark.format_benchmark_table(bench_results))"""

    cell_7_code = """# Pathological Misuse: Coordinate Rotation Shift degrades axis-aligned tree
gen = torch.Generator().manual_seed(42)
x_clean = torch.randn(200, 2, generator=gen, dtype=torch.float32)
y_clean = (x_clean[:, 0] > 0.0).float()

w_clean = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
tree_clean = SoftDecisionTree(w_clean, torch.zeros(3), 20.0, torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32))
acc_clean = ((tree_clean(x_clean)[:, 0] > 0.5).float() == y_clean).float().mean().item()

# 45-degree coordinate rotation
rot2d = torch.tensor([[math.cos(math.pi/4), -math.sin(math.pi/4)], [math.sin(math.pi/4), math.cos(math.pi/4)]], dtype=torch.float32)
acc_rot = ((tree_clean(x_clean @ rot2d)[:, 0] > 0.5).float() == y_clean).float().mean().item()
rot_err = 1.0 - acc_rot

assert rot_err >= 0.20, f"Expected tree rotation error >= 20%, got {rot_err*100:.1f}%"
print(f"[FAILURE DEMONSTRATED] Axis-aligned tree error spiked to {rot_err*100:.1f}% under 45-degree rotation!")

# Correct Pattern: Continuous models verified with core.export compile and export
ok_exp, _ = core.export.verify_torch_export(mlp_model, ref_in)
ok_comp, _ = core.export.verify_torch_compile_fullgraph(mlp_model, ref_in)
assert ok_exp and ok_comp, "Compilation or export verification failed"
print("[CORRECT] Continuous MLP compiled with torch.compile(fullgraph=True) and exported.")"""

    cell_8_md = """### Mathematical & Architectural Synthesis

<svg viewBox="0 0 650 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="5" width="640" height="110" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
  <rect x="25" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="115" y="47" fill="#38bdf8" font-size="11" font-weight="bold" text-anchor="middle">Axis-Aligned Tree</text>
  <rect x="235" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="325" y="47" fill="#10b981" font-size="11" font-weight="bold" text-anchor="middle">RBF Kernel Ridge</text>
  <rect x="445" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#a855f7" stroke-width="1.5"/>
  <text x="535" y="47" fill="#a855f7" font-size="11" font-weight="bold" text-anchor="middle">Neural MLP</text>
  <rect x="25" y="70" width="600" height="32" rx="4" fill="#0f172a" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4 4"/>
  <text x="325" y="90" fill="#f59e0b" font-size="10" text-anchor="middle">Trees excel on unrotated tabular features | Kernels &amp; MLPs preserve metric rotation equivariance</text>
</svg>

### Key Algorithmic Takeaways
1. **Axis-Aligned Inductive Bias:** Decision trees partition space via axis-orthogonal hyperplanes; invariant to monotonic coordinate scaling but fragile under coordinate rotations.
2. **Metric Smoothness Inductive Bias:** RBF kernels enforce local Euclidean decay $\\exp(-\\gamma \\|x - s\\|^2)$ and are provably invariant under orthogonal rotations $R \\in \\mathcal{O}(D)$.
3. **Adaptive Representation Inductive Bias:** Neural networks learn continuous coordinate rotations and manifold alignments dynamically via gradient descent.

---
**Curriculum Navigation:**
&larr; [Previous: 01. Generative vs. Discriminative Classifiers](01_generative_vs_discriminative_classifiers.ipynb) | [Master Index: README.md](../../README.md) | Next: 03. Discrete vs Continuous Latents & Gradient Estimators (Upcoming in Category 01) &rarr;"""

    def make_cell(cell_idx: int, cell_type: str, source: str) -> dict:
        lines = [line + "\n" for line in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
        cell = {
            "cell_type": cell_type,
            "id": f"cell-{cell_idx}",
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return cell

    nb = {
        "cells": [
            make_cell(1, "markdown", cell_1_md),
            make_cell(2, "markdown", cell_2_md),
            make_cell(3, "code", cell_3_code),
            make_cell(4, "code", cell_4_code),
            make_cell(5, "code", cell_5_code),
            make_cell(6, "code", cell_6_code),
            make_cell(7, "code", cell_7_code),
            make_cell(8, "markdown", cell_8_md),
        ],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "orig_nbformat": 4,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created Topic 02 notebook at: {nb_path}")


if __name__ == "__main__":
    update_topic_01_notebook()
    create_topic_02_notebook()
