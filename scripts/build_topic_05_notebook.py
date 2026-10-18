"""Script to build Topic 05 Category 01 notebook and update Topic 04 notebook navigation link."""

import json
from pathlib import Path


def update_topic_04_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "04_deterministic_vs_stochastic_and_mixture_density_networks.ipynb"
    )
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cell_8 = nb["cells"][7]
    new_source = []
    for line in cell_8["source"]:
        if "Next: 05. Parametric vs. Nonparametric & Gaussian Process Priors (Upcoming in Category 01)" in line:
            new_line = line.replace(
                "Next: 05. Parametric vs. Nonparametric & Gaussian Process Priors (Upcoming in Category 01)",
                "[Next: 05. Parametric vs. Nonparametric & Neural Processes](05_parametric_vs_nonparametric_and_neural_processes.ipynb)",
            )
            new_source.append(new_line)
        else:
            new_source.append(line)
    cell_8["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated Topic 04 notebook navigation link.")


def create_topic_05_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "05_parametric_vs_nonparametric_and_neural_processes.ipynb"
    )

    cell_1_md = r"""# Parametric vs. Non-Parametric & Neural Processes

[![Track](https://img.shields.io/badge/Track-PyTorch_2.x_%7C_JAX_%7C_Candle_%7C_Kernels-blue.svg)](../../README.md)
[![Parity](https://img.shields.io/badge/Parity-Transitive_%3C_1e--5-success.svg)](../../tests/test_cross_track_parity.py)

### Historical Lineage & Seminal Citations
- Fix & Hodges (1951, Nonparametric Discrimination, USAF School of Aviation Medicine): Foundational formulation of non-parametric local estimation.
- Rasmussen & Williams (2006, Gaussian Processes for Machine Learning, MIT Press): Exact Bayesian kernelized non-parametrics.
- Garnelo et al. (2018, Conditional Neural Processes, ICML): Permutation-invariant neural stochastic processes.
- Kim et al. (2019, Attentive Neural Processes, ICLR): Resolving CNP underfitting via cross-attention over context points.

### Problem Statement & Purpose
Investigates the trade-off between fixed-weight parametric models ($\mathcal{O}(1)$ memory, bounded capacity), exact non-parametric models (infinite capacity, $\mathcal{O}(N^3)$ Cholesky scaling), and semi-parametric Neural Processes that achieve dynamic context conditioning in $\mathcal{O}(C \cdot T)$ compute with $\mathcal{O}(1)$ weights.

### Decision Boundary: When NOT to Use
> [!WARNING]
> Do NOT use exact Gaussian Processes when context size $C > 10^4$; the $\mathcal{O}(C^3)$ Cholesky inversion and $\mathcal{O}(C^2)$ memory become computationally prohibitive. Do NOT use mean-pooled CNPs when the target function contains localized high-frequency features; mean-pooling washes out spatial context. Use Attentive Neural Processes (ANP) for scalable, localized context adaptation.

### Topic Connections
🔗 Upstream: [04. Deterministic vs. Stochastic & MDN](04_deterministic_vs_stochastic_and_mixture_density_networks.ipynb) | 🔗 Downstream: [06. Inductive vs. Transductive](06_inductive_vs_transductive_and_zero_shot_generalization.ipynb)"""

    cell_2_md = r"""## Mathematical Foundations & First-Principles Derivations

### 1. Exact Gaussian Process Posterior
Given context $\mathcal{D}_C = \{(x_c, y_c)\}_{c=1}^C$ and target query points $X_t$:
$$\mu(X_t) = K(X_t, X_c) \left(K(X_c, X_c) + \sigma_n^2 I\right)^{-1} Y_c$$
Complexity: $\mathcal{O}(C^3)$ for inversion / linear solve, $\mathcal{O}(C \cdot T)$ for cross-kernel evaluation.

### 2. Semi-Parametric Attentive Neural Process (Kim et al. 2019)
Context points are encoded: $v_c = \text{ReLU}([x_c, y_c] W_{\text{enc}} + b_{\text{enc}})$.
Cross-attention weights from target queries $X_t$ to context keys $X_c$:
$$A = \text{Softmax}\left(\frac{(X_t W_q)(X_c W_k)^\top}{\sqrt{D_h}}\right) \in \mathbb{R}^{T \times C}$$
The target representation $R_{\text{anp}} = A (V_c W_v)$ conditions the decoder:
$$\hat{Y}_{\text{anp}} = [X_t, R_{\text{anp}}] W_{\text{dec}} + b_{\text{dec}}$$
Complexity: $\mathcal{O}(C \cdot T \cdot D_h)$ compute, $\mathcal{O}(C + T)$ memory, $\mathcal{O}(1)$ learned parameters!"""

    cell_3_code = """import sys
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
weights_path = repo_root / "weights" / "01_paradigms_and_dichotomies" / "05_parametric_vs_nonparametric_and_neural_processes.safetensors"
weights, ref_in, ref_out, _ = core.parity.load_reference_checkpoint(weights_path, device=device)
print(f"Device: {device} | Dim: {ref_in.shape[-1]} | Checkpoint: {weights_path.name}")"""

    cell_4_code = """from jaxtyping import Float
from torch import Tensor


class ParametricMLP(nn.Module):
    def __init__(self, w1: Float[Tensor, "dx dh"], b1: Float[Tensor, "*dh"], w2: Float[Tensor, "dh dy"], b2: Float[Tensor, "*dy"]) -> None:
        super().__init__()
        self.w1, self.b1, self.w2, self.b2 = nn.Parameter(w1), nn.Parameter(b1), nn.Parameter(w2), nn.Parameter(b2)

    def forward(self, xt: Float[Tensor, "b nt dx"]) -> Float[Tensor, "b nt dy"]:
        return F.relu(xt @ self.w1 + self.b1) @ self.w2 + self.b2


class ExactGaussianProcess(nn.Module):
    def __init__(self, alpha: Float[Tensor, "b nc dy"], lengthscale: float = 1.0) -> None:
        super().__init__()
        self.alpha, self.ls = nn.Parameter(alpha), lengthscale

    def forward(self, xt: Float[Tensor, "b nt dx"], xc: Float[Tensor, "b nc dx"]) -> Float[Tensor, "b nt dy"]:
        xt_sq = torch.sum(xt**2, dim=-1, keepdim=True)
        xc_sq = torch.sum(xc**2, dim=-1, keepdim=True)
        dist_sq = torch.clamp(xt_sq + xc_sq.transpose(-2, -1) - 2.0 * (xt @ xc.transpose(-2, -1)), min=0.0)
        k_tc = torch.exp(-0.5 * dist_sq / (self.ls**2))
        return k_tc @ self.alpha


class AttentiveNeuralProcess(nn.Module):
    def __init__(self, w_enc: Float[Tensor, "d_in dh"], b_enc: Float[Tensor, "*dh"], w_q: Float[Tensor, "dx dh"], w_k: Float[Tensor, "dx dh"], w_v: Float[Tensor, "dh dh"], w_dec: Float[Tensor, "d_dec dy"], b_dec: Float[Tensor, "*dy"]) -> None:
        super().__init__()
        self.w_enc, self.b_enc = nn.Parameter(w_enc), nn.Parameter(b_enc)
        self.w_q, self.w_k, self.w_v = nn.Parameter(w_q), nn.Parameter(w_k), nn.Parameter(w_v)
        self.w_dec, self.b_dec = nn.Parameter(w_dec), nn.Parameter(b_dec)

    def forward(self, xc: Float[Tensor, "b nc dx"], yc: Float[Tensor, "b nc dy"], xt: Float[Tensor, "b nt dx"]) -> tuple[Float[Tensor, "b nt dy"], Float[Tensor, "b nt nc"]]:
        v_c = F.relu(torch.cat([xc, yc], dim=-1) @ self.w_enc + self.b_enc)
        q = xt @ self.w_q
        k = xc @ self.w_k
        v = v_c @ self.w_v
        scale = 1.0 / (float(q.shape[-1]) ** 0.5)
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        r = attn @ v
        out = torch.cat([xt, r], dim=-1) @ self.w_dec + self.b_dec
        return out, attn


mlp = ParametricMLP(weights["w_mlp1"], weights["b_mlp1"], weights["w_mlp2"], weights["b_mlp2"]).to(device)
gp = ExactGaussianProcess(weights["alpha"], float(weights["gp_lengthscale"].item())).to(device)
anp = AttentiveNeuralProcess(weights["w_enc"], weights["b_enc"], weights["w_q"], weights["w_k"], weights["w_v"], weights["w_dec"], weights["b_dec"]).to(device)
print("ParametricMLP, ExactGaussianProcess, and AttentiveNeuralProcess initialized.")"""

    cell_5_code = """# 1. 4-Way Transitive Parity (< 1e-5) against golden checkpoint
xc, yc, xt = weights["x_context"], weights["y_context"], weights["x_target"]

y_mlp = mlp(xt)
assert float(torch.max(torch.abs(y_mlp - weights["y_parametric"])).item()) < 1e-5

y_gp = gp(xt, xc)
assert float(torch.max(torch.abs(y_gp - weights["y_gp"])).item()) < 1e-5

y_anp, attn_w = anp(xc, yc, xt)
assert float(torch.max(torch.abs(y_anp - weights["y_anp"])).item()) < 1e-5
assert float(torch.max(torch.abs(attn_w - weights["attn_weights"])).item()) < 1e-5

# 2. Mathematical Invariants: Attention Simplex Axiom
assert torch.allclose(attn_w.sum(-1), torch.ones(attn_w.shape[0], attn_w.shape[1], device=device), atol=1e-5)
print("All mathematical invariants (Transitive Parity + Attention Simplex) verified.")"""

    cell_6_code = """# Multi-Track Micro-Benchmark Table
def run_mlp():
    return mlp(xt)

def run_gp():
    return gp(xt, xc)

def run_anp():
    return anp(xc, yc, xt)

track_fns = {
    "Parametric MLP": run_mlp,
    "Exact Gaussian Process": run_gp,
    "Attentive Neural Process": run_anp,
}
bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=8)
print(core.benchmark.format_benchmark_table(bench_results))"""

    cell_7_code = """# Pathological Misuse: Uniform Mean Pooling Bottleneck (CNP vs. ANP)
gen = torch.Generator(device=device).manual_seed(42)
# Reconstruct context points: xt = xc
v_c = F.relu(torch.cat([xc, yc], dim=-1) @ anp.w_enc + anp.b_enc)

# Vanilla CNP: Uniform mean pooling across all context points
r_cnp = torch.mean(v_c, dim=1, keepdim=True).expand(-1, xc.shape[1], -1)
y_cnp_rec = torch.cat([xc, r_cnp], dim=-1) @ anp.w_dec + anp.b_dec
err_cnp = (y_cnp_rec - yc).abs().mean().item()

# Attentive NP: Target queries context via cross-attention
y_anp_rec, attn_rec = anp(xc, yc, xc)
err_anp = (y_anp_rec - yc).abs().mean().item()

assert err_cnp >= 0.40, f"Expected CNP pooling error >= 0.40, got {err_cnp:.3f}"
print(f"[FAILURE DEMONSTRATED] CNP uniform mean-pooling underfitted context points with reconstruction error {err_cnp:.3f}!")

# Correct Pattern: Export verification on Parametric model
sample_input = xt[:, 0, :]
class SimpleMLP(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m
    def forward(self, x: Tensor) -> Tensor:
        return self.m(x.unsqueeze(1)).squeeze(1)

export_m = SimpleMLP(mlp)
ok_exp, _ = core.export.verify_torch_export(export_m, sample_input)
ok_comp, _ = core.export.verify_torch_compile_fullgraph(export_m, sample_input)
assert ok_exp and ok_comp, "Compilation or export verification failed"
print("[CORRECT] Parametric Model compiled with torch.compile(fullgraph=True) and exported.")"""

    cell_8_md = r"""### Mathematical & Architectural Synthesis

<svg viewBox="0 0 650 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="5" width="640" height="110" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
  <rect x="25" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="115" y="47" fill="#38bdf8" font-size="11" font-weight="bold" text-anchor="middle">Parametric MLP</text>
  <rect x="235" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="325" y="47" fill="#10b981" font-size="11" font-weight="bold" text-anchor="middle">Gaussian Process</text>
  <rect x="445" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#a855f7" stroke-width="1.5"/>
  <text x="535" y="47" fill="#a855f7" font-size="11" font-weight="bold" text-anchor="middle">Attentive NP (ANP)</text>
  <rect x="25" y="70" width="600" height="32" rx="4" fill="#0f172a" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4 4"/>
  <text x="325" y="90" fill="#f59e0b" font-size="10" text-anchor="middle">MLP: O(1) memory, fixed capacity | GP: O(C^3) exact Bayes | ANP: O(C*T) cross-attention adaptation</text>
</svg>

### Key Algorithmic Takeaways
1. **Parametric Capacity Bound:** Fixed-weight neural networks cannot condition on arbitrary runtime context points without retraining or fine-tuning.
2. **Non-Parametric Cholesky Bottleneck:** Gaussian Processes offer exact Bayesian posterior conditioning but scale cubically $\mathcal{O}(C^3)$ with context size.
3. **Semi-Parametric Cross-Attention:** Attentive Neural Processes resolve the mean-pooling bottleneck of CNPs, achieving localized in-context adaptation in $\mathcal{O}(C \cdot T)$ compute with constant $\mathcal{O}(1)$ parameter memory.

---
**Curriculum Navigation:**
&larr; [Previous: 04. Deterministic vs. Stochastic & MDN](04_deterministic_vs_stochastic_and_mixture_density_networks.ipynb) | [Master Index: README.md](../../README.md) | Next: 06. Supervised vs RL vs Self-Supervised Learning (Upcoming in Category 01) &rarr;"""

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
    print(f"Created Topic 05 notebook at: {nb_path}")


if __name__ == "__main__":
    update_topic_04_notebook()
    create_topic_05_notebook()
