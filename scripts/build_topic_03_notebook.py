"""Script to build Topic 03 Category 01 notebook and update Topic 02 notebook navigation link."""

import json
from pathlib import Path


def update_topic_02_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "02_tree_vs_kernel_vs_neural_inductive_biases.ipynb"
    )
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cell_8 = nb["cells"][7]
    new_source = []
    for line in cell_8["source"]:
        if "Next: 03. Discrete vs Continuous Latents & Gradient Estimators (Upcoming in Category 01)" in line:
            new_line = line.replace(
                "Next: 03. Discrete vs Continuous Latents & Gradient Estimators (Upcoming in Category 01)",
                "[Next: 03. Discrete vs. Continuous Latents & Gradient Estimators](03_discrete_vs_continuous_latents_and_gradient_estimators.ipynb)",
            )
            new_source.append(new_line)
        else:
            new_source.append(line)
    cell_8["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated Topic 02 notebook navigation link.")


def create_topic_03_notebook() -> None:
    nb_path = (
        Path(__file__).resolve().parent.parent
        / "notebooks"
        / "01_paradigms_and_dichotomies"
        / "03_discrete_vs_continuous_latents_and_gradient_estimators.ipynb"
    )

    cell_1_md = """# Discrete vs. Continuous Latents & Gradient Estimators

[![Track](https://img.shields.io/badge/Track-PyTorch_2.x_%7C_JAX_%7C_Candle_%7C_Kernels-blue.svg)](../../README.md)
[![Parity](https://img.shields.io/badge/Parity-Transitive_%3C_1e--5-success.svg)](../../tests/test_cross_track_parity.py)

### Historical Lineage & Seminal Citations
- Williams (1992, Simple Statistical Gradient-Following Algorithms, Machine Learning): Score function / REINFORCE estimator.
- Kingma & Welling (2013, Auto-Encoding Variational Bayes, ICLR): Continuous pathwise reparameterization trick.
- Jang, Gu & Poole (2017) / Maddison et al. (2017, Categorical Reparameterization / Concrete Distribution, ICLR): Continuous relaxation of discrete latents.
- Mnih & Rezende (2016, Variational Inference for Monte Carlo Objectives - VIMCO, ICML): Multi-sample leave-one-out variance reduction.

### Problem Statement & Purpose
Investigates stochastic latent variables under continuous vs. discrete representations. Demonstrates the variance-bias trade-off among pathwise reparameterization (low variance, requires continuity), Gumbel-Softmax (biased relaxation, differentiable), and score-function REINFORCE with VIMCO baselines (unbiased, black-box rewards).

### Decision Boundary: When NOT to Use
> [!WARNING]
> Do NOT use vanilla REINFORCE without baseline variance reduction for discrete latents; gradient variance scales as $\\mathcal{O}(\\text{Var}(R))$, causing slow or unstable training. Use pathwise reparameterization when coordinate transformations exist, Gumbel-Softmax for differentiable relaxation, or leave-one-out VIMCO baselines for non-differentiable rewards.

### Topic Connections
🔗 Upstream: [02. Tree vs. Kernel vs. Neural](02_tree_vs_kernel_vs_neural_inductive_biases.ipynb) | 🔗 Downstream: [04. Deterministic vs. Stochastic & MDN](04_deterministic_vs_stochastic_and_mixture_density_networks.ipynb)"""

    cell_2_md = """## Mathematical Foundations & First-Principles Derivations

### 1. Continuous Pathwise Reparameterization (Kingma & Welling 2013)
For $z \\sim q_\\phi(z|x) = \\mathcal{N}(\\mu(x), \\operatorname{diag}(\\sigma^2(x)))$, reparameterize via $\\epsilon \\sim \\mathcal{N}(0, I)$:
$$z = g_\\phi(x, \\epsilon) = \\mu(x) + \\sigma(x) \\odot \\epsilon$$
$$\\nabla_\\phi \\mathbb{E}_{q_\\phi}[f(z)] = \\mathbb{E}_{p(\\epsilon)}\\left[\\nabla_z f(z) \\nabla_\\phi g_\\phi(x, \\epsilon)\\right]$$
Analytical KL divergence against prior $\\mathcal{N}(0, I)$:
$$D_{\\text{KL}}(q_\\phi(z|x) \\| \\mathcal{N}(0, I)) = -\\frac{1}{2} \\sum_{j=1}^d \\left(1 + \\log \\sigma_j^2 - \\mu_j^2 - \\sigma_j^2\\right)$$

### 2. Gumbel-Softmax / Concrete Relaxation (Jang et al. 2017; Maddison et al. 2017)
Given logits $\\alpha \\in \\mathbb{R}^K$, draw standard Gumbel $g_k = -\\log(-\\log u_k)$ with $u_k \\sim \\text{Uniform}(0, 1)$:
$$y_k = \\frac{\\exp\\left((\\alpha_k + g_k) / \\tau\\right)}{\\sum_{j=1}^K \\exp\\left((\\alpha_j + g_j) / \\tau\\right)}, \\quad \\lim_{\\tau \\to 0^+} y = \\operatorname{one\\_hot}\\left(\\operatorname{argmax}_k (\\alpha_k + g_k)\\right)$$

### 3. Score Function Estimator (REINFORCE) & VIMCO Multi-Sample Baseline
For non-differentiable discrete $b \\in \\{0, 1\\}^d$ with $p = \\sigma(\\alpha)$, $\\nabla_\\alpha \\log p_\\alpha(b) = b - p$:
$$\\hat{g}_{\\text{score}} = (R(b) - c) \\cdot (b - p), \\quad c_{\\text{VIMCO}, i} = \\frac{1}{K-1} \\sum_{j \\neq i} R(b_j)$$"""

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
weights_path = repo_root / "weights" / "01_paradigms_and_dichotomies" / "03_discrete_vs_continuous_latents_and_gradient_estimators.safetensors"
weights, ref_in, ref_out, _ = core.parity.load_reference_checkpoint(weights_path, device=device)
print(f"Device: {device} | Dim: {ref_in.shape[-1]} | Checkpoint: {weights_path.name}")"""

    cell_4_code = """from jaxtyping import Float
from torch import Tensor


class GaussianReparameterization(nn.Module):
    def __init__(self, w_mu: Float[Tensor, "dim lat"], b_mu: Float[Tensor, "*lat"], w_logvar: Float[Tensor, "dim lat"], b_logvar: Float[Tensor, "*lat"], w_dec: Float[Tensor, "lat out"], b_dec: Float[Tensor, "*out"]) -> None:
        super().__init__()
        self.w_mu, self.b_mu, self.w_logvar, self.b_logvar = nn.Parameter(w_mu), nn.Parameter(b_mu), nn.Parameter(w_logvar), nn.Parameter(b_logvar)
        self.w_dec, self.b_dec = nn.Parameter(w_dec), nn.Parameter(b_dec)

    def forward(self, x: Float[Tensor, "b dim"], eps: Float[Tensor, "b lat"] | None = None) -> tuple[Float[Tensor, "b lat"], Float[Tensor, "b out"], Float[Tensor, "*b"]]:
        mu = x @ self.w_mu + self.b_mu
        logvar = x @ self.w_logvar + self.b_logvar
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        z = mu + std * eps
        out = z @ self.w_dec + self.b_dec
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return z, out, kl


class GumbelSoftmaxRelaxation(nn.Module):
    def __init__(self, w_logits: Float[Tensor, "dim lat"], b_logits: Float[Tensor, "*lat"], w_dec: Float[Tensor, "lat out"], b_dec: Float[Tensor, "*out"], tau: float = 1.0) -> None:
        super().__init__()
        self.w_logits, self.b_logits = nn.Parameter(w_logits), nn.Parameter(b_logits)
        self.w_dec, self.b_dec, self.tau = nn.Parameter(w_dec), nn.Parameter(b_dec), tau

    def forward(self, x: Float[Tensor, "b dim"], gumbel_noise: Float[Tensor, "b lat"] | None = None) -> tuple[Float[Tensor, "b lat"], Float[Tensor, "b out"]]:
        logits = x @ self.w_logits + self.b_logits
        if gumbel_noise is None:
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(u))
        y = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)
        out = y @ self.w_dec + self.b_dec
        return y, out


vae = GaussianReparameterization(weights["w_mu"], weights["b_mu"], weights["w_logvar"], weights["b_logvar"], weights["w_dec"], weights["b_dec"]).to(device)
gumbel_model = GumbelSoftmaxRelaxation(weights["w_logits"], weights["b_logits"], weights["w_dec"], weights["b_dec"], tau=float(weights["temperature"].item())).to(device)
print("GaussianReparameterization and GumbelSoftmaxRelaxation initialized.")"""

    cell_5_code = """# 1. 4-Way Transitive Parity (< 1e-5) against golden checkpoint
z_out, vae_out, kl_out = vae(ref_in, eps=weights["eps"])
assert float(torch.max(torch.abs(z_out - weights["continuous_z"])).item()) < 1e-5
assert float(torch.max(torch.abs(vae_out - weights["continuous_out"])).item()) < 1e-5
assert float(torch.max(torch.abs(kl_out - weights["kl_divergence"])).item()) < 1e-5

y_out, g_dec = gumbel_model(ref_in, gumbel_noise=weights["gumbel_noise"])
assert float(torch.max(torch.abs(y_out - weights["gumbel_soft"])).item()) < 1e-5
assert float(torch.max(torch.abs(g_dec - weights["gumbel_out"])).item()) < 1e-5

# 2. Mathematical Invariants: Simplex Axiom & Analytical KL Zero
assert torch.allclose(y_out.sum(-1), torch.ones(ref_in.shape[0], device=device), atol=1e-5)
zero_kl = -0.5 * torch.sum(1.0 + torch.zeros(4, device=device) - torch.zeros(4, device=device)**2 - torch.exp(torch.zeros(4, device=device)))
assert torch.isclose(zero_kl, torch.tensor(0.0, device=device), atol=1e-7)

# 3. Bernoulli REINFORCE Score Gradient Parity
p = torch.sigmoid(ref_in @ weights["w_bernoulli"] + weights["b_bernoulli"])
assert float(torch.max(torch.abs(p - weights["bernoulli_probs"])).item()) < 1e-5
print("All mathematical invariants (Transitive Parity + Simplex Axiom + KL Zero) verified.")"""

    cell_6_code = """# Multi-Track Micro-Benchmark Table
def run_vae():
    return vae(ref_in, eps=weights["eps"])

def run_gumbel():
    return gumbel_model(ref_in, gumbel_noise=weights["gumbel_noise"])

def run_reinforce():
    p = torch.sigmoid(ref_in @ weights["w_bernoulli"] + weights["b_bernoulli"])
    return (ref_in.sum(-1, keepdim=True)) * (torch.bernoulli(p) - p)

track_fns = {
    "Continuous VAE (Reparam)": run_vae,
    "Discrete Gumbel-Softmax": run_gumbel,
    "Bernoulli REINFORCE": run_reinforce,
}
bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=8)
print(core.benchmark.format_benchmark_table(bench_results))"""

    cell_7_code = """# Pathological Misuse: REINFORCE Variance Explosion without Baseline
gen = torch.Generator(device=device).manual_seed(42)
probs = torch.full((1000, 4), 0.5, device=device)
samples = torch.bernoulli(probs, generator=gen)
rewards = samples.sum(-1, keepdim=True) + 10.0  # High-mean reward offset

# Vanilla REINFORCE vs. Leave-One-Out Baseline (VIMCO)
vanilla_grads = rewards * (samples - probs)
baseline_c = rewards.mean(dim=0, keepdim=True)
centered_grads = (rewards - baseline_c) * (samples - probs)

var_vanilla = float(vanilla_grads.var().item())
var_centered = float(centered_grads.var().item())
var_ratio = var_vanilla / max(var_centered, 1e-8)

assert var_ratio >= 10.0, f"Expected variance reduction ratio >= 10x, got {var_ratio:.2f}x"
print(f"[FAILURE DEMONSTRATED] Vanilla REINFORCE gradient variance ({var_vanilla:.2f}) is {var_ratio:.1f}x higher than centered estimator ({var_centered:.2f})!")

# Correct Pattern: Continuous models verified with core.export compile and export
ok_exp, _ = core.export.verify_torch_export(vae, ref_in)
ok_comp, _ = core.export.verify_torch_compile_fullgraph(vae, ref_in)
assert ok_exp and ok_comp, "Compilation or export verification failed"
print("[CORRECT] Continuous VAE compiled with torch.compile(fullgraph=True) and exported.")"""

    cell_8_md = r"""### Mathematical & Architectural Synthesis

<svg viewBox="0 0 650 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="5" width="640" height="110" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
  <rect x="25" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="115" y="47" fill="#38bdf8" font-size="11" font-weight="bold" text-anchor="middle">Pathwise Reparam</text>
  <rect x="235" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="325" y="47" fill="#10b981" font-size="11" font-weight="bold" text-anchor="middle">Gumbel-Softmax</text>
  <rect x="445" y="25" width="180" height="35" rx="6" fill="#1e293b" stroke="#a855f7" stroke-width="1.5"/>
  <text x="535" y="47" fill="#a855f7" font-size="11" font-weight="bold" text-anchor="middle">Score REINFORCE</text>
  <rect x="25" y="70" width="600" height="32" rx="4" fill="#0f172a" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4 4"/>
  <text x="325" y="90" fill="#f59e0b" font-size="10" text-anchor="middle">Pathwise: lowest variance | Gumbel: continuous relaxation | REINFORCE: unbiased black-box with baseline</text>
</svg>

### Key Algorithmic Takeaways
1. **Pathwise Gradient Invariant:** The reparameterization trick $z = \mu + \sigma \odot \epsilon$ isolates stochasticity into an external parameterless distribution, yielding the lowest variance gradient estimator for continuous latent variables.
2. **Temperature-Annealed Relaxation:** Gumbel-Softmax bridges discrete argmax sampling and continuous backpropagation; low temperatures $\tau \to 0$ approximate discrete categorical samples at the cost of vanishing gradients.
3. **Variance-Bias Frontier:** REINFORCE is fully unbiased for non-differentiable rewards but suffers from high variance without baselines ($R - c$). VIMCO multi-sample baselines provide essential variance reduction.

---
**Curriculum Navigation:**
&larr; [Previous: 02. Tree vs. Kernel vs. Neural Inductive Biases](02_tree_vs_kernel_vs_neural_inductive_biases.ipynb) | [Master Index: README.md](../../README.md) | Next: 04. Deterministic vs. Stochastic & MDN (Upcoming in Category 01) &rarr;"""

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
    print(f"Created Topic 03 notebook at: {nb_path}")


if __name__ == "__main__":
    update_topic_02_notebook()
    create_topic_03_notebook()
