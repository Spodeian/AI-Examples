"""Generate the 8-cell educational notebook for Category 04 Topic 11 (Subfolder 02 Topic 04):
Combinatorial and Discrete: Simulated Annealing, Parallel Tempering, and Sinkhorn."""

import json
from pathlib import Path


def build_notebook() -> dict:
    cells = [
        # Cell 0: Header Markdown
        {
            "id": "cell_0",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Combinatorial and Discrete: Simulated Annealing, Parallel Tempering, and Sinkhorn (C04 Topic 11)\n",
                "\n",
                "**Citations:** Kirkpatrick et al. (1983 Simulated Annealing); Swendsen & Wang (1986 Parallel Tempering); Rubinstein (1999 CEM); Mena et al. (2018 Gumbel-Sinkhorn).\n",
                "\n",
                "### Purpose & Topic Bridges\n",
                "In machine learning problems over discrete or non-differentiable domains (e.g. token alignments, bipartite matching, neural architecture search, combinatorial routing), gradient descent cannot be directly applied without continuous relaxations or blackbox stochastic sampling.\n",
                "\n",
                "**Gumbel-Sinkhorn** solves this by continuously relaxing discrete permutation matrices onto the Birkhoff polytope via entropy-regularized optimal transport with backpropagatable gradients. **Simulated Annealing (SA)** and **Replica-Exchange Parallel Tempering** use Boltzmann distributions and temperature ladders to escape local minima in glassy, rugged landscapes. The **Cross-Entropy Method (CEM)** optimizes complex blackbox systems by iteratively refitting probability distributions over top-performing elite candidates.\n",
                "\n",
                "**Decision Boundary:** Use **Gumbel-Sinkhorn** for end-to-end differentiable assignment, sorting, or matching layers. Use **Parallel Tempering** when optimizing rugged discrete/blackbox energy landscapes with deep barriers. Use **CEM** for policy search or blackbox parameter tuning. When NOT to use: Never quench Simulated Annealing with overly fast cooling schedules; the chain freezes into metastable local minima, whereas Parallel Tempering tunnels through barriers."
            ]
        },
        # Cell 1: Theory Markdown
        {
            "cell_type": "markdown",
            "id": "cell_1",
            "metadata": {},
            "source": [
                "## 2. Mathematical Foundations & Governing Equations\n",
                "\n",
                "1. **Gumbel-Sinkhorn Operator:** For score matrix $M \\in \\mathbb{R}^{n \\times n}$ and temperature $\tau$:\n",
                "$$S^0 = \\exp(M / \\tau), \\quad S^{l+1} = \\mathcal{T}_c(\\mathcal{T}_r(S^l)) \\quad \\text{where } \\mathcal{T}_r(S) = S \\oslash (S \\mathbf{1}\\mathbf{1}^T), \\; \\mathcal{T}_c(S) = S \\oslash (\\mathbf{1}\\mathbf{1}^T S)$$\n",
                "2. **Simulated Annealing Acceptance:** $P(\\text{accept}) = \\min(1, \\exp(-\\Delta E / T))$ where $\\Delta E = E_{\\text{prop}} - E_{\\text{curr}}$.\n",
                "3. **Parallel Tempering Replica Swap:** For replicas $i, j$ at temperatures $T_i, T_j$ with $\\beta = 1/T$:\n",
                "$$P_{\\text{swap}} = \\min(1, \\exp((\\beta_i - \\beta_j)(E_i - E_j)))$$\n",
                "4. **Cross-Entropy Method (CEM):** Top elite fraction $\\rho$ update:\n",
                "$$\\mu_{t+1} = \\frac{1}{N_e} \\sum_{i \\in \\mathcal{E}} x_i, \\quad \\sigma_{t+1}^2 = \\frac{1}{N_e} \\sum_{i \\in \\mathcal{E}} (x_i - \\mu_{t+1})^2 + \\epsilon$$"
            ]
        },
        # Cell 2: Imports & Environment Setup
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_2",
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import numpy as np, torch, torch.nn as nn\n",
                "\n",
                "repo_root = Path.cwd().resolve()\n",
                "while not (repo_root / \"core\").exists() and repo_root.parent != repo_root:\n",
                "    repo_root = repo_root.parent\n",
                "if str(repo_root) not in sys.path:\n",
                "    sys.path.insert(0, str(repo_root))\n",
                "import core\n",
                "core.setup_env(seed=42)\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "\n",
                "ckpt_path = repo_root / \"weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.safetensors\"\n",
                "weights, test_in, test_out, meta = core.parity.load_reference_checkpoint(ckpt_path, device=device)\n",
                "print(f\"Device: {device} | Checkpoint: {ckpt_path.name} | Loaded {len(weights)} tensors\")"
            ]
        },
        # Cell 3: Unified Module Definition
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_3",
            "metadata": {},
            "outputs": [],
            "source": [
                "def run_sinkhorn(log_alpha, n_iters=10, tau=1.0):\n",
                "    P = torch.exp(log_alpha / tau)\n",
                "    for _ in range(n_iters):\n",
                "        P = P / (P.sum(dim=-1, keepdim=True) + 1e-12)\n",
                "        P = P / (P.sum(dim=-2, keepdim=True) + 1e-12)\n",
                "    return P\n",
                "\n",
                "def step_cem(candidates, scores, elite_frac=0.25):\n",
                "    n_elite = max(1, int(candidates.shape[0] * elite_frac))\n",
                "    elite_idx = torch.topk(scores, n_elite, largest=False).indices\n",
                "    elite_samples = candidates[elite_idx]\n",
                "    return elite_samples.mean(dim=0), elite_samples.std(dim=0, unbiased=False) + 1e-6\n",
                "\n",
                "def sa_acceptance_prob(delta_e, temp):\n",
                "    return torch.clamp(torch.exp(-delta_e / temp), max=1.0)\n",
                "\n",
                "def parallel_tempering_swap_prob(e1, e2, temp1, temp2):\n",
                "    delta = (1.0 / temp1 - 1.0 / temp2) * (e1 - e2)\n",
                "    return float(torch.clamp(torch.exp(torch.tensor(delta, dtype=torch.float32)), max=1.0).item())\n",
                "\n",
                "class LinearAnchor(nn.Module):\n",
                "    def __init__(self, in_f, out_f):\n",
                "        super().__init__()\n",
                "        self.weight = nn.Parameter(torch.zeros(out_f, in_f))\n",
                "        self.bias = nn.Parameter(torch.zeros(out_f))\n",
                "    def forward(self, x):\n",
                "        return torch.matmul(x, self.weight.T) + self.bias\n",
                "\n",
                "print('CombinatorialUnifiedModule routines defined successfully.')"
            ]
        },
        # Cell 4: Live Verification & Invariant Checks
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_4",
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Sinkhorn\n",
                "sinkhorn_out = run_sinkhorn(weights['log_alpha'], n_iters=10, tau=0.5)\n",
                "err_sinkhorn = float(torch.max(torch.abs(sinkhorn_out - weights['golden_sinkhorn_out'])))\n",
                "\n",
                "# 2. CEM\n",
                "cem_m, cem_s = step_cem(weights['candidates'], weights['scores'], elite_frac=0.25)\n",
                "err_cem = max(float(torch.max(torch.abs(cem_m - weights['golden_cem_mean']))), float(torch.max(torch.abs(cem_s - weights['golden_cem_std']))))\n",
                "\n",
                "# 3. Simulated Annealing\n",
                "sa_probs = sa_acceptance_prob(weights['delta_e'], float(weights['sa_temp'][0]))\n",
                "err_sa = float(torch.max(torch.abs(sa_probs - weights['golden_sa_probs'])))\n",
                "\n",
                "# 4. Parallel Tempering\n",
                "pt_prob = parallel_tempering_swap_prob(float(weights['pt_e1'][0]), float(weights['pt_e2'][0]), float(weights['pt_t1'][0]), float(weights['pt_t2'][0]))\n",
                "err_pt = abs(pt_prob - float(weights['golden_pt_swap_prob'][0]))\n",
                "\n",
                "# 5. Linear Anchor\n",
                "anchor = LinearAnchor(8, 4)\n",
                "anchor.weight.data.copy_(weights['anchor_w'].T)\n",
                "anchor.bias.data.copy_(weights['anchor_b'])\n",
                "anchor_out = anchor(weights['anchor_x']).detach()\n",
                "err_anchor = float(torch.max(torch.abs(anchor_out - weights['golden_anchor_out'])))\n",
                "\n",
                "for name, err in [('Sinkhorn', err_sinkhorn), ('CEM', err_cem), ('SA', err_sa), ('PT', err_pt), ('Anchor', err_anchor)]:\n",
                "    assert err < 1e-5, f'{name} invariant failed: {err:.2e}'\n",
                "print(f'All 5 invariants verified! Sinkhorn: {err_sinkhorn:.2e}, CEM: {err_cem:.2e}, SA: {err_sa:.2e}, PT: {err_pt:.2e}, Anchor: {err_anchor:.2e}')"
            ]
        },
        # Cell 5: Performance Profile & Roofline Benchmark
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_5",
            "metadata": {},
            "outputs": [],
            "source": [
                "track_fns = {\n",
                "    'Linear Anchor': lambda: anchor(weights['anchor_x']),\n",
                "    'Sinkhorn Operator': lambda: run_sinkhorn(weights['log_alpha'], n_iters=10, tau=0.5),\n",
                "    'CEM Step': lambda: step_cem(weights['candidates'], weights['scores']),\n",
                "    'SA Acceptance': lambda: sa_acceptance_prob(weights['delta_e'], float(weights['sa_temp'][0])),\n",
                "    'PT Swap Check': lambda: parallel_tempering_swap_prob(float(weights['pt_e1'][0]), float(weights['pt_e2'][0]), float(weights['pt_t1'][0]), float(weights['pt_t2'][0])),\n",
                "}\n",
                "bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=8)\n",
                "print(core.benchmark.format_benchmark_table(bench_results))"
            ]
        },
        # Cell 6: Active Failure Mode Demonstration
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cell_6",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Active Failure Mode: Fast Quenching in Simulated Annealing vs. Parallel Tempering\n",
                "# Multimodal energy landscape: E(x) = (x^2 - 4)^2 + 0.5*x. Deep barrier at x=0 (E=16).\n",
                "# Global minimum: x ~ -2.0 (E = -1.0). Deceptive local trap: x ~ +2.0 (E = +1.0).\n",
                "def energy_fn(x): return (x**2 - 4.0)**2 + 0.5 * x\n",
                "torch.manual_seed(42)\n",
                "\n",
                "# Fast quenching: starts at x=1.8 near local trap, cooled too rapidly (gamma=0.6):\n",
                "x_quench = torch.tensor(1.8, device=device); T = 2.0\n",
                "for _ in range(40):\n",
                "    prop = x_quench + torch.randn(1, device=device).squeeze() * 0.5\n",
                "    dE = energy_fn(prop) - energy_fn(x_quench)\n",
                "    if dE < 0 or torch.rand(1, device=device).item() < torch.exp(-dE / T).item(): x_quench = prop\n",
                "    T *= 0.6\n",
                "e_quench = energy_fn(x_quench).item()\n",
                "\n",
                "# Parallel Tempering: cold chain T1=0.2, hot barrier-crossing chain T2=8.0:\n",
                "x1, x2 = torch.tensor(1.8, device=device), torch.tensor(1.8, device=device)\n",
                "for _ in range(80):\n",
                "    p1 = x1 + torch.randn(1, device=device).squeeze() * 0.3\n",
                "    if energy_fn(p1) < energy_fn(x1) or torch.rand(1, device=device).item() < torch.exp(-(energy_fn(p1) - energy_fn(x1)) / 0.2).item(): x1 = p1\n",
                "    p2 = x2 + torch.randn(1, device=device).squeeze() * 1.5\n",
                "    if energy_fn(p2) < energy_fn(x2) or torch.rand(1, device=device).item() < torch.exp(-(energy_fn(p2) - energy_fn(x2)) / 8.0).item(): x2 = p2\n",
                "    delta = (1.0 / 0.2 - 1.0 / 8.0) * (energy_fn(x1) - energy_fn(x2)).item()\n",
                "    if delta >= 0 or torch.rand(1, device=device).item() < np.exp(delta): x1, x2 = x2, x1\n",
                "e_pt = energy_fn(x1).item()\n",
                "\n",
                "print(f'[FAILURE DEMONSTRATED] Quenching froze in local trap: x={x_quench.item():.3f}, E={e_quench:.3f}')\n",
                "print(f'[CORRECT] Parallel Tempering crossed barrier to global minimum: x={x1.item():.3f}, E={e_pt:.3f}')\n",
                "assert e_quench > 0.0 and e_pt < -0.5, 'Parallel Tempering failed to escape barrier.'\n",
                "print('[REMEDY] Use Replica-Exchange Parallel Tempering instead of single-chain fast cooling on glassy landscapes.')"
            ]
        },
        # Cell 7: Summary Markdown
        {
            "cell_type": "markdown",
            "id": "cell_7",
            "metadata": {},
            "source": [
                "## 8. Algorithmic Takeaways & Cross-Domain Isomorphisms\n",
                "\n",
                "1. **Birkhoff Polytope Bridges Discrete to Continuous:** Gumbel-Sinkhorn normalizations allow discrete permutation decisions to be trained end-to-end with backpropagation.\n",
                "2. **Replica Exchange Solves Glassy Stalling:** Parallel Tempering uses high-temperature replicas to tunnel through rugged energetic barriers without losing low-temperature refinement.\n",
                "3. **Physical-Statistical Isomorphism:** Thermodynamic annealing maps directly to importance sampling and optimal transport across physical and neural spaces.\n",
                "\n",
                "---\n",
                "**Curriculum Navigation:** [Master Index](file:///c:/Users/Liam/Code/AI-Examples/README.md) | [Previous: 03_coordinate_alternating_projections_and_majorization_minimization](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization.ipynb) | [Next: 05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed.ipynb)"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb


def main():
    nb = build_notebook()
    out_dir = Path("notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out_path} with {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
