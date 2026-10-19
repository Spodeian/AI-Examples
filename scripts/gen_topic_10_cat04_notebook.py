"""Generate the 8-cell educational notebook for Category 04 Topic 10 (Subfolder 02 Topic 03):
Coordinate, Alternating Projections, and Majorization-Minimization."""

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
                "# Coordinate, Alternating Projections, and Majorization-Minimization (C04 Topic 10)\n",
                "\n",
                "**Citations:** Dykstra (1983 Alternating Projections); Tseng (2001 Block Coordinate Descent); Yuille & Rangarajan (2003 CCCP); Lange (2016 MM Principle).\n",
                "\n",
                "### Purpose & Topic Bridges\n",
                "When objectives or constraints lack simple joint gradients, decomposition methods decompose complex problems into tractable subproblems: **Block Coordinate Descent (BCD)** minimizes one block of coordinates at a time via exact 1D closed-form updates. **Dykstra's Alternating Projections** finds the true Euclidean projection onto an intersection of convex sets $C_1 \\cap C_2$ using dual normal increment corrections ($p_k, q_k$). **Majorization-Minimization (MM)** and the **Concave-Convex Procedure (CCCP)** optimize non-convex or non-smooth objectives by iteratively minimizing locally tight convex surrogate upper bounds with monotonic descent guarantees.\n",
                "\n",
                "**Decision Boundary:** Use **BCD** when coordinate subproblems are closed-form (e.g. Lasso, matrix factorization). Use **Dykstra's Algorithm** when projecting onto intersecting convex sets (e.g. polyhedral cones, hypercubes). Use **MM/IRLS** for non-smooth or difference-of-convex losses. When NOT to use: Never use **Von Neumann Alternating Projections** without dual corrections to find metric projections; Von Neumann stops at an arbitrary feasible point with suboptimal distance."
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
                "1. **Dykstra's Dual Incremental Projections:** Given sets $C_1, C_2$ and starting point $y_0 = x_0, p_0 = 0, q_0 = 0$:\n",
                "$$x_k = \\Pi_{C_1}(y_{k-1} + p_{k-1}), \\quad p_k = y_{k-1} + p_{k-1} - x_k$$\n",
                "$$y_k = \\Pi_{C_2}(x_k + q_{k-1}), \\quad q_k = x_k + q_{k-1} - y_k$$\n",
                "2. **Block Coordinate Descent (BCD):** For quadratic $f(x) = \\frac{1}{2} x^T A x - b^T x$, coordinate update:\n",
                "$$x_i^{(k+1)} = \\frac{b_i - \\sum_{j \\ne i} A_{ij} x_j}{A_{ii}}$$\n",
                "3. **Iteratively Reweighted Least Squares (IRLS):** Majorizes $L_1$ residual $|r|$ via weights $w_i = 1 / (|r_i| + \\epsilon)$:\n",
                "$$x_{k+1} = (w \\cdot t + \\mu x_k) / (w + \\mu)$$\n",
                "4. **Concave-Convex Procedure (CCCP):** Decompose $f(x) = u(x) - v(x)$ with $u, v$ convex. Set $\\nabla u(x_{k+1}) = \\nabla v(x_k)$. For $f(x) = \\frac{1}{2} x^2 - \\ln(1 + x^2)$, this gives $x_{k+1} = \\frac{2 x_k}{1 + x_k^2}$."
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
                "ckpt_path = repo_root / \"weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization.safetensors\"\n",
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
                "def proj_halfspace(x, normal, offset):\n",
                "    dot = torch.dot(x, normal)\n",
                "    if dot <= offset:\n",
                "        return x.clone()\n",
                "    return x - ((dot - offset) / torch.dot(normal, normal)) * normal\n",
                "\n",
                "def proj_box(x, lower, upper):\n",
                "    return torch.clamp(x, min=lower, max=upper)\n",
                "\n",
                "def step_dykstra(y_prev, p_prev, q_prev, normal, offset, lower=0.0, upper=2.0):\n",
                "    x_curr = proj_halfspace(y_prev + p_prev, normal, offset)\n",
                "    p_curr = y_prev + p_prev - x_curr\n",
                "    y_curr = proj_box(x_curr + q_prev, lower, upper)\n",
                "    q_curr = x_curr + q_prev - y_curr\n",
                "    return x_curr, y_curr, p_curr, q_curr\n",
                "\n",
                "def step_bcd(param, b, a_diag, coupling=0.2):\n",
                "    return (b - coupling * param) / a_diag\n",
                "\n",
                "def step_irls(param, target, mu=0.5, eps=1e-4):\n",
                "    r = torch.abs(param - target)\n",
                "    w = 1.0 / (r + eps)\n",
                "    return (w * target + mu * param) / (w + mu)\n",
                "\n",
                "def step_cccp(param):\n",
                "    return (2.0 * param) / (1.0 + param ** 2)\n",
                "\n",
                "class LinearAnchor(nn.Module):\n",
                "    def __init__(self, in_f, out_f):\n",
                "        super().__init__()\n",
                "        self.weight = nn.Parameter(torch.zeros(out_f, in_f))\n",
                "        self.bias = nn.Parameter(torch.zeros(out_f))\n",
                "    def forward(self, x):\n",
                "        return torch.matmul(x, self.weight.T) + self.bias\n",
                "\n",
                "print('CoordinateMMUnifiedModule routines defined successfully.')"
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
                "# 1. Dykstra\n",
                "p0, q0 = torch.zeros_like(weights['x0']), torch.zeros_like(weights['x0'])\n",
                "x1, y1, p1, q1 = step_dykstra(weights['x0'], p0, q0, weights['normal'], float(weights['offset'][0]), 0.0, 2.0)\n",
                "err_dykstra = max(float(torch.max(torch.abs(x1 - weights['golden_dykstra_x']))), float(torch.max(torch.abs(y1 - weights['golden_dykstra_y']))), float(torch.max(torch.abs(p1 - weights['golden_dykstra_p']))), float(torch.max(torch.abs(q1 - weights['golden_dykstra_q']))))\n",
                "\n",
                "# 2. BCD\n",
                "p_bcd = step_bcd(weights['param_bcd'], weights['b_bcd'], weights['A_diag'], 0.2)\n",
                "err_bcd = float(torch.max(torch.abs(p_bcd - weights['golden_param_bcd'])))\n",
                "\n",
                "# 3. IRLS\n",
                "p_irls = step_irls(weights['param_irls'], weights['target_irls'], 0.5, 1e-4)\n",
                "err_irls = float(torch.max(torch.abs(p_irls - weights['golden_param_irls'])))\n",
                "\n",
                "# 4. CCCP\n",
                "p_cccp = step_cccp(weights['param_cccp'])\n",
                "err_cccp = float(torch.max(torch.abs(p_cccp - weights['golden_param_cccp'])))\n",
                "\n",
                "# 5. Linear Anchor\n",
                "anchor = LinearAnchor(8, 4)\n",
                "anchor.weight.data.copy_(weights['anchor_w'].T)\n",
                "anchor.bias.data.copy_(weights['anchor_b'])\n",
                "anchor_out = anchor(weights['anchor_x']).detach()\n",
                "err_anchor = float(torch.max(torch.abs(anchor_out - weights['golden_anchor_out'])))\n",
                "\n",
                "for name, err in [('Dykstra', err_dykstra), ('BCD', err_bcd), ('IRLS', err_irls), ('CCCP', err_cccp), ('Anchor', err_anchor)]:\n",
                "    assert err < 1e-5, f'{name} invariant failed: {err:.2e}'\n",
                "print(f'All 5 invariants verified! Dykstra: {err_dykstra:.2e}, BCD: {err_bcd:.2e}, IRLS: {err_irls:.2e}, CCCP: {err_cccp:.2e}, Anchor: {err_anchor:.2e}')"
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
                "    'Dykstra Proj Step': lambda: step_dykstra(weights['x0'], p0, q0, weights['normal'], float(weights['offset'][0])),\n",
                "    'BCD Coord Step': lambda: step_bcd(weights['param_bcd'], weights['b_bcd'], weights['A_diag']),\n",
                "    'IRLS Weight Step': lambda: step_irls(weights['param_irls'], weights['target_irls']),\n",
                "    'CCCP MM Step': lambda: step_cccp(weights['param_cccp']),\n",
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
                "# Active Failure Mode: Von Neumann Alternating Projections Failure on Metric Projections\n",
                "# Consider intersecting convex halfspaces: C1 = {x | x1 + 2*x2 <= 0}, C2 = {x | 2*x1 + x2 <= 0}.\n",
                "# Starting from x0 = (3.0, 1.0), find the closest Euclidean projection in C1 cap C2.\n",
                "a1 = torch.tensor([1.0, 2.0], device=device); a2 = torch.tensor([2.0, 1.0], device=device)\n",
                "x0_pt = torch.tensor([3.0, 1.0], device=device)\n",
                "\n",
                "# Von Neumann Projections (no dual normal vector accumulation):\n",
                "y_vn = x0_pt.clone()\n",
                "for _ in range(50):\n",
                "    x_vn = proj_halfspace(y_vn, a1, 0.0)\n",
                "    y_vn = proj_halfspace(x_vn, a2, 0.0)\n",
                "dist_vn = float(torch.sum((y_vn - x0_pt)**2).item())\n",
                "\n",
                "# Dykstra's Alternating Projections (with dual increment corrections):\n",
                "y_dyk, p_d, q_d = x0_pt.clone(), torch.zeros(2, device=device), torch.zeros(2, device=device)\n",
                "for _ in range(50):\n",
                "    x_dyk = proj_halfspace(y_dyk + p_d, a1, 0.0); p_d = y_dyk + p_d - x_dyk\n",
                "    y_dyk = proj_halfspace(x_dyk + q_d, a2, 0.0); q_d = x_dyk + q_d - y_dyk\n",
                "dist_dyk = float(torch.sum((y_dyk - x0_pt)**2).item())\n",
                "\n",
                "print(f'[FAILURE DEMONSTRATED] Von Neumann alternating projections stopped prematurely at non-optimal point: {y_vn.cpu().numpy().round(3)}, dist^2={dist_vn:.4f}')\n",
                "print(f'[CORRECT] Dykstra dual increments discovered true Euclidean projection: {y_dyk.cpu().numpy().round(3)}, dist^2={dist_dyk:.4f}')\n",
                "assert dist_vn > dist_dyk + 1.0, 'Von Neumann failed to exhibit distance suboptimality.'\n",
                "print('[REMEDY] Always use Dykstra dual increments (p, q) when computing metric projections onto intersecting convex sets.')"
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
                "1. **Dual Corrections Restore Metric Projections:** Dykstra's algorithm prevents path drift by re-injecting normal cone components discarded during sequential projections.\n",
                "2. **Decomposition Beats Joint Gradients:** Block coordinate descent achieves instant superlinear convergence on problems with separable coordinate structures.\n",
                "3. **MM Unifies Diverse Algorithms:** Expectation-Maximization (EM), IRLS, and CCCP are all special cases of Majorization-Minimization with monotonic descent guarantees.\n",
                "\n",
                "---\n",
                "**Curriculum Navigation:** [Master Index](file:///c:/Users/Liam/Code/AI-Examples/README.md) | [Previous: 02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.ipynb) | [Next: 04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn.ipynb)"
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
    out_path = out_dir / "03_coordinate_alternating_projections_and_majorization_minimization.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out_path} with {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
