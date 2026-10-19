"""Generate the 8-cell educational notebook for Category 04 Topic 09 (Subfolder 02 Topic 02):
Multi-Objective Pareto, MGDA, Tchebycheff, and Gradient Surgery."""

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
                "# Multi-Objective Pareto, MGDA, Tchebycheff, and Gradient Surgery (C04 Topic 09)\n",
                "\n",
                "**Citations:** Miettinen (1999 Tchebycheff); Désidéri (2012 MGDA); Sener & Koltun (2018 Multi-Task MOO); Yu et al. (2020 PCGrad); Liu et al. (2021 CAGrad).\n",
                "\n",
                "### Purpose & Topic Bridges\n",
                "In multi-task deep architectures (e.g. multi-modal encoders, perception + control trunks), optimizing tasks simultaneously via linear loss weighting $\\sum w_i \\mathcal{L}_i$ induces destructive gradient conflicts: $\\langle \\nabla \\mathcal{L}_i, \\nabla \\mathcal{L}_j \\rangle < 0$. Improving one task degrades the other, causing task dominance or catastrophic interference.\n",
                "\n",
                "**Multiple Gradient Descent Algorithm (MGDA)** computes the minimum-norm vector in the convex hull of task gradients, identifying a common descent direction guaranteed not to worsen any task. **PCGrad (Projecting Conflicting Gradients)** projects conflicting task gradients onto each other's tangent planes, eliminating negative interference. **Augmented Tchebycheff Scalarization** minimizes $L_\\infty$ distance to a utopian reference point, unlocking the ability to find every point along non-convex Pareto fronts.\n",
                "\n",
                "**Decision Boundary:** Use **PCGrad** for fast multi-task training with conflicting gradients. Use **MGDA** when tasks must improve simultaneously without manual weighting. Use **Tchebycheff** when exploring the full Pareto front. When NOT to use: Never use **Linear Scalarization** on non-convex/concave Pareto surfaces; linear hyperplanes only touch extreme boundary corners, completely missing all balanced interior trade-offs."
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
                "1. **MGDA Min-Norm Convex Combination:** $\\min_{\\alpha \\in \\Delta^m} \\|\\sum_{i=1}^m \\alpha_i g_i\\|_2^2$. For 2 tasks ($u = g_1 - g_2$):\n",
                "$$\\alpha^* = \\text{clamp}\\left(\\frac{\\langle g_2 - g_1, g_2\\rangle}{\\|g_1 - g_2\\|^2}, 0, 1\\right), \\quad g_{\\text{MGDA}} = \\alpha^* g_1 + (1 - \\alpha^*) g_2$$\n",
                "2. **PCGrad Conflict Projection:** If $\\langle g_1, g_2\\rangle < 0$:\n",
                "$$g_1 \\leftarrow g_1 - \\frac{\\langle g_1, g_2\\rangle}{\\|g_2\\|^2} g_2, \\quad g_2 \\leftarrow g_2 - \\frac{\\langle g_2, g_1\\rangle}{\\|g_1\\|^2} g_1, \\quad g_{\\text{PC}} = \\frac{1}{2}(g_1 + g_2)$$\n",
                "3. **Augmented Tchebycheff:** Minimizes distance to ideal point $z^*$:\n",
                "$$S(L) = \\max_{i} w_i (L_i - z_i^*) + \\rho \\sum_i w_i (L_i - z_i^*)$$\n",
                "Subgradient: $\\nabla_\\theta S = (w_{i^*} e_{i^*} + \\rho w)^T \\nabla_\\theta L$ where $i^* = \\arg\\max_i w_i (L_i - z_i^*)$."
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
                "ckpt_path = repo_root / \"weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.safetensors\"\n",
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
                "def step_mgda_2task(param, g1, g2, lr=0.01, eps=1e-12):\n",
                "    u = g1 - g2\n",
                "    alpha = float(torch.clamp(torch.dot(g2 - g1, g2) / (torch.dot(u, u) + eps), 0.0, 1.0).item())\n",
                "    g_mgda = alpha * g1 + (1.0 - alpha) * g2\n",
                "    return param - lr * g_mgda, g_mgda, alpha\n",
                "\n",
                "def step_pcgrad_2task(param, g1, g2, lr=0.01, eps=1e-12):\n",
                "    dot_12 = torch.dot(g1, g2)\n",
                "    if dot_12 < 0.0:\n",
                "        g1_proj = g1 - (dot_12 / (torch.dot(g2, g2) + eps)) * g2\n",
                "        g2_proj = g2 - (dot_12 / (torch.dot(g1, g1) + eps)) * g1\n",
                "        g_pc = 0.5 * (g1_proj + g2_proj)\n",
                "    else:\n",
                "        g_pc = 0.5 * (g1 + g2)\n",
                "    return param - lr * g_pc, g_pc\n",
                "\n",
                "def step_tchebycheff(param, g1, g2, losses, z_star, weights_tch, rho=0.05, lr=0.01):\n",
                "    diff = weights_tch * (losses - z_star)\n",
                "    idx_max = torch.argmax(diff)\n",
                "    ds_dl = rho * weights_tch.clone()\n",
                "    ds_dl[idx_max] += weights_tch[idx_max]\n",
                "    g_tch = ds_dl[0] * g1 + ds_dl[1] * g2\n",
                "    return param - lr * g_tch, g_tch\n",
                "\n",
                "def step_linear_scalarization(param, g1, g2, weights_lin, lr=0.01):\n",
                "    g_lin = weights_lin[0] * g1 + weights_lin[1] * g2\n",
                "    return param - lr * g_lin, g_lin\n",
                "\n",
                "class LinearAnchor(nn.Module):\n",
                "    def __init__(self, in_f, out_f):\n",
                "        super().__init__()\n",
                "        self.weight = nn.Parameter(torch.zeros(out_f, in_f))\n",
                "        self.bias = nn.Parameter(torch.zeros(out_f))\n",
                "    def forward(self, x):\n",
                "        return torch.matmul(x, self.weight.T) + self.bias\n",
                "\n",
                "print('MultiObjectiveUnifiedModule routines defined successfully.')"
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
                "# 1. MGDA\n",
                "p_mgda, g_mgda, alpha = step_mgda_2task(weights['param'], weights['g1'], weights['g2'], lr=0.01)\n",
                "err_mgda = max(abs(alpha - float(weights['golden_alpha_mgda'][0])), float(torch.max(torch.abs(g_mgda - weights['golden_g_mgda']))), float(torch.max(torch.abs(p_mgda - weights['golden_param_mgda']))))\n",
                "\n",
                "# 2. PCGrad\n",
                "p_pc, g_pc = step_pcgrad_2task(weights['param'], weights['g1'], weights['g2'], lr=0.01)\n",
                "err_pc = max(float(torch.max(torch.abs(g_pc - weights['golden_g_pc']))), float(torch.max(torch.abs(p_pc - weights['golden_param_pc']))))\n",
                "\n",
                "# 3. Tchebycheff Scalarization\n",
                "p_tch, g_tch = step_tchebycheff(weights['param'], weights['g1'], weights['g2'], weights['L'], weights['z_star'], weights['weights_tch'], rho=0.05, lr=0.01)\n",
                "err_tch = max(float(torch.max(torch.abs(g_tch - weights['golden_g_tch']))), float(torch.max(torch.abs(p_tch - weights['golden_param_tch']))))\n",
                "\n",
                "# 4. Linear Scalarization\n",
                "p_lin, g_lin = step_linear_scalarization(weights['param'], weights['g1'], weights['g2'], weights['weights_lin'], lr=0.01)\n",
                "err_lin = max(float(torch.max(torch.abs(g_lin - weights['golden_g_lin']))), float(torch.max(torch.abs(p_lin - weights['golden_param_lin']))))\n",
                "\n",
                "# 5. Linear Anchor\n",
                "anchor = LinearAnchor(8, 4)\n",
                "anchor.weight.data.copy_(weights['anchor_w'].T)\n",
                "anchor.bias.data.copy_(weights['anchor_b'])\n",
                "anchor_out = anchor(weights['anchor_x']).detach()\n",
                "err_anchor = float(torch.max(torch.abs(anchor_out - weights['golden_anchor_out'])))\n",
                "\n",
                "for name, err in [('MGDA', err_mgda), ('PCGrad', err_pc), ('Tchebycheff', err_tch), ('Linear', err_lin), ('Anchor', err_anchor)]:\n",
                "    assert err < 1e-5, f'{name} invariant failed: {err:.2e}'\n",
                "print(f'All 5 invariants verified! MGDA: {err_mgda:.2e}, PCGrad: {err_pc:.2e}, Tchebycheff: {err_tch:.2e}, Linear: {err_lin:.2e}, Anchor: {err_anchor:.2e}')"
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
                "    'MGDA Step': lambda: step_mgda_2task(weights['param'], weights['g1'], weights['g2']),\n",
                "    'PCGrad Step': lambda: step_pcgrad_2task(weights['param'], weights['g1'], weights['g2']),\n",
                "    'Tchebycheff Step': lambda: step_tchebycheff(weights['param'], weights['g1'], weights['g2'], weights['L'], weights['z_star'], weights['weights_tch']),\n",
                "    'Linear Multi-Task': lambda: step_linear_scalarization(weights['param'], weights['g1'], weights['g2'], weights['weights_lin']),\n",
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
                "# Active Failure Mode: Linear Scalarization Blindness on Concave Pareto Fronts\n",
                "# Consider f1(theta) = theta, f2(theta) = 1 - theta^2 for theta in [0, 1]. Pareto front f2 = 1 - f1^2 is strictly concave.\n",
                "# Linear scalarization: min w1*theta + w2*(1 - theta^2) has negative curvature (d2/dtheta2 = -2*w2 < 0).\n",
                "# For any w1, w2 > 0, linear scalarization collapses to corner boundaries (theta=0 or theta=1), missing theta=0.618!\n",
                "theta_lin = torch.tensor(0.55, requires_grad=True, device=device)\n",
                "w1, w2 = 0.5, 0.5\n",
                "for _ in range(30):\n",
                "    loss_lin = w1 * theta_lin + w2 * (1.0 - theta_lin**2); loss_lin.backward()\n",
                "    with torch.no_grad():\n",
                "        theta_lin -= 0.1 * theta_lin.grad; theta_lin.clamp_(0.0, 1.0); theta_lin.grad.zero_()\n",
                "corner_collapse = float(theta_lin.item()) in (0.0, 1.0)\n",
                "\n",
                "# Correct: Augmented Tchebycheff scalarization with z* = [0, 0], w = [0.5, 0.5]:\n",
                "theta_tch = torch.tensor(0.2, requires_grad=True, device=device)\n",
                "for _ in range(40):\n",
                "    f1, f2 = theta_tch, 1.0 - theta_tch**2\n",
                "    loss_tch = torch.maximum(0.5 * f1, 0.5 * f2) + 0.05 * (0.5 * f1 + 0.5 * f2); loss_tch.backward()\n",
                "    with torch.no_grad():\n",
                "        theta_tch -= 0.05 * theta_tch.grad; theta_tch.clamp_(0.0, 1.0); theta_tch.grad.zero_()\n",
                "balanced_solution = float(theta_tch.item())\n",
                "\n",
                "print(f'[FAILURE DEMONSTRATED] Linear scalarization collapsed to extreme boundary corner: theta={theta_lin.item():.1f}!')\n",
                "print(f'[CORRECT] Augmented Tchebycheff discovered balanced interior compromise: theta={balanced_solution:.4f} (f1={balanced_solution:.4f}, f2={1-balanced_solution**2:.4f})')\n",
                "assert corner_collapse and 0.55 < balanced_solution < 0.65, 'Tchebycheff failed to discover balanced non-convex Pareto point.'\n",
                "print('[REMEDY] Use Augmented Tchebycheff or Nash bargaining scalarization when Pareto front convexity is unverified.')"
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
                "1. **Gradient Surgery Prevents Negative Transfer:** PCGrad projects away conflicting components $\\langle g_i, g_j \\rangle < 0$, ensuring simultaneous task progression in shared trunks.\n",
                "2. **Tchebycheff Reaches Non-Convex Surfaces:** $L_\\infty$ level sets are right angles, enabling exploration of concave Pareto surfaces inaccessible to linear scalarization.\n",
                "3. **Game-Theoretic Isomorphism:** Multi-task Pareto optimization maps directly to cooperative Nash bargaining and Wardrop equilibria in multi-agent economic transport networks.\n",
                "\n",
                "---\n",
                "**Curriculum Navigation:** [Master Index](file:///c:/Users/Liam/Code/AI-Examples/README.md) | [Previous: 01_constrained_proximal_frank_wolfe_admm_and_mirror_descent](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.ipynb) | [Next: 03_coordinate_alternating_projections_and_majorization_minimization](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/03_coordinate_alternating_projections_and_majorization_minimization.ipynb)"
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
    out_path = out_dir / "02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out_path} with {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
