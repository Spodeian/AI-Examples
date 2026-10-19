"""Generate the 8-cell educational notebook for Category 04 Topic 08 (Subfolder 02 Topic 01):
Constrained, Proximal, Frank-Wolfe, ADMM, and Mirror Descent."""

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
                "# Constrained, Proximal, Frank-Wolfe, ADMM, and Mirror Descent (C04 Topic 08)\n",
                "\n",
                "**Citations:** Frank & Wolfe (1956); Nemirovsky & Yudin (1983 Mirror Descent); Beck & Teboulle (2009 FISTA); Boyd et al. (2011 ADMM).\n",
                "\n",
                "### Purpose & Topic Bridges\n",
                "Standard deep learning operates in unconstrained Euclidean space $\\mathbb{R}^d$. However, real-world models enforce structural manifolds: sparse neural pruning ($L_1$ ball), adversarial perturbation bounds ($L_2/L_\\infty$ balls), probability simplices in attention/game theory ($\\Delta^d$), and orthogonal weights ($W^T W = I$).\n",
                "\n",
                "**Projected Gradient Descent (PGD)** applies Euclidean gradient updates followed by projection onto convex sets. **Frank-Wolfe (Conditional Gradient)** eliminates expensive $\\mathcal{O}(d^3)$ projections by solving a linear minimization oracle $\\min_{s \\in \\mathcal{C}} \\langle s, \\nabla f\\rangle$, generating sparse extreme-point updates. **Proximal ISTA** produces exact mathematical zeros via soft-thresholding on composite non-smooth objectives. **Entropic Mirror Descent** maps updates through the negative entropy Bregman divergence, achieving dimension-independent $\\mathcal{O}(\\sqrt{\\log d})$ convergence on probability simplices.\n",
                "\n",
                "**Decision Boundary:** Use **Frank-Wolfe** for structured matrix completion and nuclear norm balls. Use **ISTA** for sparse feature selection and weight pruning. Use **Mirror Descent** for mixture weights and probability distributions. When NOT to use: Never use unconstrained gradient descent with quadratic penalties $\\lambda \\|h(x)\\|^2$ to enforce hard equality constraints; driving $\\lambda \\to \\infty$ creates condition numbers $\\kappa(H) \\to \\infty$, causing catastrophic numerical stalling. Always use **Augmented Lagrangian** or **ADMM** instead."
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
                "1. **Projected Gradient Descent (PGD):** Trial step $v = x_t - \\eta \\nabla f(x_t)$, projection $x_{t+1} = \\Pi_{\\mathcal{C}}(v) = \\arg\\min_{x \\in \\mathcal{C}} \\|x - v\\|_2^2$. For $L_2$ ball: $x_{t+1} = v \\cdot \\min(1.0, R / \\|v\\|_2)$.\n",
                "2. **Frank-Wolfe (Conditional Gradient):** Linear subproblem $s_t = \\arg\\min_{s \\in \\mathcal{C}} \\langle s, \\nabla f(x_t) \\rangle$. On $L_1$ ball: $s_t = -\\tau \\text{sign}(g_{i^*}) e_{i^*}$ where $i^* = \\arg\\max |g|$. Update: $x_{t+1} = (1 - \\gamma) x_t + \\gamma s_t$.\n",
                "3. **Proximal Gradient (ISTA):** Soft-thresholding $\\mathcal{S}_{\\eta \\lambda}(v_i) = \\text{sign}(v_i) \\max(0, |v_i| - \\eta \\lambda)$ on $v = x_t - \\eta \\nabla f(x_t)$.\n",
                "4. **Entropic Mirror Descent:** Multiplicative update on simplex: $p_{t+1, i} = \\frac{p_{t, i} \\exp(-\\eta g_i)}{\\sum_j p_{t, j} \\exp(-\\eta g_j)}$.\n",
                "5. **ADMM:** Augmented Lagrangian $\\mathcal{L}_\\rho(x, z, u) = f(x) + g(z) + \\frac{\\rho}{2}\\|Ax + Bz - c + u\\|^2$ with alternating $x, z$ steps and dual ascent $u \\leftarrow u + Ax + Bz - c$."
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
                "ckpt_path = repo_root / \"weights/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.safetensors\"\n",
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
                "def step_pgd(param, grad, lr=0.1, radius=1.0, eps=1e-8):\n",
                "    v = param - lr * grad\n",
                "    norm_v = torch.norm(v)\n",
                "    return v if norm_v <= radius else v * (radius / (norm_v + eps))\n",
                "\n",
                "def step_frank_wolfe(param, grad, tau=1.0, gamma=0.2):\n",
                "    idx_max = torch.argmax(torch.abs(grad))\n",
                "    s = torch.zeros_like(param)\n",
                "    s[idx_max] = -tau * torch.sign(grad[idx_max])\n",
                "    return (1.0 - gamma) * param + gamma * s\n",
                "\n",
                "def step_ista(param, grad, lr=0.05, lam=0.1):\n",
                "    v = param - lr * grad\n",
                "    return torch.sign(v) * torch.clamp(torch.abs(v) - (lr * lam), min=0.0)\n",
                "\n",
                "def step_mirror_descent(param, grad, lr=0.1, eps=1e-12):\n",
                "    unnorm = param * torch.exp(-lr * grad)\n",
                "    return unnorm / (torch.sum(unnorm) + eps)\n",
                "\n",
                "class LinearAnchor(nn.Module):\n",
                "    def __init__(self, in_f, out_f):\n",
                "        super().__init__()\n",
                "        self.weight = nn.Parameter(torch.zeros(out_f, in_f))\n",
                "        self.bias = nn.Parameter(torch.zeros(out_f))\n",
                "    def forward(self, x):\n",
                "        return torch.matmul(x, self.weight.T) + self.bias\n",
                "\n",
                "print('ConstrainedOptimUnifiedModule routines defined successfully.')"
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
                "# 1. PGD\n",
                "p_pgd = step_pgd(weights['param_pgd'], weights['grad_pgd'], lr=0.1, radius=1.0)\n",
                "err_pgd = float(torch.max(torch.abs(p_pgd - weights['golden_param_pgd'])))\n",
                "\n",
                "# 2. Frank-Wolfe\n",
                "p_fw = step_frank_wolfe(weights['param_fw'], weights['grad_fw'], tau=1.0, gamma=0.2)\n",
                "err_fw = float(torch.max(torch.abs(p_fw - weights['golden_param_fw'])))\n",
                "\n",
                "# 3. ISTA Proximal\n",
                "p_ista = step_ista(weights['param_ista'], weights['grad_ista'], lr=0.05, lam=0.1)\n",
                "err_ista = float(torch.max(torch.abs(p_ista - weights['golden_param_ista'])))\n",
                "\n",
                "# 4. Entropic Mirror Descent\n",
                "p_md = step_mirror_descent(weights['param_md'], weights['grad_md'], lr=0.1)\n",
                "err_md = float(torch.max(torch.abs(p_md - weights['golden_param_md'])))\n",
                "\n",
                "# 5. Linear Anchor\n",
                "anchor = LinearAnchor(8, 4)\n",
                "anchor.weight.data.copy_(weights['anchor_w'].T)\n",
                "anchor.bias.data.copy_(weights['anchor_b'])\n",
                "anchor_out = anchor(weights['anchor_x']).detach()\n",
                "err_anchor = float(torch.max(torch.abs(anchor_out - weights['golden_anchor_out'])))\n",
                "\n",
                "for name, err in [('PGD', err_pgd), ('Frank-Wolfe', err_fw), ('ISTA', err_ista), ('Mirror Descent', err_md), ('Anchor', err_anchor)]:\n",
                "    assert err < 1e-5, f'{name} invariant failed: {err:.2e}'\n",
                "print(f'All 5 invariants verified! PGD: {err_pgd:.2e}, FW: {err_fw:.2e}, ISTA: {err_ista:.2e}, MD: {err_md:.2e}, Anchor: {err_anchor:.2e}')"
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
                "    'PGD Step': lambda: step_pgd(weights['param_pgd'], weights['grad_pgd'], 0.1, 1.0),\n",
                "    'Frank-Wolfe Step': lambda: step_frank_wolfe(weights['param_fw'], weights['grad_fw'], 1.0, 0.2),\n",
                "    'ISTA Step': lambda: step_ista(weights['param_ista'], weights['grad_ista'], 0.05, 0.1),\n",
                "    'Mirror Descent': lambda: step_mirror_descent(weights['param_md'], weights['grad_md'], 0.1),\n",
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
                "# Active Failure Mode: Quadratic Penalty Method Ill-Conditioning vs. Augmented Lagrangian\n",
                "# Consider min 0.5*(x1^2 + x2^2) subject to x1 + x2 = 1. Analytical solution: x* = [0.5, 0.5].\n",
                "# Penalty method: min 0.5*||x||^2 + 0.5*lambda*(x1 + x2 - 1)^2. As lambda -> 500, cond(H) -> 1000:\n",
                "x_pen = torch.zeros(2, device=device)\n",
                "lam = 500.0  # high penalty required to approach feasibility\n",
                "lr = 0.005   # standard gradient step\n",
                "for _ in range(50):\n",
                "    g = x_pen + lam * (x_pen.sum() - 1.0) * torch.ones(2, device=device)\n",
                "    x_pen = x_pen - lr * g\n",
                "err_penalty = float(torch.norm(x_pen - torch.tensor([0.5, 0.5], device=device)))\n",
                "\n",
                "# Correct: Augmented Lagrangian with finite rho=2.0 and dual multiplier y:\n",
                "x_al = torch.zeros(2, device=device)\n",
                "y = 0.0\n",
                "rho = 2.0\n",
                "for _ in range(10):\n",
                "    # Solve primal subproblem with well-conditioned Hessian (rho=2 -> cond(H) = 5):\n",
                "    for _ in range(10):\n",
                "        g = x_al + (y + rho * (x_al.sum() - 1.0)) * torch.ones(2, device=device)\n",
                "        x_al = x_al - 0.1 * g\n",
                "    y += rho * float(x_al.sum().item() - 1.0)  # Dual ascent update\n",
                "err_al = float(torch.norm(x_al - torch.tensor([0.5, 0.5], device=device)))\n",
                "\n",
                "print(f'[FAILURE DEMONSTRATED] Quadratic penalty (lambda={lam}) error after 50 steps: {err_penalty:.4f} (oscillating ill-conditioned!)')\n",
                "print(f'[CORRECT] Augmented Lagrangian (rho={rho}) error: {err_al:.4e} ({err_penalty/err_al:.1f}x lower error!)')\n",
                "assert err_al < 1e-3 and err_penalty > err_al, 'Augmented Lagrangian failed to outperform ill-conditioned penalty method.'\n",
                "print('[REMEDY] Use Augmented Lagrangian or ADMM with dual multipliers instead of bare quadratic penalties.')"
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
                "1. **Zero-Projection Frank-Wolfe:** By querying extreme atoms $s = \\arg\\min \\langle s, g \\rangle$, Frank-Wolfe avoids $\\mathcal{O}(d^3)$ projections while generating sparse representations.\n",
                "2. **Mirror Descent Dimension Invariance:** Using negative entropy Bregman divergences yields exponential simplex updates scaling as $\\mathcal{O}(\\sqrt{\\log d})$ instead of Euclidean $\\mathcal{O}(\\sqrt{d})$.\n",
                "3. **Dual Multipliers Prevent Ill-Conditioning:** Augmented Lagrangian and ADMM decouple hard constraints from penalty conditioning explosions.\n",
                "\n",
                "---\n",
                "**Curriculum Navigation:** [Master Index](file:///c:/Users/Liam/Code/AI-Examples/README.md) | [Previous: 01_gradient_and_curvature_optimisers/07_learning_rate_schedulers_and_lr_finder](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/07_learning_rate_schedulers_and_lr_finder.ipynb) | [Next: 02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery.ipynb)"
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
    out_path = out_dir / "01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out_path} with {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
