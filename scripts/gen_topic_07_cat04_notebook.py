"""Generate the 8-cell educational notebook for Category 04 Topic 07: Learning Rate Schedulers and LR Finder."""

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
                "# Learning Rate Schedulers and LR Finder (C04 Topic 07)\n",
                "\n",
                "**Citations:** Smith (2015 LR Range Test); Loshchilov & Hutter (2016 SGDR/Cosine); Smith (2017 OneCycle); Seetharaman et al. (2020 AutoClip); Hu et al. (2024 MiniCPM/WSD).\n",
                "\n",
                "### Purpose & Topic Bridges\n",
                "While coordinate-wise adaptive optimizers (AdamW) scale individual parameter updates, the global learning rate $\\eta_t$ governs macroeconomic traversal through non-convex loss surfaces. Linear warmup prevents early divergence on random initializations; cosine annealing cools exploration into sharp vs flat generalization basins.\n",
                "\n",
                "**Warmup-Stable-Decay (WSD)** solves the fixed-horizon limitation of cosine schedules by maintaining a stable peak $\\eta_{\\max}$ indefinitely and deferring decay to a declared final token horizon ($T_{\\text{decay}}$). **OneCycleLR** delivers superconvergence via asymmetric cyclical momentum, and **AutoClip** eliminates manual clipping thresholds by dynamically tracking historical gradient norm percentiles.\n",
                "\n",
                "**Decision Boundary:** Use **WSD** for large language models, continual pretraining, and streaming datasets where total token budget is uncertain. Use **Cosine Annealing** for fixed-epoch computer vision/NLP. Use **OneCycleLR** for fast, compute-constrained fine-tuning. When NOT to use: Never step an epoch-based scheduler ($T_{\\max} = \\text{epochs}$) inside a batch loop; this freezes learning prematurely at step $T_{\\max}$."
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
                "1. **Cosine Annealing with Warmup:** $\\eta_t = \\eta_{\\min} + \\frac{t}{T_w}(\\eta_{\\max} - \\eta_{\\min})$ for $t \\le T_w$. For $t > T_w$:\n",
                "$$\\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min})\\left(1 + \\cos\\left(\\pi \\frac{t - T_w}{T - T_w}\\right)\\right)$$\n",
                "2. **Warmup-Stable-Decay (WSD):** Flat plateau $\\eta_t = \\eta_{\\max}$ for $t \\in [T_w, T - T_d]$. In the final $T_d$ steps:\n",
                "$$\\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min})\\left(1 + \\cos\\left(\\pi \\frac{t - (T - T_d)}{T_d}\\right)\\right)$$\n",
                "3. **OneCycleLR:** Initial ramp $\\eta_{\\text{init}} \\to \\eta_{\\max}$ over $T_{\\text{peak}}$, followed by cosine cooldown to $\\eta_{\\min}$.\n",
                "4. **AutoClip:** Adaptive clipping threshold $\\tau_t = \\text{percentile}(\\mathcal{H}, p)$, $\\tilde{g}_t = g_t \\cdot \\min(1.0, \\tau_t / (\\|g_t\\| + \\epsilon))$."
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
                "import math, sys\n",
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
                "ckpt_path = repo_root / \"weights/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/07_learning_rate_schedulers_and_lr_finder.safetensors\"\n",
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
                "def schedule_cosine_warmup(step, total=100, warmup=10, eta_min=1e-5, eta_max=0.01):\n",
                "    if step <= warmup:\n",
                "        return eta_min + (step / max(1, warmup)) * (eta_max - eta_min)\n",
                "    ratio = min(1.0, (step - warmup) / max(1, total - warmup))\n",
                "    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * ratio))\n",
                "\n",
                "def schedule_wsd(step, total=100, warmup=10, decay=20, eta_min=1e-4, eta_max=0.01):\n",
                "    if step <= warmup:\n",
                "        return eta_min + (step / max(1, warmup)) * (eta_max - eta_min)\n",
                "    decay_start = total - decay\n",
                "    if step <= decay_start:\n",
                "        return eta_max\n",
                "    ratio = (step - decay_start) / max(1, decay)\n",
                "    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * ratio))\n",
                "\n",
                "def schedule_onecycle(step, total=100, peak=30, eta_max=0.01, div=25.0, final_div=1000.0):\n",
                "    eta_init, eta_min = eta_max / div, (eta_max / div) / final_div\n",
                "    if step <= peak:\n",
                "        return eta_init + 0.5 * (eta_max - eta_init) * (1.0 - math.cos(math.pi * (step / max(1, peak))))\n",
                "    ratio = (step - peak) / max(1, total - peak)\n",
                "    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * ratio))\n",
                "\n",
                "def step_autoclip(param, grad, norm_history, percentile=90.0, lr=0.01, eps=1e-8):\n",
                "    tau = float(np.percentile(norm_history.cpu().numpy(), percentile))\n",
                "    grad_norm = float(torch.norm(grad).item())\n",
                "    clip_ratio = min(1.0, tau / (grad_norm + eps))\n",
                "    clipped_grad = grad * clip_ratio\n",
                "    return param - lr * clipped_grad, clipped_grad, tau\n",
                "\n",
                "class LinearAnchor(nn.Module):\n",
                "    def __init__(self, in_f, out_f):\n",
                "        super().__init__()\n",
                "        self.weight = nn.Parameter(torch.zeros(out_f, in_f))\n",
                "        self.bias = nn.Parameter(torch.zeros(out_f))\n",
                "    def forward(self, x):\n",
                "        return torch.matmul(x, self.weight.T) + self.bias\n",
                "\n",
                "print('LRSchedulersUnifiedModule routines defined successfully.')"
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
                "# 1. Cosine Annealing with Warmup\n",
                "lr_cos = schedule_cosine_warmup(30, 100, 10, 1e-5, 0.01)\n",
                "p_cos = weights['param_cosine'] - lr_cos * weights['grad_cosine']\n",
                "err_cos = max(abs(lr_cos - float(weights['golden_lr_cosine'][0])), float(torch.max(torch.abs(p_cos - weights['golden_param_cosine']))))\n",
                "\n",
                "# 2. WSD\n",
                "lr_wsd = schedule_wsd(85, 100, 10, 20, 1e-4, 0.01)\n",
                "p_wsd = weights['param_wsd'] - lr_wsd * weights['grad_wsd']\n",
                "err_wsd = max(abs(lr_wsd - float(weights['golden_lr_wsd'][0])), float(torch.max(torch.abs(p_wsd - weights['golden_param_wsd']))))\n",
                "\n",
                "# 3. OneCycleLR\n",
                "lr_oc = schedule_onecycle(40, 100, 30, 0.01, 25.0, 1000.0)\n",
                "p_oc = weights['param_onecycle'] - lr_oc * weights['grad_onecycle']\n",
                "err_oc = max(abs(lr_oc - float(weights['golden_lr_onecycle'][0])), float(torch.max(torch.abs(p_oc - weights['golden_param_onecycle']))))\n",
                "\n",
                "# 4. AutoClip\n",
                "p_ac, clipped_g, _ = step_autoclip(weights['param_autoclip'], weights['grad_autoclip'], weights['norm_history'], percentile=90.0, lr=0.01)\n",
                "err_ac = max(float(torch.max(torch.abs(clipped_g - weights['golden_clipped_grad']))), float(torch.max(torch.abs(p_ac - weights['golden_param_autoclip']))))\n",
                "\n",
                "# 5. Linear Anchor\n",
                "anchor = LinearAnchor(8, 4)\n",
                "anchor.weight.data.copy_(weights['anchor_w'].T)\n",
                "anchor.bias.data.copy_(weights['anchor_b'])\n",
                "anchor_out = anchor(weights['anchor_x']).detach()\n",
                "err_anchor = float(torch.max(torch.abs(anchor_out - weights['golden_anchor_out'])))\n",
                "\n",
                "for name, err in [('Cosine', err_cos), ('WSD', err_wsd), ('OneCycle', err_oc), ('AutoClip', err_ac), ('Anchor', err_anchor)]:\n",
                "    assert err < 1e-5, f'{name} invariant failed: {err:.2e}'\n",
                "print(f'All 5 invariants verified! Cosine: {err_cos:.2e}, WSD: {err_wsd:.2e}, OneCycle: {err_oc:.2e}, AutoClip: {err_ac:.2e}, Anchor: {err_anchor:.2e}')"
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
                "    'Cosine Schedule': lambda: schedule_cosine_warmup(50, 100, 10),\n",
                "    'WSD Schedule': lambda: schedule_wsd(50, 100, 10, 20),\n",
                "    'OneCycle Schedule': lambda: schedule_onecycle(50, 100, 30),\n",
                "    'AutoClip Step': lambda: step_autoclip(weights['param_autoclip'], weights['grad_autoclip'], weights['norm_history']),\n",
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
                "# Active Failure Mode: Stepping Epoch-Configured Scheduler Inside Batch Loop\n",
                "# Consider 10 epochs with 10 batches per epoch (total 100 steps).\n",
                "# If T_max is configured for 10 epochs, stepping it inside the batch loop reaches eta_min at step 10!\n",
                "total_epochs, batches_per_epoch = 10, 10\n",
                "total_batches = total_epochs * batches_per_epoch\n",
                "\n",
                "# Buggy: T_max = total_epochs, stepped inside batch loop\n",
                "lrs_buggy = [schedule_cosine_warmup(t, total=total_epochs, warmup=2, eta_min=1e-5, eta_max=0.01) for t in range(total_batches)]\n",
                "# Correct: T_max = total_batches, stepped inside batch loop\n",
                "lrs_correct = [schedule_cosine_warmup(t, total=total_batches, warmup=20, eta_min=1e-5, eta_max=0.01) for t in range(total_batches)]\n",
                "\n",
                "frozen_ratio = sum(1 for lr in lrs_buggy if lr <= 1.01e-5) / len(lrs_buggy)\n",
                "print(f'[FAILURE DEMONSTRATED] Epoch scheduler stepped inside batch loop: {frozen_ratio*100:.1f}% of training frozen at eta_min!')\n",
                "print(f'[DIAGNOSTIC] Step 10 lr: Buggy = {lrs_buggy[10]:.6f} (frozen!) vs Correct = {lrs_correct[10]:.6f} (active warmup)')\n",
                "assert frozen_ratio > 0.85, 'Buggy schedule failed to demonstrate premature freezing.'\n",
                "print('[REMEDY] Always scale T_max by steps_per_epoch when stepping schedulers per mini-batch.')"
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
                "1. **WSD Eliminates Horizon Bias:** Decoupling stable pretraining from the final decay phase makes Warmup-Stable-Decay the gold standard for large language model foundation training.\n",
                "2. **AutoClip Replaces Hard Constants:** Percentile-based gradient norm clipping dynamically tracks gradient scale changes across diverse architectural depths.\n",
                "3. **Thermal Annealing Isomorphism:** Learning rate cooling maps directly to simulated annealing temperature schedules in statistical physics and combinatorial spin glasses.\n",
                "\n",
                "---\n",
                "**Curriculum Navigation:** [Master Index](file:///c:/Users/Liam/Code/AI-Examples/README.md) | [Previous: 06_schedule_free_and_parameter_free_optimisers](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers/06_schedule_free_and_parameter_free_optimisers.ipynb) | [Next: 02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent](file:///c:/Users/Liam/Code/AI-Examples/notebooks/04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/01_constrained_proximal_frank_wolfe_admm_and_mirror_descent.ipynb)"
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
    out_dir = Path("notebooks/04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "07_learning_rate_schedulers_and_lr_finder.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out_path} with {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
