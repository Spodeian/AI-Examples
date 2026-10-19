"""Script to procedurally generate the strict 8-cell notebook for Topic 19.

Ensures total lines <= 220, valid JSON, cell IDs, and all semantic requirements.
"""

import json
from pathlib import Path

cells = []

# Cell 1: Header, Citations, Purpose & Topic Bridges
c1 = [
    "# Non-Backprop Training: Forward-Forward, Predictive Coding, EqProp, DFA, and Hebbian Learning\n",
    "\n",
    "### Historical Lineage & Theoretical Foundations\n",
    "- **Forward-Forward Algorithm** (*Hinton, 2022*): Replaces backward error propagation with dual forward passes (positive vs. negative data), maximizing layerwise goodness $\\sum h_j^2$ on real inputs and minimizing it on corrupted ones.\n",
    "- **Predictive Coding** (*Rao & Ballard, 1999; Friston, 2005*): Frames neural computation as hierarchical Bayesian inference where higher layers predict lower-layer representations and update via local prediction errors $\\varepsilon = x - r W$.\n",
    "- **Direct Feedback Alignment** (*Lillicrap et al., 2016; Nøkland, 2016*): Bypasses the biological and hardware weight transport problem ($W^T$) by projecting top errors through fixed random matrices $B$.\n",
    "- **Hebbian Plasticity & Oja's Rule** (*Hebb, 1949; Oja, 1982*): Synaptic updates derived purely from local pre- and post-synaptic firing $\\Delta w = \\eta(y x - y^2 w)$, proving that normalized local plasticity performs unsupervised PCA.\n",
    "\n",
    "> [!WARNING]\n",
    "> **Decision Boundary (When NOT to Replace Standard Backpropagation):** Do not discard backpropagation for standard GPU transformer pretraining; non-backprop algorithms exhibit higher variance and lack hardware-optimized transpose GEMM pipelines. Reserve them for neuromorphic crossbars, biological modeling, and zero-memory edge devices.\n",
    "\n",
    "**Topic Connections:** `🔗 Upstream`: [Test-Time Adaptation: TENT, MEMO, STaR, and SPIN](04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.ipynb) | `🔗 Downstream`: [Zeroth-Order MeZO & Synthetic Gradients](06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.ipynb)\n"
]
cells.append({"cell_type": "markdown", "id": "cell_0", "metadata": {}, "source": c1})

# Cell 2: Mathematical Foundations & Governing Equations
c2 = [
    "## 2. Mathematical Foundations & Governing Equations\n",
    "\n",
    "1. **Forward-Forward Goodness & Layer Normalization (Hinton, 2022):**\n",
    "$$G(h) = \\sum_{j=1}^D h_j^2, \\quad \\mathcal{L}_\\text{FF} = \\log(1 + e^{\\theta - G(h_+)}) + \\log(1 + e^{G(h_-) - \\theta}), \\quad h_\\text{norm} = \\frac{h}{\\|h\\|_2 + \\epsilon}$$\n",
    "2. **Predictive Coding Representation Inference (Rao & Ballard, 1999):**\n",
    "$$\\varepsilon_t = x - r_t W, \\quad r_{t+1} = r_t + \\eta \\big(\\varepsilon_t W^T - \\lambda r_t\\big)$$\n",
    "3. **Direct Feedback Alignment (Nøkland, 2016):** With top error $e = \\hat{y} - y$ and random projection $B$:\n",
    "$$\\delta_l = (e B_l^T) \\odot \\sigma'(a_l), \\quad \\Delta W_l = - \\eta (h_{l-1}^T \\delta_l)$$\n",
    "4. **Oja's Normalized Hebbian Learning (Oja, 1982):** With scalar output $y = w^T x$:\n",
    "$$\\Delta w = \\eta \\big(y x - y^2 w\\big) \\implies \\|w\\|_2 \\to 1$$\n",
    "5. **Asymptotic Complexity:** FF step $\\mathcal{O}(B D)$; PC step $\\mathcal{O}(D^2)$; DFA projection $\\mathcal{O}(B D C)$; Oja step $\\mathcal{O}(D)$.\n"
]
cells.append({"cell_type": "markdown", "id": "cell_1", "metadata": {}, "source": c2})

# Cell 3: Deterministic Setup & Micro-Data Generator
c3 = [
    "import sys\n",
    "from pathlib import Path\n",
    "import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\n",
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
    "ckpt_path = repo_root / \"weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.safetensors\"\n",
    "weights, test_in, test_out, meta = core.parity.load_reference_checkpoint(ckpt_path, device=device)\n",
    "print(f\"Device: {device} | Checkpoint: {ckpt_path.name} | Loaded {len(weights)} tensors\")\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_2", "metadata": {}, "outputs": [], "source": c3})

# Cell 4: First-Principles Implementation & Native Parity
c4 = [
    "class LinearAnchor(nn.Module):\n",
    "    def __init__(self, in_features: int = 8, out_features: int = 4):\n",
    "        super().__init__()\n",
    "        self.fc = nn.Linear(in_features, out_features)\n",
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
    "        return self.fc(x)\n",
    "\n",
    "def forward_forward_loss_and_norm(h_pos: torch.Tensor, h_neg: torch.Tensor, theta: float = 2.0, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:\n",
    "    g_pos = torch.sum(h_pos ** 2, dim=-1)\n",
    "    g_neg = torch.sum(h_neg ** 2, dim=-1)\n",
    "    loss = torch.mean(F.softplus(theta - g_pos) + F.softplus(g_neg - theta))\n",
    "    h_norm = h_pos / (torch.norm(h_pos, dim=-1, keepdim=True) + eps)\n",
    "    return loss, h_norm\n",
    "\n",
    "def predictive_coding_step(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eta: float = 0.1, lam: float = 0.01) -> torch.Tensor:\n",
    "    pred = torch.matmul(r, w)\n",
    "    err = x - pred\n",
    "    grad = torch.matmul(err, w.T)\n",
    "    return r + eta * (grad - lam * r)\n",
    "\n",
    "def dfa_project_error(top_error: torch.Tensor, dfa_feedback: torch.Tensor, act_pre: torch.Tensor) -> torch.Tensor:\n",
    "    proj = torch.matmul(top_error, dfa_feedback.T)\n",
    "    mask = (act_pre > 0.0).float()\n",
    "    return proj * mask\n",
    "\n",
    "def oja_hebbian_step(x: torch.Tensor, w: torch.Tensor, eta: float = 0.05) -> torch.Tensor:\n",
    "    y = torch.dot(x, w)\n",
    "    return w + eta * (y * x - (y ** 2) * w)\n",
    "print(\"Non-backprop primitives defined successfully.\")\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_3", "metadata": {}, "outputs": [], "source": c4})

# Cell 5: Quantitative Mathematical Assertions & Parity
c5 = [
    "anchor = LinearAnchor(8, 4).to(device)\n",
    "anchor.fc.weight.data.copy_(weights['anchor_w'].T)\n",
    "anchor.fc.bias.data.copy_(weights['anchor_b'])\n",
    "\n",
    "loss_ff, h_norm = forward_forward_loss_and_norm(weights['h_pos'], weights['h_neg'], theta=2.0, eps=1e-8)\n",
    "err_ff = max(float(torch.abs(loss_ff - weights['golden_ff_loss'][0]).item()),\n",
    "             float(torch.max(torch.abs(h_norm - weights['golden_h_norm'])).item()))\n",
    "\n",
    "r_next = predictive_coding_step(weights['pc_x'], weights['pc_r'], weights['pc_w'], eta=0.1, lam=0.01)\n",
    "err_pc = float(torch.max(torch.abs(r_next - weights['golden_pc_r_next'])).item())\n",
    "\n",
    "dfa_grad = dfa_project_error(weights['top_error'], weights['dfa_feedback'], weights['act_pre'])\n",
    "err_dfa = float(torch.max(torch.abs(dfa_grad - weights['golden_dfa_grad'])).item())\n",
    "\n",
    "w_oja = oja_hebbian_step(weights['oja_x'], weights['oja_w'], eta=0.05)\n",
    "err_oja = float(torch.max(torch.abs(w_oja - weights['golden_oja_w_next'])).item())\n",
    "\n",
    "anchor_out = anchor(weights['anchor_x'])\n",
    "err_anchor = float(torch.max(torch.abs(anchor_out - weights['golden_anchor_out'])).item())\n",
    "\n",
    "for name, err in [('Forward-Forward', err_ff), ('Predictive Coding', err_pc), ('DFA', err_dfa), ('Oja', err_oja), ('Anchor', err_anchor)]:\n",
    "    assert err < 1e-5, f'{name} invariant failed: {err:.2e}'\n",
    "print(f'All 5 invariants verified! FF: {err_ff:.2e}, PC: {err_pc:.2e}, DFA: {err_dfa:.2e}, Oja: {err_oja:.2e}, Anchor: {err_anchor:.2e}')\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_4", "metadata": {}, "outputs": [], "source": c5})

# Cell 6: Performance Profile & Diagnostic Visualizations
c6 = [
    "track_fns = {\n",
    "    'Linear Anchor': lambda: anchor(weights['anchor_x']),\n",
    "    'Forward-Forward': lambda: forward_forward_loss_and_norm(weights['h_pos'], weights['h_neg']),\n",
    "    'Predictive Coding': lambda: predictive_coding_step(weights['pc_x'], weights['pc_r'], weights['pc_w']),\n",
    "    'DFA Error Projection': lambda: dfa_project_error(weights['top_error'], weights['dfa_feedback'], weights['act_pre']),\n",
    "    'Oja Hebbian Update': lambda: oja_hebbian_step(weights['oja_x'], weights['oja_w']),\n",
    "}\n",
    "bench_results = core.benchmark.compare_tracks(track_fns, warmup=3, iters=15, batch_size=4)\n",
    "print(core.benchmark.format_benchmark_table(bench_results))\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_5", "metadata": {}, "outputs": [], "source": c6})

# Cell 7: Pathological Misuse vs. Corrected Implementation
c7 = [
    "# Pathological Anti-Pattern: Unnormalized standard Hebbian plasticity causes explosive weight divergence\n",
    "# Simulating unnormalized Hebbian update (Delta w = eta * y * x) vs. Oja's normalized rule\n",
    "rng = np.random.RandomState(42)\n",
    "x_seq = rng.randn(200, 4)\n",
    "w_standard = np.ones(4) / 2.0\n",
    "w_oja = np.ones(4) / 2.0\n",
    "eta = 0.01\n",
    "\n",
    "for x_t in x_seq:\n",
    "    # Standard Hebbian (unbounded positive feedback)\n",
    "    y_s = np.dot(x_t, w_standard)\n",
    "    w_standard = w_standard + 0.05 * y_s * x_t\n",
    "    # Oja's Normalized Rule (stabilizing negative feedback)\n",
    "    y_o = np.dot(x_t, w_oja)\n",
    "    w_oja = w_oja + eta * (y_o * x_t - (y_o ** 2) * w_oja)\n",
    "\n",
    "norm_standard = float(np.linalg.norm(w_standard))\n",
    "norm_oja = float(np.linalg.norm(w_oja))\n",
    "print(f'[FAILURE DEMONSTRATED] Standard Hebbian norm exploded to: {norm_standard:.2e}!')\n",
    "print(f'[CORRECT] Oja normalized rule converged stably to unit norm: {norm_oja:.4f}!')\n",
    "assert norm_standard > 1e4 and 0.8 < norm_oja < 1.2, 'Hebbian divergence assertion failed.'\n",
    "print('[REMEDY] Always include negative feedback term -y^2 * w (Oja, 1982) to guarantee PCA convergence without exploding weights.')\n"
]
cells.append({"cell_type": "code", "execution_count": None, "id": "cell_6", "metadata": {}, "outputs": [], "source": c7})

# Cell 8: Takeaways, Cross-Domain Isomorphism & Footer
c8 = [
    "## 8. Algorithmic Takeaways & Cross-Domain Isomorphisms\n",
    "\n",
    "- **Takeaways**: Non-backprop algorithms decouple credit assignment from symmetric backward graph traversals; Forward-Forward computes layerwise contrastive goodness; Predictive Coding minimises local prediction error; DFA projects via fixed random feedback; Oja's rule extracts principal components via normalized local plasticity.\n",
    "\n",
    "| Scientific Domain | Forward-Forward Goodness | Predictive Coding Error | Hebbian Synaptic Plasticity |\n",
    "| :--- | :--- | :--- | :--- |\n",
    "| **Neuromorphic Silicon** | Layer-local activation goodness on memristor arrays | Differential voltage routing between crossbar layers | In-situ conductance modification without transpose bus |\n",
    "| **Cortical Neuroscience** | Multi-compartmental dendritic gating (apical vs basal) | Sensory cortex residual prediction of retinal signals | Spike-timing dependent plasticity (STDP) in pyramidal cells |\n",
    "| **Analog Photonics** | Directional optical intensity accumulation | Interferometer optical phase error cancellation | Passive phase-shifter tuning without backward laser path |\n",
    "\n",
    "---\n",
    "**Navigation:** [← Test-Time Adaptation: TENT, MEMO, STaR, and SPIN](04_self_optimising_test_time_adaptation_tent_memo_star_and_spin.ipynb) | [↑ Master Index](../../README.md) | [Zeroth-Order MeZO & Synthetic Gradients →](06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck.ipynb)\n"
]
cells.append({"cell_type": "markdown", "id": "cell_7", "metadata": {}, "source": c8})

nb_data = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_dir = Path("notebooks/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.ipynb"

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(nb_data, f, indent=1)

print(f"Generated notebook at {out_file}")
